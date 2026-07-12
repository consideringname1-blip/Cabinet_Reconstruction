from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from artifact_layout import REALTIME_TRACKING_ROOT
from config import MAX_REALTIME_TRACKED_DISPLAY_OBJECTS


MODE_LIVE = "live"
MODE_HISTORY = "history"
VALID_MODES = {MODE_LIVE, MODE_HISTORY}


@dataclass(frozen=True)
class TrackingJobToken:
    display_object_id: str
    observation_seq: int
    tracking_epoch: int
    mode_epoch: int
    model_revision: int
    hololens_pose_revision: int
    startup_session_id: str
    ingress_session_id: str


@dataclass
class PendingObservation:
    token: TrackingJobToken
    payload: dict[str, Any]
    enqueued_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class ObjectRuntimeState:
    display_object_id: str
    tracking_epoch: int = 0
    next_observation_seq: int = 0
    pending: PendingObservation | None = None
    latest_seen: PendingObservation | None = None
    running: TrackingJobToken | None = None
    last_accepted_seq: int = 0
    last_attempted_seq: int = 0
    activated_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class SessionModeState:
    startup_session_id: str
    mode: str = MODE_LIVE
    mode_epoch: int = 0
    request_generation: int = 0
    request_initialized: bool = False
    ingress_session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class RealtimeTrackingCoordinator:
    """Ephemeral latest-wins state for Shigure-driven pose tracking.

    The coordinator intentionally persists neither temporary Shigure bindings
    nor pending/running work.  Restarting the server therefore clears the live
    tracking set while durable model/pose history remains in SQLite.
    """

    def __init__(self, max_objects: int = MAX_REALTIME_TRACKED_DISPLAY_OBJECTS) -> None:
        self.max_objects = max(1, int(max_objects))
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._session: SessionModeState | None = None
        self._objects: "OrderedDict[str, ObjectRuntimeState]" = OrderedDict()

    @property
    def ingress_session_id(self) -> str:
        with self._lock:
            self._ensure_session("")
            assert self._session is not None
            return self._session.ingress_session_id

    def _ensure_session(self, startup_session_id: str) -> SessionModeState:
        startup_session_id = str(startup_session_id or "").strip()
        if self._session is None:
            self._session = SessionModeState(startup_session_id=startup_session_id)
        elif startup_session_id and self._session.startup_session_id != startup_session_id:
            self._reset_no_lock(startup_session_id=startup_session_id)
        return self._session

    def _reset_no_lock(self, *, startup_session_id: str) -> None:
        self._session = SessionModeState(startup_session_id=startup_session_id)
        self._objects.clear()
        self._condition.notify_all()

    def reset(self, *, startup_session_id: str = "") -> None:
        with self._condition:
            self._reset_no_lock(startup_session_id=str(startup_session_id or ""))

    def activate_display_object(self, display_object_id: str, *, reanchor: bool = False) -> list[str]:
        """Add/touch a HoloLens-confirmed object and evict old runtime entries."""

        display_object_id = str(display_object_id or "").strip()
        if not display_object_id:
            return []
        evicted: list[str] = []
        with self._condition:
            state = self._objects.pop(display_object_id, None)
            if state is None:
                state = ObjectRuntimeState(display_object_id=display_object_id)
            elif reanchor:
                state.tracking_epoch += 1
                state.pending = None
                state.latest_seen = None
                state.running = None
            state.activated_monotonic = time.monotonic()
            self._objects[display_object_id] = state
            while len(self._objects) > self.max_objects:
                old_id, old = self._objects.popitem(last=False)
                old.tracking_epoch += 1
                old.pending = None
                old.latest_seen = None
                evicted.append(old_id)
            self._condition.notify_all()
        return evicted

    def active_display_object_ids(self) -> list[str]:
        with self._lock:
            return list(self._objects.keys())

    def set_mode(
        self,
        *,
        startup_session_id: str,
        mode: str,
        request_generation: int,
    ) -> dict[str, Any]:
        mode = str(mode or "").strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported realtime tracking mode: {mode}")
        request_generation = max(0, int(request_generation or 0))
        with self._condition:
            session = self._ensure_session(startup_session_id)

            if session.request_initialized:
                if request_generation < session.request_generation:
                    return {
                        **self._mode_payload_no_lock(),
                        "request_ignored": True,
                        "request_ignored_reason": "stale_request_generation",
                        "ignored_request_generation": request_generation,
                        "ignored_requested_mode": mode,
                    }
                if request_generation == session.request_generation:
                    if mode == session.mode:
                        return self._mode_payload_no_lock()
                    return {
                        **self._mode_payload_no_lock(),
                        "request_ignored": True,
                        "request_ignored_reason": "generation_mode_conflict",
                        "ignored_request_generation": request_generation,
                        "ignored_requested_mode": mode,
                    }

            session.request_initialized = True
            session.request_generation = request_generation
            if session.mode == mode:
                return self._mode_payload_no_lock()

            session.mode_epoch += 1
            session.mode = mode
            for state in self._objects.values():
                state.tracking_epoch += 1
                state.pending = None
                # The Shigure reader is deliberately not polled in History
                # mode.  On resume it scans the recorder updates accumulated
                # since the pause and coalesces them to the newest event per
                # temporary Shigure id.  Re-queuing ``latest_seen`` here would
                # start an already obsolete FoundationPose job just before
                # that scan, wasting the non-preemptive GPU slot for up to one
                # full inference.  Drop the pre-pause snapshot and let the
                # first post-resume cache scan submit only the newest pose.
                state.latest_seen = None

            response = self._mode_payload_no_lock()
            self._condition.notify_all()
            return response

    def mode_status(self, startup_session_id: str = "") -> dict[str, Any]:
        with self._lock:
            self._ensure_session(startup_session_id)
            return self._mode_payload_no_lock()

    def _mode_payload_no_lock(self) -> dict[str, Any]:
        assert self._session is not None
        return {
            "mode": self._session.mode,
            "mode_epoch": int(self._session.mode_epoch),
            "request_generation": int(self._session.request_generation),
            "request_initialized": bool(self._session.request_initialized),
            "startup_session_id": self._session.startup_session_id,
            "ingress_session_id": self._session.ingress_session_id,
            "active_display_object_ids": list(self._objects.keys()),
            "pending_count": sum(1 for item in self._objects.values() if item.pending is not None),
            "running_count": sum(1 for item in self._objects.values() if item.running is not None),
        }

    def submit_observation(
        self,
        *,
        startup_session_id: str,
        display_object_id: str,
        model_revision: int,
        hololens_pose_revision: int,
        payload: Mapping[str, Any],
    ) -> TrackingJobToken | None:
        display_object_id = str(display_object_id or "").strip()
        if not display_object_id:
            return None
        with self._condition:
            session = self._ensure_session(startup_session_id)
            if display_object_id not in self._objects:
                return None
            state = self._objects[display_object_id]
            state.next_observation_seq += 1
            token = TrackingJobToken(
                display_object_id=display_object_id,
                observation_seq=state.next_observation_seq,
                tracking_epoch=state.tracking_epoch,
                mode_epoch=session.mode_epoch,
                model_revision=int(model_revision),
                hololens_pose_revision=int(hololens_pose_revision),
                startup_session_id=session.startup_session_id,
                ingress_session_id=session.ingress_session_id,
            )
            observation = PendingObservation(token=token, payload=copy.deepcopy(dict(payload)))
            state.latest_seen = observation
            if session.mode == MODE_LIVE:
                state.pending = observation
                self._condition.notify_all()
                return token
            return None

    def take_next_pending(self, timeout: float = 1.0) -> PendingObservation | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                if self._session is not None and self._session.mode == MODE_LIVE:
                    for state in self._objects.values():
                        if state.running is not None or state.pending is None:
                            continue
                        observation = state.pending
                        state.pending = None
                        state.running = observation.token
                        state.last_attempted_seq = observation.token.observation_seq
                        return observation
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def _should_accept_no_lock(
        self,
        token: TrackingJobToken,
        *,
        active_model_revision: int,
        active_hololens_pose_revision: int,
    ) -> bool:
        if self._session is None or self._session.mode != MODE_LIVE:
            return False
        if self._session.startup_session_id != token.startup_session_id:
            return False
        if self._session.ingress_session_id != token.ingress_session_id:
            return False
        if self._session.mode_epoch != token.mode_epoch:
            return False
        state = self._objects.get(token.display_object_id)
        if state is None or state.tracking_epoch != token.tracking_epoch:
            return False
        if int(active_model_revision) != int(token.model_revision):
            return False
        if int(active_hololens_pose_revision) != int(token.hololens_pose_revision):
            return False
        if state.latest_seen is not None and state.latest_seen.token.observation_seq > token.observation_seq:
            return False
        return True

    def should_accept(
        self,
        token: TrackingJobToken,
        *,
        active_model_revision: int,
        active_hololens_pose_revision: int,
    ) -> bool:
        with self._lock:
            return self._should_accept_no_lock(
                token,
                active_model_revision=active_model_revision,
                active_hololens_pose_revision=active_hololens_pose_revision,
            )

    def commit_if_current(
        self,
        token: TrackingJobToken,
        *,
        active_model_revision: int,
        active_hololens_pose_revision: int,
        commit: Callable[[], Any],
    ) -> Any | None:
        """Run the durable commit while submission/mode state is frozen.

        A separate ``should_accept`` followed by a database write has a race:
        a newer observation can arrive between the check and write.  Holding
        the coordinator lock through the short SQLite transaction closes that
        window without trying to cancel an already-running FoundationPose job.
        """

        with self._condition:
            if not self._should_accept_no_lock(
                token,
                active_model_revision=active_model_revision,
                active_hololens_pose_revision=active_hololens_pose_revision,
            ):
                return None
            return commit()

    def finish(self, token: TrackingJobToken, *, accepted: bool) -> None:
        with self._condition:
            state = self._objects.get(token.display_object_id)
            if state is not None and state.running == token:
                state.running = None
                if accepted:
                    state.last_accepted_seq = max(state.last_accepted_seq, token.observation_seq)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            mode = self._mode_payload_no_lock() if self._session is not None else None
            return {
                "mode": mode,
                "objects": {
                    key: {
                        "tracking_epoch": value.tracking_epoch,
                        "next_observation_seq": value.next_observation_seq,
                        "last_attempted_seq": value.last_attempted_seq,
                        "last_accepted_seq": value.last_accepted_seq,
                        "pending_seq": value.pending.token.observation_seq if value.pending else None,
                        "running_seq": value.running.observation_seq if value.running else None,
                    }
                    for key, value in self._objects.items()
                },
            }


_journal_lock = threading.RLock()


def append_tracking_journal(display_object_id: str, payload: Mapping[str, Any]) -> Path:
    """Keep the complete FoundationPose transition history outside cache slots."""

    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(display_object_id)) or "unbound"
    path = REALTIME_TRACKING_ROOT / safe_id / "pose_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n"
    with _journal_lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(line)
    return path


coordinator = RealtimeTrackingCoordinator()
