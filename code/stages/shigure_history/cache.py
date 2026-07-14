from __future__ import annotations

import base64
from copy import deepcopy
import json
import socket
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping

import numpy as np

from . import settings


@dataclass(frozen=True)
class RosStamp:
    sec: int
    nanosec: int = 0

    def __post_init__(self) -> None:
        if int(self.sec) < 0 or not 0 <= int(self.nanosec) < 1_000_000_000:
            raise ValueError("ROS stamp values are out of range")

    @property
    def seconds(self) -> float:
        return float(self.sec) + float(self.nanosec) / 1_000_000_000.0

    def to_dict(self) -> dict[str, int]:
        return {"sec": int(self.sec), "nanosec": int(self.nanosec)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RosStamp":
        if "sec" not in payload or "nanosec" not in payload:
            raise ValueError("ROS stamp requires sec and nanosec")
        return cls(sec=int(payload["sec"]), nanosec=int(payload["nanosec"]))


@dataclass(frozen=True)
class CachedRgbdSample:
    stamp: RosStamp
    rgb_bgr: np.ndarray
    depth: np.ndarray
    camera_info: dict[str, Any]

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb_bgr)
        depth = np.asarray(self.depth)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            raise ValueError("RGB cache frame must be a non-empty uint8 HxWx3 array")
        if depth.dtype != np.uint16 or depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth cache frame must be a non-empty uint16 HxW array")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth cache frame dimensions must match")
        camera_info = self.camera_info
        try:
            matrix = np.asarray(camera_info["k"], dtype=np.float64).reshape(3, 3)
            width = int(camera_info["width"])
            height = int(camera_info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("camera_info requires k, width, and height") from exc
        if (
            not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0.0
            or matrix[1, 1] <= 0.0
            or (height, width) != depth.shape
        ):
            raise ValueError("camera_info does not match the RGB-D frame")

    @property
    def key(self) -> str:
        return sample_key(self.stamp)

    def to_dict(self) -> dict[str, Any]:
        return {"stamp": self.stamp.to_dict()}


@dataclass(frozen=True)
class CachedShigureFrame:
    """Canonical exact-stamp Shigure frame produced by the server adapter."""

    source_stamp: RosStamp
    source_incarnation_id: str
    frame_id: str
    received_utc: str
    received_monotonic: float
    schema_version: int
    input_states: dict[str, str]
    events: list[dict[str, Any]]
    tracked_objects: list[dict[str, Any]]
    recovery_candidates: list[dict[str, Any]]
    people: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    sequence: int = 0

    def __post_init__(self) -> None:
        if int(self.schema_version) != 2:
            raise ValueError("unsupported canonical Shigure schema version")
        if not str(self.source_incarnation_id).strip():
            raise ValueError("canonical frame source_incarnation_id is required")
        if int(self.sequence) < 0:
            raise ValueError("canonical frame sequence must be non-negative")
        valid_states = {"missing", "explicit_empty", "present"}
        if any(str(value) not in valid_states for value in self.input_states.values()):
            raise ValueError("canonical frame contains an invalid input state")
        for label, value in (
            ("events", self.events),
            ("tracked_objects", self.tracked_objects),
            ("recovery_candidates", self.recovery_candidates),
            ("people", self.people),
            ("diagnostics", self.diagnostics),
        ):
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError(f"canonical frame {label} must be an array of objects")

    @property
    def key(self) -> str:
        return sample_key(self.source_stamp)

    def to_dict(self, *, include_masks: bool = False) -> dict[str, Any]:
        events = deepcopy(self.events)
        recovery_candidates = deepcopy(self.recovery_candidates)
        if not include_masks:
            for event in events:
                detection = event.get("detection")
                if isinstance(detection, dict):
                    detection.pop("mask_b64", None)
            for candidate in recovery_candidates:
                candidate.pop("mask_b64", None)
        return {
            "schema_version": int(self.schema_version),
            "source_stamp": self.source_stamp.to_dict(),
            "source_incarnation_id": str(self.source_incarnation_id),
            "frame_id": str(self.frame_id),
            "sequence": int(self.sequence),
            "received_utc": str(self.received_utc),
            "received_monotonic": float(self.received_monotonic),
            "input_states": deepcopy(self.input_states),
            "events": events,
            "tracked_objects": deepcopy(self.tracked_objects),
            "recovery_candidates": recovery_candidates,
            "people": deepcopy(self.people),
            "diagnostics": deepcopy(self.diagnostics),
        }


class RecentRawSampleBuffer:
    def __init__(self, *, max_seconds: float, max_samples: int) -> None:
        self.max_seconds = max(0.0, float(max_seconds))
        self.max_samples = max(0, int(max_samples))
        self._samples: OrderedDict[str, CachedRgbdSample] = OrderedDict()

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, sample: CachedRgbdSample) -> None:
        if self.max_samples <= 0:
            return
        copied = CachedRgbdSample(
            stamp=sample.stamp,
            rgb_bgr=np.asarray(sample.rgb_bgr).copy(),
            depth=np.asarray(sample.depth).copy(),
            camera_info=dict(sample.camera_info),
        )
        self._samples[copied.key] = copied
        self._samples.move_to_end(copied.key)
        self._prune(newest_seconds=copied.stamp.seconds)

    def iter_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedRgbdSample]:
        start_seconds = start.seconds if start is not None else None
        end_seconds = end.seconds if end is not None else None
        for sample in list(self._samples.values()):
            seconds = sample.stamp.seconds
            if start_seconds is not None and seconds < start_seconds:
                continue
            if end_seconds is not None and seconds > end_seconds:
                continue
            yield sample

    def newest_sample(self) -> CachedRgbdSample | None:
        if not self._samples:
            return None
        return next(reversed(self._samples.values()))

    def get_sample(self, stamp: RosStamp) -> CachedRgbdSample | None:
        target = stamp.seconds
        samples = list(self._samples.values())
        if not samples:
            return None
        return min(samples, key=lambda sample: abs(sample.stamp.seconds - target))

    def clear(self) -> None:
        self._samples.clear()

    def _prune(self, *, newest_seconds: float) -> None:
        cutoff = newest_seconds - self.max_seconds if self.max_seconds > 0 else None
        while len(self._samples) > self.max_samples:
            self._samples.popitem(last=False)
        if cutoff is not None:
            for key, sample in list(self._samples.items()):
                if sample.stamp.seconds < cutoff:
                    self._samples.pop(key, None)


class RecentShigureFrameBuffer:
    def __init__(self, *, max_seconds: float, max_frames: int) -> None:
        self.max_seconds = max(0.0, float(max_seconds))
        self.max_frames = max(0, int(max_frames))
        self._frames: OrderedDict[str, CachedShigureFrame] = OrderedDict()

    def __len__(self) -> int:
        return len(self._frames)

    def append(self, frame: CachedShigureFrame) -> None:
        if self.max_frames <= 0:
            return
        copied = CachedShigureFrame(
            source_stamp=frame.source_stamp,
            source_incarnation_id=str(frame.source_incarnation_id),
            frame_id=str(frame.frame_id),
            received_utc=str(frame.received_utc),
            received_monotonic=float(frame.received_monotonic),
            schema_version=int(frame.schema_version),
            input_states=deepcopy(frame.input_states),
            events=deepcopy(frame.events),
            tracked_objects=deepcopy(frame.tracked_objects),
            recovery_candidates=deepcopy(frame.recovery_candidates),
            people=deepcopy(frame.people),
            diagnostics=deepcopy(frame.diagnostics),
            sequence=int(frame.sequence),
        )
        self._frames[copied.key] = copied
        # Canonical frames are an arrival-ordered update log.  A control frame
        # that rotates the source incarnation may legitimately arrive after a
        # frame with a larger ROS stamp, so source time must never decide which
        # incarnation is authoritative.
        self._frames.move_to_end(copied.key)
        self._prune(newest_received_monotonic=copied.received_monotonic)

    def iter_updates_after(self, sequence: int) -> Iterable[CachedShigureFrame]:
        minimum = max(0, int(sequence))
        yield from sorted(
            (frame for frame in self._frames.values() if int(frame.sequence) > minimum),
            key=lambda frame: int(frame.sequence),
        )

    def newest_frame(self) -> CachedShigureFrame | None:
        if not self._frames:
            return None
        return max(self._frames.values(), key=lambda frame: int(frame.sequence))

    def newest_recovery_frame(self) -> CachedShigureFrame | None:
        candidates = [
            frame
            for frame in self._frames.values()
            if (
                frame.input_states.get("camera_info") == "present"
                and frame.input_states.get("segments") in {"present", "explicit_empty"}
            )
        ]
        return (
            max(candidates, key=lambda frame: int(frame.sequence))
            if candidates
            else None
        )

    def _prune(self, *, newest_received_monotonic: float) -> None:
        cutoff = (
            float(newest_received_monotonic) - self.max_seconds
            if self.max_seconds > 0
            else None
        )
        while len(self._frames) > self.max_frames:
            oldest_key = min(
                self._frames,
                key=lambda key: int(self._frames[key].sequence),
            )
            self._frames.pop(oldest_key, None)
        if cutoff is not None:
            latest_sequence = max(
                (int(frame.sequence) for frame in self._frames.values()),
                default=0,
            )
            for key, frame in list(self._frames.items()):
                if (
                    int(frame.sequence) != latest_sequence
                    and float(frame.received_monotonic) < cutoff
                ):
                    self._frames.pop(key, None)


class ShigureMemoryStore:
    """Thread-safe in-process RGB-D and canonical Shigure frame buffers."""

    def __init__(self, *, max_seconds: float, max_samples: int, max_frames: int | None = None) -> None:
        self._buffer = RecentRawSampleBuffer(max_seconds=max_seconds, max_samples=max_samples)
        self._frame_buffer = RecentShigureFrameBuffer(
            max_seconds=max_seconds,
            max_frames=max_samples if max_frames is None else max_frames,
        )
        self._lock = RLock()
        self.created_at = utc_now()
        self.last_append_at: str | None = None
        self.last_frame_append_at: str | None = None
        self._next_frame_sequence = 0

    def append(self, sample: CachedRgbdSample) -> None:
        with self._lock:
            self._buffer.append(sample)
            self.last_append_at = utc_now()

    def iter_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> list[CachedRgbdSample]:
        with self._lock:
            return list(self._buffer.iter_samples(start=start, end=end))

    def newest_sample(self) -> CachedRgbdSample | None:
        with self._lock:
            return self._buffer.newest_sample()

    def get_sample(self, stamp: RosStamp) -> CachedRgbdSample | None:
        with self._lock:
            return self._buffer.get_sample(stamp)

    def append_canonical_frame(self, frame: CachedShigureFrame) -> CachedShigureFrame:
        with self._lock:
            self._next_frame_sequence += 1
            stored = replace(frame, sequence=self._next_frame_sequence)
            self._frame_buffer.append(stored)
            self.last_frame_append_at = utc_now()
            return stored

    def iter_canonical_updates_after(self, sequence: int) -> list[CachedShigureFrame]:
        with self._lock:
            return list(self._frame_buffer.iter_updates_after(sequence))

    def latest_canonical_frame(self) -> CachedShigureFrame | None:
        with self._lock:
            return self._frame_buffer.newest_frame()

    def latest_recovery_frame(self) -> CachedShigureFrame | None:
        with self._lock:
            return self._frame_buffer.newest_recovery_frame()

    def status(self) -> dict[str, Any]:
        with self._lock:
            newest = self._buffer.newest_sample()
            latest_frame = self._frame_buffer.newest_frame()
            latest_recovery = self._frame_buffer.newest_recovery_frame()
            return {
                "created_at": self.created_at,
                "last_append_at": self.last_append_at,
                "last_frame_append_at": self.last_frame_append_at,
                "sample_count": len(self._buffer),
                "canonical_frame_count": len(self._frame_buffer),
                "retention_seconds": self._buffer.max_seconds,
                "max_samples": self._buffer.max_samples,
                "max_frames": self._frame_buffer.max_frames,
                "latest_canonical_sequence": int(self._next_frame_sequence),
                "source_incarnation_id": (
                    latest_frame.source_incarnation_id if latest_frame is not None else None
                ),
                "newest_sample": newest.to_dict() if newest is not None else None,
                "latest_canonical_frame": latest_frame.to_dict() if latest_frame is not None else None,
                "latest_recovery_frame": latest_recovery.to_dict() if latest_recovery is not None else None,
            }


def sample_key(stamp: RosStamp) -> str:
    return f"{int(stamp.sec):010d}_{int(stamp.nanosec):09d}"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            tmp = Path(file.name)
            json.dump(dict(payload), file, ensure_ascii=False, indent=2)
            file.write("\n")
        tmp.replace(target)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp_to_wire(stamp: RosStamp | None) -> dict[str, int] | None:
    return stamp.to_dict() if stamp is not None else None


def _stamp_from_wire(payload: Mapping[str, Any] | None) -> RosStamp | None:
    return RosStamp.from_dict(payload) if isinstance(payload, Mapping) else None


def _array_to_wire(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "data_b64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _array_from_wire(payload: Mapping[str, Any] | None) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise ValueError("array payload is required")
    if "data_b64" not in payload or "dtype" not in payload or "shape" not in payload:
        raise ValueError("array payload requires data_b64, dtype, and shape")
    data = base64.b64decode(str(payload["data_b64"]), validate=True)
    dtype = np.dtype(str(payload["dtype"]))
    shape = tuple(int(value) for value in payload["shape"])
    if not shape:
        raise ValueError("array shape cannot be empty")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if len(data) != expected_bytes:
        raise ValueError("array byte length does not match dtype and shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _sample_to_wire(
    sample: CachedRgbdSample,
    *,
    include_rgb: bool = True,
    include_depth: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stamp": sample.stamp.to_dict(),
        "camera_info": sample.camera_info,
    }
    if include_rgb:
        payload["rgb_bgr"] = _array_to_wire(sample.rgb_bgr)
    if include_depth:
        payload["depth"] = _array_to_wire(sample.depth)
    return payload


def _sample_from_wire(payload: Mapping[str, Any]) -> CachedRgbdSample:
    camera_info = payload.get("camera_info")
    if not isinstance(camera_info, dict):
        raise ValueError("RGB-D cache sample is missing camera_info")
    return CachedRgbdSample(
        stamp=RosStamp.from_dict(payload.get("stamp") if isinstance(payload.get("stamp"), Mapping) else {}),
        rgb_bgr=_array_from_wire(payload.get("rgb_bgr") if isinstance(payload.get("rgb_bgr"), Mapping) else None),
        depth=_array_from_wire(payload.get("depth") if isinstance(payload.get("depth"), Mapping) else None),
        camera_info=camera_info,
    )


def _canonical_frame_to_wire(frame: CachedShigureFrame, *, include_masks: bool = False) -> dict[str, Any]:
    return frame.to_dict(include_masks=include_masks)


def _canonical_frame_from_wire(payload: Mapping[str, Any]) -> CachedShigureFrame:
    required = {
        "schema_version",
        "source_stamp",
        "source_incarnation_id",
        "frame_id",
        "received_utc",
        "received_monotonic",
        "input_states",
        "events",
        "tracked_objects",
        "recovery_candidates",
        "people",
        "diagnostics",
        "sequence",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"canonical Shigure frame is missing fields: {sorted(missing)}")
    input_states = payload["input_states"]
    if not isinstance(input_states, Mapping):
        raise ValueError("canonical frame input_states must be an object")

    def object_list(name: str) -> list[dict[str, Any]]:
        values = payload[name]
        if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
            raise ValueError(f"canonical frame {name} must be an array of objects")
        return [dict(item) for item in values]

    return CachedShigureFrame(
        schema_version=int(payload["schema_version"]),
        source_stamp=RosStamp.from_dict(payload["source_stamp"]),
        source_incarnation_id=str(payload["source_incarnation_id"]),
        frame_id=str(payload["frame_id"]),
        received_utc=str(payload["received_utc"]),
        received_monotonic=float(payload["received_monotonic"]),
        input_states={str(key): str(value) for key, value in input_states.items()},
        events=object_list("events"),
        tracked_objects=object_list("tracked_objects"),
        recovery_candidates=object_list("recovery_candidates"),
        people=object_list("people"),
        diagnostics=object_list("diagnostics"),
        sequence=int(payload["sequence"]),
    )


def store_request(store: ShigureMemoryStore, request: Mapping[str, Any]) -> dict[str, Any]:
    action = str(request.get("action") or "")
    start = _stamp_from_wire(request.get("start") if isinstance(request.get("start"), Mapping) else None)
    end = _stamp_from_wire(request.get("end") if isinstance(request.get("end"), Mapping) else None)
    if action == "status":
        return {"ok": True, "status": store.status()}
    if action == "iter_samples":
        samples = store.iter_samples(start=start, end=end)
        return {"ok": True, "samples": [_sample_to_wire(item, include_rgb=True, include_depth=True) for item in samples]}
    if action == "newest_sample":
        sample = store.newest_sample()
        return {"ok": True, "sample": _sample_to_wire(sample, include_rgb=True, include_depth=True) if sample is not None else None}
    if action == "get_sample":
        stamp = _stamp_from_wire(request.get("stamp") if isinstance(request.get("stamp"), Mapping) else None)
        if stamp is None:
            return {"ok": False, "error": "stamp is required"}
        sample = store.get_sample(stamp)
        return {"ok": True, "sample": _sample_to_wire(sample, include_rgb=True, include_depth=True) if sample is not None else None}
    if action == "iter_canonical_updates_after":
        include_masks = bool(request.get("include_masks", False))
        frames = store.iter_canonical_updates_after(int(request.get("sequence") or 0))
        return {"ok": True, "frames": [_canonical_frame_to_wire(item, include_masks=include_masks) for item in frames]}
    if action == "latest_canonical_frame":
        include_masks = bool(request.get("include_masks", False))
        frame = store.latest_canonical_frame()
        return {"ok": True, "frame": _canonical_frame_to_wire(frame, include_masks=include_masks) if frame is not None else None}
    if action == "latest_recovery_frame":
        include_masks = bool(request.get("include_masks", False))
        frame = store.latest_recovery_frame()
        return {"ok": True, "frame": _canonical_frame_to_wire(frame, include_masks=include_masks) if frame is not None else None}
    return {"ok": False, "error": f"unsupported action: {action}"}


def resolve_socket_path() -> Path:
    return Path(settings.SHIGURE_HISTORY_SOCKET_PATH)


def _send_socket_request(socket_path: Path, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall((json.dumps(dict(payload), ensure_ascii=False) + "\n").encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No response from Shigurei history socket: {socket_path}")
    return json.loads(b"".join(chunks).decode("utf-8").splitlines()[0])


class ShigureRgbdCache:
    """Client for the online Shigurei RGB-D and canonical frame cache."""

    def __init__(
        self,
        socket_path: str | Path | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.socket_path = Path(socket_path) if socket_path is not None else resolve_socket_path()
        self.timeout_seconds = float(timeout_seconds if timeout_seconds is not None else settings.SHIGURE_HISTORY_SOCKET_TIMEOUT_SECONDS)
        self.last_error: str | None = None

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.socket_path.exists():
            self.last_error = f"socket_not_found:{self.socket_path}"
            return None
        try:
            response = _send_socket_request(self.socket_path, payload, timeout=self.timeout_seconds)
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, TimeoutError, OSError) as exc:
            self.last_error = str(exc)
            return None
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Shigurei history socket request failed"))
        self.last_error = None
        return response

    def iter_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedRgbdSample]:
        response = self._request({"action": "iter_samples", "start": _stamp_to_wire(start), "end": _stamp_to_wire(end)})
        if response is None:
            return
        for payload in response.get("samples") or []:
            if isinstance(payload, Mapping):
                yield _sample_from_wire(payload)

    def newest_sample(self) -> CachedRgbdSample | None:
        response = self._request({"action": "newest_sample"})
        if response is None or response.get("sample") is None:
            return None
        sample_payload = response.get("sample")
        return _sample_from_wire(sample_payload) if isinstance(sample_payload, Mapping) else None

    def get_sample(self, stamp: RosStamp) -> CachedRgbdSample | None:
        response = self._request({"action": "get_sample", "stamp": stamp.to_dict()})
        if response is None or response.get("sample") is None:
            return None
        sample_payload = response.get("sample")
        return _sample_from_wire(sample_payload) if isinstance(sample_payload, Mapping) else None

    def iter_canonical_updates_after(
        self, sequence: int, *, include_masks: bool = False
    ) -> Iterable[CachedShigureFrame]:
        """Poll exact-stamp canonical frame revisions."""

        response = self._request(
            {
                "action": "iter_canonical_updates_after",
                "sequence": max(0, int(sequence)),
                "include_masks": bool(include_masks),
            }
        )
        if response is None:
            return
        for payload in response.get("frames") or []:
            if isinstance(payload, Mapping):
                yield _canonical_frame_from_wire(payload)

    def latest_canonical_frame(self, *, include_masks: bool = False) -> CachedShigureFrame | None:
        response = self._request({"action": "latest_canonical_frame", "include_masks": bool(include_masks)})
        if response is None or response.get("frame") is None:
            return None
        frame_payload = response.get("frame")
        return _canonical_frame_from_wire(frame_payload) if isinstance(frame_payload, Mapping) else None

    def latest_recovery_frame(self, *, include_masks: bool = False) -> CachedShigureFrame | None:
        response = self._request({"action": "latest_recovery_frame", "include_masks": bool(include_masks)})
        if response is None or response.get("frame") is None:
            return None
        frame_payload = response.get("frame")
        return _canonical_frame_from_wire(frame_payload) if isinstance(frame_payload, Mapping) else None

    def status(self) -> dict[str, Any] | None:
        response = self._request({"action": "status"})
        return response.get("status") if isinstance(response, dict) else None
