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
class CachedShigureEvent:
    """A Shigure event joined by the exact ROS source timestamp.

    ``contacted_state`` and ``object_detection_state`` are deliberately kept
    separate from the payloads.  This lets consumers distinguish an explicit
    empty list from a topic that was not received for the source timestamp.
    """

    source_stamp: RosStamp
    received_utc: str
    received_monotonic: float
    contacted_state: str
    object_detection_state: str
    contacted: dict[str, Any] | None = None
    object_detection: dict[str, Any] | None = None
    contact_object_matches: list[dict[str, Any]] | None = None
    sequence: int = 0

    def __post_init__(self) -> None:
        valid_states = {"missing", "explicit_empty", "present"}
        if self.contacted_state not in valid_states or self.object_detection_state not in valid_states:
            raise ValueError("invalid Shigure event topic state")
        self._validate_payload(self.contacted_state, self.contacted, "contact_count", "contacted")
        self._validate_payload(
            self.object_detection_state,
            self.object_detection,
            "object_count",
            "object_detection",
        )
        if int(self.sequence) < 0:
            raise ValueError("Shigure event sequence must be non-negative")

    @staticmethod
    def _validate_payload(state: str, payload: dict[str, Any] | None, count_key: str, label: str) -> None:
        if state == "missing":
            if payload is not None:
                raise ValueError(f"{label} payload must be absent when state is missing")
            return
        if not isinstance(payload, dict) or count_key not in payload:
            raise ValueError(f"{label} payload requires {count_key}")
        count = int(payload[count_key])
        if (state == "explicit_empty") != (count == 0):
            raise ValueError(f"{label} state does not match {count_key}")

    @property
    def key(self) -> str:
        return sample_key(self.source_stamp)

    def to_dict(self, *, include_masks: bool = False) -> dict[str, Any]:
        object_detection = deepcopy(self.object_detection)
        if object_detection is not None and not include_masks:
            for item in object_detection.get("objects") or []:
                if isinstance(item, dict):
                    item.pop("mask_b64", None)
        if self.contacted_state == "missing":
            join_state = "waiting_contacted"
        elif self.object_detection_state == "missing":
            join_state = "waiting_object_detection"
        else:
            join_state = "complete"
        return {
            "source_stamp": self.source_stamp.to_dict(),
            "sequence": int(self.sequence),
            "received_utc": self.received_utc,
            "received_monotonic": float(self.received_monotonic),
            "contacted_state": self.contacted_state,
            "object_detection_state": self.object_detection_state,
            "join_state": join_state,
            "contacted": deepcopy(self.contacted),
            "object_detection": object_detection,
            "contact_object_matches": deepcopy(self.contact_object_matches or []),
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


class RecentShigureEventBuffer:
    def __init__(self, *, max_seconds: float, max_events: int) -> None:
        self.max_seconds = max(0.0, float(max_seconds))
        self.max_events = max(0, int(max_events))
        self._events: OrderedDict[str, CachedShigureEvent] = OrderedDict()

    def __len__(self) -> int:
        return len(self._events)

    def append(self, event: CachedShigureEvent) -> None:
        if self.max_events <= 0:
            return
        copied = CachedShigureEvent(
            source_stamp=event.source_stamp,
            received_utc=str(event.received_utc),
            received_monotonic=float(event.received_monotonic),
            contacted_state=str(event.contacted_state),
            object_detection_state=str(event.object_detection_state),
            contacted=deepcopy(event.contacted),
            object_detection=deepcopy(event.object_detection),
            contact_object_matches=deepcopy(event.contact_object_matches or []),
            sequence=int(event.sequence),
        )
        self._events[copied.key] = copied
        self._events = OrderedDict(
            sorted(
                self._events.items(),
                key=lambda item: (
                    item[1].source_stamp.sec,
                    item[1].source_stamp.nanosec,
                    item[1].received_monotonic,
                ),
            )
        )
        newest = self.newest_event()
        if newest is not None:
            self._prune(newest_seconds=newest.source_stamp.seconds)

    def iter_event_updates_after(self, sequence: int) -> Iterable[CachedShigureEvent]:
        minimum = max(0, int(sequence))
        yield from sorted(
            (event for event in self._events.values() if int(event.sequence) > minimum),
            key=lambda event: int(event.sequence),
        )

    def newest_event(self) -> CachedShigureEvent | None:
        if not self._events:
            return None
        return next(reversed(self._events.values()))

    def _prune(self, *, newest_seconds: float) -> None:
        cutoff = newest_seconds - self.max_seconds if self.max_seconds > 0 else None
        while len(self._events) > self.max_events:
            self._events.popitem(last=False)
        if cutoff is not None:
            for key, event in list(self._events.items()):
                if event.source_stamp.seconds < cutoff:
                    self._events.pop(key, None)


class ShigureMemoryStore:
    """Thread-safe in-process RGB-D and correlated Shigure event buffers."""

    def __init__(self, *, max_seconds: float, max_samples: int, max_events: int | None = None) -> None:
        self._buffer = RecentRawSampleBuffer(max_seconds=max_seconds, max_samples=max_samples)
        self._event_buffer = RecentShigureEventBuffer(
            max_seconds=max_seconds,
            max_events=max_samples if max_events is None else max_events,
        )
        self._lock = RLock()
        self.created_at = utc_now()
        self.last_append_at: str | None = None
        self.last_event_append_at: str | None = None
        self._next_event_sequence = 0

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

    def append_event(self, event: CachedShigureEvent) -> CachedShigureEvent:
        with self._lock:
            self._next_event_sequence += 1
            stored = replace(event, sequence=self._next_event_sequence)
            self._event_buffer.append(stored)
            self.last_event_append_at = utc_now()
            return stored

    def iter_event_updates_after(self, sequence: int) -> list[CachedShigureEvent]:
        with self._lock:
            return list(self._event_buffer.iter_event_updates_after(sequence))

    def latest_event(self) -> CachedShigureEvent | None:
        with self._lock:
            return self._event_buffer.newest_event()

    def status(self) -> dict[str, Any]:
        with self._lock:
            newest = self._buffer.newest_sample()
            latest_event = self._event_buffer.newest_event()
            return {
                "created_at": self.created_at,
                "last_append_at": self.last_append_at,
                "last_event_append_at": self.last_event_append_at,
                "sample_count": len(self._buffer),
                "event_count": len(self._event_buffer),
                "retention_seconds": self._buffer.max_seconds,
                "max_samples": self._buffer.max_samples,
                "max_events": self._event_buffer.max_events,
                "latest_event_sequence": int(self._next_event_sequence),
                "newest_sample": newest.to_dict() if newest is not None else None,
                "latest_event": latest_event.to_dict() if latest_event is not None else None,
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


def _event_to_wire(event: CachedShigureEvent, *, include_masks: bool = False) -> dict[str, Any]:
    return event.to_dict(include_masks=include_masks)


def _event_from_wire(payload: Mapping[str, Any]) -> CachedShigureEvent:
    required = {
        "source_stamp",
        "received_utc",
        "received_monotonic",
        "contacted_state",
        "object_detection_state",
        "contacted",
        "object_detection",
        "contact_object_matches",
        "sequence",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Shigure event is missing fields: {sorted(missing)}")
    contacted = payload["contacted"]
    object_detection = payload["object_detection"]
    matches = payload["contact_object_matches"]
    if contacted is not None and not isinstance(contacted, dict):
        raise ValueError("contacted must be an object or null")
    if object_detection is not None and not isinstance(object_detection, dict):
        raise ValueError("object_detection must be an object or null")
    if not isinstance(matches, list):
        raise ValueError("contact_object_matches must be an array")
    return CachedShigureEvent(
        source_stamp=RosStamp.from_dict(payload["source_stamp"]),
        received_utc=str(payload["received_utc"]),
        received_monotonic=float(payload["received_monotonic"]),
        contacted_state=str(payload["contacted_state"]),
        object_detection_state=str(payload["object_detection_state"]),
        contacted=contacted,
        object_detection=object_detection,
        contact_object_matches=[dict(item) for item in matches if isinstance(item, Mapping)],
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
    if action == "iter_event_updates_after":
        include_masks = bool(request.get("include_masks", False))
        events = store.iter_event_updates_after(int(request.get("sequence") or 0))
        return {"ok": True, "events": [_event_to_wire(item, include_masks=include_masks) for item in events]}
    if action == "latest_event":
        include_masks = bool(request.get("include_masks", False))
        event = store.latest_event()
        return {"ok": True, "event": _event_to_wire(event, include_masks=include_masks) if event is not None else None}
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
    """Client for the online Shigurei RGB-D and correlated event cache."""

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

    def iter_event_updates_after(self, sequence: int, *, include_masks: bool = False) -> Iterable[CachedShigureEvent]:
        """Poll event updates without losing a late exact-stamp join update."""

        response = self._request(
            {
                "action": "iter_event_updates_after",
                "sequence": max(0, int(sequence)),
                "include_masks": bool(include_masks),
            }
        )
        if response is None:
            return
        for payload in response.get("events") or []:
            if isinstance(payload, Mapping):
                yield _event_from_wire(payload)

    def latest_event(self, *, include_masks: bool = False) -> CachedShigureEvent | None:
        response = self._request({"action": "latest_event", "include_masks": bool(include_masks)})
        if response is None or response.get("event") is None:
            return None
        event_payload = response.get("event")
        return _event_from_wire(event_payload) if isinstance(event_payload, Mapping) else None

    def status(self) -> dict[str, Any] | None:
        response = self._request({"action": "status"})
        return response.get("status") if isinstance(response, dict) else None
