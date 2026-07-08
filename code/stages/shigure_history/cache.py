from __future__ import annotations

import base64
import json
import os
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass
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

    @property
    def seconds(self) -> float:
        return float(self.sec) + float(self.nanosec) / 1_000_000_000.0

    def to_dict(self) -> dict[str, int]:
        return {"sec": int(self.sec), "nanosec": int(self.nanosec)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RosStamp":
        return cls(sec=int(payload.get("sec", 0)), nanosec=int(payload.get("nanosec", 0)))


@dataclass(frozen=True)
class CachedRgbdSample:
    stamp: RosStamp
    rgb_bgr: np.ndarray
    depth: np.ndarray
    camera_info_path: Path | None
    camera_info: dict[str, Any] | None = None
    yolo_path: Path | None = None
    yolo: dict[str, Any] | None = None
    yolo_hash: str | None = None
    chunk_id: str | None = None
    frame_index: int = 0
    rgb_path: Path | None = None
    depth_path: Path | None = None

    @property
    def key(self) -> str:
        return sample_key(self.stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stamp": self.stamp.to_dict(),
            "chunk_id": self.chunk_id,
            "frame_index": self.frame_index,
            "camera_info_path": str(self.camera_info_path) if self.camera_info_path else None,
            "yolo_hash": self.yolo_hash,
            "yolo_path": str(self.yolo_path) if self.yolo_path else None,
            "has_yolo": self.yolo is not None,
        }


@dataclass(frozen=True)
class CachedSampleMetadata:
    stamp: RosStamp
    camera_info_path: Path | None
    camera_info: dict[str, Any] | None = None
    yolo_path: Path | None = None
    yolo_hash: str | None = None
    chunk_id: str | None = None
    frame_index: int = 0
    yolo: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return sample_key(self.stamp)

    def load_yolo(self) -> dict[str, Any] | None:
        if self.yolo is not None:
            return dict(self.yolo)
        if self.yolo_path is None or not self.yolo_path.is_file():
            return None
        return load_json(self.yolo_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stamp": self.stamp.to_dict(),
            "chunk_id": self.chunk_id,
            "frame_index": self.frame_index,
            "camera_info_path": str(self.camera_info_path) if self.camera_info_path else None,
            "yolo_hash": self.yolo_hash,
            "yolo_path": str(self.yolo_path) if self.yolo_path else None,
            "has_yolo": self.yolo is not None,
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
            camera_info_path=sample.camera_info_path,
            camera_info=dict(sample.camera_info) if sample.camera_info is not None else None,
            yolo_path=sample.yolo_path,
            yolo=dict(sample.yolo) if sample.yolo is not None else None,
            yolo_hash=sample.yolo_hash,
            chunk_id=sample.chunk_id,
            frame_index=sample.frame_index,
            rgb_path=sample.rgb_path,
            depth_path=sample.depth_path,
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

    def iter_sample_metadata(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedSampleMetadata]:
        for sample in self.iter_samples(start=start, end=end):
            yield CachedSampleMetadata(
                stamp=sample.stamp,
                camera_info_path=sample.camera_info_path,
                camera_info=dict(sample.camera_info) if sample.camera_info is not None else None,
                yolo_path=sample.yolo_path,
                yolo_hash=sample.yolo_hash,
                chunk_id=sample.chunk_id,
                frame_index=sample.frame_index,
                yolo=dict(sample.yolo) if sample.yolo is not None else None,
            )

    def iter_samples_after(self, stamp: RosStamp | None) -> Iterable[CachedRgbdSample]:
        minimum = stamp.seconds if stamp is not None else None
        for sample in self.iter_samples(start=stamp):
            if minimum is not None and sample.stamp.seconds <= minimum:
                continue
            yield sample

    def newest_sample(self) -> CachedRgbdSample | None:
        if not self._samples:
            return None
        return next(reversed(self._samples.values()))

    def get_sample(self, stamp: RosStamp, *, mode: str = "nearest") -> CachedRgbdSample | None:
        target = stamp.seconds
        samples = list(self._samples.values())
        if not samples:
            return None
        if mode == "before":
            before = [sample for sample in samples if sample.stamp.seconds <= target]
            return before[-1] if before else None
        if mode == "after":
            for sample in samples:
                if sample.stamp.seconds >= target:
                    return sample
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


class ShigureMemoryStore:
    """Thread-safe in-process RGB-D/object-detection ring buffer."""

    def __init__(self, *, max_seconds: float, max_samples: int) -> None:
        self._buffer = RecentRawSampleBuffer(max_seconds=max_seconds, max_samples=max_samples)
        self._lock = RLock()
        self.created_at = utc_now()
        self.last_append_at: str | None = None

    def append(self, sample: CachedRgbdSample) -> None:
        with self._lock:
            self._buffer.append(sample)
            self.last_append_at = utc_now()

    def iter_sample_metadata(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> list[CachedSampleMetadata]:
        with self._lock:
            return list(self._buffer.iter_sample_metadata(start=start, end=end))

    def iter_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> list[CachedRgbdSample]:
        with self._lock:
            return list(self._buffer.iter_samples(start=start, end=end))

    def iter_samples_after(self, stamp: RosStamp | None) -> list[CachedRgbdSample]:
        with self._lock:
            return list(self._buffer.iter_samples_after(stamp))

    def iter_depth_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> list[CachedRgbdSample]:
        return self.iter_samples(start=start, end=end)

    def newest_sample(self) -> CachedRgbdSample | None:
        with self._lock:
            return self._buffer.newest_sample()

    def get_sample(self, stamp: RosStamp, *, mode: str = "nearest") -> CachedRgbdSample | None:
        with self._lock:
            return self._buffer.get_sample(stamp, mode=mode)

    def status(self) -> dict[str, Any]:
        with self._lock:
            newest = self._buffer.newest_sample()
            return {
                "created_at": self.created_at,
                "last_append_at": self.last_append_at,
                "sample_count": len(self._buffer),
                "retention_seconds": self._buffer.max_seconds,
                "max_samples": self._buffer.max_samples,
                "newest_sample": newest.to_dict() if newest is not None else None,
            }


def sample_key(stamp: RosStamp) -> str:
    return f"{int(stamp.sec):010d}_{int(stamp.nanosec):09d}"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp.replace(target)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


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


def _array_from_wire(payload: Mapping[str, Any] | None, *, default: np.ndarray) -> np.ndarray:
    if not isinstance(payload, Mapping):
        return default
    data = base64.b64decode(str(payload.get("data_b64") or ""))
    dtype = np.dtype(str(payload.get("dtype") or default.dtype))
    shape = tuple(int(value) for value in payload.get("shape") or default.shape)
    if not shape:
        return default
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _metadata_to_wire(metadata: CachedSampleMetadata) -> dict[str, Any]:
    return {
        "stamp": metadata.stamp.to_dict(),
        "camera_info": metadata.camera_info,
        "camera_info_path": str(metadata.camera_info_path) if metadata.camera_info_path else None,
        "yolo": metadata.yolo,
        "yolo_hash": metadata.yolo_hash,
        "yolo_path": str(metadata.yolo_path) if metadata.yolo_path else None,
        "chunk_id": metadata.chunk_id,
        "frame_index": int(metadata.frame_index),
    }


def _metadata_from_wire(payload: Mapping[str, Any]) -> CachedSampleMetadata:
    camera_info_path = payload.get("camera_info_path")
    yolo_path = payload.get("yolo_path")
    yolo = payload.get("yolo") if isinstance(payload.get("yolo"), dict) else None
    camera_info = payload.get("camera_info") if isinstance(payload.get("camera_info"), dict) else None
    return CachedSampleMetadata(
        stamp=RosStamp.from_dict(payload.get("stamp") or {}),
        camera_info_path=Path(str(camera_info_path)) if camera_info_path else None,
        camera_info=camera_info,
        yolo_path=Path(str(yolo_path)) if yolo_path else None,
        yolo_hash=str(payload.get("yolo_hash")) if payload.get("yolo_hash") else None,
        chunk_id=str(payload.get("chunk_id")) if payload.get("chunk_id") else None,
        frame_index=int(payload.get("frame_index") or 0),
        yolo=yolo,
    )


def _sample_to_wire(
    sample: CachedRgbdSample,
    *,
    include_rgb: bool = True,
    include_depth: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stamp": sample.stamp.to_dict(),
        "camera_info": sample.camera_info,
        "camera_info_path": str(sample.camera_info_path) if sample.camera_info_path else None,
        "yolo": sample.yolo,
        "yolo_hash": sample.yolo_hash,
        "yolo_path": str(sample.yolo_path) if sample.yolo_path else None,
        "chunk_id": sample.chunk_id,
        "frame_index": int(sample.frame_index),
        "rgb_path": str(sample.rgb_path) if sample.rgb_path else None,
        "depth_path": str(sample.depth_path) if sample.depth_path else None,
    }
    if include_rgb:
        payload["rgb_bgr"] = _array_to_wire(sample.rgb_bgr)
    if include_depth:
        payload["depth"] = _array_to_wire(sample.depth)
    return payload


def _sample_from_wire(payload: Mapping[str, Any]) -> CachedRgbdSample:
    camera_info_path = payload.get("camera_info_path")
    yolo_path = payload.get("yolo_path")
    rgb_path = payload.get("rgb_path")
    depth_path = payload.get("depth_path")
    yolo = payload.get("yolo") if isinstance(payload.get("yolo"), dict) else None
    camera_info = payload.get("camera_info") if isinstance(payload.get("camera_info"), dict) else None
    return CachedRgbdSample(
        stamp=RosStamp.from_dict(payload.get("stamp") or {}),
        rgb_bgr=_array_from_wire(payload.get("rgb_bgr"), default=np.empty((0, 0, 3), dtype=np.uint8)),
        depth=_array_from_wire(payload.get("depth"), default=np.empty((0, 0), dtype=np.uint16)),
        camera_info_path=Path(str(camera_info_path)) if camera_info_path else None,
        camera_info=camera_info,
        yolo_path=Path(str(yolo_path)) if yolo_path else None,
        yolo=yolo,
        yolo_hash=str(payload.get("yolo_hash")) if payload.get("yolo_hash") else None,
        chunk_id=str(payload.get("chunk_id")) if payload.get("chunk_id") else None,
        frame_index=int(payload.get("frame_index") or 0),
        rgb_path=Path(str(rgb_path)) if rgb_path else None,
        depth_path=Path(str(depth_path)) if depth_path else None,
    )


def store_request(store: ShigureMemoryStore, request: Mapping[str, Any]) -> dict[str, Any]:
    action = str(request.get("action") or "status")
    start = _stamp_from_wire(request.get("start") if isinstance(request.get("start"), Mapping) else None)
    end = _stamp_from_wire(request.get("end") if isinstance(request.get("end"), Mapping) else None)
    if action == "status":
        return {"ok": True, "status": store.status()}
    if action == "iter_sample_metadata":
        return {"ok": True, "metadata": [_metadata_to_wire(item) for item in store.iter_sample_metadata(start=start, end=end)]}
    if action == "iter_samples":
        samples = store.iter_samples(start=start, end=end)
        return {"ok": True, "samples": [_sample_to_wire(item, include_rgb=True, include_depth=True) for item in samples]}
    if action == "iter_depth_samples":
        samples = store.iter_depth_samples(start=start, end=end)
        return {"ok": True, "samples": [_sample_to_wire(item, include_rgb=False, include_depth=True) for item in samples]}
    if action == "iter_samples_after":
        stamp = _stamp_from_wire(request.get("stamp") if isinstance(request.get("stamp"), Mapping) else None)
        samples = store.iter_samples_after(stamp)
        return {"ok": True, "samples": [_sample_to_wire(item, include_rgb=True, include_depth=True) for item in samples]}
    if action == "newest_sample":
        sample = store.newest_sample()
        return {"ok": True, "sample": _sample_to_wire(sample, include_rgb=True, include_depth=True) if sample is not None else None}
    if action == "get_sample":
        stamp = _stamp_from_wire(request.get("stamp") if isinstance(request.get("stamp"), Mapping) else None)
        if stamp is None:
            return {"ok": False, "error": "stamp is required"}
        sample = store.get_sample(stamp, mode=str(request.get("mode") or "nearest"))
        return {"ok": True, "sample": _sample_to_wire(sample, include_rgb=True, include_depth=True) if sample is not None else None}
    if action in {"prune", "prune_yolo_payloads", "clear_decoded_cache"}:
        return {"ok": True}
    return {"ok": False, "error": f"unsupported action: {action}"}


def resolve_socket_path(root: str | Path | None = None) -> Path:
    raw = os.environ.get("SHIGURE_HISTORY_SOCKET_PATH")
    if raw:
        return Path(raw)
    configured = getattr(settings, "SHIGURE_HISTORY_SOCKET_PATH", None)
    if configured:
        return Path(configured)
    if root is not None:
        return Path(root) / "shigure_history.sock"
    return Path("/tmp/shigure_history.sock")


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
    """Client for the online Shigurei RGB-D/object-detection memory cache."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        socket_path: str | Path | None = None,
        timeout_seconds: float | None = None,
        **_: Any,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.socket_path = Path(socket_path) if socket_path is not None else resolve_socket_path(root)
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

    def clear_decoded_cache(self) -> None:
        self._request({"action": "clear_decoded_cache"})

    def iter_sample_metadata(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedSampleMetadata]:
        response = self._request({"action": "iter_sample_metadata", "start": _stamp_to_wire(start), "end": _stamp_to_wire(end)})
        if response is None:
            return
        for payload in response.get("metadata") or []:
            if isinstance(payload, Mapping):
                yield _metadata_from_wire(payload)

    def iter_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedRgbdSample]:
        response = self._request({"action": "iter_samples", "start": _stamp_to_wire(start), "end": _stamp_to_wire(end)})
        if response is None:
            return
        for payload in response.get("samples") or []:
            if isinstance(payload, Mapping):
                yield _sample_from_wire(payload)

    def iter_samples_after(self, stamp: RosStamp | None) -> Iterable[CachedRgbdSample]:
        response = self._request({"action": "iter_samples_after", "stamp": _stamp_to_wire(stamp)})
        if response is None:
            return
        for payload in response.get("samples") or []:
            if isinstance(payload, Mapping):
                yield _sample_from_wire(payload)

    def iter_depth_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedRgbdSample]:
        response = self._request({"action": "iter_depth_samples", "start": _stamp_to_wire(start), "end": _stamp_to_wire(end)})
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

    def get_sample(self, stamp: RosStamp, *, mode: str = "nearest") -> CachedRgbdSample | None:
        response = self._request({"action": "get_sample", "stamp": stamp.to_dict(), "mode": mode})
        if response is None or response.get("sample") is None:
            return None
        sample_payload = response.get("sample")
        return _sample_from_wire(sample_payload) if isinstance(sample_payload, Mapping) else None

    def status(self) -> dict[str, Any] | None:
        response = self._request({"action": "status"})
        return response.get("status") if isinstance(response, dict) else None

    def prune(self, *, newest_stamp: RosStamp | None = None, retention_seconds: float | None = None) -> None:
        self._request(
            {
                "action": "prune",
                "newest_stamp": _stamp_to_wire(newest_stamp),
                "retention_seconds": retention_seconds,
            }
        )

    def prune_yolo_payloads(self) -> None:
        self._request({"action": "prune_yolo_payloads"})
