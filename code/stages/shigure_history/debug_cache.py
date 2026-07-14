from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .cache import CachedRgbdSample, CachedShigureFrame, sample_key


MAX_DEBUG_CACHE_RETENTION_SECONDS = 600.0
MAX_PENDING_DEBUG_WRITES = 64
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def validate_debug_cache_limits(*, retention_seconds: float, max_entries: int) -> tuple[float, int]:
    retention = float(retention_seconds)
    entry_limit = int(max_entries)
    if not 0.0 < retention <= MAX_DEBUG_CACHE_RETENTION_SECONDS:
        raise ValueError("debug cache retention_seconds must be in (0, 600]")
    if entry_limit <= 0:
        raise ValueError("debug cache max_entries must be positive")
    return retention, entry_limit


def _safe_token(value: str, *, fallback: str) -> str:
    token = _SAFE_TOKEN_RE.sub("_", str(value).strip()).strip("._")
    return (token or fallback)[:120]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _encode_png(image: np.ndarray, *, label: str) -> bytes:
    ok, encoded = cv2.imencode(".png", np.asarray(image))
    if not ok:
        raise ValueError(f"failed to encode debug {label} PNG")
    return encoded.tobytes()


class ShigureDebugDiskRing:
    """Best-effort, write-only diagnostic ring outside the runtime data path.

    Online consumers have no API for loading these files. Every I/O failure is
    contained here so enabling diagnostics cannot change the in-memory cache.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool,
        retention_seconds: float,
        max_entries: int,
        session_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        retention, entry_limit = validate_debug_cache_limits(
            retention_seconds=retention_seconds,
            max_entries=max_entries,
        )
        self.root = Path(root)
        self.entries_root = self.root / "entries"
        self.enabled = bool(enabled)
        self.retention_seconds = retention
        self.max_entries = entry_limit
        self.session_id = _safe_token(session_id, fallback="session")
        self._clock = clock
        self._lock = RLock()
        self._slots = BoundedSemaphore(MAX_PENDING_DEBUG_WRITES)
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future[bool]] = set()
        self._closed = False
        self.last_error: str | None = None
        self.write_count = 0
        self.dropped_write_count = 0
        if self.enabled:
            self._best_effort(self._initialize)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shigure-debug-writer")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "root": str(self.root) if self.enabled else None,
                "retention_seconds": self.retention_seconds,
                "max_entries": self.max_entries,
                "session_id": self.session_id,
                "pending_writes": len(self._futures),
                "write_count": self.write_count,
                "dropped_write_count": self.dropped_write_count,
                "last_error": self.last_error,
            }

    def record_rgbd(self, sample: CachedRgbdSample) -> bool:
        if not self.enabled:
            return False
        recorded_at = float(self._clock())

        def write() -> None:
            entry = self._entry_path(sample.stamp)
            _atomic_write_bytes(entry / "rgb.png", _encode_png(sample.rgb_bgr, label="RGB"))
            _atomic_write_bytes(entry / "depth.png", _encode_png(sample.depth, label="depth"))
            _atomic_write_json(
                entry / "rgbd.json",
                {
                    "schema_version": 1,
                    "diagnostic_only": True,
                    "session_id": self.session_id,
                    "source_stamp": sample.stamp.to_dict(),
                    "rgb": {"file": "rgb.png", "shape": list(sample.rgb_bgr.shape), "dtype": str(sample.rgb_bgr.dtype)},
                    "depth": {"file": "depth.png", "shape": list(sample.depth.shape), "dtype": str(sample.depth.dtype)},
                    "camera_info": sample.camera_info,
                },
            )
            self._finish_entry(entry, recorded_at=recorded_at)

        return self._submit(write)

    def record_canonical(self, frame: CachedShigureFrame) -> bool:
        if not self.enabled:
            return False
        recorded_at = float(self._clock())

        def write() -> None:
            entry = self._entry_path(frame.source_stamp)
            _atomic_write_json(
                entry / "canonical.json",
                {
                    "schema_version": 1,
                    "diagnostic_only": True,
                    "session_id": self.session_id,
                    "canonical_frame": frame.to_dict(include_masks=True),
                },
            )
            self._finish_entry(entry, recorded_at=recorded_at)

        return self._submit(write)

    def prune(self) -> bool:
        if not self.enabled:
            return False
        return self._submit(self._prune)

    def _submit(self, operation: Callable[[], None]) -> bool:
        with self._lock:
            executor = self._executor
            if self._closed or executor is None:
                return False
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self.dropped_write_count += 1
            return False
        try:
            future = executor.submit(self._best_effort, operation)
        except Exception as exc:
            self._slots.release()
            with self._lock:
                self.last_error = f"{exc.__class__.__name__}: {exc}"
            return False
        with self._lock:
            self._futures.add(future)

        def completed(completed_future: Future[bool]) -> None:
            with self._lock:
                self._futures.discard(completed_future)
            self._slots.release()

        future.add_done_callback(completed)
        return True

    def flush(self, timeout: float = 5.0) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            futures = set(self._futures)
        if not futures:
            return True
        _done, pending = wait(futures, timeout=max(0.0, float(timeout)))
        return not pending

    def close(self, timeout: float = 5.0) -> bool:
        with self._lock:
            if self._closed:
                return not self._futures
            self._closed = True
            executor = self._executor
        completed = self.flush(timeout=timeout)
        if executor is not None:
            executor.shutdown(wait=completed, cancel_futures=not completed)
        return completed

    def _best_effort(self, operation: Callable[[], None]) -> bool:
        try:
            operation()
        except Exception as exc:
            with self._lock:
                self.last_error = f"{exc.__class__.__name__}: {exc}"
            return False
        with self._lock:
            self.last_error = None
        return True

    def _initialize(self) -> None:
        self.entries_root.mkdir(parents=True, exist_ok=True)
        self._prune()

    def _entry_path(self, stamp: Any) -> Path:
        key = _safe_token(sample_key(stamp), fallback="sample")
        return self.entries_root / f"{self.session_id}--{key}"

    def _finish_entry(self, entry: Path, *, recorded_at: float) -> None:
        now = float(recorded_at)
        os.utime(entry, (now, now))
        with self._lock:
            self.write_count += 1
        self._prune(now=now)

    def _prune(self, *, now: float | None = None) -> None:
        self.entries_root.mkdir(parents=True, exist_ok=True)
        current_time = float(self._clock()) if now is None else float(now)
        cutoff = current_time - self.retention_seconds
        entries: list[tuple[float, str, Path]] = []
        for path in self.entries_root.iterdir():
            if not path.is_dir():
                continue
            try:
                modified = float(path.stat().st_mtime)
            except FileNotFoundError:
                continue
            entries.append((modified, path.name, path))
        entries.sort(key=lambda item: (item[0], item[1]))

        retained: list[tuple[float, str, Path]] = []
        for modified, name, path in entries:
            if modified < cutoff:
                shutil.rmtree(path, ignore_errors=True)
            else:
                retained.append((modified, name, path))
        excess = max(0, len(retained) - self.max_entries)
        for _modified, _name, path in retained[:excess]:
            shutil.rmtree(path, ignore_errors=True)


__all__ = ["MAX_DEBUG_CACHE_RETENTION_SECONDS", "ShigureDebugDiskRing", "validate_debug_cache_limits"]
