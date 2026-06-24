from __future__ import annotations

import atexit
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from config import CONSOLE_OUTPUT_LOG_ENABLE
from artifact_layout import CONSOLE_OUTPUT_LOG_ROOT


_INSTALLED = False
_LOCK = threading.RLock()
_LOG_FILE: TextIO | None = None
_ORIGINAL_STDOUT: TextIO | None = None
_ORIGINAL_STDERR: TextIO | None = None
_LOG_PATH: Path | None = None


def _timestamp_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ.txt")


class _TeeTextIO:
    def __init__(self, original: TextIO, log_file: TextIO, lock: threading.RLock) -> None:
        self._original = original
        self._log_file = log_file
        self._lock = lock

    def write(self, text: str) -> int:
        written = self._original.write(text)
        with self._lock:
            self._log_file.write(text)
        return written

    def flush(self) -> None:
        self._original.flush()
        with self._lock:
            self._log_file.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self) -> str | None:
        return self._original.encoding

    @property
    def errors(self) -> str | None:
        return self._original.errors

    def __getattr__(self, name: str):
        return getattr(self._original, name)


def install_console_output_log() -> Path | None:
    global _INSTALLED, _LOG_FILE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _LOG_PATH
    if _INSTALLED:
        return _LOG_PATH
    if not CONSOLE_OUTPUT_LOG_ENABLE:
        return None

    CONSOLE_OUTPUT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONSOLE_OUTPUT_LOG_ROOT / _timestamp_name()
    suffix = 1
    while path.exists():
        path = CONSOLE_OUTPUT_LOG_ROOT / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')}_{suffix}.txt"
        suffix += 1

    log_file = path.open("a", encoding="utf-8", buffering=1)
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    _LOG_FILE = log_file
    _LOG_PATH = path
    sys.stdout = _TeeTextIO(_ORIGINAL_STDOUT, log_file, _LOCK)  # type: ignore[assignment]
    sys.stderr = _TeeTextIO(_ORIGINAL_STDERR, log_file, _LOCK)  # type: ignore[assignment]
    _INSTALLED = True
    atexit.register(_close_console_output_log)
    return path


def _close_console_output_log() -> None:
    global _INSTALLED, _LOG_FILE
    if not _INSTALLED:
        return
    try:
        if sys.stdout is not _ORIGINAL_STDOUT and _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if sys.stderr is not _ORIGINAL_STDERR and _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
        if _LOG_FILE is not None:
            _LOG_FILE.flush()
            os.fsync(_LOG_FILE.fileno())
            _LOG_FILE.close()
    finally:
        _LOG_FILE = None
        _INSTALLED = False
