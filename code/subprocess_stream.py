from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


GPU_LEASE_USAGE_PID_PREFIX = "__GPU_LEASE_USAGE_PID__="


def parse_gpu_lease_usage_pid_line(line: str) -> int | None:
    """Parse the private child-PID marker while ignoring ordinary output."""

    text = str(line).strip()
    if not text.startswith(GPU_LEASE_USAGE_PID_PREFIX):
        return None
    raw_pid = text[len(GPU_LEASE_USAGE_PID_PREFIX):]
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid GPU lease usage PID marker") from exc
    if pid <= 0:
        raise ValueError("GPU lease usage PID must be positive")
    return pid


def stream_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    echo: bool = True,
    on_start: Callable[[int], None] | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> str:
    """Run a command while optionally forwarding combined stdout/stderr line by line."""
    process_env = dict(os.environ if env is None else env)
    process_env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        if on_start is not None:
            on_start(int(process.pid))
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        raise

    output_parts: list[str] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output_parts.append(line)
            if on_output_line is not None:
                on_output_line(line)
            if echo:
                print(line, end="", flush=True)
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        process.stdout.close()
        raise

    process.stdout.close()
    return_code = process.wait()
    output = "".join(output_parts)
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, [str(part) for part in command], output=output)
    return output
