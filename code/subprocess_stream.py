from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


def stream_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> str:
    """Run a command while forwarding combined stdout/stderr line by line."""
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

    output_parts: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_parts.append(line)
        print(line, end="", flush=True)

    return_code = process.wait()
    output = "".join(output_parts)
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, [str(part) for part in command], output=output)
    return output
