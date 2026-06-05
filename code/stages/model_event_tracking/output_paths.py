from __future__ import annotations

import re
from pathlib import Path


TIMESTAMP_NAME_RE = re.compile(r"^\d{8}_\d{6}_\d{6}Z$")


def safe_output_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    text = text.strip("._")
    return text or "model"


def task_output_name(json_path: str | Path | None, *, fallback: str) -> str:
    if json_path:
        stem = Path(json_path).stem
        if stem.endswith("_meta"):
            stem = stem[:-5]
        if TIMESTAMP_NAME_RE.fullmatch(stem):
            return stem
    return safe_output_name(fallback)


def task_output_dir(
    output_root: str | Path,
    *,
    task_id: str,
    json_path: str | Path | None = None,
) -> Path:
    return Path(output_root) / task_output_name(json_path, fallback=task_id)
