from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap

from task_json import load_task_json, resolve_task_json_path


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_python(python_path: str) -> str:
    return python_path or sys.executable


def load_stage_task(
    argv: list[str],
    *,
    usage: str,
    stage_name: str,
    valid_lengths: tuple[int, ...] = (2,),
    json_index: int = 1,
) -> tuple[Path, dict]:
    if len(argv) not in valid_lengths:
        print(usage, file=sys.stderr)
        raise SystemExit(2)

    json_path = ensure_file(resolve_task_json_path(argv[json_index]), "JSON file")
    task = load_task_json(json_path)
    print(f"[STAGE] {stage_name} : {json_path}")
    return json_path, task


def parse_blender_stage_args(argv: list[str], *, usage: str, expected_count: int) -> list[str]:
    args = argv[argv.index("--") + 1 :] if "--" in argv else []
    if len(args) != expected_count:
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    return args
