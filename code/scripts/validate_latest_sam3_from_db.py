from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import CODE_ROOT

from config import (
    HOLOLENS2_OUTPUT_DEPTH_IMAGES,
    SAM3_BOX_MASK_RUN,
    SAM3_OUTPUT_ROOT,
    SAM3_PY,
    UPLOAD_FOLDER,
)
from task_db import get_latest_10_records, get_task_by_task_id, initialize_task_table
from task_json import load_task_json, resolve_task_json_path


def _resolve_python(python_path: str | None) -> str:
    return str(python_path or sys.executable)


def _is_object_reconstruction(task_json: dict[str, Any]) -> bool:
    return str(task_json.get("purpose") or "object_reconstruction") == "object_reconstruction"


def _has_sam3_inputs(task_json: dict[str, Any]) -> bool:
    pv = task_json.get("PVCamera") or {}
    depth = task_json.get("DepthCamera") or {}
    selection = task_json.get("SelectionBox") or {}
    return bool(
        pv.get("name")
        and pv.get("width")
        and pv.get("height")
        and depth.get("align_depth_name")
        and selection.get("top_left")
        and selection.get("bottom_right")
    )


def _find_latest_task(history_offset: int = 0, task_id: str | None = None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    initialize_task_table()

    if task_id:
        row = get_task_by_task_id(task_id)
        if row is None:
            raise RuntimeError(f"Task not found in database: {task_id}")
        json_path = resolve_task_json_path(row["json_path"])
        task_json = load_task_json(json_path)
        return row, json_path, task_json

    matches: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for row in get_latest_10_records():
        json_path = resolve_task_json_path(row["json_path"])
        task_json = load_task_json(json_path)
        if not _is_object_reconstruction(task_json):
            continue
        if not _has_sam3_inputs(task_json):
            continue
        matches.append((row, json_path, task_json))

    if history_offset >= len(matches):
        raise RuntimeError(
            f"No matching object reconstruction task at history offset {history_offset}. "
            f"Matched {len(matches)} task(s) in the latest 10 database rows."
        )
    return matches[history_offset]


def _require_input_files(task_json: dict[str, Any]) -> None:
    pv = task_json.get("PVCamera") or {}
    depth = task_json.get("DepthCamera") or {}
    color_path = UPLOAD_FOLDER / str(pv.get("name"))
    depth_path = HOLOLENS2_OUTPUT_DEPTH_IMAGES / str(depth.get("align_depth_name"))
    missing = [path for path in [color_path, depth_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SAM3 input file(s): " + ", ".join(str(path) for path in missing))


def _run_sam3(json_path: Path, python_path: str | None = None) -> None:
    result = subprocess.run(
        [_resolve_python(python_path or SAM3_PY), str(SAM3_BOX_MASK_RUN), str(json_path)],
        cwd=str(SAM3_BOX_MASK_RUN.parent),
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def _verify_outputs(json_path: Path) -> dict[str, Path]:
    task_json = load_task_json(json_path)
    sam3_name = task_json.get("sam3Name") or {}
    required = {
        "mask": sam3_name.get("mask"),
        "color": sam3_name.get("color"),
        "depth": sam3_name.get("depth"),
        "overlay": sam3_name.get("overlay"),
    }
    missing_keys = [key for key, value in required.items() if not value]
    if missing_keys:
        raise RuntimeError(f"SAM3 did not write expected JSON keys: {missing_keys}")

    outputs = {key: SAM3_OUTPUT_ROOT / str(value) for key, value in required.items()}
    missing_files = [path for path in outputs.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing SAM3 output file(s): " + ", ".join(str(path) for path in missing_files))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load the latest object reconstruction task from tasks.db and validate the SAM3 stage."
    )
    parser.add_argument("--task-id", help="Run a specific task_id instead of the latest matching task.")
    parser.add_argument(
        "--history-offset",
        type=int,
        default=0,
        help="Use an older matching task from the latest 10 rows. 0 means latest.",
    )
    parser.add_argument("--python", dest="python_path", help="Override the Python executable used for SAM3.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the selected task and input files.")
    args = parser.parse_args()

    row, json_path, task_json = _find_latest_task(args.history_offset, args.task_id)
    _require_input_files(task_json)

    print(f"[TASK] id          : {row['task_id']}")
    print(f"[TASK] status      : {row['status']}")
    print(f"[TASK] json        : {json_path}")
    print(f"[TASK] color input : {UPLOAD_FOLDER / str((task_json.get('PVCamera') or {}).get('name'))}")
    print(
        "[TASK] depth input : "
        f"{HOLOLENS2_OUTPUT_DEPTH_IMAGES / str((task_json.get('DepthCamera') or {}).get('align_depth_name'))}"
    )

    if args.dry_run:
        return 0

    _run_sam3(json_path, args.python_path)
    outputs = _verify_outputs(json_path)
    for key, path in outputs.items():
        print(f"[OK] SAM3 {key:7s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
