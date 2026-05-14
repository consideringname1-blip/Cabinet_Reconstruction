from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import CODE_ROOT

from config import (
    ENABLE_INSTANTMESH_VIDEO_OUTPUT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_VIDEOS,
    INSTANTMESH_STAGE_RUN,
    SAM3_BOX_MASK_RUN,
    SAM3_OUTPUT_ROOT,
)
from task_db import get_latest_10_records, get_task_by_task_id, initialize_task_table
from task_json import load_task_json, resolve_task_json_path


def _resolve_python(python_path: str | None, fallback: str) -> str:
    return str(python_path or fallback or sys.executable)


def _is_object_reconstruction(task_json: dict[str, Any]) -> bool:
    return str(task_json.get("purpose") or "object_reconstruction") == "object_reconstruction"


def _has_instantmesh_inputs(task_json: dict[str, Any]) -> bool:
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
        if not _has_instantmesh_inputs(task_json):
            continue
        matches.append((row, json_path, task_json))

    if history_offset >= len(matches):
        raise RuntimeError(
            f"No matching object reconstruction task at history offset {history_offset}. "
            f"Matched {len(matches)} task(s) in the latest 10 database rows."
        )
    return matches[history_offset]


def _run_stage(
    python_path: str,
    script_path: Path,
    json_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(
        [python_path, str(script_path), str(json_path)],
        cwd=str(script_path.parent),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def _sam3_color_path(task_json: dict[str, Any]) -> Path | None:
    color_name = (task_json.get("sam3Name") or {}).get("color")
    if not color_name:
        return None
    return SAM3_OUTPUT_ROOT / str(color_name)


def _ensure_sam3_color(json_path: Path, *, sam3_python: str | None, skip_sam3: bool) -> Path:
    task_json = load_task_json(json_path)
    color_path = _sam3_color_path(task_json)
    if color_path is not None and color_path.is_file():
        print(f"[OK] SAM3 color input exists: {color_path}")
        return color_path

    if skip_sam3:
        raise FileNotFoundError(
            "InstantMesh needs sam3Name.color, but it is missing. "
            "Run validate_latest_sam3_from_db.py first or omit --skip-sam3."
        )

    print("[INFO] SAM3 color input is missing; running SAM3 first.")
    _run_stage(_resolve_python(sam3_python, sys.executable), SAM3_BOX_MASK_RUN, json_path)

    task_json = load_task_json(json_path)
    color_path = _sam3_color_path(task_json)
    if color_path is None or not color_path.is_file():
        raise FileNotFoundError(f"SAM3 did not produce a usable color image: {color_path}")
    return color_path


def _run_instantmesh(json_path: Path, python_path: str | None = None, imesh_python: str | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("IMESH_PY", _resolve_python(imesh_python, sys.executable))
    _run_stage(_resolve_python(python_path, sys.executable), INSTANTMESH_STAGE_RUN, json_path, env=env)


def _verify_outputs(json_path: Path) -> dict[str, Path | None]:
    task_json = load_task_json(json_path)
    instantmesh = task_json.get("InstantMesh") or {}
    required = {
        "mesh": instantmesh.get("mesh"),
        "mtl": instantmesh.get("mtl"),
        "image": instantmesh.get("image"),
    }
    missing_keys = [key for key, value in required.items() if not value]
    if missing_keys:
        raise RuntimeError(f"InstantMesh did not write expected JSON keys: {missing_keys}")

    outputs: dict[str, Path | None] = {
        "mesh": INSTANTMESH_OUTPUT_MESHES / str(required["mesh"]),
        "mtl": INSTANTMESH_OUTPUT_MESHES / str(required["mtl"]),
        "image": INSTANTMESH_OUTPUT_MESHES / str(required["image"]),
        "video": None,
    }
    video_name = instantmesh.get("video")
    if ENABLE_INSTANTMESH_VIDEO_OUTPUT and video_name:
        outputs["video"] = INSTANTMESH_OUTPUT_VIDEOS / str(video_name)

    missing_files = [path for path in outputs.values() if path is not None and not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing InstantMesh output file(s): " + ", ".join(str(path) for path in missing_files))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load the latest object reconstruction task from tasks.db and validate the InstantMesh stage."
    )
    parser.add_argument("--task-id", help="Run a specific task_id instead of the latest matching task.")
    parser.add_argument(
        "--history-offset",
        type=int,
        default=0,
        help="Use an older matching task from the latest 10 rows. 0 means latest.",
    )
    parser.add_argument(
        "--python",
        dest="python_path",
        help="Override the Python executable used for the stage wrapper. Defaults to the current Python.",
    )
    parser.add_argument(
        "--sam3-python",
        help="Override the Python executable used when SAM3 must be run first. Defaults to the current Python.",
    )
    parser.add_argument(
        "--imesh-python",
        help="Override the Python executable used by InstantMesh itself. Defaults to the current Python.",
    )
    parser.add_argument(
        "--skip-sam3",
        action="store_true",
        help="Do not run SAM3 first when sam3Name.color is missing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print the selected task and prerequisite state.")
    args = parser.parse_args()

    row, json_path, task_json = _find_latest_task(args.history_offset, args.task_id)
    print(f"[TASK] id          : {row['task_id']}")
    print(f"[TASK] status      : {row['status']}")
    print(f"[TASK] json        : {json_path}")
    print(f"[TASK] stage python: {args.python_path or sys.executable}")
    print(f"[TASK] imesh python: {args.imesh_python or os.environ.get('IMESH_PY') or sys.executable}")

    sam3_color = _sam3_color_path(task_json)
    print(f"[TASK] sam3 color  : {sam3_color or '<missing in JSON>'}")
    if args.dry_run:
        return 0

    _ensure_sam3_color(json_path, sam3_python=args.sam3_python, skip_sam3=args.skip_sam3)
    _run_instantmesh(json_path, args.python_path, args.imesh_python)
    outputs = _verify_outputs(json_path)
    for key, path in outputs.items():
        if path is not None:
            print(f"[OK] InstantMesh {key:5s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
