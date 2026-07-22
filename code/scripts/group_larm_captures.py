from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from config import UPLOAD_FOLDER
from task_db import get_task_by_task_id, initialize_task_table
from task_json import load_task_json, normalize_path_for_storage, resolve_project_path, resolve_task_json_path, save_task_json
from task_worker import _intrinsics_3x3, _larm_transform_from_frame


def _capture_from_task_json(task_id: str, task_json: dict[str, Any]) -> dict[str, Any]:
    pv = task_json.get("PVCamera") or {}
    image_name = pv.get("name")
    if not image_name:
        raise RuntimeError(f"Task {task_id} does not contain PVCamera.name")

    image_path = resolve_project_path(image_name, default_base=UPLOAD_FOLDER)
    frame = {
        "k": pv.get("k"),
        "pose": pv.get("pose"),
        "larm_transform_matrix": pv.get("larm_transform_matrix"),
        "transform_matrix": pv.get("transform_matrix"),
    }
    larm_config = task_json.get("LARM") if isinstance(task_json.get("LARM"), dict) else {}
    capture = {
        "task_id": task_id,
        "task_name": task_json.get("task_name"),
        "server_received_utc": task_json.get("server_received_utc"),
        "intrinsics": _intrinsics_3x3(frame.get("k")),
        "transform_matrix": _larm_transform_from_frame(frame),
        "image_path": str(image_path.resolve()),
        "qpos": None,
        "qpos_token": None,
        "joint_type": str(larm_config.get("joint_type") or "revolute"),
        "source_image": normalize_path_for_storage(image_path),
    }
    for key in ("object_id", "joint_index", "joint_name"):
        if larm_config.get(key) not in (None, ""):
            capture[key] = larm_config.get(key)
    return capture


def _load_capture_from_task(task_id: str) -> dict[str, Any]:
    row = get_task_by_task_id(task_id)
    if row is None:
        raise RuntimeError(f"Task not found: {task_id}")

    task_json = load_task_json(resolve_task_json_path(row["json_path"]))
    purpose = str(task_json.get("purpose") or "object_reconstruction")
    if purpose not in {"larm_input", "object_reconstruction"}:
        raise RuntimeError(f"Task {task_id} is not a supported capture task: {purpose}")

    larm_input = task_json.get("LARMInput") or {}
    capture_json = larm_input.get("capture_json")
    if capture_json:
        capture_path = resolve_project_path(capture_json)
        capture = load_task_json(capture_path)
        capture["task_id"] = capture.get("task_id") or task_id
        return capture

    return _capture_from_task_json(task_id, task_json)


def _qpos_token(value: float) -> str:
    return f"{float(value):.2f}"


def _parse_task_ids(values: list[str]) -> list[str]:
    task_ids: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            part = part.strip()
            if part:
                task_ids.append(part)
    if not task_ids:
        raise ValueError("At least one task id is required")
    return task_ids


def _resolve_split(count: int, state0_count: int | None) -> int:
    if state0_count is None:
        if count % 2 != 0:
            raise ValueError("Odd capture count; pass --state0-count explicitly")
        state0_count = count // 2
    if state0_count < 3 or count - state0_count < 3:
        raise ValueError("LARM grouping requires at least 3 captures for each state")
    return state0_count


def build_larm_group(
    task_ids: list[str],
    *,
    state0_count: int | None,
    output_name: str | None,
    joint_type: str | None,
    qpos0: float,
    qpos1: float,
) -> dict[str, Any]:
    initialize_task_table()
    captures = [_load_capture_from_task(task_id) for task_id in task_ids]
    split = _resolve_split(len(captures), state0_count)

    if output_name is None:
        output_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ_larm_group")
    output_name = str(output_name).strip()
    if not output_name:
        raise ValueError("output name is empty")

    larm_root = UPLOAD_FOLDER / "larm" / output_name
    images_root = larm_root / "images"
    depths_root = larm_root / "depths"
    images_root.mkdir(parents=True, exist_ok=True)

    first_intrinsics = captures[0].get("intrinsics")
    selected_joint_type = joint_type or str(captures[0].get("joint_type") or "revolute")
    qpos_values = [float(qpos0), float(qpos1)]
    qpos_tokens = [_qpos_token(qpos0), _qpos_token(qpos1)]
    qpos_image_counts = {qpos_tokens[0]: 0, qpos_tokens[1]: 0}
    inputs: dict[str, dict[str, dict[str, Any]]] = {qpos_tokens[0]: {}, qpos_tokens[1]: {}}
    grouped_captures: list[dict[str, Any]] = []

    for index, capture in enumerate(captures):
        state_index = 0 if index < split else 1
        token = qpos_tokens[state_index]
        qpos = qpos_values[state_index]
        frame_index = qpos_image_counts[token]
        qpos_image_counts[token] += 1

        source_image = resolve_project_path(capture.get("image_path"))
        image_name = f"color_{token}_in_{frame_index:03d}.png"
        dest_image = images_root / image_name
        shutil.copy2(source_image, dest_image)

        source_depth = None
        dest_depth = None
        if capture.get("depth_path"):
            source_depth = resolve_project_path(capture.get("depth_path"))
            depths_root.mkdir(parents=True, exist_ok=True)
            depth_name = f"depth_{token}_in_{frame_index:03d}.png"
            dest_depth = depths_root / depth_name
            shutil.copy2(source_depth, dest_depth)

        frame_key = f"input_frame_{frame_index}"
        inputs[token][frame_key] = {
            "transform_matrix": capture.get("transform_matrix"),
            "image_path": str(dest_image.resolve()),
            "qpos": qpos,
        }
        if dest_depth is not None:
            inputs[token][frame_key]["depth_path"] = str(dest_depth.resolve())

        grouped_capture = {
            "task_id": capture.get("task_id"),
            "source_capture": normalize_path_for_storage(source_image),
            "path": normalize_path_for_storage(dest_image),
            "qpos": qpos,
            "frame_key": frame_key,
        }
        if source_depth is not None and dest_depth is not None:
            grouped_capture["source_depth"] = normalize_path_for_storage(source_depth)
            grouped_capture["depth_path"] = normalize_path_for_storage(dest_depth)
        grouped_captures.append(grouped_capture)

    metadata = {
        "intrinsics": first_intrinsics,
        "joint_type": selected_joint_type,
        "inputs": inputs,
        "source_task_ids": task_ids,
        "grouped_captures": grouped_captures,
    }
    for key in ("object_id", "joint_index", "joint_name"):
        value = captures[0].get(key)
        if value not in (None, ""):
            metadata[key] = value

    metadata_path = larm_root / f"{output_name}_larm_input.json"
    save_task_json(metadata_path, metadata)

    datalist_path = larm_root / "data.txt"
    datalist_path.write_text(str(metadata_path.resolve()) + "\n", encoding="utf-8")

    return {
        "root": normalize_path_for_storage(larm_root),
        "metadata_json": normalize_path_for_storage(metadata_path),
        "datalist_path": normalize_path_for_storage(datalist_path),
        "image_count": len(grouped_captures),
        "qpos_values": qpos_tokens,
        "state0_count": split,
        "state1_count": len(captures) - split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group single-image capture tasks into one LARM inference input dataset."
    )
    parser.add_argument("--task-ids", nargs="+", required=True, help="Task ids in capture order. Commas are also accepted.")
    parser.add_argument("--state0-count", type=int, help="Number of leading captures that belong to qpos 0.00. Defaults to half for even counts.")
    parser.add_argument("--output-name", help="Output folder name under data/upload/larm/.")
    parser.add_argument("--joint-type", help="Override joint_type, e.g. revolute or prismatic.")
    parser.add_argument("--qpos0", type=float, default=0.0)
    parser.add_argument("--qpos1", type=float, default=1.0)
    args = parser.parse_args()

    result = build_larm_group(
        _parse_task_ids(args.task_ids),
        state0_count=args.state0_count,
        output_name=args.output_name,
        joint_type=args.joint_type,
        qpos0=args.qpos0,
        qpos1=args.qpos1,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
