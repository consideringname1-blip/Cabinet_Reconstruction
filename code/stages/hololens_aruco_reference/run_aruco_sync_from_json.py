from __future__ import annotations

import sys

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from hololens3d_reconstruction.pose_math import (
    quat_xyzw_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
)
from task_db import (
    get_completed_tasks_for_startup,
    get_latest_aruco_reference,
    update_task_aruco_coordinate_synced,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json

try:
    from aruco_common import (
        invert_pose,
        load_json_payload,
        pose_to_payload,
    )
except ModuleNotFoundError:
    from .aruco_common import (
        invert_pose,
        load_json_payload,
        pose_to_payload,
    )


def _write_debug(task: dict, aruco_stage: dict) -> None:
    debug_section = dict(task.get("debug") or {})
    pose_transform_stages = dict(debug_section.get("pose_transform_stages") or {})
    pose_transform_stages["aruco_stage"] = aruco_stage
    debug_section["pose_transform_stages"] = pose_transform_stages
    task["debug"] = debug_section


def _extract_pose(pose: dict) -> tuple[np.ndarray, np.ndarray, list[float] | None]:
    position = np.asarray(pose.get("position"), dtype=np.float64)
    quaternion = np.asarray(pose.get("rotation_quaternion_xyzw"), dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("pose.position must have 3 values")
    if quaternion.shape != (4,):
        raise ValueError("pose quaternion must have 4 values")
    rotation = quat_xyzw_to_rotation_matrix(quaternion)
    scale_value = pose.get("scale")
    scale = [float(v) for v in scale_value] if isinstance(scale_value, list) and len(scale_value) == 3 else None
    return position.astype(np.float64), rotation.astype(np.float64), scale


def _minimal_pose_payload(pose: dict, *, include_scale: bool) -> dict:
    position, rotation, scale = _extract_pose(pose)
    payload = {
        "position": [float(v) for v in position],
        "rotation_quaternion_xyzw": [
            float(v) for v in rotation_matrix_to_quat_xyzw(rotation)
        ],
    }
    if include_scale and scale is not None:
        payload["scale"] = [float(v) for v in scale]
    return payload


def sync_task_json_with_latest_reference(json_path_arg: str) -> bool:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = str(task.get("task_id") or "")
    startup_session_id = str((task.get("device") or {}).get("startup_session_id") or "").strip()
    object_world = task.get("object_world")
    if object_world is not None:
        object_world = _minimal_pose_payload(object_world, include_scale=True)

    debug_section = dict(task.get("debug") or {})
    pose_transform_stages = dict(debug_section.get("pose_transform_stages") or {})
    aruco_stage = dict(pose_transform_stages.get("aruco_stage") or {})
    aruco_stage["sync_stage_ran"] = True
    aruco_stage["synced_to_reference"] = False

    latest_reference_row = get_latest_aruco_reference(startup_session_id) if startup_session_id else None
    aruco_reference_raw = (
        load_json_payload(latest_reference_row.get("marker_pose_json")) if latest_reference_row else None
    )
    aruco_reference = (
        _minimal_pose_payload(aruco_reference_raw, include_scale=False)
        if aruco_reference_raw is not None
        else None
    )

    if aruco_reference is None:
        aruco_stage["sync_reason"] = "reference_not_found"
        _write_debug(task, aruco_stage)
        save_task_json(json_path, task)
        if task_id:
            update_task_aruco_coordinate_synced(task_id, False)
        print("[INFO] aruco_sync : no ArUco reference found for this startup session")
        print("[OK] aruco_sync")
        return False

    task["aruco_reference"] = aruco_reference
    aruco_stage["reference_task_id"] = latest_reference_row.get("task_id")
    aruco_stage["reference_created_at"] = latest_reference_row.get("created_at")

    if object_world is None:
        aruco_stage["sync_reason"] = "object_world_missing"
        _write_debug(task, aruco_stage)
        save_task_json(json_path, task)
        if task_id:
            update_task_aruco_coordinate_synced(task_id, False)
        print("[INFO] aruco_sync : attached reference only because object_world is missing")
        print("[OK] aruco_sync")
        return False

    object_world_position, object_world_rotation, object_scale = _extract_pose(object_world)
    if object_scale is None:
        raise ValueError("object_world.scale must have 3 values")
    marker_world_position, marker_world_rotation, _marker_scale = _extract_pose(aruco_reference)
    marker_inverse_rotation, marker_inverse_translation = invert_pose(
        marker_world_rotation,
        marker_world_position,
    )
    object_local_position = (marker_inverse_rotation @ object_world_position) + marker_inverse_translation
    object_local_rotation = marker_inverse_rotation @ object_world_rotation
    object_aruco = pose_to_payload(
        object_local_rotation,
        object_local_position,
        scale=object_scale,
    )

    task["object_world"] = object_world
    task["object_aruco"] = object_aruco
    aruco_stage["synced_to_reference"] = True
    aruco_stage["sync_reason"] = "reference_applied"
    aruco_stage["object_aruco"] = object_aruco

    _write_debug(task, aruco_stage)
    save_task_json(json_path, task)

    if task_id:
        update_task_aruco_coordinate_synced(task_id, True)

    print("[INFO] aruco_sync : object pose converted into ArUco-local coordinates")
    print("[OK] aruco_sync")
    return True


def sync_completed_tasks_for_startup(startup_session_id: str) -> int:
    synced_count = 0
    for task_row in get_completed_tasks_for_startup(startup_session_id, require_unsynced=True):
        json_path = task_row.get("json_path")
        if not json_path:
            continue
        try:
            if sync_task_json_with_latest_reference(str(json_path)):
                synced_count += 1
        except Exception as exc:
            print(
                f"[WARN] aruco_sync : failed to retro-sync task {task_row.get('task_id')}: {exc}",
                file=sys.stderr,
            )
    return synced_count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: python code/stages/hololens_aruco_reference/run_aruco_sync_from_json.py <task_meta.json or filename>",
            file=sys.stderr,
        )
        raise SystemExit(2)

    sync_task_json_with_latest_reference(argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
