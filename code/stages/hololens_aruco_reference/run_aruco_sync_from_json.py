from __future__ import annotations

import sys

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from task_db import (
    get_latest_aruco_reference,
    update_task_aruco_coordinate_synced,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json
from stages.hololens3d_reconstruction.model_bounds import compute_and_store_model_bounds
from spatial_transforms import (
    aruco_pose_to_hololens_pose,
    hololens_point_to_aruco,
    hololens_pose_to_aruco_pose,
    minimal_pose_payload,
    resolve_hololens_original_pose,
)

try:
    from aruco_common import load_json_payload
except ModuleNotFoundError:
    from .aruco_common import load_json_payload


def _write_debug(task: dict, aruco_stage: dict) -> None:
    debug_section = dict(task.get("debug") or {})
    pose_transform_stages = dict(debug_section.get("pose_transform_stages") or {})
    pose_transform_stages["aruco_stage"] = aruco_stage
    debug_section["pose_transform_stages"] = pose_transform_stages
    task["debug"] = debug_section



def _task_startup_session_id(task: dict) -> str:
    return str((task.get("device") or {}).get("startup_session_id") or "").strip()


def _sync_model_bounds(json_path) -> bool:
    try:
        compute_and_store_model_bounds(json_path)
        return True
    except Exception as exc:
        print(f"[WARN] aruco_sync : model_bounds refresh failed: {exc}", file=sys.stderr)
        return False


def _sync_sam3_spatial_box_to_aruco(task: dict, aruco_reference: dict) -> dict | None:
    """Persist the preview box in the server's canonical ArUco space.

    ``Sam3SpatialBox`` is produced before model alignment and its ``*_world``
    fields are HoloLens-local coordinates for the capture startup.  Keeping the
    original fields and deriving canonical corners here makes the early box
    usable by Shigure candidate gating, and rerunning ArUco sync refreshes the
    derivation from the immutable HoloLens values.
    """

    raw = task.get("Sam3SpatialBox")
    if not isinstance(raw, dict) or str(raw.get("status") or "") != "ready":
        return None
    try:
        minimum = np.asarray(raw.get("aabb_min_world"), dtype=np.float64).reshape(3)
        maximum = np.asarray(raw.get("aabb_max_world"), dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        return None

    corners_hololens = np.asarray(
        [[x, y, z] for x in (minimum[0], maximum[0]) for y in (minimum[1], maximum[1]) for z in (minimum[2], maximum[2])],
        dtype=np.float64,
    )
    corners_aruco = np.asarray(
        [hololens_point_to_aruco(point, aruco_reference) for point in corners_hololens],
        dtype=np.float64,
    )
    center_hololens = (minimum + maximum) * 0.5
    center_aruco = hololens_point_to_aruco(center_hololens, aruco_reference)
    diagonal_m = float(np.linalg.norm(maximum - minimum))

    spatial_box = dict(raw)
    spatial_box["canonical_coordinate_space"] = "aruco_local"
    spatial_box["corners_aruco"] = corners_aruco.astype(float).tolist()
    spatial_box["center_aruco"] = center_aruco.astype(float).tolist()
    spatial_box["diagonal_m"] = diagonal_m
    spatial_box["aruco_sync_source"] = "holoLens_aabb_min_max_world"
    task["Sam3SpatialBox"] = spatial_box
    return spatial_box


def sync_task_json_with_latest_reference(json_path_arg: str, *, refresh_model_bounds: bool = False) -> bool:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = str(task.get("task_id") or "")
    startup_session_id = _task_startup_session_id(task)

    debug_section = dict(task.get("debug") or {})
    pose_transform_stages = dict(debug_section.get("pose_transform_stages") or {})
    aruco_stage = dict(pose_transform_stages.get("aruco_stage") or {})
    aruco_stage["sync_stage_ran"] = True
    aruco_stage["synced_to_reference"] = False
    aruco_stage["source_pose_field"] = None

    latest_reference_row = get_latest_aruco_reference(startup_session_id) if startup_session_id else None
    aruco_reference_raw = (
        load_json_payload(latest_reference_row.get("marker_pose_json")) if latest_reference_row else None
    )
    aruco_reference = (
        minimal_pose_payload(aruco_reference_raw, include_scale=False)
        if isinstance(aruco_reference_raw, dict)
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

    original_pose_source = "object_hololens_original" if isinstance(task.get("object_hololens_original"), dict) else None
    original_pose_raw = resolve_hololens_original_pose(task)
    if original_pose_raw is None:
        aruco_stage["reference_task_id"] = latest_reference_row.get("task_id")
        aruco_stage["reference_created_at"] = latest_reference_row.get("created_at")
        aruco_stage["sync_reason"] = "object_hololens_original_missing"
        _write_debug(task, aruco_stage)
        save_task_json(json_path, task)
        if task_id:
            update_task_aruco_coordinate_synced(task_id, False)
        print("[INFO] aruco_sync : attached reference only because object HoloLens pose is missing")
        print("[OK] aruco_sync")
        return False

    original_pose = minimal_pose_payload(original_pose_raw, include_scale=True)
    object_aruco = hololens_pose_to_aruco_pose(original_pose, aruco_reference)
    object_hololens_current = aruco_pose_to_hololens_pose(object_aruco, aruco_reference)

    task["object_hololens_original"] = original_pose
    task["object_hololens_current"] = object_hololens_current
    task["aruco_reference"] = aruco_reference
    task["object_aruco"] = object_aruco
    synced_spatial_box = _sync_sam3_spatial_box_to_aruco(task, aruco_reference)

    aruco_stage["reference_task_id"] = latest_reference_row.get("task_id")
    aruco_stage["reference_created_at"] = latest_reference_row.get("created_at")
    aruco_stage["source_pose_field"] = original_pose_source
    aruco_stage["synced_to_reference"] = True
    aruco_stage["sync_reason"] = "reference_applied_from_hololens_original"
    aruco_stage["object_aruco"] = object_aruco
    aruco_stage["object_hololens_current"] = object_hololens_current
    aruco_stage["sam3_spatial_box_aruco_synced"] = synced_spatial_box is not None

    _write_debug(task, aruco_stage)
    save_task_json(json_path, task)

    if task_id:
        update_task_aruco_coordinate_synced(task_id, True)

    model_bounds_refreshed = False
    if refresh_model_bounds:
        model_bounds_refreshed = _sync_model_bounds(json_path)
    aruco_stage["model_bounds_refreshed"] = bool(model_bounds_refreshed)
    if refresh_model_bounds:
        refreshed_task = load_task_json(json_path)
        _write_debug(refreshed_task, aruco_stage)
        save_task_json(json_path, refreshed_task)

    print("[INFO] aruco_sync : object pose converted from HoloLens-local into ArUco-local coordinates")
    print("[OK] aruco_sync")
    return True


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
