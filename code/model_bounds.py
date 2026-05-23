from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from config import BLENDER_FBX_DIR
from model_generation_common import resolve_model_source_from_stage, resolve_runtime_or_generated_source
from object_alignment_common import MODEL_INPUT_TO_FBX_RUNTIME_LOCAL, read_obj_vertices
from task_db import (
    get_latest_ready_model_bounds,
    get_ready_model_bounds_in_range,
    get_task_by_task_id,
    upsert_model_bounds,
)
from task_json import (
    load_task_json,
    normalize_path_for_storage,
    resolve_project_path,
    resolve_task_json_path,
    save_task_json,
)
from unity_coordinate_utils import quat_xyzw_to_rotation_matrix


DEFAULT_RAY_MAX_DISTANCE_M = 10.0
RAY_EPSILON = 1.0e-8


def _json_or_none(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _vector3(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{label} must have exactly 3 numeric values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector.astype(np.float64)


def _pose_components(pose: dict, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(pose, dict):
        raise ValueError(f"{label} must be an object")

    position = _vector3(pose.get("position"), f"{label}.position")
    quaternion = np.asarray(pose.get("rotation_quaternion_xyzw"), dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"{label}.rotation_quaternion_xyzw must have 4 values")
    rotation = quat_xyzw_to_rotation_matrix(quaternion)

    scale_value = pose.get("scale")
    scale = None
    if isinstance(scale_value, list) and len(scale_value) == 3:
        scale = _vector3(scale_value, f"{label}.scale")
    return position, rotation, scale


def _resolve_uploaded_at(task: dict, task_row: dict | None) -> str | None:
    if task_row and task_row.get("created_at"):
        return str(task_row.get("created_at"))
    value = task.get("server_received_utc")
    return str(value) if value else None


def _resolve_fbx_name(task: dict) -> str | None:
    blender_info = task.get("Blender") or {}
    fbx_name = blender_info.get("fbx")
    return str(fbx_name) if fbx_name else None


def _resolve_final_source_model(task: dict) -> tuple[Path, str]:
    blender_info = task.get("Blender") or {}
    source_stage = str(blender_info.get("source_stage") or "").strip()
    source_mesh = str(blender_info.get("source_mesh") or "").strip()
    if source_mesh:
        source = resolve_model_source_from_stage(task, source_stage, source_mesh)
        return source.mesh_path, source.source_stage

    source = resolve_runtime_or_generated_source(task, require_mtl_image=False)
    return source.mesh_path, source.source_stage


def _aabb_corners(min_corner: np.ndarray, max_corner: np.ndarray) -> list[list[float]]:
    x0, y0, z0 = [float(v) for v in min_corner]
    x1, y1, z1 = [float(v) for v in max_corner]
    return [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]


def _row_task_id(task: dict) -> str:
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is missing")
    return task_id


def compute_and_store_model_bounds(json_path_arg: str | Path) -> dict:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = _row_task_id(task)
    task_row = get_task_by_task_id(task_id)
    uploaded_at = _resolve_uploaded_at(task, task_row)
    model_name = str(task.get("task_name") or task_id)
    fbx_name = _resolve_fbx_name(task)
    aruco_reference_task_id = None
    aruco_stage = (
        (task.get("debug") or {})
        .get("pose_transform_stages", {})
        .get("aruco_stage", {})
    )
    if isinstance(aruco_stage, dict):
        reference_value = aruco_stage.get("reference_task_id")
        aruco_reference_task_id = str(reference_value) if reference_value else None

    source_path: Path | None = None
    try:
        source_path, _source_stage = _resolve_final_source_model(task)
        source_path = resolve_project_path(source_path)
        source_model_path = normalize_path_for_storage(source_path)
    except Exception:
        source_model_path = None

    upsert_model_bounds(
        task_id=task_id,
        status="pending",
        uploaded_at=uploaded_at,
        model_name=model_name,
        fbx_name=fbx_name,
        aruco_reference_task_id=aruco_reference_task_id,
        source_model_path=source_model_path,
    )

    object_aruco = task.get("object_aruco")
    if not isinstance(object_aruco, dict):
        row = upsert_model_bounds(
            task_id=task_id,
            status="pending_reference",
            uploaded_at=uploaded_at,
            model_name=model_name,
            fbx_name=fbx_name,
            aruco_reference_task_id=aruco_reference_task_id,
            source_model_path=source_model_path,
            error_message="object_aruco is missing; wait for a valid ArUco reference",
        )
        task["ModelBounds"] = {
            "status": "pending_reference",
            "coordinate_space": "aruco",
            "error_message": row.get("error_message"),
        }
        save_task_json(json_path, task)
        return row

    try:
        if source_path is None:
            source_path, _source_stage = _resolve_final_source_model(task)
            source_path = resolve_project_path(source_path)
            source_model_path = normalize_path_for_storage(source_path)

        object_position, object_rotation, object_scale = _pose_components(
            object_aruco,
            "object_aruco",
        )
        if object_scale is None:
            object_world = task.get("object_world") or {}
            _world_position, _world_rotation, object_scale = _pose_components(
                object_world,
                "object_world",
            )
        if object_scale is None:
            raise ValueError("object_aruco.scale or object_world.scale must have 3 values")

        vertices = read_obj_vertices(source_path).astype(np.float64)
        runtime_local = vertices @ np.asarray(MODEL_INPUT_TO_FBX_RUNTIME_LOCAL, dtype=np.float64).T
        runtime_local = runtime_local * object_scale.reshape(1, 3)
        aruco_points = (object_rotation @ runtime_local.T).T + object_position.reshape(1, 3)
        if aruco_points.size == 0:
            raise ValueError(f"No vertices found for bounds: {source_path}")

        min_corner = aruco_points.min(axis=0)
        max_corner = aruco_points.max(axis=0)
        corners = _aabb_corners(min_corner, max_corner)

        row = upsert_model_bounds(
            task_id=task_id,
            status="ready",
            uploaded_at=uploaded_at,
            model_name=model_name,
            fbx_name=fbx_name,
            aruco_reference_task_id=aruco_reference_task_id,
            object_aruco_json=object_aruco,
            aabb_min_aruco_json=[float(v) for v in min_corner],
            aabb_max_aruco_json=[float(v) for v in max_corner],
            corners_aruco_json=corners,
            source_model_path=source_model_path,
        )
        task["ModelBounds"] = {
            "status": "ready",
            "coordinate_space": "aruco",
            "source_model_path": source_model_path,
            "aabb_min_aruco": [float(v) for v in min_corner],
            "aabb_max_aruco": [float(v) for v in max_corner],
            "corners_aruco": corners,
        }
        save_task_json(json_path, task)
        return row
    except Exception as exc:
        row = upsert_model_bounds(
            task_id=task_id,
            status="failed",
            uploaded_at=uploaded_at,
            model_name=model_name,
            fbx_name=fbx_name,
            aruco_reference_task_id=aruco_reference_task_id,
            object_aruco_json=object_aruco,
            source_model_path=source_model_path,
            error_message=str(exc),
        )
        task["ModelBounds"] = {
            "status": "failed",
            "coordinate_space": "aruco",
            "source_model_path": source_model_path,
            "error_message": str(exc),
        }
        save_task_json(json_path, task)
        raise


def decode_model_bounds_row(row: dict) -> dict:
    decoded = dict(row)
    decoded["object_aruco"] = row.get("object_aruco") or _json_or_none(row.get("object_aruco_json"))
    decoded["aabb_min_aruco"] = row.get("aabb_min_aruco") or _json_or_none(row.get("aabb_min_aruco_json"))
    decoded["aabb_max_aruco"] = row.get("aabb_max_aruco") or _json_or_none(row.get("aabb_max_aruco_json"))
    decoded["corners_aruco"] = row.get("corners_aruco") or _json_or_none(row.get("corners_aruco_json"))
    return decoded


def _normalize_ray(
    payload: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    origin = _vector3(payload.get("origin_aruco"), "origin_aruco")
    direction = _vector3(payload.get("direction_aruco"), "direction_aruco")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= RAY_EPSILON:
        raise ValueError("direction_aruco must be non-zero")
    direction = direction / direction_norm

    requested_max = payload.get("max_distance_m", DEFAULT_RAY_MAX_DISTANCE_M)
    try:
        max_distance = float(requested_max)
    except Exception as exc:
        raise ValueError("max_distance_m must be numeric") from exc
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        max_distance = DEFAULT_RAY_MAX_DISTANCE_M

    endpoint = None
    if payload.get("end_aruco") is not None:
        endpoint = _vector3(payload.get("end_aruco"), "end_aruco")
        endpoint_distance = float(np.linalg.norm(endpoint - origin))
        if endpoint_distance <= RAY_EPSILON:
            raise ValueError("end_aruco must be different from origin_aruco")
        max_distance = min(max_distance, endpoint_distance)

    return origin, direction, endpoint, max_distance


def ray_aabb_intersection_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    max_distance: float,
) -> float | None:
    t_min = 0.0
    t_max = float(max_distance)

    for axis in range(3):
        axis_origin = float(origin[axis])
        axis_direction = float(direction[axis])
        axis_min = float(min_corner[axis])
        axis_max = float(max_corner[axis])

        if abs(axis_direction) <= RAY_EPSILON:
            if axis_origin < axis_min or axis_origin > axis_max:
                return None
            continue

        inv_direction = 1.0 / axis_direction
        near = (axis_min - axis_origin) * inv_direction
        far = (axis_max - axis_origin) * inv_direction
        if near > far:
            near, far = far, near

        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return None

    if t_min < 0.0 or t_min > max_distance:
        return None
    return float(t_min)


def query_first_ray_hit(payload: dict, rows: list[dict]) -> dict:
    origin, direction, _endpoint, max_distance = _normalize_ray(payload)
    best_row = None
    best_distance = None

    for row in rows:
        decoded = decode_model_bounds_row(row)
        min_corner = _vector3(decoded.get("aabb_min_aruco"), "aabb_min_aruco")
        max_corner = _vector3(decoded.get("aabb_max_aruco"), "aabb_max_aruco")
        hit_distance = ray_aabb_intersection_distance(
            origin,
            direction,
            min_corner,
            max_corner,
            max_distance,
        )
        if hit_distance is None:
            continue
        if best_distance is None or hit_distance < best_distance:
            best_distance = hit_distance
            best_row = decoded

    if best_row is None or best_distance is None:
        return {
            "hit": False,
            "candidates_checked": len(rows),
            "max_distance_m": float(max_distance),
        }

    hit_point = origin + (direction * best_distance)
    return {
        "hit": True,
        "candidates_checked": len(rows),
        "hit_distance_m": float(best_distance),
        "hit_point_aruco": [float(v) for v in hit_point],
        "row": best_row,
        "max_distance_m": float(max_distance),
    }


def latest_bounds_for_ray(payload: dict) -> tuple[list[dict], dict]:
    limit = max(1, min(int(payload.get("limit", 5) or 5), 50))
    rows = get_latest_ready_model_bounds(limit)
    return rows, query_first_ray_hit(payload, rows)


def range_bounds_for_ray(payload: dict) -> tuple[list[dict], dict]:
    start = str(payload.get("start") or "").strip()
    end = str(payload.get("end") or "").strip()
    limit = max(1, min(int(payload.get("limit", 50) or 50), 200))
    rows = get_ready_model_bounds_in_range(start, end, limit=limit)
    return rows, query_first_ray_hit(payload, rows)
