"""Model-bound calculation for HoloLens reconstruction outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stages.hololens3d_reconstruction.model_generation_common import resolve_runtime_mesh_source
from coordinate_systems import MODEL_INPUT_TO_FBX_RUNTIME_LOCAL, quat_xyzw_to_rotation_matrix
from stages.hololens3d_reconstruction.object_alignment_common import read_obj_vertices
from task_db import (
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


def _resolve_final_source_model(task: dict) -> Path:
    source = resolve_runtime_mesh_source(task, require_mtl_image=False)
    if source is None:
        raise ValueError("RuntimeMesh.mesh is missing")
    return source.mesh_path


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

    source_path = resolve_project_path(_resolve_final_source_model(task))
    source_model_path = normalize_path_for_storage(source_path)

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
        object_position, object_rotation, object_scale = _pose_components(
            object_aruco,
            "object_aruco",
        )
        if object_scale is None:
            object_hololens_original = task.get("object_hololens_original") or {}
            _world_position, _world_rotation, object_scale = _pose_components(
                object_hololens_original,
                "object_hololens_original",
            )
        if object_scale is None:
            raise ValueError("object_aruco.scale or object_hololens_original.scale must have 3 values")

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
