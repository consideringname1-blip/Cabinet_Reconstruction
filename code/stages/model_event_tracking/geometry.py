from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS, quat_xyzw_to_rotation_matrix

from .schemas import ProjectedBox


def _load_json(path_or_payload: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, Mapping):
        return dict(path_or_payload)
    path = Path(path_or_payload)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _parse_float_array(value: Any, expected_count: int, label: str) -> np.ndarray:
    if isinstance(value, str):
        normalized = re.sub(r"[\[\],]+", " ", value)
        array = np.fromstring(normalized, sep=" ", dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != expected_count:
        raise ValueError(f"{label} must contain {expected_count} numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def load_camera_matrix(camera_info_json: str | Path | Mapping[str, Any]) -> np.ndarray:
    payload = _load_json(camera_info_json)
    message = payload.get("message") if isinstance(payload.get("message"), Mapping) else payload
    values = message.get("k") or message.get("K") or message.get("camera_matrix")
    if values is None:
        raise ValueError("camera info JSON does not contain k/K/camera_matrix")
    return _parse_float_array(values, 9, "camera matrix").reshape(3, 3)


def load_marker_pose_opencv(
    marker_pose_json: str | Path | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(marker_pose_json)
    pose = payload.get("opencv_camera_pose") if isinstance(payload.get("opencv_camera_pose"), Mapping) else payload
    if not isinstance(pose, Mapping):
        raise ValueError("marker pose JSON does not contain opencv_camera_pose")

    if pose.get("rotation_matrix") is not None:
        rotation = _parse_float_array(pose.get("rotation_matrix"), 9, "rotation_matrix").reshape(3, 3)
    elif pose.get("rotation_quaternion_xyzw") is not None:
        rotation = quat_xyzw_to_rotation_matrix(
            _parse_float_array(pose.get("rotation_quaternion_xyzw"), 4, "rotation_quaternion_xyzw")
        )
    else:
        raise ValueError("marker opencv pose is missing rotation_matrix/rotation_quaternion_xyzw")

    translation_value = pose.get("tvec_m") if pose.get("tvec_m") is not None else pose.get("position")
    translation = _parse_float_array(translation_value, 3, "marker translation").reshape(3)
    return rotation.astype(np.float64), translation.astype(np.float64)


def load_model_bounds_corners_aruco(task_or_bounds: Mapping[str, Any]) -> np.ndarray:
    bounds = task_or_bounds.get("ModelBounds") if isinstance(task_or_bounds.get("ModelBounds"), Mapping) else task_or_bounds
    corners = bounds.get("corners_aruco") if isinstance(bounds, Mapping) else None
    if corners is None:
        raise ValueError("ModelBounds.corners_aruco is missing")
    array = np.asarray(corners, dtype=np.float64)
    if array.shape != (8, 3):
        raise ValueError("ModelBounds.corners_aruco must have shape (8, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("ModelBounds.corners_aruco contains non-finite values")
    return array


def transform_aruco_unity_points_to_camera_opencv(
    points_aruco_unity: np.ndarray,
    marker_rotation_camera_marker_cv: np.ndarray,
    marker_translation_camera_marker_cv: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_aruco_unity, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_aruco_unity must have shape (N, 3)")

    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    points_marker_cv = (basis @ points.T).T
    rotation = np.asarray(marker_rotation_camera_marker_cv, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(marker_translation_camera_marker_cv, dtype=np.float64).reshape(3)
    return (rotation @ points_marker_cv.T).T + translation.reshape(1, 3)


def project_camera_points(
    camera_matrix: np.ndarray,
    points_camera_m: np.ndarray,
    *,
    min_z_m: float = 1.0e-6,
) -> np.ndarray:
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    points = np.asarray(points_camera_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_m must have shape (N, 3)")

    out = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    z = points[:, 2]
    valid = z > float(min_z_m)
    if np.any(valid):
        out[valid, 0] = k[0, 0] * points[valid, 0] / z[valid] + k[0, 2]
        out[valid, 1] = k[1, 1] * points[valid, 1] / z[valid] + k[1, 2]
    return out


def pixel_bbox_from_points(
    pixel_points: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
    padding_px: float = 0.0,
) -> tuple[float, float, float, float]:
    points = np.asarray(pixel_points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        raise ValueError("no projected box points are in front of the camera")
    selected = points[valid]
    x0 = float(selected[:, 0].min() - padding_px)
    y0 = float(selected[:, 1].min() - padding_px)
    x1 = float(selected[:, 0].max() + padding_px)
    y1 = float(selected[:, 1].max() + padding_px)
    if image_size is not None:
        width, height = image_size
        x0 = min(max(x0, 0.0), max(0.0, float(width - 1)))
        x1 = min(max(x1, 0.0), max(0.0, float(width - 1)))
        y0 = min(max(y0, 0.0), max(0.0, float(height - 1)))
        y1 = min(max(y1, 0.0), max(0.0, float(height - 1)))
    return (x0, y0, x1, y1)


def project_model_bounds_to_shigurei(
    task_or_bounds: Mapping[str, Any],
    marker_pose_json: str | Path | Mapping[str, Any],
    camera_matrix: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
    padding_px: float = 0.0,
) -> ProjectedBox:
    corners_aruco_unity = load_model_bounds_corners_aruco(task_or_bounds)
    marker_rotation, marker_translation = load_marker_pose_opencv(marker_pose_json)
    corners_camera = transform_aruco_unity_points_to_camera_opencv(
        corners_aruco_unity,
        marker_rotation,
        marker_translation,
    )
    pixel_points = project_camera_points(camera_matrix, corners_camera)
    bbox = pixel_bbox_from_points(pixel_points, image_size=image_size, padding_px=padding_px)
    return ProjectedBox(corners_camera_m=corners_camera, pixel_points=pixel_points, bbox_xyxy=bbox)


def oriented_box_axes_from_corners(corners_camera_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(corners_camera_m, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError("corners_camera_m must have shape (8, 3)")
    origin = corners[0]
    edge_vectors = np.vstack([corners[1] - origin, corners[3] - origin, corners[4] - origin])
    lengths = np.linalg.norm(edge_vectors, axis=1)
    if np.any(lengths <= 1.0e-9):
        raise ValueError("box has a near-zero edge length")
    axes = edge_vectors / lengths.reshape(3, 1)
    half_extents = lengths * 0.5
    center = origin + 0.5 * edge_vectors.sum(axis=0)
    return center, axes, half_extents


def point_to_oriented_box_signed_distance(
    point_camera_m: np.ndarray,
    corners_camera_m: np.ndarray,
    *,
    margin_m: float = 0.0,
) -> float:
    point = np.asarray(point_camera_m, dtype=np.float64).reshape(3)
    center, axes, half_extents = oriented_box_axes_from_corners(corners_camera_m)
    expanded_half_extents = half_extents + max(0.0, float(margin_m))
    local = axes @ (point - center)
    delta = np.abs(local) - expanded_half_extents
    outside = np.maximum(delta, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance
    return float(np.max(delta))


def point_inside_oriented_box(
    point_camera_m: np.ndarray,
    corners_camera_m: np.ndarray,
    *,
    margin_m: float = 0.0,
) -> bool:
    return point_to_oriented_box_signed_distance(
        point_camera_m,
        corners_camera_m,
        margin_m=margin_m,
    ) <= 0.0


def finite_distance(value: float | None) -> float:
    if value is None:
        return math.inf
    try:
        value = float(value)
    except Exception:
        return math.inf
    return value if math.isfinite(value) else math.inf
