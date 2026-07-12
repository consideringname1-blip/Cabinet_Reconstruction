from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from coordinate_systems import (
    UNITY_TO_OPENCV_CAMERA_BASIS,
    orthonormalize_rotation,
    quat_xyzw_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
)


def vector3(value: Any, label: str = "vector") -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{label} must have exactly 3 numeric values")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains non-finite values")
    return vector.astype(np.float64)


def parse_camera_matrix(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)]
    else:
        try:
            values = list(np.asarray(raw, dtype=np.float64).reshape(-1))
        except Exception:
            return None
    if len(values) != 9:
        return None
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] == 0 or matrix[1, 1] == 0:
        return None
    return matrix


def camera_matrix_from_info(camera_info: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(camera_info, Mapping):
        return None
    return parse_camera_matrix(camera_info.get("k"))


def camera_info_image_shape(camera_info: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(camera_info, Mapping):
        return None
    try:
        height = int(camera_info.get("height") or 0)
        width = int(camera_info.get("width") or 0)
    except Exception:
        return None
    return (height, width) if height > 0 and width > 0 else None


def pose_to_rt(pose: Mapping[str, Any], label: str = "pose") -> tuple[np.ndarray, np.ndarray, list[float] | None]:
    if not isinstance(pose, Mapping):
        raise ValueError(f"{label} must be an object")
    position = vector3(pose.get("position"), f"{label}.position")
    quaternion = np.asarray(pose.get("rotation_quaternion_xyzw"), dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"{label}.rotation_quaternion_xyzw must have 4 values")
    rotation = quat_xyzw_to_rotation_matrix(quaternion)

    scale_value = pose.get("scale")
    scale = None
    if isinstance(scale_value, (list, tuple)) and len(scale_value) == 3:
        scale = [float(v) for v in vector3(scale_value, f"{label}.scale")]
    return position, rotation.astype(np.float64), scale


def rt_to_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    scale: list[float] | tuple[float, float, float] | np.ndarray | None = None,
) -> dict[str, Any]:
    rotation = orthonormalize_rotation(np.asarray(rotation, dtype=np.float64))
    translation = vector3(translation, "pose.position")
    payload: dict[str, Any] = {
        "position": [float(v) for v in translation],
        "rotation_quaternion_xyzw": [float(v) for v in rotation_matrix_to_quat_xyzw(rotation)],
    }
    if scale is not None:
        payload["scale"] = [float(v) for v in vector3(scale, "pose.scale")]
    return payload


def minimal_pose_payload(pose: Mapping[str, Any], *, include_scale: bool = True) -> dict[str, Any]:
    position, rotation, scale = pose_to_rt(pose)
    return rt_to_pose(rotation, position, scale=scale if include_scale else None)


def invert_pose_rt(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = orthonormalize_rotation(np.asarray(rotation, dtype=np.float64))
    translation = vector3(translation, "pose.position")
    inverse_rotation = rotation.T
    inverse_translation = -(inverse_rotation @ translation)
    return inverse_rotation.astype(np.float64), inverse_translation.astype(np.float64)


def compose_pose_rt(
    parent_rotation: np.ndarray,
    parent_translation: np.ndarray,
    child_rotation: np.ndarray,
    child_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rotation = orthonormalize_rotation(parent_rotation)
    child_rotation = orthonormalize_rotation(child_rotation)
    parent_translation = vector3(parent_translation, "parent.position")
    child_translation = vector3(child_translation, "child.position")
    rotation = orthonormalize_rotation(parent_rotation @ child_rotation)
    translation = (parent_rotation @ child_translation) + parent_translation
    return rotation.astype(np.float64), translation.astype(np.float64)


def resolve_hololens_original_pose(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    pose = task.get("object_hololens_original")
    return pose if isinstance(pose, Mapping) else None


def hololens_pose_to_aruco_pose(
    object_hololens_pose: Mapping[str, Any],
    aruco_reference_hololens_pose: Mapping[str, Any],
) -> dict[str, Any]:
    object_position, object_rotation, object_scale = pose_to_rt(object_hololens_pose, "object_hololens")
    marker_position, marker_rotation, _marker_scale = pose_to_rt(aruco_reference_hololens_pose, "aruco_reference")
    marker_inverse_rotation, marker_inverse_translation = invert_pose_rt(marker_rotation, marker_position)
    aruco_rotation, aruco_translation = compose_pose_rt(
        marker_inverse_rotation,
        marker_inverse_translation,
        object_rotation,
        object_position,
    )
    return rt_to_pose(aruco_rotation, aruco_translation, scale=object_scale)


def aruco_pose_to_hololens_pose(
    object_aruco_pose: Mapping[str, Any],
    aruco_reference_hololens_pose: Mapping[str, Any],
) -> dict[str, Any]:
    object_position, object_rotation, object_scale = pose_to_rt(object_aruco_pose, "object_aruco")
    marker_position, marker_rotation, _marker_scale = pose_to_rt(aruco_reference_hololens_pose, "aruco_reference")
    hololens_rotation, hololens_translation = compose_pose_rt(
        marker_rotation,
        marker_position,
        object_rotation,
        object_position,
    )
    return rt_to_pose(hololens_rotation, hololens_translation, scale=object_scale)


def hololens_point_to_aruco(point_hololens: Any, aruco_reference_hololens_pose: Mapping[str, Any]) -> np.ndarray:
    point = vector3(point_hololens, "point_hololens")
    marker_position, marker_rotation, _marker_scale = pose_to_rt(aruco_reference_hololens_pose, "aruco_reference")
    return (marker_rotation.T @ (point - marker_position)).astype(np.float64)


def hololens_vector_to_aruco(vector_hololens: Any, aruco_reference_hololens_pose: Mapping[str, Any]) -> np.ndarray:
    vector = vector3(vector_hololens, "vector_hololens")
    _marker_position, marker_rotation, _marker_scale = pose_to_rt(aruco_reference_hololens_pose, "aruco_reference")
    return (marker_rotation.T @ vector).astype(np.float64)


def aruco_point_to_hololens(point_aruco: Any, aruco_reference_hololens_pose: Mapping[str, Any]) -> np.ndarray:
    point = vector3(point_aruco, "point_aruco")
    marker_position, marker_rotation, _marker_scale = pose_to_rt(aruco_reference_hololens_pose, "aruco_reference")
    return (marker_rotation @ point + marker_position).astype(np.float64)


def aruco_points_to_hololens(points_aruco: Any, aruco_reference_hololens_pose: Mapping[str, Any]) -> np.ndarray:
    points = np.asarray(points_aruco, dtype=np.float64).reshape(-1, 3)
    marker_position, marker_rotation, _marker_scale = pose_to_rt(aruco_reference_hololens_pose, "aruco_reference")
    return ((marker_rotation @ points.T).T + marker_position.reshape(1, 3)).astype(np.float64)


def hololens_ray_to_aruco_payload(
    payload: Mapping[str, Any],
    aruco_reference_hololens_pose: Mapping[str, Any],
) -> dict[str, Any]:
    origin = hololens_point_to_aruco(payload.get("origin_hololens"), aruco_reference_hololens_pose)
    direction = hololens_vector_to_aruco(payload.get("direction_hololens"), aruco_reference_hololens_pose)
    converted = dict(payload)
    converted["origin_aruco"] = [float(v) for v in origin]
    converted["direction_aruco"] = [float(v) for v in direction]
    if payload.get("end_hololens") is not None:
        endpoint = hololens_point_to_aruco(payload.get("end_hololens"), aruco_reference_hololens_pose)
        converted["end_aruco"] = [float(v) for v in endpoint]
    converted["input_coordinate_space"] = "hololens_current_local"
    converted["query_coordinate_space"] = "aruco"
    return converted


def aruco_points_to_shigure_camera(
    points_aruco: Any,
    marker_rotation_camera_marker_cv: np.ndarray,
    marker_translation_camera_marker_cv: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_aruco, dtype=np.float64).reshape(-1, 3)
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    points_marker_cv = (basis @ points.T).T
    marker_rotation = orthonormalize_rotation(marker_rotation_camera_marker_cv)
    marker_translation = vector3(marker_translation_camera_marker_cv, "marker_translation_camera_marker_cv")
    return ((marker_rotation @ points_marker_cv.T).T + marker_translation.reshape(1, 3)).astype(np.float64)


def shigure_camera_points_to_aruco(
    points_camera_m: Any,
    marker_rotation_camera_marker_cv: np.ndarray,
    marker_translation_camera_marker_cv: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_camera_m, dtype=np.float64).reshape(-1, 3)
    marker_rotation = orthonormalize_rotation(marker_rotation_camera_marker_cv)
    marker_translation = vector3(marker_translation_camera_marker_cv, "marker_translation_camera_marker_cv")
    marker_cv = (marker_rotation.T @ (points - marker_translation.reshape(1, 3)).T).T
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    return (basis @ marker_cv.T).T.astype(np.float64)


def project_camera_points_to_pixels(points_camera_m: Any, camera_matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera_m, dtype=np.float64).reshape(-1, 3)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    z = points[:, 2]
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    visible = z > 1.0e-6
    pixels[visible, 0] = (k[0, 0] * points[visible, 0] / z[visible]) + k[0, 2]
    pixels[visible, 1] = (k[1, 1] * points[visible, 1] / z[visible]) + k[1, 2]
    return pixels, visible


def project_aruco_points_to_shigure_pixels(
    points_aruco: Any,
    marker_rotation_camera_marker_cv: np.ndarray,
    marker_translation_camera_marker_cv: np.ndarray,
    camera_matrix: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = aruco_points_to_shigure_camera(
        points_aruco,
        marker_rotation_camera_marker_cv,
        marker_translation_camera_marker_cv,
    )
    pixels, visible = project_camera_points_to_pixels(points_camera, camera_matrix)
    return points_camera, pixels, visible


def pixel_depth_to_shigure_camera(pixel_xy: tuple[float, float], depth_m: float, camera_matrix: Any) -> np.ndarray:
    if not np.isfinite(depth_m) or float(depth_m) <= 0.0:
        raise ValueError("depth_m must be positive")
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    z = float(depth_m)
    return np.asarray([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], dtype=np.float64)


def pixel_depth_to_aruco(
    pixel_xy: tuple[float, float],
    depth_m: float,
    camera_matrix: Any,
    marker_rotation_camera_marker_cv: np.ndarray,
    marker_translation_camera_marker_cv: np.ndarray,
) -> np.ndarray:
    camera_point = pixel_depth_to_shigure_camera(pixel_xy, depth_m, camera_matrix)
    return shigure_camera_points_to_aruco(
        camera_point.reshape(1, 3),
        marker_rotation_camera_marker_cv,
        marker_translation_camera_marker_cv,
    ).reshape(3)
