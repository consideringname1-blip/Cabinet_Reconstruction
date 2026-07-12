"""Projection helpers for Shigure RGB-D observations."""

from __future__ import annotations

import math
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spatial_transforms import camera_matrix_from_info, project_aruco_points_to_shigure_pixels
from stages.shigure_history.marker_history import latest_marker_pose_path


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _marker_pose() -> tuple[np.ndarray, np.ndarray, Path]:
    path = latest_marker_pose_path()
    if path is None:
        raise FileNotFoundError("Shigure ArUco marker pose is unavailable")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    pose = payload.get("opencv_camera_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("marker pose must contain opencv_camera_pose")
    rotation = _vector(pose.get("rotation_matrix"), 9, "rotation_matrix").reshape(3, 3)
    translation = _vector(pose.get("tvec_m"), 3, "tvec_m")
    return rotation, translation, path


def _spatial_box(task: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    box = task.get("Sam3SpatialBox")
    if not isinstance(box, Mapping) or box.get("status") != "ready":
        raise ValueError("Sam3SpatialBox is not ready")
    center = _vector(box.get("center_aruco"), 3, "Sam3SpatialBox.center_aruco")
    corners = np.asarray(box.get("corners_aruco"), dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("Sam3SpatialBox.corners_aruco must be an 8x3 finite array")
    return center, corners


def _circle_mask(
    image_shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius_px: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_shape
    center_x, center_y = center_xy
    radius = max(1.0, float(radius_px))
    bbox = (
        int(max(0, math.floor(center_x - radius))),
        int(max(0, math.floor(center_y - radius))),
        int(min(width, math.ceil(center_x + radius))),
        int(min(height, math.ceil(center_y + radius))),
    )
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=255,
    )
    return np.asarray(image, dtype=np.uint8) > 0, bbox


def project_spatial_box_circle_to_shigure(
    task: Mapping[str, Any],
    camera_info: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Project the canonical Sam3 preview box into the Shigure RGB image."""

    try:
        camera_matrix = camera_matrix_from_info(camera_info)
        if camera_matrix is None:
            raise ValueError("camera_info.k is required")
        center_aruco, corners_aruco = _spatial_box(task)
        marker_rotation, marker_translation, marker_path = _marker_pose()
    except Exception as exc:
        return None, {"source": "sam3_spatial_box_circle", "reason": str(exc)}

    corner_camera, corner_pixels, corner_visible = project_aruco_points_to_shigure_pixels(
        corners_aruco,
        marker_rotation,
        marker_translation,
        camera_matrix,
    )
    center_camera, center_pixels, center_visible = project_aruco_points_to_shigure_pixels(
        center_aruco.reshape(1, 3),
        marker_rotation,
        marker_translation,
        camera_matrix,
    )
    center_xyz = center_camera[0]
    center_pixel = center_pixels[0]
    height, width = image_shape
    if (
        not bool(center_visible[0])
        or not np.isfinite(center_pixel).all()
        or center_pixel[0] < 0
        or center_pixel[1] < 0
        or center_pixel[0] >= width
        or center_pixel[1] >= height
    ):
        return None, {"source": "sam3_spatial_box_circle", "reason": "center_not_visible"}

    visible_pixels = corner_pixels[corner_visible]
    if visible_pixels.size == 0:
        return None, {"source": "sam3_spatial_box_circle", "reason": "corners_not_visible"}
    distances = np.linalg.norm(visible_pixels - center_pixel.reshape(1, 2), axis=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return None, {"source": "sam3_spatial_box_circle", "reason": "circle_radius_invalid"}

    radius_px = float(np.max(distances))
    mask, bbox = _circle_mask(
        image_shape,
        (float(center_pixel[0]), float(center_pixel[1])),
        radius_px,
    )
    visible_depths = corner_camera[corner_visible, 2]
    diagonal_m = float(np.linalg.norm(np.max(corners_aruco, axis=0) - np.min(corners_aruco, axis=0)))
    return mask, {
        "source": "sam3_spatial_box_circle",
        "reason": "ready",
        "marker_pose_path": str(marker_path),
        "center_pixel_xy": center_pixel.astype(float).tolist(),
        "center_depth_m": float(center_xyz[2]),
        "box_depth_min_m": float(np.min(visible_depths)),
        "box_depth_max_m": float(np.max(visible_depths)),
        "box_diagonal_m": diagonal_m,
        "circle_radius_px": radius_px,
        "bbox_xyxy": list(bbox),
        "projected_pixels": int(np.count_nonzero(mask)),
    }
