from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from artifact_layout import aruco_worker_frame_color
from hololens3d_reconstruction.pose_math import (
    serialize_pose,
)
from coordinate_systems import (
    convert_hololens_pv_pose_matrix_to_unity_pose_components,
    convert_opencv_camera_pose_to_unity_camera_pose,
)


def resolve_marker_configs(db_markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for marker in db_markers:
        if not isinstance(marker, dict):
            raise ValueError("ArUco marker database row must be an object")
        try:
            marker_id = int(marker["marker_id"])
            marker_size_mm = float(marker["marker_size_mm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ArUco marker requires marker_id and marker_size_mm") from exc
        dictionary = str(marker.get("dictionary") or "").strip()
        if not dictionary or marker_size_mm <= 0.0:
            raise ValueError(f"invalid ArUco marker configuration: {marker_id}")
        markers.append(
            {
                "marker_id": marker_id,
                "dictionary": dictionary,
                "marker_size_mm": marker_size_mm,
                "reference_image_name": marker.get("reference_image_name") or "",
            }
        )
    return markers


def resolve_task_name(task: dict[str, Any]) -> str:
    task_name = str(task.get("task_name") or "").strip()
    if not task_name:
        raise ValueError("task_name is required")
    return task_name


def resolve_pv_frames(task: dict[str, Any]) -> list[dict[str, Any]]:
    frames = task.get("PVCameraFrames")
    if not isinstance(frames, list) or not frames or not all(isinstance(frame, dict) for frame in frames):
        raise ValueError("PVCameraFrames must contain at least one frame object")
    return [dict(frame) for frame in frames]


def resolve_pv_image_path(frame: dict[str, Any]) -> Path:
    if str(frame.get("artifact_root") or "").strip() != "aruco_worker":
        raise ValueError("PVCamera frame must reference aruco_worker artifacts")
    task_timestamp = str(frame.get("task_timestamp") or "").strip()
    frame_timestamp = str(frame.get("artifact_timestamp") or "").strip()
    if not task_timestamp or not frame_timestamp:
        raise ValueError("aruco_worker frame requires task_timestamp and artifact_timestamp")
    artifact_candidate = aruco_worker_frame_color(task_timestamp, frame_timestamp).resolve()
    if artifact_candidate.is_file():
        return artifact_candidate
    raise FileNotFoundError(f"PVCamera image not found: {artifact_candidate}")


def resolve_pv_camera_matrix(frame: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(frame.get("k"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("PVCamera frame k must be a finite 3x3 matrix")
    return matrix


def resolve_selection_roi(task: dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    purpose = str(task.get("purpose") or "").strip()
    selection = task.get("SelectionBox") or {}
    top_left = np.asarray(selection.get("top_left"), dtype=np.float64)
    bottom_right = np.asarray(selection.get("bottom_right"), dtype=np.float64)
    if top_left.shape != (2,) or bottom_right.shape != (2,):
        if purpose == "aruco_reference":
            return 0, 0, int(image_width), int(image_height)
        raise ValueError("SelectionBox.top_left and bottom_right must each contain 2 values")

    left = float(np.clip(min(top_left[0], bottom_right[0]), 0.0, 1.0))
    right = float(np.clip(max(top_left[0], bottom_right[0]), 0.0, 1.0))
    top = float(np.clip(min(top_left[1], bottom_right[1]), 0.0, 1.0))
    bottom = float(np.clip(max(top_left[1], bottom_right[1]), 0.0, 1.0))

    x0 = int(np.floor(left * image_width))
    x1 = int(np.ceil(right * image_width))
    y0 = int(np.floor(top * image_height))
    y1 = int(np.ceil(bottom * image_height))

    x0 = int(np.clip(x0, 0, max(image_width - 1, 0)))
    y0 = int(np.clip(y0, 0, max(image_height - 1, 0)))
    x1 = int(np.clip(x1, x0 + 1, image_width))
    y1 = int(np.clip(y1, y0 + 1, image_height))
    return x0, y0, x1, y1


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    normalized = u @ vh
    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vh
    return normalized.astype(np.float64)


def resolve_pv_camera_world_pose(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pv_pose = np.asarray(frame.get("pose"), dtype=np.float64)
    if pv_pose.shape != (4, 4) or not np.isfinite(pv_pose).all():
        raise ValueError("PVCamera frame pose must be a finite 4x4 matrix")
    translation, rotation, _quat_xyzw = convert_hololens_pv_pose_matrix_to_unity_pose_components(pv_pose)
    return translation.astype(np.float64), rotation.astype(np.float64)


def convert_cv_pose_to_unity_pose(rotation_cv: np.ndarray, translation_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_unity, translation_unity = convert_opencv_camera_pose_to_unity_camera_pose(
        rotation_cv,
        translation_cv,
    )
    return rotation_unity.astype(np.float64), translation_unity.astype(np.float64)


def compose_world_pose(
    camera_world_translation: np.ndarray,
    camera_world_rotation: np.ndarray,
    local_translation: np.ndarray,
    local_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_translation = (camera_world_rotation @ np.asarray(local_translation, dtype=np.float64)) + np.asarray(
        camera_world_translation,
        dtype=np.float64,
    )
    world_rotation = orthonormalize_rotation(
        np.asarray(camera_world_rotation, dtype=np.float64) @ np.asarray(local_rotation, dtype=np.float64)
    )
    return world_translation.astype(np.float64), world_rotation.astype(np.float64)


def invert_pose(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = orthonormalize_rotation(rotation)
    translation = np.asarray(translation, dtype=np.float64)
    inverse_rotation = rotation.T
    inverse_translation = -(inverse_rotation @ translation)
    return inverse_rotation.astype(np.float64), inverse_translation.astype(np.float64)


def pose_to_payload(
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    scale: list[float] | None = None,
) -> dict[str, Any]:
    payload = serialize_pose(rotation, translation)
    if scale is not None:
        payload["scale"] = [float(v) for v in scale]
    return payload


def load_json_payload(raw_json: str | None) -> dict[str, Any] | None:
    if not raw_json:
        return None
    loaded = json.loads(raw_json)
    return loaded if isinstance(loaded, dict) else None
