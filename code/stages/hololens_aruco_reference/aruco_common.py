from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from config import ARUCO_RAW_ROOT, ARUCO_TEMPLATE_PATH, UPLOAD_FOLDER
from hololens3d_reconstruction.pose_math import (
    quat_xyzw_to_rotation_matrix,
    serialize_pose,
)
from unity_coordinate_utils import (
    convert_hololens_pv_pose_matrix_to_unity_pose_components,
    convert_opencv_camera_pose_to_unity_camera_pose,
)


ARUCO_LOCAL_COORDINATE_BASIS = "aruco_local_x_right_y_up_z_forward"
UNITY_WORLD_COORDINATE_BASIS = "unity_world_x_right_y_up_z_forward"
WINDOWS_POSE_TO_UNITY_TRANSFORM = "windows_spatial_to_unity_flip_z"
OPENCV_CAMERA_TO_UNITY_TRANSFORM = "opencv_camera_to_unity_camera_flip_y"


def default_aruco_template() -> dict[str, Any]:
    return {
        "enabled": False,
        "dictionary": "",
        "marker_id": None,
        "marker_size_mm": None,
        "reference_image_name": "",
        "markers": [],
        "notes": "",
    }


def load_aruco_template() -> dict[str, Any]:
    if not ARUCO_TEMPLATE_PATH.is_file():
        return default_aruco_template()

    with ARUCO_TEMPLATE_PATH.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"ArUco template must be a JSON object: {ARUCO_TEMPLATE_PATH}")
    template = default_aruco_template()
    template.update(loaded)
    return template


def evaluate_aruco_template(template: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(template.get("enabled"))
    dictionary = str(template.get("dictionary") or "").strip()
    marker_id = template.get("marker_id")
    marker_size_mm = template.get("marker_size_mm")

    configured = (
        enabled
        and bool(dictionary)
        and marker_id is not None
        and marker_size_mm is not None
        and float(marker_size_mm) > 0.0
    )

    reason = ""
    if not enabled:
        reason = "template_disabled"
    elif not dictionary:
        reason = "dictionary_missing"
    elif marker_id is None:
        reason = "marker_id_missing"
    elif marker_size_mm is None:
        reason = "marker_size_mm_missing"
    elif float(marker_size_mm) <= 0.0:
        reason = "marker_size_mm_invalid"

    return {
        "enabled": enabled,
        "configured": configured,
        "reason": reason,
    }


def resolve_marker_configs(template: dict[str, Any], db_markers: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for marker in db_markers or []:
        try:
            marker_id = int(marker.get("marker_id"))
            marker_size_mm = float(marker.get("marker_size_mm"))
        except Exception:
            continue
        dictionary = str(marker.get("dictionary") or "").strip()
        if not dictionary or marker_size_mm <= 0.0:
            continue
        markers.append(
            {
                "marker_id": marker_id,
                "dictionary": dictionary,
                "marker_size_mm": marker_size_mm,
                "reference_image_name": marker.get("reference_image_name") or "",
                "source": "database",
            }
        )

    if markers:
        return markers

    configured_markers = template.get("markers")
    if isinstance(configured_markers, list):
        for marker in configured_markers:
            if not isinstance(marker, dict) or not bool(marker.get("enabled", True)):
                continue
            try:
                marker_id = int(marker.get("marker_id", marker.get("id")))
                marker_size_mm = float(marker.get("marker_size_mm") or template.get("marker_size_mm"))
            except Exception:
                continue
            dictionary = str(marker.get("dictionary") or template.get("dictionary") or "").strip()
            if not dictionary or marker_size_mm <= 0.0:
                continue
            markers.append(
                {
                    "marker_id": marker_id,
                    "dictionary": dictionary,
                    "marker_size_mm": marker_size_mm,
                    "reference_image_name": marker.get("reference_image_name") or "",
                    "source": "template",
                }
            )

    template_state = evaluate_aruco_template(template)
    if not markers and template_state["configured"]:
        markers.append(
            {
                "marker_id": int(template["marker_id"]),
                "dictionary": str(template["dictionary"]),
                "marker_size_mm": float(template["marker_size_mm"]),
                "reference_image_name": template.get("reference_image_name") or "",
                "source": "template_legacy",
            }
        )
    return markers


def resolve_task_name(task: dict[str, Any], fallback: str) -> str:
    return str(task.get("task_name") or fallback)


def resolve_pv_frames(task: dict[str, Any]) -> list[dict[str, Any]]:
    frames = task.get("PVCameraFrames")
    if isinstance(frames, list) and frames:
        return [dict(frame) for frame in frames if isinstance(frame, dict)]
    pv_info = task.get("PVCamera") or {}
    return [dict(pv_info)] if pv_info else []


def resolve_pv_image_path(task_or_frame: dict[str, Any]) -> Path:
    pv_info = task_or_frame.get("PVCamera") or task_or_frame
    name = str(pv_info.get("name") or "").strip()
    if not name:
        raise ValueError("PVCamera.name is required")

    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    upload_candidate = (UPLOAD_FOLDER / candidate).resolve()
    if upload_candidate.is_file():
        return upload_candidate

    raise FileNotFoundError(f"PVCamera image not found: {name}")


def resolve_pv_camera_matrix(task_or_frame: dict[str, Any]) -> np.ndarray:
    pv_info = task_or_frame.get("PVCamera") or task_or_frame
    k = np.asarray(pv_info.get("k"), dtype=np.float64)
    if k.shape == (3, 3):
        return k.astype(np.float64)
    if k.ndim == 2 and k.shape[0] >= 3 and k.shape[1] >= 3:
        return k[:3, :3].astype(np.float64)

    flat = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float64).reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3).astype(np.float64)

    raise ValueError("PVCamera.k must be a 3x3 matrix or contain at least 9 values")


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


def ensure_raw_output_dir(task_name: str) -> Path:
    target_dir = ARUCO_RAW_ROOT / task_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    normalized = u @ vh
    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vh
    return normalized.astype(np.float64)


def resolve_pv_camera_world_pose(task_or_frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pv_info = task_or_frame.get("PVCamera") or task_or_frame
    pv_pose = np.asarray(pv_info.get("pose"), dtype=np.float64)
    if pv_pose.shape == (4, 4):
        translation, rotation, _quat_xyzw = convert_hololens_pv_pose_matrix_to_unity_pose_components(
            pv_pose
        )
        return translation.astype(np.float64), rotation.astype(np.float64)

    translation = np.asarray(pv_info.get("position"), dtype=np.float64)
    quat_xyzw = np.asarray(pv_info.get("rotation_quaternion_xyzw"), dtype=np.float64)

    if translation.shape == (3,) and quat_xyzw.shape == (4,):
        rotation = quat_xyzw_to_rotation_matrix(quat_xyzw)
        return translation.astype(np.float64), rotation.astype(np.float64)

    raise ValueError(
        "PVCamera must include either pose(4x4) or position+rotation_quaternion_xyzw"
    )


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
    coordinate_basis: str,
    *,
    scale: list[float] | None = None,
) -> dict[str, Any]:
    payload = serialize_pose(rotation, translation, coordinate_basis)
    payload["rotation"] = list(payload["rotation_quaternion_xyzw"])
    if scale is not None:
        payload["scale"] = [float(v) for v in scale]
    return payload


def load_json_payload(raw_json: str | None) -> dict[str, Any] | None:
    if not raw_json:
        return None
    loaded = json.loads(raw_json)
    return loaded if isinstance(loaded, dict) else None
