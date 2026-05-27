from __future__ import annotations

import sys

import _bootstrap
import numpy as np

from config import (
    SKIP_ICP_POSE_USE_CAMERA_PITCH,
    SKIP_ICP_POSE_USE_CAMERA_ROLL,
    SKIP_ICP_POSE_USE_CAMERA_YAW,
)
from object_alignment_common import (
    FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY,
    RUNTIME_AXIS_CONTRACT,
    model_pose_canonical_rh_to_unity_camera,
)
from pose_math import (
    quat_xyzw_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
    serialize_pose,
)
from stage_common import load_stage_task
from task_json import save_task_json
from unity_coordinate_utils import convert_hololens_pv_pose_matrix_to_unity_pose_components


def resolve_runtime_local_to_unity_rotation() -> np.ndarray:
    # RuntimeMesh now bakes generated model axes into Unity's runtime local
    # contract (+Z forward, +Y up). Pose composition therefore should not carry
    # any model-axis compatibility correction.
    return np.asarray(FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY, dtype=np.float64)


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return vector / norm


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize_vector(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _look_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = _normalize_vector(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    up_axis = _normalize_vector(up, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis = np.cross(up_axis, z_axis)
    if float(np.linalg.norm(x_axis)) <= 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = _normalize_vector(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), up_axis)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float64)


def resolve_camera_rotation_for_object(task: dict, camera_rotation: np.ndarray) -> tuple[np.ndarray, dict]:
    alignment = task.get("object_alignment") or {}
    alignment_mode = str(alignment.get("alignment_mode") or alignment.get("icp_mode") or "").strip().lower()
    if alignment_mode != "off":
        return camera_rotation, {"mode": "full_camera_rotation", "reason": "alignment_mode_not_off"}

    use_yaw = bool(SKIP_ICP_POSE_USE_CAMERA_YAW)
    use_pitch = bool(SKIP_ICP_POSE_USE_CAMERA_PITCH)
    use_roll = bool(SKIP_ICP_POSE_USE_CAMERA_ROLL)
    if use_yaw and use_pitch and use_roll:
        return camera_rotation, {
            "mode": "full_camera_rotation",
            "use_yaw": use_yaw,
            "use_pitch": use_pitch,
            "use_roll": use_roll,
        }

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    camera_forward = camera_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    flat_forward = np.array([camera_forward[0], 0.0, camera_forward[2]], dtype=np.float64)
    yaw_rotation = _look_rotation(flat_forward, world_up)

    residual = yaw_rotation.T @ camera_rotation
    pitch_rad = float(np.arctan2(-residual[1, 2], residual[2, 2]))
    roll_rad = float(np.arctan2(-residual[0, 1], residual[0, 0]))

    filtered = np.eye(3, dtype=np.float64)
    if use_yaw:
        filtered = filtered @ yaw_rotation
    if use_pitch:
        filtered = filtered @ _axis_angle_rotation(np.array([1.0, 0.0, 0.0]), pitch_rad)
    if use_roll:
        filtered = filtered @ _axis_angle_rotation(np.array([0.0, 0.0, 1.0]), roll_rad)

    return filtered.astype(np.float64), {
        "mode": "filtered_camera_rotation",
        "use_yaw": use_yaw,
        "use_pitch": use_pitch,
        "use_roll": use_roll,
        "pitch_deg": float(np.degrees(pitch_rad)),
        "roll_deg": float(np.degrees(roll_rad)),
        "camera_forward": [float(v) for v in camera_forward],
        "flat_forward": [float(v) for v in flat_forward],
    }


def resolve_local_camera_pose(task: dict) -> tuple[np.ndarray, np.ndarray]:
    alignment = task.get("object_alignment") or {}
    position = np.asarray(alignment.get("camera_local_position"), dtype=np.float64)
    quat = np.asarray(alignment.get("camera_local_rotation_quaternion_xyzw"), dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("object_alignment.camera_local_position must have 3 values")
    if quat.shape != (4,):
        raise ValueError("object_alignment.camera_local_rotation_quaternion_xyzw must have 4 values")
    rotation = quat_xyzw_to_rotation_matrix(quat)
    return position.astype(np.float64), rotation.astype(np.float64)


def resolve_pv_camera_world_pose(task: dict) -> tuple[np.ndarray, np.ndarray]:
    pv_info = task.get("PVCamera") or {}
    pv_pose = np.asarray(pv_info.get("pose"), dtype=np.float64)
    if pv_pose.shape == (4, 4):
        translation, rotation, _quat_xyzw = convert_hololens_pv_pose_matrix_to_unity_pose_components(
            pv_pose
        )
        return translation, rotation

    translation = np.asarray(pv_info.get("position"), dtype=np.float64)
    quat_xyzw = np.asarray(pv_info.get("rotation_quaternion_xyzw"), dtype=np.float64)

    if translation.shape == (3,) and quat_xyzw.shape == (4,):
        rotation = quat_xyzw_to_rotation_matrix(quat_xyzw)
        return translation.astype(np.float64), rotation.astype(np.float64)

    raise ValueError(
        "PVCamera must include either pose(4x4) or position+rotation_quaternion_xyzw"
    )

def compute_world_pose(task: dict) -> dict[str, list[float]]:
    alignment = task.get("object_alignment") or {}

    local_position_rh, local_rotation_rh = resolve_local_camera_pose(task)
    local_rotation, local_position = model_pose_canonical_rh_to_unity_camera(
        local_rotation_rh,
        local_position_rh,
    )
    local_rotation = local_rotation.astype(np.float64)
    local_position = local_position.astype(np.float64)

    model_scale = float(alignment.get("model_real_scale") or 0.0)
    if model_scale <= 0:
        raise ValueError("object_alignment.model_real_scale must be positive")

    t_cam, R_cam_raw = resolve_pv_camera_world_pose(task)
    R_cam, _camera_rotation_filter = resolve_camera_rotation_for_object(task, R_cam_raw)
    runtime_local_to_unity = resolve_runtime_local_to_unity_rotation()

    world_position = (R_cam_raw @ local_position) + t_cam
    # Compose only the solved pose; generated runtime assets already use
    # Unity's +Z-forward, +Y-up local axis contract.
    world_rotation = R_cam @ local_rotation @ runtime_local_to_unity
    det_world = float(np.linalg.det(world_rotation))
    if not np.isfinite(det_world) or det_world <= 0.0:
        raise ValueError(
            f"Final world rotation must be a proper rotation, got determinant {det_world:.6f}"
        )
    world_quat = rotation_matrix_to_quat_xyzw(world_rotation)

    uniform_scale = [float(model_scale), float(model_scale), float(model_scale)]
    return {
        "position": [float(v) for v in world_position],
        "rotation_quaternion_xyzw": [float(v) for v in world_quat],
        "scale": uniform_scale,
    }


def build_pose_debug(task: dict) -> dict:
    alignment = task.get("object_alignment") or {}
    local_position_rh, local_rotation_rh = resolve_local_camera_pose(task)
    local_rotation, local_position = model_pose_canonical_rh_to_unity_camera(
        local_rotation_rh,
        local_position_rh,
    )
    pv_position, pv_rotation_raw = resolve_pv_camera_world_pose(task)
    pv_rotation, camera_rotation_filter = resolve_camera_rotation_for_object(task, pv_rotation_raw)
    runtime_local_to_unity = resolve_runtime_local_to_unity_rotation()

    world_position = (pv_rotation_raw @ local_position) + pv_position
    world_rotation = pv_rotation @ local_rotation @ runtime_local_to_unity

    return {
        "camera_local_rh": {
            "scale": float(alignment.get("model_real_scale") or 0.0),
            "pose": serialize_pose(
                local_rotation_rh,
                local_position_rh,
            ),
        },
        "camera_local_unity": {
            "scale": float(alignment.get("model_real_scale") or 0.0),
            "pose": serialize_pose(local_rotation, local_position),
        },
        "pv_camera_world": {
            "pose": serialize_pose(pv_rotation_raw, pv_position),
        },
        "camera_rotation_filter": camera_rotation_filter,
        "pv_camera_world_filtered": {
            "pose": serialize_pose(pv_rotation, pv_position),
        },
        "runtime_asset": {
            "axis_contract": RUNTIME_AXIS_CONTRACT,
            "runtime_local_to_unity_rotation": runtime_local_to_unity.astype(float).tolist(),
        },
        "final_object_world": {
            "scale": [float(alignment.get("model_real_scale") or 0.0)] * 3,
            "pose": serialize_pose(world_rotation, world_position),
        },
    }


def main(argv: list[str]) -> int:
    json_path, task = load_stage_task(
        argv,
        usage="Usage: python code/stages/hololens3d_reconstruction/run_pose_from_json.py <task_meta.json or filename>",
        stage_name="pose",
    )

    world_pose = compute_world_pose(task)
    task["object_world"] = dict(world_pose)
    debug_section = dict(task.get("debug") or {})
    pose_debug = dict(debug_section.get("pose_transform_stages") or {})
    pose_debug["pose_stage"] = build_pose_debug(task)
    debug_section["pose_transform_stages"] = pose_debug
    task["debug"] = debug_section
    save_task_json(json_path, task)

    print(
        f"[INFO] pose : position={world_pose['position']} "
        f"rotation={world_pose['rotation_quaternion_xyzw']} scale={world_pose['scale']}"
    )
    print("[OK] pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
