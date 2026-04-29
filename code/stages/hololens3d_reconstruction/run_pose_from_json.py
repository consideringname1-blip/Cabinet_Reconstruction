from __future__ import annotations

import sys

import _bootstrap
import numpy as np

from object_alignment_common import (
    FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY,
    model_pose_canonical_rh_to_unity_camera,
    model_pose_unity_camera_to_canonical_rh,
)
from pose_math import (
    quat_xyzw_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
    serialize_pose,
)
from stage_common import load_stage_task
from task_json import save_task_json
from unity_coordinate_utils import convert_hololens_pv_pose_matrix_to_unity_pose_components


CUSTOM_RUNTIME_LOCAL_AXIS_REMAP_TO_UNITY = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

def resolve_runtime_local_to_unity_rotation() -> np.ndarray:
    # Apply a transform-space axis remap on top of the runtime FBX correction so
    # the final Unity object axes match the desired debugging/orientation
    # convention:
    # new +Y = current -X
    # new +Z = current +Y
    # and therefore new +X = current -Z to keep a proper right-handed rotation.
    # A single-axis flip would become a reflection (det=-1), which cannot be
    # represented by the runtime quaternion path. So we apply a 180-degree
    # local-Z rotation after the remap: this reverses Y and X together while
    # keeping a proper rotation matrix.
    base = np.asarray(FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY, dtype=np.float64)
    rotate_180_about_local_z = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    return base @ CUSTOM_RUNTIME_LOCAL_AXIS_REMAP_TO_UNITY @ rotate_180_about_local_z


def resolve_local_camera_pose(task: dict) -> tuple[np.ndarray, np.ndarray]:
    alignment = task.get("object_alignment") or {}
    position = np.asarray(alignment.get("camera_local_position"), dtype=np.float64)
    quat = np.asarray(alignment.get("camera_local_rotation_quaternion_xyzw"), dtype=np.float64)
    if position.shape == (3,) and quat.shape == (4,):
        rotation = quat_xyzw_to_rotation_matrix(quat)
        return position.astype(np.float64), rotation.astype(np.float64)

    coordinate_basis = str(alignment.get("coordinate_basis") or "")
    if coordinate_basis == "pointcloud_input_pre_blender_import":
        pointcloud_position = np.asarray(alignment.get("model_position"), dtype=np.float64)
        pointcloud_quat = np.asarray(alignment.get("model_rotation_quaternion_xyzw"), dtype=np.float64)
        if pointcloud_position.shape != (3,):
            raise ValueError("object_alignment.model_position must have 3 values")
        if pointcloud_quat.shape != (4,):
            raise ValueError("object_alignment.model_rotation_quaternion_xyzw must have 4 values")

        old_pointcloud_to_unity = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        old_unity_to_model_input = np.array(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        pointcloud_rotation = quat_xyzw_to_rotation_matrix(pointcloud_quat)
        unity_rotation = old_pointcloud_to_unity @ pointcloud_rotation @ old_unity_to_model_input
        unity_translation = pointcloud_position @ old_pointcloud_to_unity.T
        canonical_rotation, canonical_translation = model_pose_unity_camera_to_canonical_rh(
            unity_rotation,
            unity_translation,
        )
        return canonical_translation.astype(np.float64), canonical_rotation.astype(np.float64)

    raise ValueError(
        "object_alignment must include camera_local_position and "
        "camera_local_rotation_quaternion_xyzw"
    )


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

    t_cam, R_cam = resolve_pv_camera_world_pose(task)
    runtime_local_to_unity = resolve_runtime_local_to_unity_rotation()

    world_position = (R_cam @ local_position) + t_cam
    # Compose the ICP rotation with the runtime FBX local-axis chain so the
    # loaded model is placed in the same orientation that ICP solved.
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
    pv_position, pv_rotation = resolve_pv_camera_world_pose(task)
    runtime_local_to_unity = resolve_runtime_local_to_unity_rotation()

    world_position = (pv_rotation @ local_position) + pv_position
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
            "pose": serialize_pose(pv_rotation, pv_position),
            "notes": "PVCamera world pose is reconstructed from the raw HoloLens PV pose using the pipeline's legacy Z-flip conversion before composition.",
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
    task.pop("object", None)
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
