from __future__ import annotations

import sys
import numpy as np

from _bootstrap import CODE_ROOT
from object_alignment_common import (
    BLENDER_WORLD_TO_FBX_EXPORT_LOCAL,
    FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD,
    FBX_RUNTIME_LOCAL_TO_UNITY_BASIS,
    FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY,
    ICP_OBJ_IMPORT_LOCAL_ROTATION,
    ICP_OBJ_IMPORT_TO_BLENDER_WORLD,
    MODEL_INPUT_TO_FBX_RUNTIME_LOCAL,
    MODEL_INPUT_TO_UNITY_BASIS,
    model_pose_pointcloud_input_to_unity,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


CUSTOM_RUNTIME_LOCAL_AXIS_REMAP_TO_UNITY = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero-length quaternion")
    return q / n


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")

    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def quat_xyzw_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    sy = np.sqrt((R[0, 0] * R[0, 0]) + (R[1, 0] * R[1, 0]))
    singular = sy < 1e-8

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z], dtype=np.float64))


def make_row_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[3, :3] = translation
    return matrix


def serialize_pose(rotation: np.ndarray, translation: np.ndarray, coordinate_basis: str) -> dict[str, list[float] | list[list[float]] | str]:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    quat_xyzw = rotation_matrix_to_quat_xyzw(rotation)
    euler_deg = rotation_matrix_to_euler_xyz_deg(rotation)
    matrix = make_row_transform_matrix(rotation, translation)
    return {
        "coordinate_basis": coordinate_basis,
        "position": [float(v) for v in translation],
        "rotation_euler_deg": [float(v) for v in euler_deg],
        "rotation_quaternion_xyzw": [float(v) for v in quat_xyzw],
        "transform_matrix": [[float(v) for v in row] for row in matrix],
    }


def serialize_rotation_only(
    rotation: np.ndarray,
    coordinate_basis: str,
    notes: str | None = None,
) -> dict[str, list[float] | list[list[float]] | str]:
    payload = serialize_pose(rotation, np.zeros(3, dtype=np.float64), coordinate_basis)
    if notes:
        payload["notes"] = notes
    return payload


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
    coordinate_basis = str(alignment.get("coordinate_basis") or "")
    if coordinate_basis != "pointcloud_input_pre_blender_import":
        raise ValueError(
            "object_alignment.coordinate_basis must be pointcloud_input_pre_blender_import"
        )

    pointcloud_position = np.asarray(alignment.get("model_position"), dtype=np.float64)
    pointcloud_quat = np.asarray(alignment.get("model_rotation_quaternion_xyzw"), dtype=np.float64)
    if pointcloud_position.shape != (3,):
        raise ValueError("object_alignment.model_position must have 3 values")
    if pointcloud_quat.shape != (4,):
        raise ValueError("object_alignment.model_rotation_quaternion_xyzw must have 4 values")

    pointcloud_rotation = quat_xyzw_to_rotation_matrix(pointcloud_quat)
    unity_rotation, unity_translation = model_pose_pointcloud_input_to_unity(
        pointcloud_rotation,
        pointcloud_position,
    )
    return unity_translation.astype(np.float64), unity_rotation.astype(np.float64)

def compute_world_pose(task: dict) -> dict[str, list[float]]:
    alignment = task.get("object_alignment") or {}
    device = task.get("device") or {}

    local_position, local_rotation = resolve_local_camera_pose(task)

    model_scale = float(alignment.get("model_real_scale") or 0.0)
    if model_scale <= 0:
        raise ValueError("object_alignment.model_real_scale must be positive")

    device_position = np.asarray(device.get("pose"), dtype=np.float64)
    device_rotation_quat = np.asarray(device.get("rotation"), dtype=np.float64)
    if device_position.shape != (3,):
        raise ValueError("device.pose must have 3 values")
    if device_rotation_quat.shape != (4,):
        raise ValueError("device.rotation must have 4 values")

    # Temporary fallback for debugging: use the uploaded Unity Camera.main pose
    # as the world anchor instead of PVCamera.pose so we can compare behavior.
    # When the anchor comes from a Unity quaternion, compose poses with the
    # usual Unity/column-vector convention: p_world = R_cam @ p_local + t_cam.
    R_cam = quat_xyzw_to_rotation_matrix(device_rotation_quat)
    t_cam = device_position
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
        "rotation": [float(v) for v in world_quat],
        "scale": uniform_scale,
    }


def build_pose_debug(task: dict) -> dict:
    alignment = task.get("object_alignment") or {}
    device = task.get("device") or {}

    pointcloud_position = np.asarray(alignment.get("model_position"), dtype=np.float64)
    pointcloud_quat = np.asarray(alignment.get("model_rotation_quaternion_xyzw"), dtype=np.float64)
    pointcloud_rotation = quat_xyzw_to_rotation_matrix(pointcloud_quat)

    local_position, local_rotation = resolve_local_camera_pose(task)

    device_position = np.asarray(device.get("pose"), dtype=np.float64)
    device_rotation_quat = np.asarray(device.get("rotation"), dtype=np.float64)
    if device_position.shape != (3,):
        raise ValueError("device.pose must have 3 values")
    if device_rotation_quat.shape != (4,):
        raise ValueError("device.rotation must have 4 values")
    device_rotation = quat_xyzw_to_rotation_matrix(device_rotation_quat)
    runtime_local_to_unity = resolve_runtime_local_to_unity_rotation()

    world_position = (device_rotation @ local_position) + device_position
    world_rotation = device_rotation @ local_rotation @ runtime_local_to_unity

    return {
        "camera_local_pointcloud_input": {
            "scale": float(alignment.get("model_real_scale") or 0.0),
            "pose": serialize_pose(
                pointcloud_rotation,
                pointcloud_position,
                "pointcloud_input_pre_blender_import",
            ),
        },
        "camera_local_unity": {
            "scale": float(alignment.get("model_real_scale") or 0.0),
            "pose": serialize_pose(
                local_rotation,
                local_position,
                "unity_camera_local_x_right_y_up_z_forward",
            ),
        },
        "pv_camera_world": {
            "pose": serialize_pose(
                device_rotation,
                device_position,
                "unity_world_x_right_y_up_z_forward",
            ),
            "notes": "Temporary fallback for debugging/backtracking: this field currently uses uploaded device.pose/device.rotation (Unity Camera.main), not PVCamera.pose.",
        },
        "local_rotation_compensation_chain": {
            "model_input_to_unity": serialize_rotation_only(
                MODEL_INPUT_TO_UNITY_BASIS,
                "model_input_x_unknown_y_unknown_z_unknown -> unity_camera_local_x_right_y_up_z_forward",
                notes="This is the basis used when ICP converts raw OBJ vertices into the internal Unity-aligned point set.",
            ),
            "icp_obj_import_to_blender_world": serialize_rotation_only(
                ICP_OBJ_IMPORT_TO_BLENDER_WORLD,
                "model_input_local -> blender_world_for_icp_preview",
                notes="ICP preview/rendering imports OBJ into Blender with forward=-X, up=+Z.",
            ),
            "fbx_convert_obj_import_to_blender_world": serialize_rotation_only(
                FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD,
                "model_input_local -> blender_world_for_fbx_wrapper",
                notes="Runtime FBX wrapping imports OBJ with Blender's default OBJ axes: forward=-Z, up=+Y.",
            ),
            "blender_world_to_fbx_export_local": serialize_rotation_only(
                BLENDER_WORLD_TO_FBX_EXPORT_LOCAL,
                "blender_world -> runtime_fbx_local",
                notes="FBX export uses axis_forward=-Z, axis_up=Y with bake_space_transform=True.",
            ),
            "model_input_to_runtime_fbx_local": serialize_rotation_only(
                MODEL_INPUT_TO_FBX_RUNTIME_LOCAL,
                "model_input_local -> runtime_fbx_local",
                notes="This is the composed local-axis transform of the OBJ->FBX wrapper path.",
            ),
            "runtime_fbx_local_to_unity": serialize_rotation_only(
                runtime_local_to_unity,
                "runtime_fbx_local -> unity_camera_local_x_right_y_up_z_forward",
                notes="This is the transform-space correction actually multiplied onto the final world quaternion. It is identity for the current OBJ->FBX->Unity/TriLib path.",
            ),
            "runtime_fbx_local_to_unity_basis_reference": serialize_rotation_only(
                FBX_RUNTIME_LOCAL_TO_UNITY_BASIS,
                "runtime_fbx_mesh_basis -> unity_camera_local_x_right_y_up_z_forward",
                notes="Reference basis map for mesh coordinates only. determinant=-1, so it must not be multiplied directly into a runtime world quaternion.",
            ),
            "legacy_icp_obj_import_only_compensation": serialize_rotation_only(
                ICP_OBJ_IMPORT_LOCAL_ROTATION,
                "legacy_blender_icp_local -> model_input_local",
                notes="Previous pose-stage implementation only used the ICP Blender import correction, which ignored the FBX wrapper path.",
            ),
        },
        "final_object_world": {
            "scale": [float(alignment.get("model_real_scale") or 0.0)] * 3,
            "pose": serialize_pose(
                world_rotation,
                world_position,
                "unity_world_x_right_y_up_z_forward",
            ),
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python code/stages/run_pose_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_task_json_path(argv[1])
    task = load_task_json(json_path)

    world_pose = compute_world_pose(task)
    world_pose["coordinate_basis"] = "unity_world_x_right_y_up_z_forward"
    task["object"] = world_pose
    debug_section = dict(task.get("debug") or {})
    pose_debug = dict(debug_section.get("pose_transform_stages") or {})
    pose_debug["pose_stage"] = build_pose_debug(task)
    debug_section["pose_transform_stages"] = pose_debug
    task["debug"] = debug_section
    save_task_json(json_path, task)

    print(f"[INFO] JSON            : {json_path}")
    print(f"[INFO] World position  : {world_pose['position']}")
    print(f"[INFO] World rotation  : {world_pose['rotation']}")
    print(f"[INFO] World scale     : {world_pose['scale']}")
    print("[OK] Pose stage completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
