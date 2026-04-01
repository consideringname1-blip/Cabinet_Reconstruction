from __future__ import annotations

import sys
import numpy as np

from _bootstrap import CODE_ROOT
from task_json import load_task_json, resolve_task_json_path, save_task_json


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero-length quaternion")
    return q / n


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, xyzw order
    x1, y1, z1, w1 = normalize_quat_xyzw(q1)
    x2, y2, z2, w2 = normalize_quat_xyzw(q2)

    q = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)
    return normalize_quat_xyzw(q)


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


def compute_world_pose(task: dict) -> dict[str, list[float]]:
    alignment = task.get("object_alignment") or {}
    pv = task.get("PVCamera") or {}

    local_position = np.asarray(alignment.get("model_unity_position"), dtype=np.float64)
    if local_position.shape != (3,):
        raise ValueError("object_alignment.model_unity_position must have 3 values")

    local_quat = np.asarray(alignment.get("model_unity_rotation_quaternion_xyzw"), dtype=np.float64)
    if local_quat.shape != (4,):
        raise ValueError("object_alignment.model_unity_rotation_quaternion_xyzw must have 4 values")

    model_scale = float(alignment.get("model_real_scale") or 0.0)
    if model_scale <= 0:
        raise ValueError("object_alignment.model_real_scale must be positive")

    pv_pose = np.asarray(pv.get("pose"), dtype=np.float64)
    if pv_pose.shape != (4, 4):
        raise ValueError("PVCamera.pose must be a 4x4 matrix")

    # 你的 JSON 里平移在最后一行，所以按 row-vector 约定读：
    # p_world = p_local @ R_cam + t_cam
    R_cam = pv_pose[:3, :3]
    t_cam = pv_pose[3, :3]

    world_position = local_position @ R_cam + t_cam

    # 世界旋转 = 拍摄相机世界旋转 * 物体相对相机旋转
    cam_quat = rotation_matrix_to_quat_xyzw(R_cam)
    world_quat = quat_mul_xyzw(cam_quat, local_quat)

    uniform_scale = [float(model_scale), float(model_scale), float(model_scale)]
    return {
        "position": [float(v) for v in world_position],
        "rotation": [float(v) for v in world_quat],
        "scale": uniform_scale,
    }