from __future__ import annotations

import sys

from _bootstrap import CODE_ROOT
import numpy as np
from scipy.spatial.transform import Rotation

from task_json import load_task_json, resolve_task_json_path, save_task_json


def compute_world_pose(task: dict) -> dict[str, list[float]]:
    pv_info = task.get("PVCamera") or {}
    alignment = task.get("object_alignment") or {}

    pv_pose = np.asarray(pv_info.get("pose"), dtype=np.float64)
    if pv_pose.shape != (4, 4):
        raise ValueError(f"PVCamera.pose must be 4x4, got {pv_pose.shape}")

    model_position = np.asarray(alignment.get("model_unity_position"), dtype=np.float64)
    if model_position.shape != (3,):
        raise ValueError("object_alignment.model_unity_position must have 3 values")

    model_quat = np.asarray(alignment.get("model_unity_rotation_quaternion_xyzw"), dtype=np.float64)
    if model_quat.shape != (4,):
        raise ValueError("object_alignment.model_unity_rotation_quaternion_xyzw must have 4 values")

    model_scale = float(alignment.get("model_real_scale") or 0.0)
    if model_scale <= 0:
        raise ValueError("object_alignment.model_real_scale must be positive")

    # HoloLens poses in this project use row-vector transforms:
    # p_world = p_camera @ R + t
    camera_rotation_row = pv_pose[:3, :3]
    camera_translation = pv_pose[3, :3]
    world_position = model_position @ camera_rotation_row + camera_translation

    # Alignment rotations are stored in the standard scipy convention.
    camera_rotation = camera_rotation_row.T
    model_rotation = Rotation.from_quat(model_quat).as_matrix()
    world_rotation = camera_rotation @ model_rotation
    world_quat = Rotation.from_matrix(world_rotation).as_quat()

    uniform_scale = [float(model_scale), float(model_scale), float(model_scale)]
    return {
        "position": [float(v) for v in world_position],
        "rotation": [float(v) for v in world_quat],
        "scale": uniform_scale,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python code/stages/run_pose_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_task_json_path(argv[1])
    task = load_task_json(json_path)

    if "object_alignment" not in task:
        raise ValueError("object_alignment is missing. Run ICP alignment stage first.")

    object_info = dict(task.get("object") or {})
    object_info.update(compute_world_pose(task))
    object_info["coordinate_basis"] = "unity_world_x_right_y_up_z_forward"
    task["object"] = object_info
    save_task_json(json_path, task)

    print(f"[INFO] JSON            : {json_path}")
    print(f"[INFO] Object position : {object_info['position']}")
    print(f"[INFO] Object rotation : {object_info['rotation']}")
    print(f"[INFO] Object scale    : {object_info['scale']}")
    print("[OK] Pose stage completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
