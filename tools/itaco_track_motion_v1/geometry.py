"""Coordinates and fixed known-motion utilities using T_world_camera."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .errors import Failure, Phase1Error


def load_odometry_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5:
        raise Phase1Error(Failure("manifest", "invalid_odometry_log", "Odometry log line count is not divisible by five", details={"path": str(path), "lines": len(lines)}))
    ids, poses = [], []
    for start in range(0, len(lines), 5):
        ids.append(int(lines[start].split()[0]))
        pose = np.asarray([[float(x) for x in row.split()] for row in lines[start + 1:start + 5]], dtype=np.float64)
        if pose.shape != (4, 4):
            raise Phase1Error(Failure("manifest", "invalid_pose_shape", "Odometry pose is not 4x4", details={"block": start // 5}))
        poses.append(pose)
    return np.asarray(ids, dtype=np.int64), np.stack(poses)


def load_intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(values) < 4:
        raise Phase1Error(Failure("manifest", "invalid_intrinsics", "Calibration must contain fx fy cx cy", details={"path": str(path)}))
    fx, fy, cx, cy = values[:4]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def validate_pose(pose: np.ndarray, tolerance: float) -> tuple[bool, dict[str, float]]:
    rotation = pose[:3, :3]
    ortho = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    bottom = float(np.linalg.norm(pose[3] - np.asarray([0.0, 0.0, 0.0, 1.0])))
    ok = np.isfinite(pose).all() and ortho <= tolerance and abs(determinant - 1.0) <= tolerance and bottom <= tolerance
    return bool(ok), {"orthogonality_error": ortho, "determinant": determinant, "bottom_row_error": bottom}


def rebase_poses(poses_world_camera: np.ndarray, reference_index: int) -> np.ndarray:
    return np.linalg.inv(poses_world_camera[reference_index])[None] @ poses_world_camera


def unproject_pixel(uv: np.ndarray, depth_m: float, intrinsic: np.ndarray) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])
    return np.asarray([(u - intrinsic[0, 2]) * depth_m / intrinsic[0, 0], (v - intrinsic[1, 2]) * depth_m / intrinsic[1, 1], depth_m], dtype=np.float64)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def camera_speed(poses: np.ndarray, timestamps: np.ndarray, timestamp_scale: float) -> tuple[np.ndarray, np.ndarray]:
    count = len(poses)
    linear, angular = np.zeros(count), np.zeros(count)
    for index in range(1, count):
        dt = max((timestamps[index] - timestamps[index - 1]) * timestamp_scale, 1e-9)
        linear[index] = np.linalg.norm(poses[index, :3, 3] - poses[index - 1, :3, 3]) / dt
        delta = poses[index - 1, :3, :3].T @ poses[index, :3, :3]
        angular[index] = np.linalg.norm(Rotation.from_matrix(delta).as_rotvec()) / dt
    if count > 1:
        linear[0], angular[0] = linear[1], angular[1]
    return linear, angular


class KnownMotion:
    """Fixed configured motion; no estimation or model selection occurs here."""

    def __init__(self, config: dict, poses_world_camera: np.ndarray, reference_index: int, frame_count: int):
        self.type = str(config["type"])
        axis_world = np.load(config["axis_path"]).astype(np.float64).reshape(3)
        norm = float(np.linalg.norm(axis_world))
        if not np.isfinite(norm) or norm < 1e-9:
            raise Phase1Error(Failure("known_motion", "invalid_axis", "Configured known axis is invalid"))
        axis_world /= norm
        world_to_reference = np.linalg.inv(poses_world_camera[reference_index])
        self.axis = world_to_reference[:3, :3] @ axis_world
        self.axis /= np.linalg.norm(self.axis)
        self.state = np.load(config["state_path"]).astype(np.float64).reshape(-1)
        self.canonicalization_sign = float(config["canonicalization_sign"])
        if len(self.state) != frame_count:
            raise Phase1Error(Failure("known_motion", "state_count_mismatch", "Known q_t count does not match processing frames", details={"states": len(self.state), "frames": frame_count}))
        self.origin = None
        if self.type == "revolute":
            if "origin_path" not in config:
                raise Phase1Error(Failure("known_motion", "missing_origin", "Fixed revolute motion requires origin_path"))
            self.origin = transform_point(world_to_reference, np.load(config["origin_path"]).astype(np.float64).reshape(3))
        self.source = {"type": self.type, "axis_path": config["axis_path"], "state_path": config["state_path"], "origin_path": config.get("origin_path"), "fixed": True, "estimated_by_stage1": False}

    def canonicalize(self, points_reference: np.ndarray, processing_indices: np.ndarray) -> np.ndarray:
        q = self.state[np.asarray(processing_indices, dtype=np.int64)]
        if self.type == "prismatic":
            return points_reference + self.canonicalization_sign * q[:, None] * self.axis[None]
        assert self.origin is not None
        rotations = Rotation.from_rotvec(-q[:, None] * self.axis[None]).as_matrix()
        return np.einsum("nij,nj->ni", rotations, points_reference - self.origin[None]) + self.origin[None]

    def decanonicalize(self, canonical_points: np.ndarray, processing_indices: np.ndarray) -> np.ndarray:
        """Apply the fixed known motion at requested frames; this performs no estimation."""
        q = self.state[np.asarray(processing_indices, dtype=np.int64)]
        if self.type == "prismatic":
            return canonical_points - self.canonicalization_sign * q[:, None] * self.axis[None]
        assert self.origin is not None
        rotations = Rotation.from_rotvec(q[:, None] * self.axis[None]).as_matrix()
        return np.einsum("nij,nj->ni", rotations, canonical_points - self.origin[None]) + self.origin[None]


def save_pose_text(path: Path, records: list[dict], poses: np.ndarray, convention: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# convention: {convention}\n# original_frame_id timestamp then row-major 4x4\n")
        for record, pose in zip(records, poses):
            values = " ".join(f"{value:.12g}" for value in pose.reshape(-1))
            handle.write(f"{record['original_frame_id']} {record['timestamp']} {values}\n")
