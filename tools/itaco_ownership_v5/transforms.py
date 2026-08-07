"""Generic frozen articulation transform used by v5 primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ArticulatedTransform:
    kind: str
    axis: np.ndarray
    origin: np.ndarray

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis, float).reshape(3)
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("axis must be finite and non-zero")
        object.__setattr__(self, "axis", axis / norm)
        object.__setattr__(self, "origin", np.asarray(self.origin, float).reshape(3))
        if self.kind not in ("prismatic", "revolute"):
            raise ValueError("kind must be prismatic or revolute")

    def matrix(self, q: float) -> np.ndarray:
        transform = np.eye(4)
        if self.kind == "prismatic":
            transform[:3, 3] = float(q) * self.axis
        else:
            rotation = Rotation.from_rotvec(float(q) * self.axis).as_matrix()
            transform[:3, :3] = rotation
            transform[:3, 3] = self.origin - rotation @ self.origin
        return transform

    def between(self, q_i: float, q_j: float) -> np.ndarray:
        return self.matrix(q_j) @ np.linalg.inv(self.matrix(q_i))

    def apply_between(self, points: np.ndarray, q_i: float, q_j: float) -> np.ndarray:
        points = np.asarray(points, float).reshape(-1, 3)
        transform = self.between(q_i, q_j)
        return points @ transform[:3, :3].T + transform[:3, 3]
