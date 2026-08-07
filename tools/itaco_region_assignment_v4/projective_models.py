"""Fixed-pose projective depth evidence for static and prismatic models."""
from __future__ import annotations

import numpy as np

UNOBSERVABLE = 0
SUPPORTED = 1
OCCLUDED = 2
CONTRADICTION = 3
STATUS_NAMES = {UNOBSERVABLE: "unobservable", SUPPORTED: "supported", OCCLUDED: "occluded", CONTRADICTION: "contradiction"}


def unproject_pixels(uv: np.ndarray, depth_m: np.ndarray, pose_world_camera: np.ndarray,
                     intrinsic: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, float); z = np.asarray(depth_m, float)
    camera = np.column_stack(((uv[:, 0] - intrinsic[0, 2]) * z / intrinsic[0, 0],
                              (uv[:, 1] - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
    return camera @ pose_world_camera[:3, :3].T + pose_world_camera[:3, 3]


def transform_model(points_world: np.ndarray, source_q: float, target_q: float,
                    axis_world: np.ndarray, model: str) -> np.ndarray:
    points = np.asarray(points_world, float)
    if model == "static":
        return points
    if model == "drawer":
        return points + (float(target_q) - float(source_q)) * np.asarray(axis_world, float)
    raise ValueError(f"unknown model: {model}")


def project_world(points_world: np.ndarray, pose_world_camera: np.ndarray, intrinsic: np.ndarray,
                  shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, float)
    camera = (points - pose_world_camera[:3, 3]) @ pose_world_camera[:3, :3]
    z = camera[:, 2]
    uv = np.column_stack((intrinsic[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + intrinsic[0, 2],
                          intrinsic[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + intrinsic[1, 2]))
    rounded = np.rint(uv).astype(np.int64); h, w = shape
    inside = (z > 0) & np.isfinite(camera).all(axis=1) & (rounded[:, 0] >= 0) & (rounded[:, 0] < w) & (rounded[:, 1] >= 0) & (rounded[:, 1] < h)
    return rounded, z, inside


def predicted_surface_front(uv: np.ndarray, z: np.ndarray, inside: np.ndarray,
                            shape: tuple[int, int]) -> np.ndarray:
    """Return source-point indices surviving the predicted surface z-buffer."""
    h, w = shape; ids = np.flatnonzero(inside)
    if not len(ids):
        return ids
    flat = uv[ids, 1] * w + uv[ids, 0]
    order = np.lexsort((ids, z[ids])); ordered_ids = ids[order]; ordered_flat = flat[order]
    _, first = np.unique(ordered_flat, return_index=True)
    return ordered_ids[first]


def evaluate_prediction(predicted_world: np.ndarray, target_frame: dict, intrinsic: np.ndarray,
                        cfg: dict) -> dict:
    """Classify predicted samples using registered depth; occlusion is neutral."""
    uv, z, inside = project_world(predicted_world, target_frame["pose"], intrinsic, target_frame["depth"].shape)
    front = predicted_surface_front(uv, z, inside, target_frame["depth"].shape)
    status = np.full(len(predicted_world), UNOBSERVABLE, np.uint8)
    observed = np.full(len(predicted_world), np.nan, float)
    residual = np.full(len(predicted_world), np.nan, float)
    if len(front):
        u, v = uv[front, 0], uv[front, 1]
        observable = target_frame["valid"][v, u]
        ids = front[observable]
        if len(ids):
            uo, vo = uv[ids, 0], uv[ids, 1]
            observed[ids] = target_frame["depth"][vo, uo]
            residual[ids] = observed[ids] - z[ids]
            absolute = np.abs(residual[ids])
            support = absolute <= float(cfg["depth_support_threshold_m"])
            occluded = residual[ids] < -float(cfg["occlusion_margin_m"])
            contradiction = residual[ids] > float(cfg["free_space_margin_m"])
            status[ids[support]] = SUPPORTED
            status[ids[~support & occluded]] = OCCLUDED
            status[ids[~support & ~occluded & contradiction]] = CONTRADICTION
    counts = {name: int(np.sum(status == code)) for code, name in STATUS_NAMES.items()}
    testable = counts["supported"] + counts["contradiction"]
    return {"status": status, "uv": uv, "predicted_depth_m": z, "observed_depth_m": observed,
            "signed_depth_residual_m": residual, "counts": counts, "testable_count": testable,
            "support_ratio": counts["supported"] / max(testable, 1),
            "contradiction_ratio": counts["contradiction"] / max(testable, 1),
            "observable_fraction": (counts["supported"] + counts["occluded"] + counts["contradiction"]) / max(len(status), 1)}


def evaluate_model(points_world: np.ndarray, source_q: float, target_frame: dict, target_q: float,
                   axis_world: np.ndarray, intrinsic: np.ndarray, model: str, cfg: dict) -> dict:
    predicted = transform_model(points_world, source_q, target_q, axis_world, model)
    return evaluate_prediction(predicted, target_frame, intrinsic, cfg)


def supported_template_pixels(points_world: np.ndarray, target_frame: dict, intrinsic: np.ndarray,
                              cfg: dict) -> np.ndarray:
    result = evaluate_prediction(np.asarray(points_world), target_frame, intrinsic, cfg)
    mask = np.zeros(target_frame["depth"].shape, bool)
    ids = np.flatnonzero(result["status"] == SUPPORTED)
    if len(ids):
        uv = result["uv"][ids]; mask[uv[:, 1], uv[:, 0]] = True
    return mask
