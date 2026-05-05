from __future__ import annotations

import sys
import shutil
import uuid
import warnings
from pathlib import Path

import _bootstrap
from alignment_preview import render_model_compare_preview_image, render_overlay_preview_image
from config import (
    ICP_ACCELERATION_DEVICE,
    ICP_ALIGNMENT_MODEL_MAX_POINTS,
    ICP_BBOX_SURFACE_DISTANCE_MODE,
    ICP_BBOX_SURFACE_LATERAL_MODE,
    ICP_BBOX_SURFACE_RAY_SOURCE,
    ICP_BBOX_SURFACE_THICKNESS_FACTOR,
    ICP_CAMERA_REFINE_MAX_ROTATION_DELTA_DEG,
    ICP_CAMERA_REFINE_SCALE_DELTA_RATIO,
    ICP_CAMERA_REFINE_SEED_KEEP,
    ICP_COARSE_VISIBLE_MAX_POINTS,
    ENABLE_ALIGNMENT_RENDER_OUTPUTS,
    ICP_ENABLE,
    ICP_MODE,
    ICP_FINAL_ITERATIONS,
    ICP_FINAL_VISIBLE_MAX_POINTS,
    ICP_IGNORE_INVERTED_SOLUTIONS,
    ICP_INITIAL_ROTATION_PENALTY_WEIGHT,
    ICP_TARGET_FRONT_MAX_POINTS,
)
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from object_alignment_common import (
    build_depth_pointcloud,
    build_depth_border_keep_mask,
    build_depth_pointcloud_from_valid_mask,
    compute_front_view_extents,
    extract_front_visible_points,
    model_pose_canonical_rh_to_blender_world,
    MAX_DEPTH_MM,
    MIN_DEPTH_MM,
    obj_vertices_to_canonical_rh,
    object_alignment_output_path,
    read_depth_image,
    read_mask,
    read_obj_vertices,
    resolve_task_paths,
    select_front_visible_points,
    task_prefix,
    canonical_rh_to_blender_world_vector,
    write_binary_ply,
)
from pose_math import (
    rotation_matrix_to_quat_xyzw,
    serialize_pose,
)
from stage_common import load_stage_task
from task_json import save_task_json


MODEL_UP_AXIS_UNITY = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_ICP_BACKEND: dict | None = None


def resolve_icp_backend() -> dict:
    global _ICP_BACKEND
    if _ICP_BACKEND is not None:
        return _ICP_BACKEND

    requested = str(ICP_ACCELERATION_DEVICE).strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("config.ICP_ACCELERATION_DEVICE must be one of: auto / cuda / cpu")

    reason = "forced-by-config" if requested == "cpu" else "simplified-cpu-only"
    _ICP_BACKEND = {"requested": requested, "actual": "cpu", "reason": reason, "torch_device": None}
    return _ICP_BACKEND


def nearest_neighbor_distances(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    dims: int = 3,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    query_points = np.asarray(query_points, dtype=np.float32)
    reference_points = np.asarray(reference_points, dtype=np.float32)
    if dims <= 0 or dims > query_points.shape[1] or dims > reference_points.shape[1]:
        raise ValueError(f"invalid dims={dims} for nearest-neighbor query")

    query_used = query_points[:, :dims]
    reference_used = reference_points[:, :dims]
    if len(query_used) == 0:
        empty_dist = np.empty(0, dtype=np.float32)
        empty_idx = np.empty(0, dtype=np.int32)
        return (empty_dist, empty_idx) if return_indices else empty_dist
    if len(reference_used) == 0:
        raise ValueError("reference_points must be non-empty")

    tree = cKDTree(reference_used)
    dists, idx = tree.query(query_used, k=1)
    dists = np.asarray(dists, dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int32)
    return (dists, idx) if return_indices else dists


def query_prebuilt_tree(
    query_points: np.ndarray,
    tree: cKDTree,
    *,
    dims: int,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    query_points = np.asarray(query_points, dtype=np.float32)
    if dims <= 0 or dims > query_points.shape[1]:
        raise ValueError(f"invalid dims={dims} for tree query")

    query_used = query_points[:, :dims]
    if len(query_used) == 0:
        empty_dist = np.empty(0, dtype=np.float32)
        empty_idx = np.empty(0, dtype=np.int32)
        return (empty_dist, empty_idx) if return_indices else empty_dist

    dists, idx = tree.query(query_used, k=1)
    dists = np.asarray(dists, dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int32)
    return (dists, idx) if return_indices else dists

def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def build_alignment_model_points(model_vertices_unity: np.ndarray) -> np.ndarray:
    model_vertices_unity = np.asarray(model_vertices_unity, dtype=np.float32)
    model_max_points = int(ICP_ALIGNMENT_MODEL_MAX_POINTS)
    visible_points = extract_front_visible_points(
        model_vertices_unity,
        bins=192,
        max_points=max(model_max_points * 2, 1) if model_max_points > 0 else None,
        seed=5,
    )
    if len(visible_points) == 0:
        visible_points = model_vertices_unity
    if model_max_points > 0:
        visible_points = downsample_points(visible_points, max_points=model_max_points, seed=97)
    return np.asarray(visible_points, dtype=np.float32)


def trimmed_rmse(dists: np.ndarray, trim_percentile: float = 85.0) -> float:
    dists = np.asarray(dists, dtype=np.float32).reshape(-1)
    if dists.size == 0:
        return float("inf")
    cutoff = float(np.percentile(dists, trim_percentile))
    keep = dists <= max(cutoff, 1e-6)
    if not np.any(keep):
        keep = np.ones_like(dists, dtype=bool)
    return float(np.sqrt(np.mean(np.square(dists[keep]))))


def safe_matrix_to_blender_euler_xyz_deg(rotation: np.ndarray) -> list[float]:
    # Blender's mathutils.Euler(..., "XYZ") expects intrinsic XYZ angles.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_euler("XYZ", degrees=True)
    return [float(v) for v in euler]


def model_up_y_in_unity(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float32)
    return float((rotation @ MODEL_UP_AXIS_UNITY)[1])


def rotation_delta_degrees(base_rotation: np.ndarray, rotation: np.ndarray) -> float:
    base_rotation = np.asarray(base_rotation, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    delta = rotation @ base_rotation.T
    return float(np.degrees(Rotation.from_matrix(delta).magnitude()))


def is_within_rotation_delta(base_rotation: np.ndarray, rotation: np.ndarray, max_delta_deg: float) -> bool:
    return bool(rotation_delta_degrees(base_rotation, rotation) <= float(max_delta_deg) + 1e-6)


def should_reject_inverted_solution(rotation: np.ndarray) -> bool:
    return bool(ICP_IGNORE_INVERTED_SOLUTIONS) and (model_up_y_in_unity(rotation) < 0.0)


def is_identity_rotation(rotation: np.ndarray, atol: float = 1e-5) -> bool:
    rotation = np.asarray(rotation, dtype=np.float32)
    return bool(np.allclose(rotation, np.eye(3, dtype=np.float32), atol=atol))


def compute_initial_rotation_penalty(rotation: np.ndarray | None) -> float:
    if rotation is None:
        return 0.0
    weight = float(ICP_INITIAL_ROTATION_PENALTY_WEIGHT)
    if weight <= 0.0:
        return 0.0
    angle_rad = float(Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).magnitude())
    normalized = angle_rad / np.pi
    return float(weight * normalized * normalized)


def compute_xy_center(extents: dict) -> np.ndarray:
    return np.array(
        [
            0.5 * (extents["bbox_min"][0] + extents["bbox_max"][0]),
            0.5 * (extents["bbox_min"][1] + extents["bbox_max"][1]),
        ],
        dtype=np.float32,
    )


def build_target_context(target_front_points: np.ndarray) -> dict:
    target_front_points = np.asarray(target_front_points, dtype=np.float32)
    extents = compute_front_view_extents(target_front_points)
    context = {
        "points": target_front_points,
        "extents": extents,
        "center_xy": compute_xy_center(extents),
        "front_anchor_z": float(np.median(target_front_points[:, 2])),
        "tree_3d": cKDTree(target_front_points) if len(target_front_points) > 0 else None,
        "tree_2d": cKDTree(target_front_points[:, :2]) if len(target_front_points) > 0 else None,
    }
    return context


def evaluate_alignment_geometry_only(
    transformed_front_points: np.ndarray,
    target_context: dict,
    scale: float,
    nominal_scale: float,
    rotation: np.ndarray | None = None,
) -> dict:
    extents = compute_front_view_extents(transformed_front_points)
    target_extents = target_context["extents"]
    width_error = abs(extents["width_units"] - target_extents["width_units"])
    height_error = abs(extents["height_units"] - target_extents["height_units"])
    size_error = width_error + height_error
    scale_error = abs(scale - nominal_scale) / max(nominal_scale, 1e-6)
    center_error = float(np.linalg.norm(compute_xy_center(extents) - target_context["center_xy"]))
    initial_rotation_penalty = compute_initial_rotation_penalty(rotation)
    score = 0.70 * size_error + 0.20 * center_error + 0.10 * scale_error + initial_rotation_penalty
    return {
        "score": float(score),
        "size_error": float(size_error),
        "width_error": float(width_error),
        "height_error": float(height_error),
        "center_error": center_error,
        "initial_rotation_penalty": float(initial_rotation_penalty),
        "extents": extents,
    }


def rank_scale_candidates(
    rotated_full_unit: np.ndarray,
    rotated_front_unit: np.ndarray,
    scale_candidates: np.ndarray,
    nominal_scale: float,
    target_context: dict,
    rotation: np.ndarray | None = None,
) -> list[dict]:
    ranked: list[dict] = []
    for scale in scale_candidates:
        scale_value = float(scale)
        scaled_full = rotated_full_unit * scale_value
        scaled_front = rotated_front_unit * scale_value
        translation = build_initial_translation(
            model_full_points=scaled_full,
            model_front_points=scaled_front,
            target_front_points=target_context["points"],
            target_context=target_context,
        )
        quick_metrics = evaluate_alignment_geometry_only(
            scaled_front + translation,
            target_context=target_context,
            scale=scale_value,
            nominal_scale=nominal_scale,
            rotation=rotation,
        )
        ranked.append(
            {
                "scale": scale_value,
                "translation": translation.astype(np.float32, copy=False),
                "quick_metrics": quick_metrics,
            }
        )
    ranked.sort(key=lambda item: item["quick_metrics"]["score"])
    return ranked


def best_fit_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    aa = a - centroid_a
    bb = b - centroid_b
    h = aa.T @ bb
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = centroid_b - (centroid_a @ r.T)
    return r.astype(np.float32), t.astype(np.float32)


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points * scale) @ rotation.T + translation


def run_icp(
    model_points: np.ndarray,
    target_points: np.ndarray,
    scale: float,
    init_rotation: np.ndarray,
    init_translation: np.ndarray,
    iterations: int = 25,
    visible_bins: int = 160,
    visible_max_points: int = 5000,
    reference_rotation: np.ndarray | None = None,
    max_rotation_delta_deg: float | None = None,
) -> dict:
    rotation = init_rotation.astype(np.float32).copy()
    translation = init_translation.astype(np.float32).copy()
    if reference_rotation is not None:
        reference_rotation = np.asarray(reference_rotation, dtype=np.float32)
    if max_rotation_delta_deg is not None:
        max_rotation_delta_deg = float(max_rotation_delta_deg)
    rmse = float("inf")
    inlier_ratio = 0.0
    transformed_visible = np.empty((0, 3), dtype=np.float32)

    for _ in range(iterations):
        transformed_full = transform_points(model_points, scale, rotation, translation)
        transformed_visible = extract_front_visible_points(
            transformed_full,
            bins=visible_bins,
            max_points=visible_max_points,
            seed=17,
        )
        if len(transformed_visible) < 32:
            break

        dists, idx = nearest_neighbor_distances(
            target_points,
            transformed_visible,
            dims=3,
            return_indices=True,
        )

        threshold = float(np.percentile(dists, 75))
        if threshold <= 0:
            threshold = float(dists.max()) if len(dists) else 0.0
        keep = dists <= max(threshold, 1e-4)
        if keep.sum() < 24:
            break

        matched_model = transformed_visible[idx[keep]]
        matched_target = target_points[keep]

        delta_r, delta_t = best_fit_transform(matched_model, matched_target)
        next_rotation = delta_r @ rotation
        next_translation = delta_r @ translation + delta_t
        if (
            reference_rotation is not None
            and max_rotation_delta_deg is not None
            and not is_within_rotation_delta(reference_rotation, next_rotation, max_rotation_delta_deg)
        ):
            break
        rotation = next_rotation
        translation = next_translation
        rmse = float(np.sqrt(np.mean((matched_model - matched_target) ** 2)))
        inlier_ratio = float(keep.mean())

    transformed_full = transform_points(model_points, scale, rotation, translation)
    transformed_visible = extract_front_visible_points(
        transformed_full,
        bins=visible_bins,
        max_points=visible_max_points,
        seed=19,
    )
    dists = nearest_neighbor_distances(target_points, transformed_visible, dims=3, return_indices=False)
    rmse = trimmed_rmse(dists, trim_percentile=85.0)

    return {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "rmse": rmse,
        "inlier_ratio": inlier_ratio,
        "visible_points": transformed_visible,
    }


def build_initial_translation(
    model_full_points: np.ndarray,
    model_front_points: np.ndarray,
    target_front_points: np.ndarray,
    target_context: dict | None = None,
) -> np.ndarray:
    model_extents = compute_front_view_extents(model_full_points)
    if target_context is None:
        target_extents = compute_front_view_extents(target_front_points)
        target_center = compute_xy_center(target_extents)
        target_anchor_z = float(np.median(target_front_points[:, 2]))
    else:
        target_extents = target_context["extents"]
        target_center = np.asarray(target_context["center_xy"], dtype=np.float32)
        target_anchor_z = float(target_context["front_anchor_z"])

    model_center_x, model_center_y = compute_xy_center(model_extents)
    target_center_x, target_center_y = target_center
    front_anchor_z = float(np.median(model_front_points[:, 2]))

    return np.array(
        [
            target_center_x - model_center_x,
            target_center_y - model_center_y,
            target_anchor_z - front_anchor_z,
        ],
        dtype=np.float32,
    )


def evaluate_alignment(
    transformed_full_points: np.ndarray,
    transformed_front_points: np.ndarray,
    target_front_points: np.ndarray,
    scale: float,
    nominal_scale: float,
    target_context: dict | None = None,
    rotation: np.ndarray | None = None,
) -> dict:
    if target_context is None:
        target_context = build_target_context(target_front_points)

    dists_target_to_model_3d = nearest_neighbor_distances(target_context["points"], transformed_front_points, dims=3)
    dists_target_to_model_2d = nearest_neighbor_distances(target_context["points"], transformed_front_points, dims=2)
    if target_context["tree_3d"] is not None:
        dists_model_to_target_3d = query_prebuilt_tree(transformed_front_points, target_context["tree_3d"], dims=3)
        dists_model_to_target_2d = query_prebuilt_tree(transformed_front_points, target_context["tree_2d"], dims=2)
    else:
        dists_model_to_target_3d = nearest_neighbor_distances(transformed_front_points, target_context["points"], dims=3)
        dists_model_to_target_2d = nearest_neighbor_distances(transformed_front_points, target_context["points"], dims=2)

    extents = compute_front_view_extents(transformed_front_points)
    target_extents = target_context["extents"]
    width_error = abs(extents["width_units"] - target_extents["width_units"])
    height_error = abs(extents["height_units"] - target_extents["height_units"])
    size_error = width_error + height_error
    scale_error = abs(scale - nominal_scale) / max(nominal_scale, 1e-6)
    center_error = float(np.linalg.norm(compute_xy_center(extents) - target_context["center_xy"]))

    rmse_3d = trimmed_rmse(dists_target_to_model_3d, trim_percentile=85.0)
    coverage_rmse_3d = trimmed_rmse(dists_model_to_target_3d, trim_percentile=85.0)
    surface_rmse_3d = 0.5 * (rmse_3d + coverage_rmse_3d)
    rmse_2d = 0.5 * (
        trimmed_rmse(dists_target_to_model_2d, trim_percentile=85.0)
        + trimmed_rmse(dists_model_to_target_2d, trim_percentile=85.0)
    )
    initial_rotation_penalty = compute_initial_rotation_penalty(rotation)
    score = (
        0.60 * surface_rmse_3d
        + 0.20 * rmse_2d
        + 0.10 * size_error
        + 0.05 * center_error
        + 0.05 * scale_error
        + initial_rotation_penalty
    )
    return {
        "score": float(score),
        "rmse_3d": rmse_3d,
        "coverage_rmse_3d": float(coverage_rmse_3d),
        "surface_rmse_3d": float(surface_rmse_3d),
        "rmse_2d": rmse_2d,
        "size_error": float(size_error),
        "width_error": float(width_error),
        "height_error": float(height_error),
        "center_error": center_error,
        "initial_rotation_penalty": float(initial_rotation_penalty),
        "extents": extents,
    }


def estimate_scale_from_visible_extents(
    model_front_points: np.ndarray,
    target_front_points: np.ndarray,
    fallback_scale: float,
) -> dict:
    model_extents = compute_front_view_extents(model_front_points)
    target_extents = compute_front_view_extents(target_front_points)

    width_scale = target_extents["width_units"] / max(model_extents["width_units"], 1e-6)
    height_scale = target_extents["height_units"] / max(model_extents["height_units"], 1e-6)
    candidates = [
        float(width_scale),
        float(height_scale),
        float(0.5 * (width_scale + height_scale)),
        float(np.sqrt(max(width_scale * height_scale, 1e-8))),
    ]
    candidates = [value for value in candidates if np.isfinite(value) and value > 0]
    estimated_scale = float(np.median(candidates)) if candidates else float(fallback_scale)

    min_scale = max(float(fallback_scale) * 0.65, 1e-4)
    max_scale = max(float(fallback_scale) * 1.45, min_scale + 1e-4)
    estimated_scale = float(np.clip(estimated_scale, min_scale, max_scale))
    return {
        "estimated_scale": estimated_scale,
        "width_scale": float(width_scale),
        "height_scale": float(height_scale),
    }


def build_camera_refine_scale_candidates(
    estimated_scale: float,
    width_scale: float,
    height_scale: float,
    fallback_scale: float,
) -> np.ndarray:
    delta_ratio = max(float(ICP_CAMERA_REFINE_SCALE_DELTA_RATIO), 0.0)
    min_scale = max(float(fallback_scale) * (1.0 - delta_ratio), 1e-4)
    max_scale = max(float(fallback_scale) * (1.0 + delta_ratio), min_scale + 1e-4)
    geometric_scale = float(np.sqrt(max(width_scale * height_scale, 1e-8)))
    candidates: list[float] = []
    for value in (
        float(fallback_scale),
        float(estimated_scale),
        float(width_scale),
        float(height_scale),
        float(0.5 * (width_scale + height_scale)),
        geometric_scale,
    ):
        if not np.isfinite(value) or value <= 0.0:
            continue
        candidates.append(round(float(np.clip(value, min_scale, max_scale)), 6))
    for factor in (
        1.0 - delta_ratio,
        1.0 - 0.5 * delta_ratio,
        1.0,
        1.0 + 0.5 * delta_ratio,
        1.0 + delta_ratio,
    ):
        candidates.append(round(float(np.clip(float(fallback_scale) * factor, min_scale, max_scale)), 6))
    return np.array(sorted(set(candidates)), dtype=np.float32)


def compute_confidence(task: dict, best: dict) -> float:
    depthpointcloud = task.get("depthpointcloud") or {}
    model = task.get("model") or {}

    valid_ratio = float(depthpointcloud.get("valid_depth_ratio") or 0.0)
    point_count = float(depthpointcloud.get("point_count") or 0.0)
    point_score = float(np.clip(point_count / 2500.0, 0.0, 1.0))

    scale_w = float(model.get("width_scale") or 0.0)
    scale_h = float(model.get("height_scale") or 0.0)
    scale_consistency = 1.0
    if max(scale_w, scale_h, 1e-6) > 0:
        scale_consistency = 1.0 - abs(scale_w - scale_h) / max(scale_w, scale_h, 1e-6)
    scale_consistency = float(np.clip(scale_consistency, 0.0, 1.0))

    real_w = float(depthpointcloud.get("real_width_measured") or 0.0)
    real_h = float(depthpointcloud.get("real_height_measured") or 0.0)
    size_ref = max(real_w, real_h, 1e-6)
    surface_rmse = float(best.get("surface_rmse_3d") or best["rmse"])
    coverage_rmse = float(best.get("coverage_rmse_3d") or best["rmse"])
    rmse_score = 1.0 - min(surface_rmse / size_ref, 1.0)
    coverage_score = 1.0 - min(coverage_rmse / size_ref, 1.0)
    inlier_ratio = float(np.clip(best["inlier_ratio"], 0.0, 1.0))

    confidence = (
        0.25 * valid_ratio
        + 0.15 * scale_consistency
        + 0.25 * rmse_score
        + 0.15 * coverage_score
        + 0.10 * inlier_ratio
        + 0.10 * point_score
    )
    return float(np.clip(confidence, 0.0, 1.0))


def build_final_camera_local_rh_debug(
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    metrics: dict,
) -> dict:
    debug = {
        "scale": float(scale),
        "up_y": model_up_y_in_unity(rotation),
        "pose": serialize_pose(
            rotation,
            translation,
        ),
        "front_view_rmse_2d": float(metrics["rmse_2d"]),
        "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
        "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
        "front_view_size_error": float(metrics["size_error"]),
        "width_error": float(metrics["width_error"]),
        "height_error": float(metrics["height_error"]),
        "center_error": float(metrics["center_error"]),
        "initial_rotation_penalty": float(metrics["initial_rotation_penalty"]),
    }
    if isinstance(metrics.get("placement"), dict):
        debug["placement"] = metrics["placement"]
    return debug


def solve_camera_local_alignment(
    model_vertices_unity: np.ndarray,
    target_front_fit: np.ndarray,
    overall_scale: float,
    target_context: dict,
) -> tuple[dict, dict]:
    base_rotation = np.eye(3, dtype=np.float32)
    base_front_unit = extract_front_visible_points(
        model_vertices_unity,
        bins=180,
        max_points=ICP_COARSE_VISIBLE_MAX_POINTS,
        seed=53,
    )
    scale_estimate = estimate_scale_from_visible_extents(
        base_front_unit,
        target_context["points"],
        fallback_scale=overall_scale,
    )
    scale_candidates = build_camera_refine_scale_candidates(
        estimated_scale=float(scale_estimate["estimated_scale"]),
        width_scale=float(scale_estimate["width_scale"]),
        height_scale=float(scale_estimate["height_scale"]),
        fallback_scale=overall_scale,
    )

    quick_candidates: list[dict] = []
    scale_eval_keep = max(1, min(len(scale_candidates), 2))
    rotation_offsets_deg = (-8, -4, 0, 4, 8)
    max_rotation_delta_deg = float(ICP_CAMERA_REFINE_MAX_ROTATION_DELTA_DEG)
    for rx in rotation_offsets_deg:
        for ry in rotation_offsets_deg:
            for rz in rotation_offsets_deg:
                rotation = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix().astype(np.float32)
                if should_reject_inverted_solution(rotation):
                    continue
                if not is_within_rotation_delta(base_rotation, rotation, max_rotation_delta_deg):
                    continue

                rotated_full_unit = transform_points(
                    model_vertices_unity,
                    1.0,
                    rotation,
                    np.zeros(3, dtype=np.float32),
                )
                rotated_front_unit = extract_front_visible_points(
                    rotated_full_unit,
                    bins=180,
                    max_points=ICP_COARSE_VISIBLE_MAX_POINTS,
                    seed=59,
                )
                ranked_scales = rank_scale_candidates(
                    rotated_full_unit=rotated_full_unit,
                    rotated_front_unit=rotated_front_unit,
                    scale_candidates=scale_candidates,
                    nominal_scale=overall_scale,
                    target_context=target_context,
                    rotation=rotation,
                )
                for ranked in ranked_scales[:scale_eval_keep]:
                    quick_candidates.append(
                        {
                            "rotation": rotation,
                            "translation": ranked["translation"].astype(np.float32),
                            "scale": float(ranked["scale"]),
                            "quick_score": float(ranked["quick_metrics"]["score"]),
                            "includes_identity_lineage": bool(is_identity_rotation(rotation)),
                        }
                    )

    if not quick_candidates:
        return build_distance_only_alignment(
            model_vertices_unity=model_vertices_unity,
            target_front_fit=target_front_fit,
            overall_scale=overall_scale,
            target_context=target_context,
        )

    quick_candidates.sort(key=lambda item: item["quick_score"])
    quick_keep = max(1, min(len(quick_candidates), int(ICP_CAMERA_REFINE_SEED_KEEP)))
    selected_candidates = list(quick_candidates[:quick_keep])
    identity_candidate = next(
        (item for item in quick_candidates if bool(item.get("includes_identity_lineage"))),
        None,
    )
    if identity_candidate is not None and all(identity_candidate is not item for item in selected_candidates):
        if selected_candidates:
            selected_candidates[-1] = identity_candidate
        else:
            selected_candidates.append(identity_candidate)
        selected_candidates.sort(key=lambda item: item["quick_score"])

    best: dict | None = None
    for seeded in selected_candidates:
        result = run_icp(
            model_points=model_vertices_unity,
            target_points=target_front_fit,
            scale=float(seeded["scale"]),
            init_rotation=seeded["rotation"],
            init_translation=seeded["translation"],
            iterations=ICP_FINAL_ITERATIONS,
            visible_bins=180,
            visible_max_points=ICP_FINAL_VISIBLE_MAX_POINTS,
            reference_rotation=base_rotation,
            max_rotation_delta_deg=max_rotation_delta_deg,
        )
        final_rotation = result["rotation"]
        if should_reject_inverted_solution(final_rotation):
            continue
        if not is_within_rotation_delta(base_rotation, final_rotation, max_rotation_delta_deg):
            continue

        transformed_full = transform_points(
            model_vertices_unity,
            float(seeded["scale"]),
            final_rotation,
            result["translation"],
        )
        transformed_front = extract_front_visible_points(
            transformed_full,
            bins=180,
            max_points=ICP_FINAL_VISIBLE_MAX_POINTS,
            seed=61,
        )
        metrics = evaluate_alignment(
            transformed_full_points=transformed_full,
            transformed_front_points=transformed_front,
            target_front_points=target_front_fit,
            scale=float(seeded["scale"]),
            nominal_scale=overall_scale,
            target_context=target_context,
            rotation=final_rotation,
        )
        candidate = {
            "scale": float(seeded["scale"]),
            "rotation": final_rotation,
            "translation": result["translation"].astype(np.float32),
            "rmse": float(metrics["rmse_3d"]),
            "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
            "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
            "rmse_2d": float(metrics["rmse_2d"]),
            "score": float(metrics["score"]),
            "size_error": float(metrics["size_error"]),
            "inlier_ratio": float(result["inlier_ratio"]),
            "extents": metrics["extents"],
            "width_error": float(metrics["width_error"]),
            "height_error": float(metrics["height_error"]),
            "center_error": float(metrics["center_error"]),
            "initial_rotation_penalty": float(metrics["initial_rotation_penalty"]),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        return build_distance_only_alignment(
            model_vertices_unity=model_vertices_unity,
            target_front_fit=target_front_fit,
            overall_scale=overall_scale,
            target_context=target_context,
        )

    final_debug = build_final_camera_local_rh_debug(
        best["rotation"],
        best["translation"],
        float(best["scale"]),
        best,
    )
    return best, final_debug


def build_distance_only_alignment(
    model_vertices_unity: np.ndarray,
    target_front_fit: np.ndarray,
    overall_scale: float,
    target_context: dict,
) -> tuple[dict, dict]:
    initial_rotation = np.eye(3, dtype=np.float32)
    initial_full = transform_points(
        model_vertices_unity,
        overall_scale,
        initial_rotation,
        np.zeros(3, dtype=np.float32),
    )
    initial_front = extract_front_visible_points(
        initial_full,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=11,
    )
    initial_translation = build_initial_translation(
        model_full_points=initial_full,
        model_front_points=initial_front,
        target_front_points=target_front_fit,
        target_context=target_context,
    )
    transformed_full = initial_full + initial_translation
    transformed_front = initial_front + initial_translation
    metrics = evaluate_alignment(
        transformed_full_points=transformed_full,
        transformed_front_points=transformed_front,
        target_front_points=target_front_fit,
        scale=float(overall_scale),
        nominal_scale=overall_scale,
        target_context=target_context,
        rotation=initial_rotation,
    )
    best = {
        "scale": float(overall_scale),
        "rotation": initial_rotation,
        "translation": initial_translation.astype(np.float32),
        "rmse": float(metrics["rmse_3d"]),
        "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
        "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
        "rmse_2d": float(metrics["rmse_2d"]),
        "score": float(metrics["score"]),
        "size_error": float(metrics["size_error"]),
        "inlier_ratio": 0.0,
        "extents": metrics["extents"],
        "width_error": float(metrics["width_error"]),
        "height_error": float(metrics["height_error"]),
        "center_error": float(metrics["center_error"]),
        "initial_rotation_penalty": float(metrics["initial_rotation_penalty"]),
    }
    final_debug = build_final_camera_local_rh_debug(
        best["rotation"],
        best["translation"],
        float(best["scale"]),
        best,
    )
    return best, final_debug


def build_bbox_surface_alignment(
    model_vertices_unity: np.ndarray,
    target_points: np.ndarray,
    target_front_fit: np.ndarray,
    overall_scale: float,
    target_context: dict,
) -> tuple[dict, dict]:
    rotation = np.eye(3, dtype=np.float32)
    scaled_full = np.asarray(model_vertices_unity, dtype=np.float32) * float(overall_scale)
    bbox_min = scaled_full.min(axis=0)
    bbox_max = scaled_full.max(axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_center = 0.5 * (bbox_min + bbox_max)

    target_points = np.asarray(target_points, dtype=np.float32)
    if len(target_points) == 0:
        raise ValueError("target_points must be non-empty for bbox surface alignment")
    target_front_fit = np.asarray(target_front_fit, dtype=np.float32)
    if len(target_front_fit) == 0:
        raise ValueError("target_front_fit must be non-empty for bbox surface alignment")

    ray_source = str(ICP_BBOX_SURFACE_RAY_SOURCE or "all_points").strip().lower()
    if ray_source in {"all", "all_points", "pointcloud"}:
        ray_points = target_points
    elif ray_source in {"front", "front_points", "target_front"}:
        ray_points = target_front_fit
    else:
        raise ValueError(
            "config.ICP_BBOX_SURFACE_RAY_SOURCE must be one of: all_points / front_points"
        )

    target_centroid = ray_points.mean(axis=0)
    ray_norm = float(np.linalg.norm(target_centroid))
    if not np.isfinite(ray_norm) or ray_norm <= 1e-6:
        ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        ray_direction = (target_centroid / ray_norm).astype(np.float32)

    distance_mode = str(ICP_BBOX_SURFACE_DISTANCE_MODE or "mean_depth").strip().lower()
    if distance_mode in {"centroid_norm", "ray_centroid"}:
        front_surface_distance = ray_norm
    elif distance_mode in {"mean_depth", "pointcloud_mean_depth"}:
        front_depth = float(np.mean(-target_points[:, 2]))
        front_surface_distance = front_depth / max(-float(ray_direction[2]), 1e-6)
    elif distance_mode in {"median_depth", "pointcloud_median_depth"}:
        front_depth = float(np.median(-target_points[:, 2]))
        front_surface_distance = front_depth / max(-float(ray_direction[2]), 1e-6)
    elif distance_mode in {"front_median_depth", "front_depth"}:
        front_depth = float(np.median(-target_front_fit[:, 2]))
        front_surface_distance = front_depth / max(-float(ray_direction[2]), 1e-6)
    elif distance_mode in {"front_median_norm", "front_norm"}:
        front_surface_distance = float(np.median(np.linalg.norm(target_front_fit, axis=1)))
    else:
        raise ValueError(
            "config.ICP_BBOX_SURFACE_DISTANCE_MODE must be one of: "
            "mean_depth / median_depth / front_median_depth / centroid_norm / front_median_norm"
        )

    thickness_factor = float(ICP_BBOX_SURFACE_THICKNESS_FACTOR)
    bbox_thickness = float(np.dot(np.abs(ray_direction), bbox_size))
    thickness_offset = thickness_factor * max(bbox_thickness, 0.0)
    desired_bbox_center = ray_direction * (front_surface_distance + thickness_offset)
    lateral_mode = str(ICP_BBOX_SURFACE_LATERAL_MODE or "centroid_xy").strip().lower()
    if lateral_mode in {"centroid_xy", "pointcloud_xy"}:
        desired_bbox_center[0] = target_centroid[0]
        desired_bbox_center[1] = target_centroid[1]
    elif lateral_mode in {"ray", "ray_xy"}:
        pass
    else:
        raise ValueError(
            "config.ICP_BBOX_SURFACE_LATERAL_MODE must be one of: centroid_xy / ray"
        )
    translation = (desired_bbox_center - bbox_center).astype(np.float32)

    transformed_full = scaled_full + translation
    transformed_front = extract_front_visible_points(
        transformed_full,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=11,
    )
    metrics = evaluate_alignment(
        transformed_full_points=transformed_full,
        transformed_front_points=transformed_front,
        target_front_points=target_front_fit,
        scale=float(overall_scale),
        nominal_scale=overall_scale,
        target_context=target_context,
        rotation=rotation,
    )
    metrics["placement"] = {
        "mode": "bbox_front_surface",
        "ray_source": ray_source,
        "distance_mode": distance_mode,
        "lateral_mode": lateral_mode,
        "thickness_factor": float(thickness_factor),
        "target_centroid": [float(v) for v in target_centroid],
        "ray_direction": [float(v) for v in ray_direction],
        "front_surface_distance": float(front_surface_distance),
        "bbox_min_scaled": [float(v) for v in bbox_min],
        "bbox_max_scaled": [float(v) for v in bbox_max],
        "bbox_center_scaled": [float(v) for v in bbox_center],
        "bbox_size_scaled": [float(v) for v in bbox_size],
        "bbox_thickness_along_ray": float(bbox_thickness),
        "bbox_thickness_offset_along_ray": float(thickness_offset),
        "desired_bbox_center": [float(v) for v in desired_bbox_center],
    }
    best = {
        "scale": float(overall_scale),
        "rotation": rotation,
        "translation": translation,
        "rmse": float(metrics["rmse_3d"]),
        "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
        "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
        "rmse_2d": float(metrics["rmse_2d"]),
        "score": float(metrics["score"]),
        "size_error": float(metrics["size_error"]),
        "inlier_ratio": 0.0,
        "extents": metrics["extents"],
        "width_error": float(metrics["width_error"]),
        "height_error": float(metrics["height_error"]),
        "center_error": float(metrics["center_error"]),
        "initial_rotation_penalty": float(metrics["initial_rotation_penalty"]),
        "placement": metrics["placement"],
    }
    final_debug = build_final_camera_local_rh_debug(
        best["rotation"],
        best["translation"],
        float(best["scale"]),
        best,
    )
    return best, final_debug


def main(argv: list[str]) -> int:
    json_path, task = load_stage_task(
        argv,
        usage=(
            "Usage: python code/stages/hololens3d_reconstruction/"
            "run_object_icp_alignment_from_json.py <task_meta.json or filename> [blender_path]"
        ),
        stage_name="icpalignment",
        valid_lengths=(2, 3),
    )
    icp_backend = resolve_icp_backend()

    if "depthpointcloud" not in task:
        raise ValueError("depthpointcloud is missing. Run pointcloud stage first.")
    if "model" not in task:
        raise ValueError("model is missing. Run model scale stage first.")

    paths = resolve_task_paths(task)
    prefix = task_prefix(task, json_path)

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    mask_bool = read_mask(paths["mask_path"])
    depth_mm = read_depth_image(paths["depth_path"])
    pointcloud_points_export, pointcloud_points_unity = build_depth_pointcloud(
        depth_mm,
        mask_bool,
        k,
    )
    valid_all = mask_bool & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    discarded_mask = valid_all & ~build_depth_border_keep_mask(mask_bool)
    discarded_points_export, _discarded_points_canonical = build_depth_pointcloud_from_valid_mask(
        depth_mm,
        discarded_mask,
        k,
    )
    target_front_fit, target_front_indices = select_front_visible_points(
        pointcloud_points_unity,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=7,
    )
    icp_used_points_export = pointcloud_points_export[target_front_indices]

    model_vertices_raw = read_obj_vertices(paths["mesh_path"])
    model_vertices_unity = obj_vertices_to_canonical_rh(model_vertices_raw)
    alignment_model_points = (
        build_alignment_model_points(model_vertices_unity)
        if bool(ICP_ENABLE)
        else np.empty((0, 3), dtype=np.float32)
    )

    target_context = build_target_context(target_front_fit)
    overall_scale = float((task.get("model") or {}).get("overall_scale") or 1.0)
    preview_initial_rotation = np.eye(3, dtype=np.float32)
    preview_initial_full = transform_points(
        model_vertices_unity,
        overall_scale,
        preview_initial_rotation,
        np.zeros(3, dtype=np.float32),
    )
    preview_initial_front = extract_front_visible_points(
        preview_initial_full,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=11,
    )
    preview_initial_translation = build_initial_translation(
        model_full_points=preview_initial_full,
        model_front_points=preview_initial_front,
        target_front_points=target_front_fit,
        target_context=target_context,
    )

    if ICP_MODE == "camera_refine":
        best, final_camera_local_unity_debug = solve_camera_local_alignment(
            model_vertices_unity=alignment_model_points,
            target_front_fit=target_front_fit,
            overall_scale=overall_scale,
            target_context=target_context,
        )
    elif ICP_MODE == "off":
        best, final_camera_local_unity_debug = build_bbox_surface_alignment(
            model_vertices_unity=model_vertices_unity,
            target_points=pointcloud_points_unity,
            target_front_fit=target_front_fit,
            overall_scale=overall_scale,
            target_context=target_context,
        )
    else:
        raise ValueError(f"Unsupported ICP_MODE: {ICP_MODE}")

    camera_local_quat_xyzw = rotation_matrix_to_quat_xyzw(best["rotation"])
    blender_rotation, blender_translation = model_pose_canonical_rh_to_blender_world(
        best["rotation"],
        best["translation"],
    )
    blender_delta_euler_deg = np.asarray(
        safe_matrix_to_blender_euler_xyz_deg(blender_rotation),
        dtype=np.float32,
    )

    confidence = compute_confidence(task, best)
    overlay_preview_name = f"{prefix}_alignment_preview_pointcloud_model.png"
    overlay_preview_path = object_alignment_output_path(overlay_preview_name)
    unaligned_preview_name = f"{prefix}_alignment_preview_model_compare.png"
    unaligned_preview_path = object_alignment_output_path(unaligned_preview_name)
    preview_image_name: str | None = overlay_preview_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None
    preview_image_unaligned_name: str | None = unaligned_preview_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None

    object_alignment = {
        "camera_local_position": [float(v) for v in best["translation"]],
        "camera_local_rotation_quaternion_xyzw": [float(v) for v in camera_local_quat_xyzw],
        "model_real_scale": float(best["scale"]),
        "icp_mode": str(ICP_MODE),
        "icp_enabled": bool(ICP_ENABLE),
        "preview_image_name": preview_image_name,
        "preview_image_unaligned_name": preview_image_unaligned_name,
        "confidence": confidence,
        "icp_rmse": float(best["rmse"]),
        "icp_fit_model_point_count": int(len(alignment_model_points)) if bool(ICP_ENABLE) else 0,
        "target_point_count": int(len(pointcloud_points_unity)),
        "target_front_point_count": int(len(target_front_fit)),
        "discarded_point_count": int(len(discarded_points_export)),
    }
    preview_render_errors: list[dict[str, str]] = []
    task["object_alignment"] = object_alignment

    debug_section = dict(task.get("debug") or {})
    pose_debug = dict(debug_section.get("pose_transform_stages") or {})
    pose_debug["object_alignment"] = {
        "final_camera_local_rh": final_camera_local_unity_debug,
    }
    debug_section["pose_transform_stages"] = pose_debug
    task["debug"] = debug_section

    if ENABLE_ALIGNMENT_RENDER_OUTPUTS:
        preview_tmp_root = object_alignment_output_path("_tmp_preview_root").parent
        tmp_dir_path = preview_tmp_root / f"{prefix}_alignment_{uuid.uuid4().hex}"
        tmp_dir_path.mkdir(parents=False, exist_ok=False)
        try:
            discarded_points_preview_path = tmp_dir_path / "icp_discarded_points.ply"
            icp_points_preview_path = tmp_dir_path / "icp_used_points.ply"
            write_binary_ply(discarded_points_preview_path, discarded_points_export)
            write_binary_ply(icp_points_preview_path, icp_used_points_export)
            try:
                render_overlay_preview_image(
                    mesh_path=paths["mesh_path"],
                    discarded_pointcloud_path=discarded_points_preview_path,
                    icp_pointcloud_path=icp_points_preview_path,
                    render_path=overlay_preview_path,
                    blender_translation=blender_translation,
                    blender_delta_euler_deg=blender_delta_euler_deg,
                    scale=float(best["scale"]),
                    task=task,
                    blender_arg=argv[2] if len(argv) == 3 else None,
                    title=(
                        "Camera-local Refine Preview"
                        if bool(ICP_ENABLE)
                        else "ICP Skipped Preview"
                    ),
                    header_line=(
                        "Camera-view preview: two-color point cloud + refined model"
                        if bool(ICP_ENABLE)
                        else "Camera-view preview: two-color point cloud + initial model"
                    ),
                    model_legend_line=(
                        "Orange = mask-border-discarded points, green = ICP-used points, blue = refined model"
                        if bool(ICP_ENABLE)
                        else "Orange = mask-border-discarded points, green = ICP-used points, blue = initial model"
                    ),
                )
            except Exception as exc:
                preview_render_errors.append(
                    {
                        "preview": "overlay_preview",
                        "path": overlay_preview_name,
                        "error": str(exc),
                    }
                )
                object_alignment["preview_image_name"] = None
                if overlay_preview_path.exists():
                    overlay_preview_path.unlink()
        finally:
            shutil.rmtree(tmp_dir_path, ignore_errors=True)

        try:
            render_model_compare_preview_image(
                mesh_path=paths["mesh_path"],
                render_path=unaligned_preview_path,
                aligned_blender_translation=blender_translation,
                aligned_blender_delta_euler_deg=blender_delta_euler_deg,
                aligned_scale=float(best["scale"]),
                reference_blender_translation=canonical_rh_to_blender_world_vector(preview_initial_translation),
                reference_blender_delta_euler_deg=np.zeros(3, dtype=np.float32),
                reference_scale=float(overall_scale),
                task=task,
                blender_arg=argv[2] if len(argv) == 3 else None,
                title=(
                    "Before/After Model Preview"
                    if bool(ICP_ENABLE)
                    else "Model Preview (ICP Skipped)"
                ),
                header_line=(
                    "Camera-view preview: aligned model overlaid with initial model"
                    if bool(ICP_ENABLE)
                    else "Camera-view preview: initial model only (before/after identical)"
                ),
                model_legend_line=(
                    "Blue = aligned model, orange = initial model"
                    if bool(ICP_ENABLE)
                    else "Blue = current model, orange = initial model"
                ),
            )
        except Exception as exc:
            preview_render_errors.append(
                {
                    "preview": "model_compare_preview",
                    "path": unaligned_preview_name,
                    "error": str(exc),
                }
            )
            object_alignment["preview_image_unaligned_name"] = None
            if unaligned_preview_path.exists():
                unaligned_preview_path.unlink()
    elif overlay_preview_path.exists():
        overlay_preview_path.unlink()
        if unaligned_preview_path.exists():
            unaligned_preview_path.unlink()

    if preview_render_errors:
        object_alignment["preview_render_errors"] = preview_render_errors
        pose_debug["object_alignment"]["preview_render_errors"] = preview_render_errors

    save_task_json(json_path, task)

    print(
        f"[INFO] icpalignment : backend={icp_backend['actual']} "
        f"icp_mode={str(ICP_MODE)} "
        f"icp_enabled={bool(ICP_ENABLE)} "
        f"points={len(pointcloud_points_export)}/{len(target_front_fit)} "
        f"model_fit_points={len(alignment_model_points)} discarded={len(discarded_points_export)} "
        f"scale={object_alignment['model_real_scale']:.6f} confidence={confidence:.3f} "
        f"rmse={object_alignment['icp_rmse']:.6f} preview={object_alignment.get('preview_image_name') or 'not-generated'}"
    )
    print(
        "[INFO] pose-local-rh   : "
        f"pos={object_alignment['camera_local_position']} "
        f"quat={object_alignment['camera_local_rotation_quaternion_xyzw']}"
    )
    print("[OK] icpalignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
