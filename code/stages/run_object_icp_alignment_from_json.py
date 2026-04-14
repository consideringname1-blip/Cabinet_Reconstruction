from __future__ import annotations

import itertools
import sys
import warnings

from _bootstrap import CODE_ROOT
from alignment_preview import render_model_compare_preview_image, render_overlay_preview_image
from config import (
    ICP_ACCELERATION_DEVICE,
    ICP_AXIS_SEED_RETAIN_TOPK,
    ICP_COARSE_CANDIDATE_KEEP,
    ICP_COARSE_SCALE_EVAL_KEEP,
    ICP_COARSE_VISIBLE_MAX_POINTS,
    ENABLE_ALIGNMENT_RENDER_OUTPUTS,
    ICP_FINAL_SCALE_CANDIDATE_KEEP,
    ICP_FINAL_ITERATIONS,
    ICP_FINAL_VISIBLE_MAX_POINTS,
    ICP_FINE_VISIBLE_MAX_POINTS,
    ICP_IGNORE_INVERTED_SOLUTIONS,
    ICP_LOCAL_REFINE_CANDIDATE_KEEP,
    ICP_IGNORE_OCCLUDED_MODEL_POINTS,
    ICP_LOCAL_REFINE_ITERATIONS,
    ICP_LOCAL_REFINE_VISIBLE_MAX_POINTS,
    ICP_MEDIUM_RETAIN_TOPK,
    ICP_MEDIUM_VISIBLE_MAX_POINTS,
    ICP_TARGET_FRONT_MAX_POINTS,
)
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from object_alignment_common import (
    build_depth_border_keep_mask,
    build_depth_pointcloud_from_valid_mask,
    compute_front_view_extents,
    extract_front_visible_points,
    model_pose_unity_to_blender_world,
    model_pose_unity_to_pointcloud_input,
    MAX_DEPTH_MM,
    MIN_DEPTH_MM,
    obj_vertices_to_unity,
    object_alignment_output_path,
    pointcloud_export_to_unity,
    read_depth_image,
    read_binary_ply_points,
    read_mask,
    read_obj_vertices,
    resolve_task_paths,
    select_front_visible_points,
    task_prefix,
    unity_to_blender_world_vector,
    write_binary_ply,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


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


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero-length quaternion")
    return q / n


def make_row_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[3, :3] = translation
    return matrix


def serialize_pose(rotation: np.ndarray, translation: np.ndarray, coordinate_basis: str) -> dict:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    quat_xyzw = normalize_quat_xyzw(Rotation.from_matrix(rotation).as_quat())
    euler_deg = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    matrix = make_row_transform_matrix(rotation, translation)
    return {
        "coordinate_basis": coordinate_basis,
        "position": [float(v) for v in translation],
        "rotation_euler_deg": [float(v) for v in euler_deg],
        "rotation_quaternion_xyzw": [float(v) for v in quat_xyzw],
        "transform_matrix": [[float(v) for v in row] for row in matrix],
    }


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


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


def should_reject_inverted_solution(rotation: np.ndarray) -> bool:
    return bool(ICP_IGNORE_INVERTED_SOLUTIONS) and (model_up_y_in_unity(rotation) < 0.0)


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
) -> dict:
    extents = compute_front_view_extents(transformed_front_points)
    target_extents = target_context["extents"]
    width_error = abs(extents["width_units"] - target_extents["width_units"])
    height_error = abs(extents["height_units"] - target_extents["height_units"])
    size_error = width_error + height_error
    scale_error = abs(scale - nominal_scale) / max(nominal_scale, 1e-6)
    center_error = float(np.linalg.norm(compute_xy_center(extents) - target_context["center_xy"]))
    score = 0.70 * size_error + 0.20 * center_error + 0.10 * scale_error
    return {
        "score": float(score),
        "size_error": float(size_error),
        "width_error": float(width_error),
        "height_error": float(height_error),
        "center_error": center_error,
        "extents": extents,
    }


def rank_scale_candidates(
    rotated_full_unit: np.ndarray,
    rotated_front_unit: np.ndarray,
    scale_candidates: np.ndarray,
    nominal_scale: float,
    target_context: dict,
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
) -> dict:
    rotation = init_rotation.astype(np.float32).copy()
    translation = init_translation.astype(np.float32).copy()
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
        rotation = delta_r @ rotation
        translation = delta_r @ translation + delta_t
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
    score = (
        0.60 * surface_rmse_3d
        + 0.20 * rmse_2d
        + 0.10 * size_error
        + 0.05 * center_error
        + 0.05 * scale_error
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
        "extents": extents,
    }


def refine_pose_locally(
    model_points: np.ndarray,
    target_front_points: np.ndarray,
    scale: float,
    nominal_scale: float,
    base_rotation: np.ndarray,
    base_translation: np.ndarray,
    target_context: dict | None = None,
) -> dict:
    if target_context is None:
        target_context = build_target_context(target_front_points)

    quick_candidates: list[dict] = []
    for rx in (-9, -6, -3, 0, 3, 6, 9):
        for ry in (-9, -6, -3, 0, 3, 6, 9):
            for rz in (-9, -6, -3, 0, 3, 6, 9):
                delta = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix().astype(np.float32)
                init_rotation = delta @ base_rotation
                if should_reject_inverted_solution(init_rotation):
                    continue

                init_full = transform_points(model_points, float(scale), init_rotation, np.zeros(3, dtype=np.float32))
                init_front = extract_front_visible_points(
                    init_full,
                    bins=180,
                    max_points=ICP_LOCAL_REFINE_VISIBLE_MAX_POINTS,
                    seed=29,
                )
                init_translation = build_initial_translation(
                    model_full_points=init_full,
                    model_front_points=init_front,
                    target_front_points=target_front_points,
                    target_context=target_context,
                )
                quick_metrics = evaluate_alignment_geometry_only(
                    init_front + init_translation,
                    target_context=target_context,
                    scale=float(scale),
                    nominal_scale=nominal_scale,
                )
                quick_candidates.append(
                    {
                        "rotation": init_rotation,
                        "translation": init_translation,
                        "delta_euler_deg": [float(rx), float(ry), float(rz)],
                        "quick_score": float(quick_metrics["score"]),
                    }
                )

    if not quick_candidates:
        return {
            "rotation": base_rotation,
            "translation": base_translation,
            "delta_euler_deg": [0.0, 0.0, 0.0],
            "refinement_applied": False,
        }

    quick_candidates.sort(key=lambda item: item["quick_score"])
    candidate_keep = max(1, min(len(quick_candidates), int(ICP_LOCAL_REFINE_CANDIDATE_KEEP)))

    best_candidate: dict | None = None
    best_score = float("inf")
    best_metrics: dict | None = None

    for seeded in quick_candidates[:candidate_keep]:
        result = run_icp(
            model_points=model_points,
            target_points=target_front_points,
            scale=float(scale),
            init_rotation=seeded["rotation"],
            init_translation=seeded["translation"],
            iterations=ICP_LOCAL_REFINE_ITERATIONS,
            visible_bins=180,
            visible_max_points=ICP_LOCAL_REFINE_VISIBLE_MAX_POINTS,
        )
        rotation = result["rotation"]
        translation = result["translation"]
        if should_reject_inverted_solution(rotation):
            continue

        transformed_full = transform_points(model_points, float(scale), rotation, translation)
        transformed_front = extract_front_visible_points(
            transformed_full,
            bins=180,
            max_points=ICP_LOCAL_REFINE_VISIBLE_MAX_POINTS,
            seed=31,
        )
        metrics = evaluate_alignment(
            transformed_full_points=transformed_full,
            transformed_front_points=transformed_front,
            target_front_points=target_front_points,
            scale=float(scale),
            nominal_scale=nominal_scale,
            target_context=target_context,
        )
        score = float(metrics["score"])
        if score < best_score:
            best_score = score
            best_metrics = metrics
            best_candidate = {
                "rotation": rotation,
                "translation": translation,
                "rmse": float(metrics["rmse_3d"]),
                "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
                "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
                "rmse_2d": float(metrics["rmse_2d"]),
                "score": score,
                "size_error": float(metrics["size_error"]),
                "inlier_ratio": float(result["inlier_ratio"]),
                "extents": metrics["extents"],
                "width_error": float(metrics["width_error"]),
                "height_error": float(metrics["height_error"]),
                "center_error": float(metrics["center_error"]),
                "quick_score": float(seeded["quick_score"]),
            }

    if best_candidate is None or best_metrics is None:
        return {
            "rotation": base_rotation,
            "translation": base_translation,
            "delta_euler_deg": [0.0, 0.0, 0.0],
            "refinement_applied": False,
        }

    best_candidate["refinement_applied"] = True
    return best_candidate


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


def build_scale_candidates(
    estimated_scale: float,
    width_scale: float,
    height_scale: float,
    fallback_scale: float,
) -> np.ndarray:
    raw_scales = [
        float(fallback_scale),
        float(estimated_scale),
        float(0.5 * (width_scale + height_scale)),
    ]
    min_scale = max(float(fallback_scale) * 0.65, 1e-4)
    max_scale = max(float(fallback_scale) * 1.45, min_scale + 1e-4)

    candidates: list[float] = []
    for base in raw_scales:
        if not np.isfinite(base) or base <= 0:
            continue
        for factor in (0.95, 1.0, 1.05):
            value = float(np.clip(base * factor, min_scale, max_scale))
            candidates.append(round(value, 6))
    return np.array(sorted(set(candidates)), dtype=np.float32)


def generate_axis_aligned_rotation_seeds() -> list[dict]:
    seeds: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float32)
            for row, col in enumerate(perm):
                rotation[row, col] = signs[row]
            if np.linalg.det(rotation) < 0.5:
                continue
            key = tuple(float(v) for v in rotation.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            seeds.append(
                {
                    "rotation": rotation,
                }
            )
    return seeds


def evaluate_pose_candidate(
    model_points: np.ndarray,
    target_front_points: np.ndarray,
    nominal_scale: float,
    rotation: np.ndarray,
    visible_bins: int,
    visible_max_points: int,
    scale_candidates: np.ndarray | None = None,
    target_context: dict | None = None,
) -> dict:
    if target_context is None:
        target_context = build_target_context(target_front_points)

    rotated_full_unit = transform_points(model_points, 1.0, rotation, np.zeros(3, dtype=np.float32))
    rotated_front_unit = extract_front_visible_points(
        rotated_full_unit,
        bins=visible_bins,
        max_points=visible_max_points,
        seed=13,
    )
    scale_estimate = estimate_scale_from_visible_extents(
        rotated_front_unit,
        target_context["points"],
        fallback_scale=nominal_scale,
    )
    if scale_candidates is None:
        scale_candidates = build_scale_candidates(
            estimated_scale=scale_estimate["estimated_scale"],
            width_scale=scale_estimate["width_scale"],
            height_scale=scale_estimate["height_scale"],
            fallback_scale=nominal_scale,
        )

    ranked_scales = rank_scale_candidates(
        rotated_full_unit=rotated_full_unit,
        rotated_front_unit=rotated_front_unit,
        scale_candidates=scale_candidates,
        nominal_scale=nominal_scale,
        target_context=target_context,
    )
    scale_eval_keep = max(1, min(len(ranked_scales), int(ICP_COARSE_SCALE_EVAL_KEEP)))

    best_candidate: dict | None = None
    for ranked in ranked_scales[:scale_eval_keep]:
        scale_value = float(ranked["scale"])
        scaled_full = rotated_full_unit * scale_value
        scaled_front = rotated_front_unit * scale_value
        translation = ranked["translation"]
        translated_full = scaled_full + translation
        translated_front = scaled_front + translation
        metrics = evaluate_alignment(
            transformed_full_points=translated_full,
            transformed_front_points=translated_front,
            target_front_points=target_front_points,
            scale=scale_value,
            nominal_scale=nominal_scale,
            target_context=target_context,
        )
        candidate = {
            "rotation": rotation.astype(np.float32),
            "translation": translation.astype(np.float32),
            "scale": scale_value,
            "metrics": metrics,
            "scale_estimate": scale_estimate,
            "quick_score": float(ranked["quick_metrics"]["score"]),
        }
        if best_candidate is None or candidate["metrics"]["score"] < best_candidate["metrics"]["score"]:
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("Failed to evaluate pose candidate")
    return best_candidate


def search_initial_pose_candidates(
    model_points: np.ndarray,
    target_front_points: np.ndarray,
    nominal_scale: float,
    target_context: dict | None = None,
) -> list[dict]:
    if target_context is None:
        target_context = build_target_context(target_front_points)
    candidates: list[dict] = []

    def append_candidate(
        rotation: np.ndarray,
        stage: str,
        visible_bins: int,
        visible_max_points: int,
    ) -> None:
        if should_reject_inverted_solution(rotation):
            return
        candidate = evaluate_pose_candidate(
            model_points=model_points,
            target_front_points=target_front_points,
            nominal_scale=nominal_scale,
            rotation=rotation,
            visible_bins=visible_bins,
            visible_max_points=visible_max_points,
            target_context=target_context,
        )
        candidate["stage"] = stage
        candidates.append(candidate)

    for seed in generate_axis_aligned_rotation_seeds():
        append_candidate(
            rotation=seed["rotation"],
            stage="axis_seed",
            visible_bins=120,
            visible_max_points=2600,
        )

    if not candidates:
        raise RuntimeError("No valid coarse pose candidates remain after applying ICP constraints.")

    seed_best = sorted(candidates, key=lambda item: item["metrics"]["score"])[:ICP_AXIS_SEED_RETAIN_TOPK]
    for base in seed_best:
        for rx in (-18, 0, 18):
            for ry in (-18, 0, 18):
                for rz in (-18, 0, 18):
                    perturb = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix().astype(np.float32)
                    append_candidate(
                        rotation=perturb @ base["rotation"],
                        stage="refine_medium",
                        visible_bins=144,
                        visible_max_points=ICP_MEDIUM_VISIBLE_MAX_POINTS,
                    )

    medium_best = sorted(candidates, key=lambda item: item["metrics"]["score"])[:ICP_MEDIUM_RETAIN_TOPK]
    for base in medium_best:
        for rx in (-6, 0, 6):
            for ry in (-6, 0, 6):
                for rz in (-6, 0, 6):
                    perturb = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix().astype(np.float32)
                    append_candidate(
                        rotation=perturb @ base["rotation"],
                        stage="refine_fine",
                        visible_bins=160,
                        visible_max_points=ICP_FINE_VISIBLE_MAX_POINTS,
                    )

    candidates.sort(key=lambda item: item["metrics"]["score"])
    if not candidates:
        raise RuntimeError("No valid pose candidates remain after applying ICP constraints.")
    return candidates


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


def build_final_camera_local_unity_debug(
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    metrics: dict,
) -> dict:
    return {
        "scale": float(scale),
        "up_y": model_up_y_in_unity(rotation),
        "pose": serialize_pose(
            rotation,
            translation,
            "unity_camera_local_x_right_y_up_z_forward",
        ),
        "front_view_rmse_2d": float(metrics["rmse_2d"]),
        "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
        "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
        "front_view_size_error": float(metrics["size_error"]),
        "width_error": float(metrics["width_error"]),
        "height_error": float(metrics["height_error"]),
        "center_error": float(metrics["center_error"]),
    }


def solve_alignment(
    model_vertices_unity: np.ndarray,
    target_front_fit: np.ndarray,
    overall_scale: float,
    target_context: dict,
) -> tuple[dict, dict]:
    coarse_candidates = search_initial_pose_candidates(
        model_points=model_vertices_unity,
        target_front_points=target_front_fit,
        nominal_scale=overall_scale,
        target_context=target_context,
    )
    coarse_candidates = coarse_candidates[:ICP_COARSE_CANDIDATE_KEEP]

    best: dict | None = None
    for coarse_candidate in coarse_candidates:
        candidate_scales = build_scale_candidates(
            estimated_scale=float(coarse_candidate["scale_estimate"]["estimated_scale"]),
            width_scale=float(coarse_candidate["scale_estimate"]["width_scale"]),
            height_scale=float(coarse_candidate["scale_estimate"]["height_scale"]),
            fallback_scale=overall_scale,
        )
        rotation_seed = coarse_candidate["rotation"]
        coarse_full_unit = transform_points(model_vertices_unity, 1.0, rotation_seed, np.zeros(3, dtype=np.float32))
        coarse_front_unit = extract_front_visible_points(
            coarse_full_unit,
            bins=180,
            max_points=ICP_COARSE_VISIBLE_MAX_POINTS,
            seed=19,
        )
        ranked_scales = rank_scale_candidates(
            rotated_full_unit=coarse_full_unit,
            rotated_front_unit=coarse_front_unit,
            scale_candidates=candidate_scales,
            nominal_scale=overall_scale,
            target_context=target_context,
        )
        scale_keep = max(1, min(len(ranked_scales), int(ICP_FINAL_SCALE_CANDIDATE_KEEP)))
        for ranked in ranked_scales[:scale_keep]:
            scale = float(ranked["scale"])
            coarse_translation = ranked["translation"]
            delta_result = run_icp(
                model_points=model_vertices_unity,
                target_points=target_front_fit,
                scale=scale,
                init_rotation=rotation_seed,
                init_translation=coarse_translation,
                iterations=ICP_FINAL_ITERATIONS,
                visible_bins=180,
                visible_max_points=ICP_FINAL_VISIBLE_MAX_POINTS,
            )
            final_rotation = delta_result["rotation"]
            final_translation = delta_result["translation"]
            if should_reject_inverted_solution(final_rotation):
                continue

            transformed_full = transform_points(model_vertices_unity, scale, final_rotation, final_translation)
            transformed_front = extract_front_visible_points(
                transformed_full,
                bins=180,
                max_points=ICP_FINAL_VISIBLE_MAX_POINTS,
                seed=23,
            )
            metrics = evaluate_alignment(
                transformed_full_points=transformed_full,
                transformed_front_points=transformed_front,
                target_front_points=target_front_fit,
                scale=scale,
                nominal_scale=overall_scale,
                target_context=target_context,
            )
            candidate = {
                "scale": scale,
                "rotation": final_rotation,
                "translation": final_translation,
                "rmse": float(metrics["rmse_3d"]),
                "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
                "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
                "rmse_2d": float(metrics["rmse_2d"]),
                "score": float(metrics["score"]),
                "size_error": float(metrics["size_error"]),
                "inlier_ratio": float(delta_result["inlier_ratio"]),
                "extents": metrics["extents"],
                "width_error": float(metrics["width_error"]),
                "height_error": float(metrics["height_error"]),
                "center_error": float(metrics["center_error"]),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError("Failed to compute object alignment")

    refined = refine_pose_locally(
        model_points=model_vertices_unity,
        target_front_points=target_front_fit,
        scale=float(best["scale"]),
        nominal_scale=overall_scale,
        base_rotation=best["rotation"],
        base_translation=best["translation"],
        target_context=target_context,
    )
    if refined.get("refinement_applied") and float(refined["score"]) < float(best["score"]):
        best = {
            **best,
            **{
                "rotation": refined["rotation"],
                "translation": refined["translation"],
                "rmse": refined["rmse"],
                "surface_rmse_3d": refined["surface_rmse_3d"],
                "coverage_rmse_3d": refined["coverage_rmse_3d"],
                "rmse_2d": refined["rmse_2d"],
                "score": refined["score"],
                "size_error": refined["size_error"],
                "inlier_ratio": refined["inlier_ratio"],
                "extents": refined["extents"],
                "width_error": refined["width_error"],
                "height_error": refined["height_error"],
                "center_error": refined["center_error"],
            },
        }

    final_debug = build_final_camera_local_unity_debug(
        best["rotation"],
        best["translation"],
        float(best["scale"]),
        best,
    )
    return best, final_debug


def remove_legacy_outputs(prefix: str) -> None:
    for name in (
        f"{prefix}_size_compare.png",
        f"{prefix}_model_front.png",
        f"{prefix}_model_front_tmp.png",
        f"{prefix}_aligned_model_front.png",
    ):
        path = object_alignment_output_path(name)
        if path.exists():
            path.unlink()


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(
            "Usage: python code/stages/run_object_icp_alignment_from_json.py <task_meta.json or filename> [blender_path]",
            file=sys.stderr,
        )
        return 2

    json_path = resolve_task_json_path(argv[1])
    task = load_task_json(json_path)
    print(f"[STAGE] icpalignment : {json_path}")
    icp_backend = resolve_icp_backend()

    if "depthpointcloud" not in task:
        raise ValueError("depthpointcloud is missing. Run pointcloud stage first.")
    if "model" not in task:
        raise ValueError("model is missing. Run model scale stage first.")

    paths = resolve_task_paths(task)
    prefix = task_prefix(task, json_path)
    pointcloud_name = (task.get("depthpointcloud") or {}).get("pointcloud_name")
    if not pointcloud_name:
        raise ValueError("depthpointcloud.pointcloud_name is missing")
    precomputed_discarded_name = (task.get("depthpointcloud") or {}).get("icp_discarded_pointcloud_name")
    precomputed_used_name = (task.get("depthpointcloud") or {}).get("icp_used_pointcloud_name")

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    pointcloud_path = object_alignment_output_path(pointcloud_name)
    pointcloud_points_export = read_binary_ply_points(pointcloud_path)
    pointcloud_points_unity = pointcloud_export_to_unity(pointcloud_points_export)
    model_vertices_raw = read_obj_vertices(paths["mesh_path"])
    model_vertices_unity = obj_vertices_to_unity(model_vertices_raw)

    if precomputed_discarded_name and precomputed_used_name:
        discarded_points_export = read_binary_ply_points(object_alignment_output_path(precomputed_discarded_name))
        icp_used_points_export = read_binary_ply_points(object_alignment_output_path(precomputed_used_name))
        target_front_fit = pointcloud_export_to_unity(icp_used_points_export)
    else:
        mask_bool = read_mask(paths["mask_path"])
        depth_mm = read_depth_image(paths["depth_path"])
        valid_all = mask_bool & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
        discarded_mask = valid_all & ~build_depth_border_keep_mask(mask_bool)
        discarded_points_export, _ = build_depth_pointcloud_from_valid_mask(depth_mm, discarded_mask, k)
        target_front_fit, target_front_indices = select_front_visible_points(
            pointcloud_points_unity,
            bins=160,
            max_points=ICP_TARGET_FRONT_MAX_POINTS,
            seed=7,
        )
        icp_used_points_export = pointcloud_points_export[target_front_indices]

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

    best, final_camera_local_unity_debug = solve_alignment(
        model_vertices_unity=model_vertices_unity,
        target_front_fit=target_front_fit,
        overall_scale=overall_scale,
        target_context=target_context,
    )

    pointcloud_rotation, pointcloud_translation = model_pose_unity_to_pointcloud_input(
        best["rotation"],
        best["translation"],
    )
    pointcloud_euler_deg = Rotation.from_matrix(pointcloud_rotation).as_euler("xyz", degrees=True)
    pointcloud_quat_xyzw = Rotation.from_matrix(pointcloud_rotation).as_quat()

    blender_rotation, blender_translation = model_pose_unity_to_blender_world(
        best["rotation"],
        best["translation"],
    )
    blender_delta_euler_deg = np.asarray(
        safe_matrix_to_blender_euler_xyz_deg(blender_rotation),
        dtype=np.float32,
    )

    confidence = compute_confidence(task, best)
    discarded_points_preview_name = f"{prefix}_icp_discarded_points.ply"
    discarded_points_preview_path = object_alignment_output_path(discarded_points_preview_name)
    icp_points_preview_name = f"{prefix}_icp_used_points.ply"
    icp_points_preview_path = object_alignment_output_path(icp_points_preview_name)
    overlay_preview_name = f"{prefix}_alignment_preview_perspective.png"
    overlay_preview_path = object_alignment_output_path(overlay_preview_name)
    unaligned_preview_name = f"{prefix}_alignment_preview_unaligned_perspective.png"
    unaligned_preview_path = object_alignment_output_path(unaligned_preview_name)
    write_binary_ply(discarded_points_preview_path, discarded_points_export)
    write_binary_ply(icp_points_preview_path, icp_used_points_export)
    preview_image_name: str | None = overlay_preview_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None
    preview_image_unaligned_name: str | None = unaligned_preview_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None

    object_alignment = {
        "coordinate_basis": "pointcloud_input_pre_blender_import",
        "model_position": [float(v) for v in pointcloud_translation],
        "model_rotation_euler_deg": [float(v) for v in pointcloud_euler_deg],
        "model_rotation_quaternion_xyzw": [float(v) for v in pointcloud_quat_xyzw],
        "model_real_scale": float(best["scale"]),
        "preview_image_name": preview_image_name,
        "preview_image_unaligned_name": preview_image_unaligned_name,
        "confidence": confidence,
        "icp_rmse": float(best["rmse"]),
    }
    task["object_alignment"] = object_alignment

    debug_section = dict(task.get("debug") or {})
    pose_debug = dict(debug_section.get("pose_transform_stages") or {})
    pose_debug["object_alignment"] = {
        "final_camera_local_unity": final_camera_local_unity_debug,
    }
    debug_section["pose_transform_stages"] = pose_debug
    task["debug"] = debug_section

    if ENABLE_ALIGNMENT_RENDER_OUTPUTS:
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
        )
        render_model_compare_preview_image(
            mesh_path=paths["mesh_path"],
            render_path=unaligned_preview_path,
            aligned_blender_translation=blender_translation,
            aligned_blender_delta_euler_deg=blender_delta_euler_deg,
            aligned_scale=float(best["scale"]),
            reference_blender_translation=unity_to_blender_world_vector(preview_initial_translation),
            reference_blender_delta_euler_deg=np.zeros(3, dtype=np.float32),
            reference_scale=float(overall_scale),
            task=task,
            blender_arg=argv[2] if len(argv) == 3 else None,
            title="Aligned vs Initial Model Preview",
            header_line="Perspective preview: ICP result overlaid with initial-distance model",
            model_legend_line="Blue = ICP-aligned model, orange = initial-distance model",
        )
    elif overlay_preview_path.exists():
        overlay_preview_path.unlink()
        if unaligned_preview_path.exists():
            unaligned_preview_path.unlink()

    save_task_json(json_path, task)
    remove_legacy_outputs(prefix)

    print(
        f"[INFO] icpalignment : backend={icp_backend['actual']} "
        f"points={len(pointcloud_points_export)}/{len(target_front_fit)} discarded={len(discarded_points_export)} "
        f"scale={object_alignment['model_real_scale']:.6f} confidence={confidence:.3f} "
        f"rmse={object_alignment['icp_rmse']:.6f} preview={object_alignment.get('preview_image_name') or 'not-generated'}"
    )
    print(f"[INFO] pose-local      : pos={object_alignment['model_position']} rot={object_alignment['model_rotation_euler_deg']}")
    print("[OK] icpalignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
