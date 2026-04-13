from __future__ import annotations

import itertools
import subprocess
import sys
import warnings
from pathlib import Path

from _bootstrap import CODE_ROOT
from config import (
    ICP_ACCELERATION_DEVICE,
    ICP_AXIS_SEED_RETAIN_TOPK,
    ICP_COARSE_CANDIDATE_KEEP,
    ICP_COARSE_VISIBLE_MAX_POINTS,
    ENABLE_ALIGNMENT_RENDER_OUTPUTS,
    ICP_FINAL_ITERATIONS,
    ICP_FINAL_VISIBLE_MAX_POINTS,
    ICP_FINE_VISIBLE_MAX_POINTS,
    ICP_GPU_CDIST_CHUNK_SIZE,
    ICP_IGNORE_INVERTED_SOLUTIONS,
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
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from object_alignment_common import (
    annotate_rendered_image,
    annotate_rendered_model_front_view,
    build_depth_border_keep_mask,
    build_depth_pointcloud_from_valid_mask,
    compute_front_view_extents,
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
    render_front_view_points,
    resolve_blender_path,
    resolve_task_paths,
    rotation_unity_to_blender_world,
    task_prefix,
    unity_to_blender_world_vector,
    write_binary_ply,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


HELPER_SCRIPT = Path(__file__).resolve().with_name("blender_render_measure.py")
MODEL_UP_AXIS_UNITY = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_ICP_BACKEND: dict | None = None


def resolve_icp_backend() -> dict:
    global _ICP_BACKEND
    if _ICP_BACKEND is not None:
        return _ICP_BACKEND

    requested = str(ICP_ACCELERATION_DEVICE).strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("config.ICP_ACCELERATION_DEVICE must be one of: auto / cuda / cpu")

    if requested == "cpu":
        _ICP_BACKEND = {"requested": requested, "actual": "cpu", "reason": "forced-by-config", "torch_device": None}
        return _ICP_BACKEND

    if torch is None:
        if requested == "cuda":
            raise RuntimeError("config.ICP_ACCELERATION_DEVICE='cuda' but PyTorch is not installed")
        _ICP_BACKEND = {"requested": requested, "actual": "cpu", "reason": "torch-not-installed", "torch_device": None}
        return _ICP_BACKEND

    if torch.cuda.is_available():
        _ICP_BACKEND = {
            "requested": requested,
            "actual": "cuda",
            "reason": "cuda-available",
            "torch_device": torch.device("cuda"),
        }
        return _ICP_BACKEND

    if requested == "cuda":
        raise RuntimeError("config.ICP_ACCELERATION_DEVICE='cuda' but CUDA is not available")

    _ICP_BACKEND = {"requested": requested, "actual": "cpu", "reason": "cuda-unavailable", "torch_device": None}
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

    backend = resolve_icp_backend()
    if backend["actual"] != "cuda":
        tree = cKDTree(reference_used)
        dists, idx = tree.query(query_used, k=1)
        dists = np.asarray(dists, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int32)
        return (dists, idx) if return_indices else dists

    device = backend["torch_device"]
    query_tensor = torch.as_tensor(query_used, dtype=torch.float32, device=device)
    reference_tensor = torch.as_tensor(reference_used, dtype=torch.float32, device=device)

    chunk_size = max(1, int(ICP_GPU_CDIST_CHUNK_SIZE))
    dist_chunks: list[np.ndarray] = []
    idx_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(query_tensor), chunk_size):
            end = min(start + chunk_size, len(query_tensor))
            distances = torch.cdist(query_tensor[start:end], reference_tensor)
            chunk_dists, chunk_idx = torch.min(distances, dim=1)
            dist_chunks.append(chunk_dists.cpu().numpy().astype(np.float32, copy=False))
            if return_indices:
                idx_chunks.append(chunk_idx.cpu().numpy().astype(np.int32, copy=False))

    all_dists = np.concatenate(dist_chunks) if dist_chunks else np.empty(0, dtype=np.float32)
    if not return_indices:
        return all_dists
    all_idx = np.concatenate(idx_chunks) if idx_chunks else np.empty(0, dtype=np.int32)
    return all_dists, all_idx


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


def select_front_visible_points(
    points: np.ndarray,
    bins: int = 128,
    max_points: int | None = None,
    seed: int = 0,
    ignore_occluded_points: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points, np.empty(0, dtype=np.int32)

    if ignore_occluded_points is None:
        ignore_occluded_points = bool(ICP_IGNORE_OCCLUDED_MODEL_POINTS)

    if ignore_occluded_points:
        # Despite the legacy name, this extracts the camera-visible surface when
        # the camera looks along +Z in Unity camera-local space.
        min_xy = points[:, :2].min(axis=0)
        max_xy = points[:, :2].max(axis=0)
        span_xy = np.maximum(max_xy - min_xy, 1e-6)
        uv = np.floor((points[:, :2] - min_xy) / span_xy * (bins - 1)).astype(np.int32)
        flat = uv[:, 1] * bins + uv[:, 0]
        order = np.lexsort((points[:, 2], flat))
        flat_sorted = flat[order]

        keep = np.empty(len(order), dtype=bool)
        keep[0] = True
        keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
        selected_indices = order[keep]
    else:
        selected_indices = np.arange(len(points), dtype=np.int32)

    if max_points is not None and len(selected_indices) > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(selected_indices), size=max_points, replace=False)
        selected_indices = selected_indices[pick]

    return points[selected_indices], selected_indices.astype(np.int32, copy=False)


def extract_front_visible_points(
    points: np.ndarray,
    bins: int = 128,
    max_points: int | None = None,
    seed: int = 0,
    ignore_occluded_points: bool | None = None,
) -> np.ndarray:
    selected_points, _ = select_front_visible_points(
        points,
        bins=bins,
        max_points=max_points,
        seed=seed,
        ignore_occluded_points=ignore_occluded_points,
    )
    return selected_points


def trimmed_rmse(dists: np.ndarray, trim_percentile: float = 85.0) -> float:
    dists = np.asarray(dists, dtype=np.float32).reshape(-1)
    if dists.size == 0:
        return float("inf")
    cutoff = float(np.percentile(dists, trim_percentile))
    keep = dists <= max(cutoff, 1e-6)
    if not np.any(keep):
        keep = np.ones_like(dists, dtype=bool)
    return float(np.sqrt(np.mean(np.square(dists[keep]))))


def safe_matrix_to_euler_xyz_deg(rotation: np.ndarray) -> list[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_euler("xyz", degrees=True)
    return [float(v) for v in euler]


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


def best_fit_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    backend = resolve_icp_backend()
    if backend["actual"] != "cuda":
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

    device = backend["torch_device"]
    with torch.no_grad():
        a_tensor = torch.as_tensor(a, dtype=torch.float32, device=device)
        b_tensor = torch.as_tensor(b, dtype=torch.float32, device=device)
        centroid_a = a_tensor.mean(dim=0)
        centroid_b = b_tensor.mean(dim=0)
        aa = a_tensor - centroid_a
        bb = b_tensor - centroid_b
        h = aa.transpose(0, 1) @ bb
        u, _, vh = torch.linalg.svd(h, full_matrices=False)
        r = vh.transpose(0, 1) @ u.transpose(0, 1)
        if torch.linalg.det(r) < 0:
            vh = vh.clone()
            vh[-1, :] *= -1
            r = vh.transpose(0, 1) @ u.transpose(0, 1)
        t = centroid_b - (centroid_a @ r.transpose(0, 1))
    return (
        r.cpu().numpy().astype(np.float32, copy=False),
        t.cpu().numpy().astype(np.float32, copy=False),
    )


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
) -> np.ndarray:
    model_extents = compute_front_view_extents(model_full_points)
    target_extents = compute_front_view_extents(target_front_points)

    model_center_x, model_center_y = compute_xy_center(model_extents)
    target_center_x, target_center_y = compute_xy_center(target_extents)
    front_anchor_z = float(np.median(model_front_points[:, 2]))
    target_anchor_z = float(np.median(target_front_points[:, 2]))

    return np.array(
        [
            target_center_x - model_center_x,
            target_center_y - model_center_y,
            target_anchor_z - front_anchor_z,
        ],
        dtype=np.float32,
    )


def compose_transform(
    base_rotation: np.ndarray,
    base_translation: np.ndarray,
    delta_rotation: np.ndarray,
    delta_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    final_rotation = delta_rotation @ base_rotation
    final_translation = base_translation @ delta_rotation.T + delta_translation
    return final_rotation.astype(np.float32), final_translation.astype(np.float32)


def evaluate_alignment(
    transformed_full_points: np.ndarray,
    transformed_front_points: np.ndarray,
    target_front_points: np.ndarray,
    scale: float,
    nominal_scale: float,
) -> dict:
    dists_target_to_model_3d = nearest_neighbor_distances(target_front_points, transformed_front_points, dims=3)
    dists_model_to_target_3d = nearest_neighbor_distances(transformed_front_points, target_front_points, dims=3)
    dists_target_to_model_2d = nearest_neighbor_distances(target_front_points, transformed_front_points, dims=2)
    dists_model_to_target_2d = nearest_neighbor_distances(transformed_front_points, target_front_points, dims=2)

    extents = compute_front_view_extents(transformed_front_points)
    target_extents = compute_front_view_extents(target_front_points)
    width_error = abs(extents["width_units"] - target_extents["width_units"])
    height_error = abs(extents["height_units"] - target_extents["height_units"])
    size_error = width_error + height_error
    scale_error = abs(scale - nominal_scale) / max(nominal_scale, 1e-6)
    center_error = float(np.linalg.norm(compute_xy_center(extents) - compute_xy_center(target_extents)))

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
) -> dict:
    best_candidate: dict | None = None
    best_score = float("inf")
    best_metrics: dict | None = None

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
                )
                result = run_icp(
                    model_points=model_points,
                    target_points=target_front_points,
                    scale=float(scale),
                    init_rotation=init_rotation,
                    init_translation=init_translation,
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
                )
                score = float(metrics["score"])
                if score < best_score:
                    best_score = score
                    best_metrics = metrics
                    best_candidate = {
                        "rotation": rotation,
                        "translation": translation,
                        "delta_euler_deg": [float(rx), float(ry), float(rz)],
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
                    "euler_deg": safe_matrix_to_euler_xyz_deg(rotation),
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
) -> dict:
    rotated_full = transform_points(model_points, 1.0, rotation, np.zeros(3, dtype=np.float32))
    rotated_front = extract_front_visible_points(
        rotated_full,
        bins=visible_bins,
        max_points=visible_max_points,
        seed=11,
    )
    scale_estimate = estimate_scale_from_visible_extents(
        rotated_front,
        target_front_points,
        fallback_scale=nominal_scale,
    )
    if scale_candidates is None:
        scale_candidates = build_scale_candidates(
            estimated_scale=scale_estimate["estimated_scale"],
            width_scale=scale_estimate["width_scale"],
            height_scale=scale_estimate["height_scale"],
            fallback_scale=nominal_scale,
        )

    best_candidate: dict | None = None
    for scale in scale_candidates:
        scaled_full = transform_points(model_points, float(scale), rotation, np.zeros(3, dtype=np.float32))
        scaled_front = extract_front_visible_points(
            scaled_full,
            bins=visible_bins,
            max_points=visible_max_points,
            seed=13,
        )
        translation = build_initial_translation(
            model_full_points=scaled_full,
            model_front_points=scaled_front,
            target_front_points=target_front_points,
        )
        translated_full = scaled_full + translation
        translated_front = scaled_front + translation
        metrics = evaluate_alignment(
            transformed_full_points=translated_full,
            transformed_front_points=translated_front,
            target_front_points=target_front_points,
            scale=float(scale),
            nominal_scale=nominal_scale,
        )
        candidate = {
            "rotation": rotation.astype(np.float32),
            "translation": translation.astype(np.float32),
            "scale": float(scale),
            "metrics": metrics,
            "scale_estimate": scale_estimate,
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
) -> list[dict]:
    candidates: list[dict] = []

    def append_candidate(
        rotation: np.ndarray,
        seed_euler_deg: list[float],
        local_euler_deg: list[float],
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
        )
        candidate["seed_euler_deg"] = [float(v) for v in seed_euler_deg]
        candidate["local_euler_deg"] = [float(v) for v in local_euler_deg]
        candidate["euler_deg"] = safe_matrix_to_euler_xyz_deg(rotation)
        candidate["stage"] = stage
        candidates.append(candidate)

    for seed in generate_axis_aligned_rotation_seeds():
        append_candidate(
            rotation=seed["rotation"],
            seed_euler_deg=seed["euler_deg"],
            local_euler_deg=[0.0, 0.0, 0.0],
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
                        seed_euler_deg=base["seed_euler_deg"],
                        local_euler_deg=[float(rx), float(ry), float(rz)],
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
                        seed_euler_deg=base["seed_euler_deg"],
                        local_euler_deg=[
                            float(base["local_euler_deg"][0] + rx),
                            float(base["local_euler_deg"][1] + ry),
                            float(base["local_euler_deg"][2] + rz),
                        ],
                        stage="refine_fine",
                        visible_bins=160,
                        visible_max_points=ICP_FINE_VISIBLE_MAX_POINTS,
                    )

    candidates.sort(key=lambda item: item["metrics"]["score"])
    if not candidates:
        raise RuntimeError("No valid pose candidates remain after applying ICP constraints.")
    return candidates


def search_initial_pose(
    model_points: np.ndarray,
    target_front_points: np.ndarray,
    nominal_scale: float,
) -> dict:
    return search_initial_pose_candidates(
        model_points=model_points,
        target_front_points=target_front_points,
        nominal_scale=nominal_scale,
    )[0]


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


def render_aligned_model_image(
    blender_path: Path | None,
    mesh_path: Path,
    render_path: Path,
    blender_translation: np.ndarray,
    blender_euler_deg: np.ndarray,
    scale: float,
    transformed_model_unity: np.ndarray,
    task: dict,
) -> str:
    depthpointcloud = task.get("depthpointcloud") or {}
    model = task.get("model") or {}
    object_alignment = task.get("object_alignment") or {}

    info_lines = [
        "Basis: orthographic front view",
        "Model import in Blender: forward=-X, up=+Z",
        f"Model width  : {float(model.get('width_measured') or 0.0) * scale:.4f} m",
        f"Model height : {float(model.get('height_measured') or 0.0) * scale:.4f} m",
        f"Point depth   : {float(depthpointcloud.get('mean_depth_measured') or 0.0):.4f} m",
        f"ICP rmse      : {float(object_alignment.get('icp_rmse') or 0.0):.4f} m",
        f"Confidence    : {float(object_alignment.get('confidence') or 0.0):.3f}",
    ]

    if blender_path is None:
        render_front_view_points(
            transformed_model_unity,
            render_path,
            title="Aligned Model Front View",
            info_lines=info_lines + ["Render: software fallback"],
            point_color=(210, 210, 210),
        )
        return "software-fallback"

    command = [
        str(blender_path),
        "--background",
        "--python",
        str(HELPER_SCRIPT),
        "--",
        "front_model",
        str(mesh_path),
        str(render_path),
        *[f"{float(v):.9f}" for v in blender_translation],
        *[f"{float(v):.9f}" for v in blender_euler_deg],
        f"{float(scale):.9f}",
    ]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Blender aligned-model render failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    annotate_rendered_model_front_view(render_path, "Aligned Model Front View", info_lines)
    return str(blender_path)


def render_overlay_preview_image(
    blender_path: Path | None,
    mesh_path: Path,
    discarded_pointcloud_path: Path,
    icp_pointcloud_path: Path,
    render_path: Path,
    blender_translation: np.ndarray,
    blender_delta_euler_deg: np.ndarray,
    scale: float,
    task: dict,
) -> str:
    depthpointcloud = task.get("depthpointcloud") or {}
    object_alignment = task.get("object_alignment") or {}

    info_lines = [
        "Perspective preview: point cloud + aligned model",
        "Point cloud import: forward=-X, up=+Y",
        "Model import: forward=-X, up=+Z",
        "Orange = mask-border-discarded points, green = ICP-used points, blue = aligned model",
        f"Depth mean    : {float(depthpointcloud.get('mean_depth_measured') or 0.0):.4f} m",
        f"ICP rmse      : {float(object_alignment.get('icp_rmse') or 0.0):.4f} m",
        f"Confidence    : {float(object_alignment.get('confidence') or 0.0):.3f}",
    ]

    if blender_path is None:
        raise FileNotFoundError("Blender is required for perspective overlay preview.")

    command = [
        str(blender_path),
        "--background",
        "--python",
        str(HELPER_SCRIPT),
        "--",
        "overlay_preview",
        str(mesh_path),
        str(discarded_pointcloud_path),
        str(icp_pointcloud_path),
        str(render_path),
        *[f"{float(v):.9f}" for v in blender_translation],
        *[f"{float(v):.9f}" for v in blender_delta_euler_deg],
        f"{float(scale):.9f}",
    ]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Blender overlay preview render failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    annotate_rendered_image(render_path, "Alignment Perspective Preview", info_lines)
    return str(blender_path)


def remove_legacy_outputs(prefix: str) -> None:
    for name in (
        f"{prefix}_size_compare.png",
        f"{prefix}_model_front.png",
        f"{prefix}_model_front_tmp.png",
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
    print(f"[STAGE] Object alignment start : {json_path}")
    icp_backend = resolve_icp_backend()
    print(
        f"[STAGE] ICP acceleration     : requested={icp_backend['requested']}, "
        f"actual={icp_backend['actual']}, reason={icp_backend['reason']}"
    )

    if "depthpointcloud" not in task:
        raise ValueError("depthpointcloud is missing. Run pointcloud stage first.")
    if "model" not in task:
        raise ValueError("model is missing. Run model scale stage first.")

    paths = resolve_task_paths(task)
    prefix = task_prefix(task, json_path)
    pointcloud_name = (task.get("depthpointcloud") or {}).get("pointcloud_name")
    if not pointcloud_name:
        raise ValueError("depthpointcloud.pointcloud_name is missing")

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    mask_bool = read_mask(paths["mask_path"])
    depth_mm = read_depth_image(paths["depth_path"])
    print("[STAGE] Preparing point cloud and mesh inputs")
    valid_all = mask_bool & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    depth_keep_mask = build_depth_border_keep_mask(mask_bool)
    valid_cropped = valid_all & ~depth_keep_mask
    discarded_points_export, _ = build_depth_pointcloud_from_valid_mask(depth_mm, valid_cropped, k)

    pointcloud_path = object_alignment_output_path(pointcloud_name)
    pointcloud_points_export = read_binary_ply_points(pointcloud_path)
    pointcloud_points_unity = pointcloud_export_to_unity(pointcloud_points_export)
    model_vertices_raw = read_obj_vertices(paths["mesh_path"])
    model_vertices_unity = obj_vertices_to_unity(model_vertices_raw)
    target_front_fit, target_front_indices = select_front_visible_points(
        pointcloud_points_unity,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=7,
    )
    icp_used_points_export = pointcloud_points_export[target_front_indices]
    print(
        f"[STAGE] Front visible points  : source={len(pointcloud_points_export)}, "
        f"selected={len(target_front_fit)}, max_selected={ICP_TARGET_FRONT_MAX_POINTS}"
    )

    overall_scale = float((task.get("model") or {}).get("overall_scale") or 1.0)

    print("[STAGE] Running coarse pose search")
    coarse_candidates = search_initial_pose_candidates(
        model_points=model_vertices_unity,
        target_front_points=target_front_fit,
        nominal_scale=overall_scale,
    )
    coarse_candidates = coarse_candidates[:ICP_COARSE_CANDIDATE_KEEP]
    print(
        f"[STAGE] Coarse candidates     : total={len(coarse_candidates)}, "
        f"keep={ICP_COARSE_CANDIDATE_KEEP}"
    )

    best_result: dict | None = None
    best_debug: dict | None = None
    for coarse_rank, best_coarse in enumerate(coarse_candidates, start=1):
        print(
            f"[STAGE] ICP candidate        : rank={coarse_rank}, "
            f"search_stage={best_coarse['stage']}, seed={best_coarse['seed_euler_deg']}"
        )
        candidate_scales = build_scale_candidates(
            estimated_scale=float(best_coarse["scale_estimate"]["estimated_scale"]),
            width_scale=float(best_coarse["scale_estimate"]["width_scale"]),
            height_scale=float(best_coarse["scale_estimate"]["height_scale"]),
            fallback_scale=overall_scale,
        )
        for scale in candidate_scales:
            rotation_seed = best_coarse["rotation"]
            coarse_full = transform_points(model_vertices_unity, float(scale), rotation_seed, np.zeros(3, dtype=np.float32))
            coarse_front = extract_front_visible_points(
                coarse_full,
                bins=180,
                max_points=ICP_COARSE_VISIBLE_MAX_POINTS,
                seed=19,
            )
            coarse_translation = build_initial_translation(
                model_full_points=coarse_full,
                model_front_points=coarse_front,
                target_front_points=target_front_fit,
            )
            coarse_full = coarse_full + coarse_translation
            coarse_front = coarse_front + coarse_translation

            delta_result = run_icp(
                model_points=model_vertices_unity,
                target_points=target_front_fit,
                scale=float(scale),
                init_rotation=rotation_seed,
                init_translation=coarse_translation,
                iterations=ICP_FINAL_ITERATIONS,
                visible_bins=180,
                visible_max_points=ICP_FINAL_VISIBLE_MAX_POINTS,
            )
            final_rotation = delta_result["rotation"]
            final_translation = delta_result["translation"]
            delta_rotation = final_rotation @ rotation_seed.T
            delta_translation = final_translation - (coarse_translation @ delta_rotation.T)

            if should_reject_inverted_solution(final_rotation):
                continue

            transformed_full = transform_points(model_vertices_unity, float(scale), final_rotation, final_translation)
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
                scale=float(scale),
                nominal_scale=overall_scale,
            )

            candidate = {
                "scale": float(scale),
                "rotation": final_rotation,
                "translation": final_translation,
                "rmse": metrics["rmse_3d"],
                "surface_rmse_3d": metrics["surface_rmse_3d"],
                "coverage_rmse_3d": metrics["coverage_rmse_3d"],
                "rmse_2d": metrics["rmse_2d"],
                "score": metrics["score"],
                "size_error": metrics["size_error"],
                "inlier_ratio": float(delta_result["inlier_ratio"]),
                "extents": metrics["extents"],
                "width_error": metrics["width_error"],
                "height_error": metrics["height_error"],
            }
            if best_result is None or candidate["score"] < best_result["score"]:
                best_result = candidate
                best_debug = {
                    "coarse_search": {
                        "candidate_rank": int(coarse_rank),
                        "scale": float(scale),
                        "up_y": model_up_y_in_unity(rotation_seed),
                        "seed_euler_deg": [float(v) for v in best_coarse["seed_euler_deg"]],
                        "local_refine_euler_deg": [float(v) for v in best_coarse["local_euler_deg"]],
                        "search_stage": str(best_coarse["stage"]),
                        "pose": serialize_pose(
                            rotation_seed,
                            coarse_translation,
                            "unity_camera_local_x_right_y_up_z_forward",
                        ),
                        "score": float(best_coarse["metrics"]["score"]),
                        "rmse_3d": float(best_coarse["metrics"]["rmse_3d"]),
                        "surface_rmse_3d": float(best_coarse["metrics"]["surface_rmse_3d"]),
                        "coverage_rmse_3d": float(best_coarse["metrics"]["coverage_rmse_3d"]),
                        "rmse_2d": float(best_coarse["metrics"]["rmse_2d"]),
                        "size_error": float(best_coarse["metrics"]["size_error"]),
                        "center_error": float(best_coarse["metrics"]["center_error"]),
                        "estimated_scale": float(best_coarse["scale_estimate"]["estimated_scale"]),
                        "width_scale": float(best_coarse["scale_estimate"]["width_scale"]),
                        "height_scale": float(best_coarse["scale_estimate"]["height_scale"]),
                    },
                    "icp_delta": {
                        "scale": 1.0,
                        "pose": serialize_pose(
                            delta_rotation,
                            delta_translation,
                            "unity_camera_local_delta_x_right_y_up_z_forward",
                        ),
                        "rmse": float(delta_result["rmse"]),
                        "inlier_ratio": float(delta_result["inlier_ratio"]),
                    },
                    "final_camera_local_unity": {
                        "scale": float(scale),
                        "up_y": model_up_y_in_unity(final_rotation),
                        "pose": serialize_pose(
                            final_rotation,
                            final_translation,
                            "unity_camera_local_x_right_y_up_z_forward",
                        ),
                        "front_view_rmse_2d": float(metrics["rmse_2d"]),
                        "surface_rmse_3d": float(metrics["surface_rmse_3d"]),
                        "coverage_rmse_3d": float(metrics["coverage_rmse_3d"]),
                        "front_view_size_error": float(metrics["size_error"]),
                        "width_error": float(metrics["width_error"]),
                        "height_error": float(metrics["height_error"]),
                        "center_error": float(metrics["center_error"]),
                    },
                }

    if best_result is None:
        raise RuntimeError("Failed to compute object alignment")
    if best_debug is None:
        raise RuntimeError("Failed to collect object alignment debug data")

    best = best_result
    print("[STAGE] Running local pose refinement")
    refined = refine_pose_locally(
        model_points=model_vertices_unity,
        target_front_points=target_front_fit,
        scale=float(best["scale"]),
        nominal_scale=overall_scale,
        base_rotation=best["rotation"],
        base_translation=best["translation"],
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
            },
        }
        best_debug["local_rotation_refine"] = {
            "delta_euler_deg": [float(v) for v in refined["delta_euler_deg"]],
            "center_error": float(refined["center_error"]),
            "score": float(refined["score"]),
            "surface_rmse_3d": float(refined["surface_rmse_3d"]),
            "coverage_rmse_3d": float(refined["coverage_rmse_3d"]),
            "front_view_rmse_2d": float(refined["rmse_2d"]),
            "up_y": model_up_y_in_unity(refined["rotation"]),
        }
        best_debug["final_camera_local_unity"] = {
            "scale": float(best["scale"]),
            "up_y": model_up_y_in_unity(best["rotation"]),
            "pose": serialize_pose(
                best["rotation"],
                best["translation"],
                "unity_camera_local_x_right_y_up_z_forward",
            ),
            "front_view_rmse_2d": float(best["rmse_2d"]),
            "surface_rmse_3d": float(best["surface_rmse_3d"]),
            "coverage_rmse_3d": float(best["coverage_rmse_3d"]),
            "front_view_size_error": float(best["size_error"]),
            "width_error": float(best["width_error"]),
            "height_error": float(best["height_error"]),
            "center_error": float(refined["center_error"]),
        }

    pointcloud_rotation, pointcloud_translation = model_pose_unity_to_pointcloud_input(
        best["rotation"],
        best["translation"],
    )
    pointcloud_euler_deg = Rotation.from_matrix(pointcloud_rotation).as_euler("xyz", degrees=True)
    pointcloud_quat_xyzw = Rotation.from_matrix(pointcloud_rotation).as_quat()

    blender_rotation = rotation_unity_to_blender_world(best["rotation"])
    blender_translation = unity_to_blender_world_vector(best["translation"])
    blender_delta_euler_deg = np.asarray(
        safe_matrix_to_blender_euler_xyz_deg(blender_rotation),
        dtype=np.float32,
    )

    confidence = compute_confidence(task, best)
    aligned_model_image_name = f"{prefix}_aligned_model_front.png"
    aligned_model_image_path = object_alignment_output_path(aligned_model_image_name)
    overlay_preview_name = f"{prefix}_alignment_preview_perspective.png"
    overlay_preview_path = object_alignment_output_path(overlay_preview_name)
    discarded_points_preview_name = f"{prefix}_icp_discarded_points.ply"
    discarded_points_preview_path = object_alignment_output_path(discarded_points_preview_name)
    icp_points_preview_name = f"{prefix}_icp_used_points.ply"
    icp_points_preview_path = object_alignment_output_path(icp_points_preview_name)
    write_binary_ply(
        discarded_points_preview_path,
        discarded_points_export,
    )
    write_binary_ply(
        icp_points_preview_path,
        icp_used_points_export,
    )

    front_view_image_name: str | None = aligned_model_image_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None
    preview_image_name: str | None = overlay_preview_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None
    if not ENABLE_ALIGNMENT_RENDER_OUTPUTS:
        if aligned_model_image_path.exists():
            aligned_model_image_path.unlink()
        if overlay_preview_path.exists():
            overlay_preview_path.unlink()

    object_alignment = {
        "coordinate_basis": "pointcloud_input_pre_blender_import",
        "translation_axes_relative_to_unity": {
            "x": "-Z",
            "y": "-Y",
            "z": "-X",
        },
        "rotation_axes_relative_to_unity": {
            "x": "-Z",
            "y": "+Y",
            "z": "-X",
        },
        "model_position": [float(v) for v in pointcloud_translation],
        "model_rotation_euler_deg": [float(v) for v in pointcloud_euler_deg],
        "model_rotation_quaternion_xyzw": [float(v) for v in pointcloud_quat_xyzw],
        "model_real_scale": float(best["scale"]),
        "front_view_image_name": front_view_image_name,
        "preview_image_name": preview_image_name,
        "icp_discarded_pointcloud_name": discarded_points_preview_name,
        "icp_used_pointcloud_name": icp_points_preview_name,
        "discarded_count": int(len(discarded_points_export)),
        "used_count": int(len(icp_used_points_export)),
        "confidence": confidence,
        "icp_rmse": float(best["rmse"]),
        "surface_rmse_3d": float(best.get("surface_rmse_3d") or 0.0),
        "coverage_rmse_3d": float(best.get("coverage_rmse_3d") or 0.0),
        "front_view_rmse_2d": float(best.get("rmse_2d") or 0.0),
        "front_view_size_error": float(best.get("size_error") or 0.0),
        "icp_inlier_ratio": float(best["inlier_ratio"]),
        "icp_constraints": {
            "acceleration_device_requested": str(icp_backend["requested"]),
            "acceleration_device_actual": str(icp_backend["actual"]),
            "acceleration_reason": str(icp_backend["reason"]),
            "ignore_inverted_solutions": bool(ICP_IGNORE_INVERTED_SOLUTIONS),
            "ignore_occluded_model_points": bool(ICP_IGNORE_OCCLUDED_MODEL_POINTS),
            "target_front_max_points": int(ICP_TARGET_FRONT_MAX_POINTS),
            "coarse_visible_max_points": int(ICP_COARSE_VISIBLE_MAX_POINTS),
            "medium_visible_max_points": int(ICP_MEDIUM_VISIBLE_MAX_POINTS),
            "fine_visible_max_points": int(ICP_FINE_VISIBLE_MAX_POINTS),
            "local_refine_visible_max_points": int(ICP_LOCAL_REFINE_VISIBLE_MAX_POINTS),
            "final_visible_max_points": int(ICP_FINAL_VISIBLE_MAX_POINTS),
            "final_iterations": int(ICP_FINAL_ITERATIONS),
            "local_refine_iterations": int(ICP_LOCAL_REFINE_ITERATIONS),
            "coarse_candidate_keep": int(ICP_COARSE_CANDIDATE_KEEP),
            "axis_seed_retain_topk": int(ICP_AXIS_SEED_RETAIN_TOPK),
            "medium_retain_topk": int(ICP_MEDIUM_RETAIN_TOPK),
        },
    }
    task["object_alignment"] = object_alignment

    debug_section = dict(task.get("debug") or {})
    pose_debug = dict(debug_section.get("pose_transform_stages") or {})
    pose_debug["object_alignment"] = {
        "coordinate_notes": {
            "unity_camera_local": "model pose in PVCamera local space, Unity basis (X right, Y up, Z forward)",
            "pointcloud_input_pre_blender_import": "legacy/export basis currently consumed by pose stage",
        },
        "coarse_search": best_debug["coarse_search"],
        "icp_delta": best_debug["icp_delta"],
        "final_camera_local_unity": best_debug["final_camera_local_unity"],
        "final_camera_local_pointcloud_input": {
            "scale": float(best["scale"]),
            "pose": serialize_pose(
                pointcloud_rotation,
                pointcloud_translation,
                "pointcloud_input_pre_blender_import",
            ),
        },
    }
    debug_section["pose_transform_stages"] = pose_debug
    task["debug"] = debug_section

    blender_label = "disabled-by-config"
    if ENABLE_ALIGNMENT_RENDER_OUTPUTS:
        blender_label = "software-fallback"
        try:
            blender_path = resolve_blender_path(argv[2] if len(argv) == 3 else None)
            blender_label = render_aligned_model_image(
                blender_path=blender_path,
                mesh_path=paths["mesh_path"],
                render_path=aligned_model_image_path,
                blender_translation=blender_translation,
                blender_euler_deg=blender_delta_euler_deg,
                scale=float(best["scale"]),
                transformed_model_unity=transform_points(model_vertices_unity, best["scale"], best["rotation"], best["translation"]),
                task=task,
            )
            render_overlay_preview_image(
                blender_path=blender_path,
                mesh_path=paths["mesh_path"],
                discarded_pointcloud_path=discarded_points_preview_path,
                icp_pointcloud_path=icp_points_preview_path,
                render_path=overlay_preview_path,
                blender_translation=blender_translation,
                blender_delta_euler_deg=blender_delta_euler_deg,
                scale=float(best["scale"]),
                task=task,
            )
        except FileNotFoundError:
            blender_label = render_aligned_model_image(
                blender_path=None,
                mesh_path=paths["mesh_path"],
                render_path=aligned_model_image_path,
                blender_translation=blender_translation,
                blender_euler_deg=blender_delta_euler_deg,
                scale=float(best["scale"]),
                transformed_model_unity=transform_points(model_vertices_unity, best["scale"], best["rotation"], best["translation"]),
                task=task,
            )
            object_alignment["preview_image_name"] = None

    save_task_json(json_path, task)
    remove_legacy_outputs(prefix)

    print(f"[INFO] JSON            : {json_path}")
    print(f"[INFO] Pointcloud      : {pointcloud_path}")
    print(f"[INFO] Mesh            : {paths['mesh_path']}")
    print(f"[INFO] Render          : {aligned_model_image_path if object_alignment.get('front_view_image_name') else 'not-generated'}")
    print(f"[INFO] Preview         : {overlay_preview_path if object_alignment.get('preview_image_name') else 'not-generated'}")
    print(f"[INFO] Blender         : {blender_label}")
    print(
        f"[INFO] ICP backend     : requested={icp_backend['requested']}, "
        f"actual={icp_backend['actual']}, reason={icp_backend['reason']}"
    )
    print(
        f"[INFO] Point usage      : source={len(pointcloud_points_export)}, "
        f"selected={len(target_front_fit)}, discarded={len(discarded_points_export)}"
    )
    print(
        f"[INFO] Output pose     : "
        f"pos={object_alignment['model_position']}, "
        f"rot={object_alignment['model_rotation_euler_deg']}"
    )
    print(
        f"[INFO] Final scale     : {object_alignment['model_real_scale']:.6f}, "
        f"confidence={confidence:.3f}, icp_iter={ICP_FINAL_ITERATIONS}"
    )
    print("[OK] Object alignment stage completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
