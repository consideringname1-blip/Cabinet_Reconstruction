from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import CODE_ROOT
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from object_alignment_common import (
    annotate_rendered_image,
    annotate_rendered_model_front_view,
    compute_front_view_extents,
    model_pose_unity_to_pointcloud_input,
    obj_vertices_to_unity,
    object_alignment_output_path,
    pointcloud_export_to_unity,
    read_binary_ply_points,
    read_obj_vertices,
    render_front_view_points,
    resolve_blender_path,
    resolve_task_paths,
    rotation_unity_to_blender_world,
    task_prefix,
    unity_to_blender_world_vector,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


HELPER_SCRIPT = Path(__file__).resolve().with_name("blender_render_measure.py")


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


def extract_front_visible_points(
    points: np.ndarray,
    bins: int = 128,
    max_points: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points

    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    span_xy = np.maximum(max_xy - min_xy, 1e-6)
    uv = np.floor((points[:, :2] - min_xy) / span_xy * (bins - 1)).astype(np.int32)
    flat = uv[:, 1] * bins + uv[:, 0]
    order = np.lexsort((points[:, 2], flat))
    flat_sorted = flat[order]

    keep = np.empty(len(order), dtype=bool)
    keep[:-1] = flat_sorted[:-1] != flat_sorted[1:]
    keep[-1] = True
    selected = points[order[keep]]

    if max_points is not None:
        selected = downsample_points(selected, max_points=max_points, seed=seed)
    return selected


def best_fit_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
) -> dict:
    rotation = init_rotation.astype(np.float32).copy()
    translation = init_translation.astype(np.float32).copy()
    rmse = float("inf")
    inlier_ratio = 0.0

    for _ in range(iterations):
        transformed = transform_points(model_points, scale, rotation, translation)
        tree = cKDTree(transformed)
        dists, idx = tree.query(target_points, k=1)

        threshold = float(np.percentile(dists, 80))
        if threshold <= 0:
            threshold = float(dists.max()) if len(dists) else 0.0
        keep = dists <= max(threshold, 1e-4)
        if keep.sum() < 16:
            break

        matched_model = transformed[idx[keep]]
        matched_target = target_points[keep]

        delta_r, delta_t = best_fit_transform(matched_model, matched_target)
        rotation = delta_r @ rotation
        translation = delta_r @ translation + delta_t
        rmse = float(np.sqrt(np.mean((matched_model - matched_target) ** 2)))
        inlier_ratio = float(keep.mean())

    transformed = transform_points(model_points, scale, rotation, translation)
    tree = cKDTree(transformed)
    dists, _ = tree.query(target_points, k=1)
    rmse = float(np.sqrt(np.mean(dists ** 2)))

    return {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "rmse": rmse,
        "inlier_ratio": inlier_ratio,
    }


def build_initial_translation(
    model_full_points: np.ndarray,
    model_front_points: np.ndarray,
    target_front_points: np.ndarray,
    mean_depth: float,
) -> np.ndarray:
    model_extents = compute_front_view_extents(model_full_points)
    target_extents = compute_front_view_extents(target_front_points)

    model_center_x = 0.5 * (model_extents["bbox_min"][0] + model_extents["bbox_max"][0])
    model_center_y = 0.5 * (model_extents["bbox_min"][1] + model_extents["bbox_max"][1])
    target_center_x = 0.5 * (target_extents["bbox_min"][0] + target_extents["bbox_max"][0])
    target_center_y = 0.5 * (target_extents["bbox_min"][1] + target_extents["bbox_max"][1])
    front_anchor_z = float(model_front_points[:, 2].mean())

    return np.array(
        [
            target_center_x - model_center_x,
            target_center_y - model_center_y,
            mean_depth - front_anchor_z,
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
    real_width: float,
    real_height: float,
    scale: float,
    nominal_scale: float,
) -> dict:
    tree_3d = cKDTree(transformed_front_points)
    dists_3d, _ = tree_3d.query(target_front_points, k=1)

    tree_2d = cKDTree(transformed_front_points[:, :2])
    dists_2d, _ = tree_2d.query(target_front_points[:, :2], k=1)

    extents = compute_front_view_extents(transformed_full_points)
    width_error = abs(extents["width_units"] - real_width)
    height_error = abs(extents["height_units"] - real_height)
    size_error = width_error + height_error
    scale_error = abs(scale - nominal_scale) / max(nominal_scale, 1e-6)

    rmse_3d = float(np.sqrt(np.mean(np.square(dists_3d))))
    rmse_2d = float(np.sqrt(np.mean(np.square(dists_2d))))
    score = rmse_3d + 0.35 * rmse_2d + 0.45 * size_error + 0.10 * scale_error
    return {
        "score": float(score),
        "rmse_3d": rmse_3d,
        "rmse_2d": rmse_2d,
        "size_error": float(size_error),
        "width_error": float(width_error),
        "height_error": float(height_error),
        "extents": extents,
    }


def search_initial_pose(
    model_points: np.ndarray,
    target_front_points: np.ndarray,
    nominal_scale: float,
    mean_depth: float,
    real_width: float,
    real_height: float,
) -> dict:
    candidates: list[dict] = []

    def try_grid(center: tuple[float, float, float], step: int, radius: int) -> None:
        for rx in range(int(center[0] - radius), int(center[0] + radius + 1), step):
            for ry in range(int(center[1] - radius), int(center[1] + radius + 1), step):
                for rz in range(int(center[2] - radius), int(center[2] + radius + 1), step):
                    rotation = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix().astype(np.float32)
                    coarse_full = transform_points(model_points, nominal_scale, rotation, np.zeros(3, dtype=np.float32))
                    coarse_front = extract_front_visible_points(coarse_full, bins=128, max_points=3200, seed=11)
                    translation = build_initial_translation(
                        model_full_points=coarse_full,
                        model_front_points=coarse_front,
                        target_front_points=target_front_points,
                        mean_depth=mean_depth,
                    )
                    coarse_full = coarse_full + translation
                    coarse_front = coarse_front + translation
                    metrics = evaluate_alignment(
                        transformed_full_points=coarse_full,
                        transformed_front_points=coarse_front,
                        target_front_points=target_front_points,
                        real_width=real_width,
                        real_height=real_height,
                        scale=nominal_scale,
                        nominal_scale=nominal_scale,
                    )
                    candidates.append(
                        {
                            "rotation": rotation,
                            "translation": translation,
                            "scale": nominal_scale,
                            "metrics": metrics,
                            "euler_deg": [float(rx), float(ry), float(rz)],
                        }
                    )

    try_grid(center=(0.0, 0.0, 0.0), step=10, radius=30)
    best_coarse = min(candidates, key=lambda item: item["metrics"]["score"])
    try_grid(center=tuple(best_coarse["euler_deg"]), step=4, radius=8)
    candidates.sort(key=lambda item: item["metrics"]["score"])
    return candidates[0]


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
    rmse_score = 1.0 - min(best["rmse"] / size_ref, 1.0)
    inlier_ratio = float(np.clip(best["inlier_ratio"], 0.0, 1.0))

    confidence = (
        0.30 * valid_ratio
        + 0.20 * scale_consistency
        + 0.25 * rmse_score
        + 0.15 * inlier_ratio
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
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
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
    pointcloud_path: Path,
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
        "Dark = point cloud, light = aligned model",
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
        str(pointcloud_path),
        str(render_path),
        *[f"{float(v):.9f}" for v in blender_translation],
        *[f"{float(v):.9f}" for v in blender_delta_euler_deg],
        f"{float(scale):.9f}",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
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

    if "depthpointcloud" not in task:
        raise ValueError("depthpointcloud is missing. Run pointcloud stage first.")
    if "model" not in task:
        raise ValueError("model is missing. Run model scale stage first.")

    paths = resolve_task_paths(task)
    prefix = task_prefix(task, json_path)
    pointcloud_name = (task.get("depthpointcloud") or {}).get("pointcloud_name")
    if not pointcloud_name:
        raise ValueError("depthpointcloud.pointcloud_name is missing")

    pointcloud_path = object_alignment_output_path(pointcloud_name)
    pointcloud_points_export = read_binary_ply_points(pointcloud_path)
    pointcloud_points_unity = pointcloud_export_to_unity(pointcloud_points_export)
    model_vertices_raw = read_obj_vertices(paths["mesh_path"])
    model_vertices_unity = obj_vertices_to_unity(model_vertices_raw)
    target_front_fit = extract_front_visible_points(pointcloud_points_unity, bins=140, max_points=4200, seed=7)

    overall_scale = float((task.get("model") or {}).get("overall_scale") or 1.0)
    mean_depth = float((task.get("depthpointcloud") or {}).get("mean_depth_measured") or 0.0)
    real_width = float((task.get("depthpointcloud") or {}).get("real_width_measured") or 0.0)
    real_height = float((task.get("depthpointcloud") or {}).get("real_height_measured") or 0.0)

    best_coarse = search_initial_pose(
        model_points=model_vertices_unity,
        target_front_points=target_front_fit,
        nominal_scale=overall_scale,
        mean_depth=mean_depth,
        real_width=real_width,
        real_height=real_height,
    )

    candidate_scales = np.linspace(0.97, 1.03, 7, dtype=np.float32) * overall_scale
    best_result: dict | None = None
    best_debug: dict | None = None
    for scale in candidate_scales:
        rotation_seed = best_coarse["rotation"]
        coarse_full = transform_points(model_vertices_unity, float(scale), rotation_seed, np.zeros(3, dtype=np.float32))
        coarse_front = extract_front_visible_points(coarse_full, bins=160, max_points=5000, seed=19)
        coarse_translation = build_initial_translation(
            model_full_points=coarse_full,
            model_front_points=coarse_front,
            target_front_points=target_front_fit,
            mean_depth=mean_depth,
        )
        coarse_full = coarse_full + coarse_translation
        coarse_front = coarse_front + coarse_translation

        delta_result = run_icp(
            model_points=coarse_front,
            target_points=target_front_fit,
            scale=1.0,
            init_rotation=np.eye(3, dtype=np.float32),
            init_translation=np.zeros(3, dtype=np.float32),
            iterations=24,
        )
        final_rotation, final_translation = compose_transform(
            base_rotation=rotation_seed,
            base_translation=coarse_translation,
            delta_rotation=delta_result["rotation"],
            delta_translation=delta_result["translation"],
        )

        transformed_full = transform_points(model_vertices_unity, float(scale), final_rotation, final_translation)
        transformed_front = extract_front_visible_points(transformed_full, bins=160, max_points=5000, seed=23)
        metrics = evaluate_alignment(
            transformed_full_points=transformed_full,
            transformed_front_points=transformed_front,
            target_front_points=target_front_fit,
            real_width=real_width,
            real_height=real_height,
            scale=float(scale),
            nominal_scale=overall_scale,
        )

        candidate = {
            "scale": float(scale),
            "rotation": final_rotation,
            "translation": final_translation,
            "rmse": metrics["rmse_3d"],
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
                    "scale": float(scale),
                    "seed_euler_deg": [float(v) for v in best_coarse["euler_deg"]],
                    "pose": serialize_pose(
                        rotation_seed,
                        coarse_translation,
                        "unity_camera_local_x_right_y_up_z_forward",
                    ),
                    "score": float(best_coarse["metrics"]["score"]),
                    "rmse_3d": float(best_coarse["metrics"]["rmse_3d"]),
                    "rmse_2d": float(best_coarse["metrics"]["rmse_2d"]),
                    "size_error": float(best_coarse["metrics"]["size_error"]),
                },
                "icp_delta": {
                    "scale": 1.0,
                    "pose": serialize_pose(
                        delta_result["rotation"],
                        delta_result["translation"],
                        "unity_camera_local_delta_x_right_y_up_z_forward",
                    ),
                    "rmse": float(delta_result["rmse"]),
                    "inlier_ratio": float(delta_result["inlier_ratio"]),
                },
                "final_camera_local_unity": {
                    "scale": float(scale),
                    "pose": serialize_pose(
                        final_rotation,
                        final_translation,
                        "unity_camera_local_x_right_y_up_z_forward",
                    ),
                    "front_view_rmse_2d": float(metrics["rmse_2d"]),
                    "front_view_size_error": float(metrics["size_error"]),
                    "width_error": float(metrics["width_error"]),
                    "height_error": float(metrics["height_error"]),
                },
            }

    if best_result is None:
        raise RuntimeError("Failed to compute object alignment")
    if best_debug is None:
        raise RuntimeError("Failed to collect object alignment debug data")

    best = best_result

    pointcloud_rotation, pointcloud_translation = model_pose_unity_to_pointcloud_input(
        best["rotation"],
        best["translation"],
    )
    pointcloud_euler_deg = Rotation.from_matrix(pointcloud_rotation).as_euler("xyz", degrees=True)
    pointcloud_quat_xyzw = Rotation.from_matrix(pointcloud_rotation).as_quat()

    blender_rotation = rotation_unity_to_blender_world(best["rotation"])
    blender_translation = unity_to_blender_world_vector(best["translation"])
    blender_delta_euler_deg = Rotation.from_matrix(blender_rotation).as_euler("xyz", degrees=True)

    confidence = compute_confidence(task, best)
    aligned_model_image_name = f"{prefix}_aligned_model_front.png"
    aligned_model_image_path = object_alignment_output_path(aligned_model_image_name)
    overlay_preview_name = f"{prefix}_alignment_preview_perspective.png"
    overlay_preview_path = object_alignment_output_path(overlay_preview_name)

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
        "front_view_image_name": aligned_model_image_name,
        "preview_image_name": overlay_preview_name,
        "confidence": confidence,
        "icp_rmse": float(best["rmse"]),
        "front_view_rmse_2d": float(best.get("rmse_2d") or 0.0),
        "front_view_size_error": float(best.get("size_error") or 0.0),
        "icp_inlier_ratio": float(best["inlier_ratio"]),
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
            pointcloud_path=pointcloud_path,
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
    print(f"[INFO] Render          : {aligned_model_image_path}")
    print(f"[INFO] Preview         : {overlay_preview_path if object_alignment.get('preview_image_name') else 'not-generated'}")
    print(f"[INFO] Blender         : {blender_label}")
    print(
        f"[INFO] Output pose     : "
        f"pos={object_alignment['model_position']}, "
        f"rot={object_alignment['model_rotation_euler_deg']}"
    )
    print(
        f"[INFO] Final scale     : {object_alignment['model_real_scale']:.6f}, "
        f"confidence={confidence:.3f}"
    )
    print("[OK] Object alignment stage completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
