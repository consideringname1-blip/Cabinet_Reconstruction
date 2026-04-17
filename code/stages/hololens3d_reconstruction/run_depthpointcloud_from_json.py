from __future__ import annotations

import sys

import _bootstrap
from config import (
    DEPTHPOINTCLOUD_MAX_EXPORT_POINTS,
    ICP_TARGET_FRONT_MAX_POINTS,
)
import numpy as np

from object_alignment_common import (
    MAX_DEPTH_MM,
    MIN_DEPTH_MM,
    build_depth_border_keep_mask,
    build_depth_pointcloud,
    build_depth_pointcloud_from_valid_mask,
    compute_front_view_extents,
    compute_real_measurements,
    object_alignment_output_path,
    read_depth_image,
    read_mask,
    resolve_task_paths,
    select_front_visible_points,
    task_prefix,
    write_binary_ply,
)
from stage_common import load_stage_task
from task_json import save_task_json


def downsample_export_points(
    export_points: np.ndarray,
    unity_points: np.ndarray,
    max_points: int | None,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw_count = int(len(export_points))
    if not max_points or max_points <= 0 or raw_count <= max_points:
        return export_points, unity_points, raw_count
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(raw_count, size=max_points, replace=False))
    return export_points[picked], unity_points[picked], raw_count


def main(argv: list[str]) -> int:
    json_path, task = load_stage_task(
        argv,
        usage="Usage: python code/stages/hololens3d_reconstruction/run_depthpointcloud_from_json.py <task_meta.json or filename>",
        stage_name="depthpointcloud",
    )
    paths = resolve_task_paths(task)

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    mask_bool = read_mask(paths["mask_path"])
    depth_mm = read_depth_image(paths["depth_path"])

    measurements = compute_real_measurements(mask_bool, depth_mm, k)
    export_points, unity_points = build_depth_pointcloud(depth_mm, mask_bool, k)
    export_points, unity_points, raw_point_count = downsample_export_points(
        export_points,
        unity_points,
        DEPTHPOINTCLOUD_MAX_EXPORT_POINTS,
    )
    extents = compute_front_view_extents(unity_points)

    prefix = task_prefix(task, json_path)
    pointcloud_name = f"{prefix}_sam3_pointcloud.ply"
    icp_discarded_pointcloud_name = f"{prefix}_icp_discarded_points.ply"
    icp_used_pointcloud_name = f"{prefix}_icp_used_points.ply"
    pointcloud_path = object_alignment_output_path(pointcloud_name)
    icp_discarded_pointcloud_path = object_alignment_output_path(icp_discarded_pointcloud_name)
    icp_used_pointcloud_path = object_alignment_output_path(icp_used_pointcloud_name)

    valid_all = mask_bool & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    depth_keep_mask = build_depth_border_keep_mask(mask_bool)
    valid_cropped = valid_all & ~depth_keep_mask
    discarded_points_export, _ = build_depth_pointcloud_from_valid_mask(depth_mm, valid_cropped, k)
    icp_used_points_unity, target_front_indices = select_front_visible_points(
        unity_points,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=7,
    )
    icp_used_points_export = export_points[target_front_indices]

    write_binary_ply(pointcloud_path, export_points)
    write_binary_ply(icp_discarded_pointcloud_path, discarded_points_export)
    write_binary_ply(icp_used_pointcloud_path, icp_used_points_export)

    depthpointcloud = {
        "pointcloud_name": pointcloud_name,
        "coordinate_basis": "unity_x_right_y_up_z_forward",
        "blender_import_axes": {"forward": "-X", "up": "+Y"},
        "width_pointcloud_units": extents["width_units"],
        "height_pointcloud_units": extents["height_units"],
        "depth_pointcloud_units": extents["depth_units"],
        "bbox_min": extents["bbox_min"],
        "bbox_max": extents["bbox_max"],
        "real_width_measured": measurements["real_width_m"],
        "real_height_measured": measurements["real_height_m"],
        "mean_depth_measured": measurements["mean_depth_m"],
        "raw_point_count": int(raw_point_count),
        "point_count": int(len(export_points)),
        "export_max_points": int(DEPTHPOINTCLOUD_MAX_EXPORT_POINTS) if DEPTHPOINTCLOUD_MAX_EXPORT_POINTS else None,
        "export_downsampled": bool(len(export_points) != raw_point_count),
        "valid_depth_ratio": measurements["valid_ratio"],
        "depth_border_crop_ratio": measurements["depth_border_crop_ratio"],
        "depth_border_crop_mode": measurements["depth_border_crop_mode"],
        "depth_border_crop_margin_x_px": measurements["depth_border_crop_margin_x_px"],
        "depth_border_crop_margin_y_px": measurements["depth_border_crop_margin_y_px"],
        "mask_border_crop_threshold_px": measurements["mask_border_crop_threshold_px"],
        "mask_border_crop_min_inside_distance_px": measurements["mask_border_crop_min_inside_distance_px"],
        "mask_border_crop_max_inside_distance_px": measurements["mask_border_crop_max_inside_distance_px"],
        "usable_mask_pixels": measurements["usable_mask_pixels"],
        "cropped_mask_pixels": measurements["cropped_mask_pixels"],
        "icp_discarded_pointcloud_name": icp_discarded_pointcloud_name,
        "icp_used_pointcloud_name": icp_used_pointcloud_name,
        "discarded_count": int(len(discarded_points_export)),
        "used_count": int(len(icp_used_points_export)),
        "icp_target_front_max_points": int(ICP_TARGET_FRONT_MAX_POINTS),
    }
    task["depthpointcloud"] = depthpointcloud
    save_task_json(json_path, task)

    print(
        f"[INFO] depthpointcloud : points raw={raw_point_count} export={len(export_points)} "
        f"used={len(icp_used_points_export)} discarded={len(discarded_points_export)} "
        f"size={measurements['real_width_m']:.4f}x{measurements['real_height_m']:.4f}m "
        f"depth={measurements['mean_depth_m']:.4f}m"
    )
    print("[OK] depthpointcloud")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
