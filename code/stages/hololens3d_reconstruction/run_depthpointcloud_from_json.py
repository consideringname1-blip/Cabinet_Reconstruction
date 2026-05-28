from __future__ import annotations

import sys

import _bootstrap
from config import (
    ICP_TARGET_FRONT_MAX_POINTS,
)
import numpy as np

from object_alignment_common import (
    build_depth_border_keep_mask,
    build_depth_pointcloud,
    build_depth_pointcloud_from_valid_mask,
    compute_front_view_extents,
    compute_real_measurements,
    depth_limits_for_task,
    read_depth_image,
    read_mask,
    resolve_task_paths,
    select_front_visible_points,
)
from stage_common import load_stage_task
from task_json import save_task_json


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
    depth_limits = depth_limits_for_task(task)

    measurements = compute_real_measurements(
        mask_bool,
        depth_mm,
        k,
        min_depth_mm=depth_limits.min_depth_mm,
        max_depth_mm=depth_limits.max_reliable_depth_mm,
    )
    _export_points, canonical_points = build_depth_pointcloud(
        depth_mm,
        mask_bool,
        k,
        min_depth_mm=depth_limits.min_depth_mm,
        max_depth_mm=depth_limits.max_reliable_depth_mm,
    )
    raw_point_count = int(len(canonical_points))
    extents = compute_front_view_extents(canonical_points)

    valid_all = (
        mask_bool
        & (depth_mm >= depth_limits.min_depth_mm)
        & (depth_mm <= depth_limits.max_reliable_depth_mm)
    )
    depth_keep_mask = build_depth_border_keep_mask(mask_bool)
    valid_cropped = valid_all & ~depth_keep_mask
    discarded_points_export, _discarded_points_canonical = build_depth_pointcloud_from_valid_mask(
        depth_mm,
        valid_cropped,
        k,
    )
    _icp_used_points_canonical, target_front_indices = select_front_visible_points(
        canonical_points,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=7,
    )

    depthpointcloud = {
        "depth_sensor": depth_limits.sensor,
        "min_depth_mm": int(depth_limits.min_depth_mm),
        "max_reliable_depth_mm": int(depth_limits.max_reliable_depth_mm),
        "width_pointcloud_units": extents["width_units"],
        "height_pointcloud_units": extents["height_units"],
        "depth_pointcloud_units": extents["depth_units"],
        "real_width_measured": measurements["real_width_m"],
        "real_height_measured": measurements["real_height_m"],
        "mean_depth_measured": measurements["mean_depth_m"],
        "raw_point_count": int(raw_point_count),
        "point_count": int(raw_point_count),
        "valid_depth_ratio": measurements["valid_ratio"],
        "mask_bbox_xyxy": measurements["mask_bbox_xyxy"],
        "mask_pixels": measurements["mask_pixels"],
        "valid_depth_pixels": measurements["valid_depth_pixels"],
        "depth_border_crop_ratio": measurements["depth_border_crop_ratio"],
        "depth_border_crop_mode": measurements["depth_border_crop_mode"],
        "depth_border_crop_margin_x_px": measurements["depth_border_crop_margin_x_px"],
        "depth_border_crop_margin_y_px": measurements["depth_border_crop_margin_y_px"],
        "mask_border_crop_threshold_px": measurements["mask_border_crop_threshold_px"],
        "mask_border_crop_min_inside_distance_px": measurements["mask_border_crop_min_inside_distance_px"],
        "mask_border_crop_max_inside_distance_px": measurements["mask_border_crop_max_inside_distance_px"],
        "usable_mask_pixels": measurements["usable_mask_pixels"],
        "cropped_mask_pixels": measurements["cropped_mask_pixels"],
        "discarded_count": int(len(discarded_points_export)),
        "used_count": int(len(target_front_indices)),
        "icp_target_front_max_points": int(ICP_TARGET_FRONT_MAX_POINTS),
    }
    task["depthpointcloud"] = depthpointcloud
    save_task_json(json_path, task)

    print(
        f"[INFO] depthpointcloud : points raw={raw_point_count} "
        f"used={len(target_front_indices)} discarded={len(discarded_points_export)} "
        f"size={measurements['real_width_m']:.4f}x{measurements['real_height_m']:.4f}m "
        f"depth={measurements['mean_depth_m']:.4f}m"
    )
    print("[OK] depthpointcloud")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
