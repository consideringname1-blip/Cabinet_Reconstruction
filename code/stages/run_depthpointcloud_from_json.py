from __future__ import annotations

import sys

from _bootstrap import CODE_ROOT
from config import ENABLE_ALIGNMENT_RENDER_OUTPUTS, DEPTHPOINTCLOUD_MAX_EXPORT_POINTS
import numpy as np

from object_alignment_common import (
    build_depth_pointcloud,
    compute_front_view_extents,
    compute_real_measurements,
    object_alignment_output_path,
    read_depth_image,
    read_mask,
    render_front_view_points,
    resolve_task_paths,
    task_prefix,
    write_binary_ply,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


def remove_legacy_outputs(prefix: str) -> None:
    for name in (
        f"{prefix}_size_compare.png",
        f"{prefix}_pointcloud_measure.png",
    ):
        path = object_alignment_output_path(name)
        if path.exists():
            path.unlink()


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
    if len(argv) != 2:
        print("Usage: python code/stages/run_depthpointcloud_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_task_json_path(argv[1])
    task = load_task_json(json_path)
    paths = resolve_task_paths(task)
    print(f"[STAGE] Depth pointcloud start : {json_path}")

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    print("[STAGE] Loading depth and mask inputs")
    mask_bool = read_mask(paths["mask_path"])
    depth_mm = read_depth_image(paths["depth_path"])

    print("[STAGE] Computing measurements and building point cloud")
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
    front_view_image_name = f"{prefix}_depthpointcloud_front.png"
    pointcloud_path = object_alignment_output_path(pointcloud_name)
    front_view_path = object_alignment_output_path(front_view_image_name)

    print(
        f"[STAGE] Writing outputs      : raw_points={raw_point_count}, "
        f"export_points={len(export_points)}, max_export={DEPTHPOINTCLOUD_MAX_EXPORT_POINTS}"
    )
    write_binary_ply(pointcloud_path, export_points)
    if ENABLE_ALIGNMENT_RENDER_OUTPUTS:
        render_front_view_points(
            unity_points,
            front_view_path,
            title="Depth Point Cloud Front View",
            info_lines=[
                "Basis: Unity X-right / Y-up / Z-forward",
                "Blender import: forward=-X, up=+Y",
                f"Real width  : {measurements['real_width_m']:.4f} m",
                f"Real height : {measurements['real_height_m']:.4f} m",
                f"Mean depth  : {measurements['mean_depth_m']:.4f} m",
                f"Mask crop   : outer {measurements['depth_border_crop_ratio'] * 100.0:.0f}% inward",
                f"Point count : {len(export_points)}",
            ],
        )
    elif front_view_path.exists():
        front_view_path.unlink()
    remove_legacy_outputs(prefix)

    depthpointcloud = {
        "pointcloud_name": pointcloud_name,
        "front_view_image_name": front_view_image_name if ENABLE_ALIGNMENT_RENDER_OUTPUTS else None,
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
    }
    task["depthpointcloud"] = depthpointcloud
    save_task_json(json_path, task)

    print(f"[INFO] JSON            : {json_path}")
    print(f"[INFO] Pointcloud      : {pointcloud_path}")
    print(f"[INFO] Front image     : {front_view_path if ENABLE_ALIGNMENT_RENDER_OUTPUTS else 'not-generated'}")
    print(
        f"[INFO] Real size       : "
        f"{measurements['real_width_m']:.4f} m x {measurements['real_height_m']:.4f} m"
    )
    print(f"[INFO] Mean depth      : {measurements['mean_depth_m']:.4f} m")
    print(
        f"[INFO] Point count      : raw={raw_point_count}, "
        f"exported={len(export_points)}, downsampled={len(export_points) != raw_point_count}"
    )
    print("[OK] Depth pointcloud stage completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
