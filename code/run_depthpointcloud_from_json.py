from __future__ import annotations

import sys

import numpy as np

from object_alignment_common import (
    build_depth_pointcloud,
    compute_front_view_extents,
    compute_real_measurements,
    load_json,
    object_alignment_output_path,
    read_depth_image,
    read_mask,
    render_front_view_points,
    resolve_json_path,
    resolve_task_paths,
    save_json,
    task_prefix,
    write_binary_ply,
)


def remove_legacy_outputs(prefix: str) -> None:
    for name in (
        f"{prefix}_size_compare.png",
        f"{prefix}_pointcloud_measure.png",
    ):
        path = object_alignment_output_path(name)
        if path.exists():
            path.unlink()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python code/run_depthpointcloud_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_json_path(argv[1])
    task = load_json(json_path)
    paths = resolve_task_paths(task)

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    mask_bool = read_mask(paths["mask_path"])
    depth_mm = read_depth_image(paths["depth_path"])

    measurements = compute_real_measurements(mask_bool, depth_mm, k)
    export_points, unity_points = build_depth_pointcloud(depth_mm, mask_bool, k)
    extents = compute_front_view_extents(unity_points)

    prefix = task_prefix(task, json_path)
    pointcloud_name = f"{prefix}_sam3_pointcloud.ply"
    front_view_image_name = f"{prefix}_depthpointcloud_front.png"
    pointcloud_path = object_alignment_output_path(pointcloud_name)
    front_view_path = object_alignment_output_path(front_view_image_name)

    write_binary_ply(pointcloud_path, export_points)
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
            f"Point count : {len(export_points)}",
        ],
    )
    remove_legacy_outputs(prefix)

    depthpointcloud = {
        "pointcloud_name": pointcloud_name,
        "front_view_image_name": front_view_image_name,
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
        "point_count": int(len(export_points)),
        "valid_depth_ratio": measurements["valid_ratio"],
    }
    task["depthpointcloud"] = depthpointcloud
    save_json(json_path, task)

    print(f"[INFO] JSON            : {json_path}")
    print(f"[INFO] Pointcloud      : {pointcloud_path}")
    print(f"[INFO] Front image     : {front_view_path}")
    print(
        f"[INFO] Real size       : "
        f"{measurements['real_width_m']:.4f} m x {measurements['real_height_m']:.4f} m"
    )
    print(f"[INFO] Mean depth      : {measurements['mean_depth_m']:.4f} m")
    print("[OK] Depth pointcloud stage completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
