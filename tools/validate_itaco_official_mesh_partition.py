#!/usr/bin/env python3
"""Preflight the exact iTACO moving-map partition used by extract_mesh.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root

    depth = np.load(root / "official/preprocess/prompt_depth_video/000000.npy")
    moving_map = np.load(
        root
        / "official/prediction/refinement/monst3r/chamfer/0/prismatic/moving_map.npz"
    )["a"][0]
    metadata = json.loads((root / "official/view/metadata.json").read_text())
    intrinsic = np.asarray(metadata["K"], dtype=np.float64).reshape(3, 3).T
    yy, xx = np.indices(depth.shape)
    camera_xyz = np.stack(
        [
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    )
    pose = np.load(root / "official/preprocess/cam2world.npy")
    world = camera_xyz.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
    selected = (moving_map > 0.7).reshape(-1) & (depth.reshape(-1) > 0.2)
    moving_points = world[selected]
    moving_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moving_points))
    surface = o3d.io.read_point_cloud(str(root / "official/view/surface/surface.ply"))
    distance = np.asarray(surface.compute_point_cloud_distance(moving_pcd))
    surface_moving = distance < 0.03
    report = {
        "matches_extract_mesh_py": {
            "moving_threshold": 0.7,
            "surface_distance_threshold_m": 0.03,
            "frame": "first interaction frame, source 97",
        },
        "moving_map_full_image_fraction_above_threshold": float((moving_map > 0.7).mean()),
        "moving_pixels_with_valid_gt_depth": int(len(moving_points)),
        "surface_points": int(len(surface.points)),
        "surface_points_classified_moving": int(surface_moving.sum()),
        "surface_fraction_classified_moving": float(surface_moving.mean()),
        "surface_distance_to_selected_video_points_quantiles_m": np.quantile(
            distance, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        ).tolist(),
        "quality_warning": (
            "The official refined moving map over-selects background/cabinet regions; "
            "the official moving mesh is expected to include static cabinet geometry."
        ),
    }
    output = root / "validation/official_mesh_partition_preflight.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
