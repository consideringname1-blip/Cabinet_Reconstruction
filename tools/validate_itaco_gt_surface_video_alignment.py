#!/usr/bin/env python3
"""Validate the GT camera-to-world transform used instead of Polycam alignment."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.view_dir / "metadata.json").read_text())
    intrinsic = np.array(metadata["K"]).reshape(3, 3).T
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    depth = np.load(args.preprocess_dir / "prompt_depth_video/000000.npy")
    moving = cv2.imread(
        str(args.preprocess_dir / "monst3r/dynamic_mask_0.png"), cv2.IMREAD_GRAYSCALE
    ) > 127
    hand = np.load(args.preprocess_dir / "hand_mask/000000.npy").squeeze().astype(bool)
    camera_to_world = np.load(args.preprocess_dir / "cam2world.npy")
    yy, xx = np.indices(depth.shape)
    valid = moving & (~hand) & (depth > 0.2) & (depth < 4.0)
    camera = np.stack(
        [(xx - cx) * depth / fx, (yy - cy) * depth / fy, depth], axis=-1
    )
    world = camera[valid] @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    surface = np.asarray(
        o3d.io.read_point_cloud(str(args.view_dir / "surface/surface.ply")).points
    )
    distance, _ = cKDTree(surface).query(world, k=1, workers=-1)
    report = {
        "alignment_stage": "GT same-world alignment replacing cross-device Polycam/Record3D LoFTR alignment",
        "camera_to_world": camera_to_world.tolist(),
        "moving_pixels_with_valid_depth": int(len(world)),
        "nearest_surface_distance_m": {
            "mean": float(np.mean(distance)),
            "median": float(np.median(distance)),
            "p90": float(np.percentile(distance, 90)),
            "p95": float(np.percentile(distance, 95)),
            "max": float(np.max(distance)),
            "fraction_under_0.03m": float(np.mean(distance < 0.03)),
            "fraction_under_0.05m": float(np.mean(distance < 0.05)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
