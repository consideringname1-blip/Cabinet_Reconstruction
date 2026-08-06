#!/usr/bin/env python3
"""Report per-frame masked GT depth and world-centroid consistency."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rgb_files = sorted((args.input_dir / "jpg").glob("*.jpg"))
    depth_files = sorted((args.input_dir / "depth").glob("*.png"))
    mask_files = sorted(args.mask_dir.glob("*.npy"))
    poses = load_odometry(args.input_dir / "odometry.log")
    fx, fy, cx, cy = np.loadtxt(args.input_dir / "calibration.txt").reshape(-1)[:4]
    rows = []
    centroids = []

    for index, (depth_path, mask_path, pose) in enumerate(zip(depth_files, mask_files, poses)):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
        mask = np.load(mask_path).squeeze().astype(bool) & (depth > 0)
        y, x = np.nonzero(mask)
        z = depth[mask]
        camera = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], axis=1)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        centroid = np.median(world, axis=0)
        centroids.append(centroid)
        rows.append(
            {
                "local_index": index,
                "source_index": index + 3,
                "mask_fraction": float(mask.mean()),
                "valid_points": int(len(z)),
                "depth_median_m": float(np.median(z)),
                "depth_p10_m": float(np.percentile(z, 10)),
                "depth_p90_m": float(np.percentile(z, 90)),
                "world_centroid_m": centroid.tolist(),
            }
        )

    centroids = np.stack(centroids)
    robust_center = np.median(centroids, axis=0)
    distances = np.linalg.norm(centroids - robust_center, axis=1)
    for row, distance in zip(rows, distances):
        row["world_centroid_distance_to_median_m"] = float(distance)
    report = {
        "robust_world_centroid_m": robust_center.tolist(),
        "world_centroid_distance_percentiles_m": np.percentile(
            distances, [0, 25, 50, 75, 90, 95, 100]
        ).tolist(),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['local_index']:02d} src={row['source_index']:03d} "
            f"mask={row['mask_fraction']:.3f} depth={row['depth_median_m']:.3f}m "
            f"centroid_delta={row['world_centroid_distance_to_median_m']:.3f}m"
        )
    print("centroid distance percentiles:", report["world_centroid_distance_percentiles_m"])


if __name__ == "__main__":
    main()
