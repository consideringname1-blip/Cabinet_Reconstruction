#!/usr/bin/env python3
"""Reproject HoloLens Long Throw world-space PLYs into the saved PV pinhole view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def depth_timestamps(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def project_world_points(
    world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, dict]:
    camera = (world - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z >= depth_min) & (z <= depth_max)
    camera = camera[valid]
    z = camera[:, 2]
    x = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
    y = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]

    # Bilinear footprint: contribute to the four neighboring pixels, retaining
    # the nearest z at every pixel. This fills sampling gaps without a 3x3
    # morphological expansion across object boundaries.
    candidates = []
    for u in (np.floor(x), np.ceil(x)):
        for v in (np.floor(y), np.ceil(y)):
            ui = u.astype(np.int64)
            vi = v.astype(np.int64)
            inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            candidates.append((ui[inside], vi[inside], z[inside]))

    flat = np.full(height * width, np.inf, dtype=np.float64)
    projected_count = 0
    for ui, vi, zi in candidates:
        np.minimum.at(flat, vi * width + ui, zi)
        projected_count += len(zi)
    depth = flat.reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    positive = depth[depth > 0]
    stats = {
        "input_world_points": int(len(world)),
        "valid_camera_points": int(valid.sum()),
        "zbuffer_contributions": int(projected_count),
        "valid_pixels": int(len(positive)),
        "valid_fraction": float(len(positive) / depth.size),
        "depth_percentiles_m": (
            np.percentile(positive, [0, 1, 10, 50, 90, 99, 100]).tolist()
            if len(positive)
            else []
        ),
    }
    return depth.astype(np.float32), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=4.0)
    args = parser.parse_args()

    mapping = json.loads((args.sequence_dir / "frame_mapping.json").read_text())
    source_pinhole = args.source_root / "pinhole_projection"
    poses = load_odometry(source_pinhole / "odometry.log")
    timestamps = depth_timestamps(source_pinhole / "depth.txt")
    fx, fy, cx, cy = np.loadtxt(source_pinhole / "calibration.txt").reshape(-1)[:4]
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    first_rgb = cv2.imread(mapping[0]["rgb"], cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise FileNotFoundError(mapping[0]["rgb"])
    height, width = first_rgb.shape[:2]
    png_dir = args.output_dir / "png"
    npy_dir = args.output_dir / "npy"
    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for local_index, item in enumerate(mapping):
        source_index = int(item["source_index"])
        timestamp = timestamps[source_index]
        ply_path = args.source_root / "Depth Long Throw" / f"{timestamp}.ply"
        world = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
        depth, stats = project_world_points(
            world,
            poses[source_index],
            intrinsic,
            height,
            width,
            args.depth_min,
            args.depth_max,
        )
        np.save(npy_dir / f"{local_index:06d}.npy", depth)
        cv2.imwrite(
            str(png_dir / f"{local_index:06d}.png"),
            np.rint(depth * 1000.0).clip(0, 65535).astype(np.uint16),
        )
        record = {
            "local_index": local_index,
            "source_index": source_index,
            "depth_timestamp": timestamp,
            "source_ply": str(ply_path.resolve()),
            **stats,
        }
        records.append(record)
        print(
            f"[{local_index + 1:03d}/{len(mapping):03d}] source={source_index:03d} "
            f"valid={stats['valid_fraction']:.3f} "
            f"median={stats['depth_percentiles_m'][3]:.3f}m"
        )

    (args.output_dir / "projection_report.json").write_text(
        json.dumps(
            {
                "method": "world PLY -> inverse PV camera-to-world -> pinhole bilinear z-buffer",
                "intrinsic": intrinsic.tolist(),
                "image_size_wh": [width, height],
                "depth_range_m": [args.depth_min, args.depth_max],
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
