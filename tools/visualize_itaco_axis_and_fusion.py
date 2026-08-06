#!/usr/bin/env python3
"""Render the extended open-state point cloud and prismatic axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def equal_limits(points: np.ndarray, padding: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2
    radius = np.max(high - low) / 2 + padding
    return center - radius, center + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-static-points", type=int, default=60000)
    parser.add_argument("--max-moving-points", type=int, default=30000)
    args = parser.parse_args()

    fusion_dir = args.run_root / "extended/prismatic_open_fusion"
    report = json.loads((fusion_dir / "fusion_report.json").read_text())
    static_pcd = o3d.io.read_point_cloud(
        str(fusion_dir / "open_state_static_surface_points.ply")
    )
    moving_pcd = o3d.io.read_point_cloud(str(fusion_dir / "open_state_drawer_fused.ply"))
    static = np.asarray(static_pcd.points)
    moving = np.asarray(moving_pcd.points)
    moving_colors = np.asarray(moving_pcd.colors)
    rng = np.random.default_rng(0)
    if len(static) > args.max_static_points:
        static = static[rng.choice(len(static), args.max_static_points, replace=False)]
    if len(moving) > args.max_moving_points:
        keep = rng.choice(len(moving), args.max_moving_points, replace=False)
        moving = moving[keep]
        moving_colors = moving_colors[keep]

    axis = np.asarray(report["opening_axis_world"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    axis_center = np.median(moving, axis=0)
    axis_points = np.stack([axis_center - axis * 0.28, axis_center + axis * 0.28])
    all_points = np.concatenate([static, moving, axis_points])
    low, high = equal_limits(all_points)

    figure = plt.figure(figsize=(16, 12), dpi=140)
    views = [
        ("Perspective", 24, -62),
        ("Front-ish", 8, -88),
        ("Side", 9, 5),
        ("Top", 88, -90),
    ]
    for plot_index, (title, elevation, azimuth) in enumerate(views, start=1):
        axis3d = figure.add_subplot(2, 2, plot_index, projection="3d")
        axis3d.scatter(
            static[:, 0], static[:, 1], static[:, 2],
            s=0.15, c="#8a8a8a", alpha=0.17, depthshade=False,
        )
        axis3d.scatter(
            moving[:, 0], moving[:, 1], moving[:, 2],
            s=0.65, c=moving_colors, alpha=0.92, depthshade=False,
        )
        axis3d.plot(
            axis_points[:, 0], axis_points[:, 1], axis_points[:, 2],
            color="#00ff62", linewidth=4,
        )
        arrow_start = axis_center - axis * 0.18
        axis3d.quiver(
            arrow_start[0], arrow_start[1], arrow_start[2],
            axis[0], axis[1], axis[2],
            length=0.36, color="#00ff62", linewidth=3, arrow_length_ratio=0.12,
        )
        axis3d.set_xlim(low[0], high[0])
        axis3d.set_ylim(low[1], high[1])
        axis3d.set_zlim(low[2], high[2])
        axis3d.view_init(elev=elevation, azim=azimuth)
        axis3d.set_title(title)
        axis3d.set_xlabel("world X (m)")
        axis3d.set_ylabel("world Y (m)")
        axis3d.set_zlabel("world Z (m)")
        axis3d.set_box_aspect((1, 1, 1))

    figure.suptitle(
        "Extended open-state fusion: static cabinet (gray), drawer (RGB), "
        "estimated opening axis (green)",
        fontsize=14,
    )
    figure.tight_layout()
    output = args.run_root / "validation/extended_open_fusion_axis_views.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
