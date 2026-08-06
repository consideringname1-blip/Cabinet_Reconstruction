#!/usr/bin/env python3
"""Render the GT-plane-axis corrected open-state fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    fusion_dir = args.run_root / "extended/prismatic_open_fusion_gt_plane_axis"
    report = json.loads((fusion_dir / "fusion_report.json").read_text())
    static_pcd = o3d.io.read_point_cloud(
        str(fusion_dir / "open_state_static_surface_points.ply")
    )
    moving_pcd = o3d.io.read_point_cloud(str(fusion_dir / "open_state_drawer_fused.ply"))
    static = np.asarray(static_pcd.points)
    moving = np.asarray(moving_pcd.points)
    colors = np.asarray(moving_pcd.colors)
    rng = np.random.default_rng(0)
    if len(static) > 60000:
        static = static[rng.choice(len(static), 60000, replace=False)]
    axis = np.asarray(report["axis_world"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    center = np.median(moving, axis=0)
    line = np.stack([center - axis * 0.30, center + axis * 0.30])
    points = np.concatenate([static, moving, line])
    low, high = points.min(axis=0), points.max(axis=0)
    cube_center = (low + high) / 2
    radius = np.max(high - low) / 2 + 0.03
    low, high = cube_center - radius, cube_center + radius

    figure = plt.figure(figsize=(16, 12), dpi=140)
    for plot_index, (title, elevation, azimuth) in enumerate(
        [
            ("Perspective", 24, -62),
            ("Front-ish", 8, -88),
            ("Side", 9, 5),
            ("Top", 88, -90),
        ],
        start=1,
    ):
        panel = figure.add_subplot(2, 2, plot_index, projection="3d")
        panel.scatter(
            static[:, 0], static[:, 1], static[:, 2],
            s=0.15, c="#8a8a8a", alpha=0.17, depthshade=False,
        )
        panel.scatter(
            moving[:, 0], moving[:, 1], moving[:, 2],
            s=0.8, c=colors, alpha=0.95, depthshade=False,
        )
        panel.plot(line[:, 0], line[:, 1], line[:, 2], color="#00dfff", linewidth=4)
        panel.quiver(
            center[0] - axis[0] * 0.18,
            center[1] - axis[1] * 0.18,
            center[2] - axis[2] * 0.18,
            axis[0],
            axis[1],
            axis[2],
            length=0.36,
            color="#00dfff",
            linewidth=3,
            arrow_length_ratio=0.12,
        )
        panel.set_xlim(low[0], high[0])
        panel.set_ylim(low[1], high[1])
        panel.set_zlim(low[2], high[2])
        panel.view_init(elev=elevation, azim=azimuth)
        panel.set_box_aspect((1, 1, 1))
        panel.set_title(title)
        panel.set_xlabel("world X (m)")
        panel.set_ylabel("world Y (m)")
        panel.set_zlabel("world Z (m)")
    figure.suptitle(
        "GT-plane-axis corrected open fusion: cabinet gray, drawer RGB, axis cyan",
        fontsize=14,
    )
    figure.tight_layout()
    output = args.run_root / "validation/extended_gtplane_open_fusion_axis_views.jpg"
    figure.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
