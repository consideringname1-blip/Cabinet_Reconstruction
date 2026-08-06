#!/usr/bin/env python3
"""Plot static and moving point volumes in the fitted drawer coordinate basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def sample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), maximum, replace=False)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=250000)
    args = parser.parse_args()

    report = json.loads((args.fusion_dir / "fusion_report.json").read_text())
    geometry = report["geometry_bounds_drawer_axis_u_v_m"]
    origin = np.asarray(geometry["origin_world"])
    basis = np.stack(
        [
            np.asarray(report["motion"]["opening_axis_world"]),
            np.asarray(geometry["basis_u_world"]),
            np.asarray(geometry["basis_v_world"]),
        ],
        axis=1,
    )

    static = np.asarray(
        o3d.io.read_point_cloud(str(args.fusion_dir / "cabinet_static_points.ply")).points
    )
    drawer = np.asarray(
        o3d.io.read_point_cloud(
            str(args.fusion_dir / "drawer_canonical_open_points.ply")
        ).points
    )
    static_q = sample((static - origin) @ basis, args.max_points, 20260729)
    drawer_q = sample((drawer - origin) @ basis, args.max_points, 20260729)

    views = [
        (1, 2, "u", "v", "front: u-v"),
        (0, 2, "axis", "v", "side: axis-v"),
        (0, 1, "axis", "u", "top: axis-u"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, (x, y, xlabel, ylabel, title) in zip(axes, views):
        ax.scatter(static_q[:, x], static_q[:, y], s=0.08, c="#7891a8", alpha=0.25)
        ax.scatter(drawer_q[:, x], drawer_q[:, y], s=0.18, c="#f06b35", alpha=0.55)
        ax.set_xlabel(f"{xlabel} (m)")
        ax.set_ylabel(f"{ylabel} (m)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
    fig.suptitle("Blue=static volume, orange=open drawer volume")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
