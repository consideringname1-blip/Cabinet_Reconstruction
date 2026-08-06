#!/usr/bin/env python3
"""Remove drawer point outliers outside the fitted canonical 3D drawer box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def crop(
    cloud: o3d.geometry.PointCloud,
    origin: np.ndarray,
    basis: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> o3d.geometry.PointCloud:
    coordinates = (np.asarray(cloud.points) - origin) @ basis
    keep = np.all((coordinates >= lower) & (coordinates <= upper), axis=1)
    return cloud.select_by_index(np.flatnonzero(keep).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    args = parser.parse_args()

    report = json.loads((args.input_dir / "fusion_report.json").read_text())
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
    lower_closed = np.asarray(geometry["lower"])
    upper_closed = np.asarray(geometry["upper"])
    travel = float(report["motion"]["travel_m"])
    lower_open = lower_closed + np.asarray([travel, 0.0, 0.0])
    upper_open = upper_closed + np.asarray([travel, 0.0, 0.0])

    static = o3d.io.read_point_cloud(str(args.input_dir / "cabinet_static_points.ply"))
    drawer_closed_source = o3d.io.read_point_cloud(
        str(args.input_dir / "drawer_canonical_closed_points.ply")
    )
    drawer_open_source = o3d.io.read_point_cloud(
        str(args.input_dir / "drawer_canonical_open_points.ply")
    )
    drawer_closed = crop(
        drawer_closed_source, origin, basis, lower_closed, upper_closed
    )
    drawer_open = crop(drawer_open_source, origin, basis, lower_open, upper_open)
    combined_open = (static + drawer_open).voxel_down_sample(args.voxel_size_m)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "drawer_closed": args.output_dir / "drawer_canonical_closed_points.ply",
        "drawer_open": args.output_dir / "drawer_canonical_open_points.ply",
        "static": args.output_dir / "cabinet_static_points.ply",
        "combined_open": args.output_dir / "combined_open_points.ply",
    }
    o3d.io.write_point_cloud(str(paths["drawer_closed"]), drawer_closed)
    o3d.io.write_point_cloud(str(paths["drawer_open"]), drawer_open)
    o3d.io.write_point_cloud(str(paths["static"]), static)
    o3d.io.write_point_cloud(str(paths["combined_open"]), combined_open)

    stage = {
        "stage": "extended_non_official_drawer_box_outlier_crop",
        "source": str(args.input_dir.resolve()),
        "closed_bounds_axis_u_v_m": [lower_closed.tolist(), upper_closed.tolist()],
        "open_bounds_axis_u_v_m": [lower_open.tolist(), upper_open.tolist()],
        "points": {
            "drawer_closed_before": len(drawer_closed_source.points),
            "drawer_closed_after": len(drawer_closed.points),
            "drawer_open_before": len(drawer_open_source.points),
            "drawer_open_after": len(drawer_open.points),
            "static_unchanged": len(static.points),
            "combined_open": len(combined_open.points),
        },
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
    }
    (args.output_dir / "drawer_crop_report.json").write_text(
        json.dumps(stage, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fusion_report.json").write_text(
        json.dumps(
            {
                **report,
                "post_drawer_box_crop": stage,
                "points": {
                    **report["points"],
                    "drawer_closed": len(drawer_closed.points),
                    "drawer_open": len(drawer_open.points),
                    "combined_open": len(combined_open.points),
                },
                "outputs": {key: str(path.resolve()) for key, path in paths.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stage, indent=2))


if __name__ == "__main__":
    main()
