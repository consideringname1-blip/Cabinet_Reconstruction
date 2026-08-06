#!/usr/bin/env python3
"""Crop only the static volume in the fitted drawer basis; preserve the drawer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis-range", type=float, nargs=2, default=[-0.55, 0.20])
    parser.add_argument("--u-range", type=float, nargs=2, default=[-0.32, 0.24])
    parser.add_argument("--v-range", type=float, nargs=2, default=[-0.25, 0.25])
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    args = parser.parse_args()

    source_report = json.loads((args.input_dir / "fusion_report.json").read_text())
    geometry = source_report["geometry_bounds_drawer_axis_u_v_m"]
    origin = np.asarray(geometry["origin_world"])
    basis = np.stack(
        [
            np.asarray(source_report["motion"]["opening_axis_world"]),
            np.asarray(geometry["basis_u_world"]),
            np.asarray(geometry["basis_v_world"]),
        ],
        axis=1,
    )
    lower = np.asarray([args.axis_range[0], args.u_range[0], args.v_range[0]])
    upper = np.asarray([args.axis_range[1], args.u_range[1], args.v_range[1]])

    static = o3d.io.read_point_cloud(str(args.input_dir / "cabinet_static_points.ply"))
    static_points = np.asarray(static.points)
    coords = (static_points - origin) @ basis
    keep = np.all((coords >= lower) & (coords <= upper), axis=1)
    cropped_static = static.select_by_index(np.flatnonzero(keep).tolist())
    if len(cropped_static.points) >= 100:
        cropped_static, _ = cropped_static.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=2.5
        )

    drawer_closed = o3d.io.read_point_cloud(
        str(args.input_dir / "drawer_canonical_closed_points.ply")
    )
    drawer_open = o3d.io.read_point_cloud(
        str(args.input_dir / "drawer_canonical_open_points.ply")
    )
    combined_open = (cropped_static + drawer_open).voxel_down_sample(args.voxel_size_m)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "drawer_closed": args.output_dir / "drawer_canonical_closed_points.ply",
        "drawer_open": args.output_dir / "drawer_canonical_open_points.ply",
        "static": args.output_dir / "cabinet_static_points.ply",
        "combined_open": args.output_dir / "combined_open_points.ply",
    }
    o3d.io.write_point_cloud(str(outputs["drawer_closed"]), drawer_closed)
    o3d.io.write_point_cloud(str(outputs["drawer_open"]), drawer_open)
    o3d.io.write_point_cloud(str(outputs["static"]), cropped_static)
    o3d.io.write_point_cloud(str(outputs["combined_open"]), combined_open)

    report = {
        "stage": "extended_non_official_static_cabinet_crop",
        "source_fusion_report": str((args.input_dir / "fusion_report.json").resolve()),
        "reason": (
            "Remove floor and adjacent room surfaces admitted by the broad initial "
            "static cabinet crop; moving drawer geometry is preserved unchanged."
        ),
        "coordinate_order": ["opening_axis", "basis_u", "basis_v"],
        "static_crop_lower_m": lower.tolist(),
        "static_crop_upper_m": upper.tolist(),
        "points": {
            "static_before": len(static.points),
            "static_after": len(cropped_static.points),
            "drawer_closed_unchanged": len(drawer_closed.points),
            "drawer_open_unchanged": len(drawer_open.points),
            "combined_open": len(combined_open.points),
        },
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    (args.output_dir / "crop_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fusion_report.json").write_text(
        json.dumps(
            {
                **source_report,
                "post_static_crop": report,
                "points": {
                    **source_report["points"],
                    "static": len(cropped_static.points),
                    "combined_open": len(combined_open.points),
                },
                "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
