#!/usr/bin/env python3
"""Summarize official and extended NKSR meshes without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def summarize(path: Path) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices):
        low = vertices.min(axis=0)
        high = vertices.max(axis=0)
    else:
        low = high = np.zeros(3)
    report = {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "has_vertex_colors": bool(mesh.has_vertex_colors()),
        "has_vertex_normals": bool(mesh.has_vertex_normals()),
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "bounds_min_m": low.tolist(),
        "bounds_max_m": high.tolist(),
        "extent_m": (high - low).tolist(),
        "nonempty": bool(len(vertices) and len(triangles)),
    }
    del mesh, vertices, triangles
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    official = (
        args.run_root / "official/prediction/refinement/monst3r/chamfer/0"
    )
    extended = args.run_root / "extended/prismatic_open_fusion_gt_plane_axis"
    paths = [
        official / "surface_mesh.ply",
        official / "revolute/moving_mesh.ply",
        official / "revolute/static_mesh.ply",
        official / "prismatic/moving_mesh.ply",
        official / "prismatic/static_mesh.ply",
        extended / "open_state_drawer_nksr_mesh.ply",
        extended / "open_state_static_nksr_mesh.ply",
        extended / "open_state_full_nksr_mesh.ply",
    ]
    reports = []
    for path in paths:
        print(f"[mesh] {path}", flush=True)
        reports.append(summarize(path))
    result = {"all_meshes_valid": all(item["nonempty"] and item["finite_vertices"] for item in reports), "meshes": reports}
    output = args.run_root / "validation/nksr_mesh_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
