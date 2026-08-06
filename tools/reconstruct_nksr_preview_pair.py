#!/usr/bin/env python3
"""Create low-resolution NKSR meshes only for articulated visualization assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nksr
import numpy as np
import open3d as o3d
import torch
from pycg import vis


def reconstruct_preview(
    reconstructor: nksr.Reconstructor,
    device: torch.device,
    input_path: Path,
    output_path: Path,
    max_points: int,
    seed: int,
) -> dict:
    point_cloud = o3d.io.read_point_cloud(str(input_path))
    original_points = len(point_cloud.points)
    if original_points > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(original_points, max_points, replace=False))
        point_cloud = point_cloud.select_by_index(indices.tolist())
    xyz = torch.from_numpy(np.asarray(point_cloud.points).astype(np.float32)).to(device)
    normal = torch.from_numpy(np.asarray(point_cloud.normals).astype(np.float32)).to(device)
    color = torch.from_numpy(np.asarray(point_cloud.colors).astype(np.float32)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    field = reconstructor.reconstruct(xyz, normal, detail_level=0.0)
    field.set_texture_field(nksr.fields.PCNNField(xyz, color))
    mesh = field.extract_dual_mesh(mise_iter=0)
    output_mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), output_mesh):
        raise RuntimeError(f"Could not write {output_path}")
    report = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "purpose": "visualization/URDF preview only; not authoritative geometry",
        "original_input_points": original_points,
        "sampled_input_points": len(point_cloud.points),
        "sampling": "deterministic uniform without replacement",
        "seed": seed,
        "detail_level": 0.0,
        "mise_iter": 0,
        "mesh_vertices": len(output_mesh.vertices),
        "mesh_triangles": len(output_mesh.triangles),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del field, mesh, output_mesh, xyz, normal, color
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--drawer-max-points", type=int, default=100000)
    parser.add_argument("--static-max-points", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    device = torch.device(args.device)
    reconstructor = nksr.Reconstructor(device)
    jobs = [
        (
            "drawer",
            args.fusion_dir / "drawer_canonical_open_points.ply",
            args.output_dir / "drawer_open_preview_nksr.ply",
            args.drawer_max_points,
        ),
        (
            "static",
            args.fusion_dir / "cabinet_static_points.ply",
            args.output_dir / "cabinet_static_preview_nksr.ply",
            args.static_max_points,
        ),
    ]
    reports = {}
    for label, input_path, output_path, max_points in jobs:
        print(f"[NKSR preview] {label}", flush=True)
        reports[label] = reconstruct_preview(
            reconstructor, device, input_path, output_path, max_points, args.seed
        )
        print(json.dumps(reports[label], indent=2), flush=True)
    result = {
        "stage": "low-resolution NKSR articulated preview pair",
        "authoritative_geometry": "full detail_level=1.0, mise_iter=2 meshes remain separate",
        "jobs": reports,
    }
    report_path = args.output_dir / "nksr_preview_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
