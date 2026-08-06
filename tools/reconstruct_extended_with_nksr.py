#!/usr/bin/env python3
"""Run NKSR on the extended open-state point-cloud outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nksr
import numpy as np
import open3d as o3d
import torch
from pycg import vis


def reconstruct_one(
    reconstructor: nksr.Reconstructor,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict:
    point_cloud = o3d.io.read_point_cloud(str(input_path))
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=80)
        )
        point_cloud.orient_normals_consistent_tangent_plane(30)
    xyz_np = np.asarray(point_cloud.points).astype(np.float32)
    normal_np = np.asarray(point_cloud.normals).astype(np.float32)
    color_np = np.asarray(point_cloud.colors).astype(np.float32)
    xyz = torch.from_numpy(xyz_np).to(device)
    normal = torch.from_numpy(normal_np).to(device)
    color = torch.from_numpy(color_np).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    field = reconstructor.reconstruct(xyz, normal, detail_level=1.0)
    field.set_texture_field(nksr.fields.PCNNField(xyz, color))
    mesh = field.extract_dual_mesh(mise_iter=2)
    open3d_mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), open3d_mesh):
        raise RuntimeError(f"Could not write {output_path}")
    sampled_path = output_path.with_name(output_path.stem + "_sampled_10000.ply")
    sampled = open3d_mesh.sample_points_uniformly(number_of_points=10000)
    o3d.io.write_point_cloud(str(sampled_path), sampled)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "sampled_output": str(sampled_path),
        "input_points": int(len(xyz_np)),
        "mesh_vertices": int(len(open3d_mesh.vertices)),
        "mesh_triangles": int(len(open3d_mesh.triangles)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "detail_level": 1.0,
        "mise_iter": 2,
    }
    del field, mesh, open3d_mesh, xyz, normal, color
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    reconstructor = nksr.Reconstructor(device)
    fusion_dir = args.run_root / "extended/prismatic_open_fusion"
    jobs = [
        (
            fusion_dir / "open_state_drawer_fused.ply",
            fusion_dir / "open_state_drawer_nksr_mesh.ply",
        ),
        (
            fusion_dir / "open_state_static_surface_points.ply",
            fusion_dir / "open_state_static_nksr_mesh.ply",
        ),
        (
            fusion_dir / "open_state_full_surface_points.ply",
            fusion_dir / "open_state_full_nksr_mesh.ply",
        ),
    ]
    reports = []
    for input_path, output_path in jobs:
        print(f"[NKSR] {input_path.name} -> {output_path.name}", flush=True)
        reports.append(reconstruct_one(reconstructor, input_path, output_path, device))
        print(json.dumps(reports[-1], indent=2), flush=True)
    result = {
        "stage": "extended open-state NKSR reconstruction",
        "algorithm_parameters": "same detail_level=1.0 and mise_iter=2 as iTACO extract_mesh.py",
        "jobs": reports,
    }
    report_path = fusion_dir / "nksr_reconstruction_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
