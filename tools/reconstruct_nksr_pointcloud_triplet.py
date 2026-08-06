#!/usr/bin/env python3
"""Reconstruct moving, static, and combined point clouds with NKSR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nksr
import numpy as np
import open3d as o3d
import torch
from pycg import vis


def reconstruct(
    reconstructor: nksr.Reconstructor,
    device: torch.device,
    input_path: Path,
    output_path: Path,
    max_input_points: int | None,
    sample_seed: int,
    detail_level: float,
    mise_iter: int,
) -> dict:
    pcd = o3d.io.read_point_cloud(str(input_path))
    original_input_points = len(pcd.points)
    if max_input_points is not None and original_input_points > max_input_points:
        rng = np.random.default_rng(sample_seed)
        indices = np.sort(
            rng.choice(original_input_points, size=max_input_points, replace=False)
        )
        pcd = pcd.select_by_index(indices.tolist())
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=80)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
    xyz = torch.from_numpy(np.asarray(pcd.points).astype(np.float32)).to(device)
    normal = torch.from_numpy(np.asarray(pcd.normals).astype(np.float32)).to(device)
    color = torch.from_numpy(np.asarray(pcd.colors).astype(np.float32)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    field = reconstructor.reconstruct(xyz, normal, detail_level=detail_level)
    field.set_texture_field(nksr.fields.PCNNField(xyz, color))
    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    o3d_mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)
    o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "original_input_points": int(original_input_points),
        "input_points": int(len(pcd.points)),
        "sampling": (
            {
                "kind": "deterministic_uniform_without_replacement",
                "seed": int(sample_seed),
                "max_input_points": int(max_input_points),
            }
            if original_input_points != len(pcd.points)
            else None
        ),
        "mesh_vertices": int(len(o3d_mesh.vertices)),
        "mesh_triangles": int(len(o3d_mesh.triangles)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "detail_level": detail_level,
        "mise_iter": mise_iter,
    }
    del field, mesh, o3d_mesh, xyz, normal, color
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profile",
        choices=["legacy_open_triplet", "iphone_dual_volume", "hololens_dual_volume_closed"],
        default="legacy_open_triplet",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["drawer", "static", "combined"],
        default=["drawer", "static", "combined"],
    )
    parser.add_argument("--max-input-points", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260729)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--report-name", default="nksr_reconstruction_report.json")
    parser.add_argument("--detail-level", type=float, default=1.0)
    parser.add_argument("--mise-iter", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    reconstructor = nksr.Reconstructor(device)
    if args.profile == "iphone_dual_volume":
        all_jobs = {
            "drawer": ("drawer_canonical_open_points.ply", "drawer_canonical_open_nksr_mesh.ply"),
            "static": ("cabinet_static_points.ply", "cabinet_static_nksr_mesh.ply"),
            "combined": ("combined_open_points.ply", "combined_open_nksr_mesh.ply"),
        }
        jobs = [all_jobs[name] for name in args.components]
    elif args.profile == "hololens_dual_volume_closed":
        all_jobs = {
            "drawer": (
                "drawer_canonical_closed_points.ply",
                "drawer_canonical_closed_nksr_mesh.ply",
            ),
            "static": ("cabinet_static_points.ply", "cabinet_static_nksr_mesh.ply"),
            "combined": ("combined_open_points.ply", "combined_open_nksr_mesh.ply"),
        }
        jobs = [all_jobs[name] for name in args.components]
    else:
        jobs = [
            ("open_state_drawer_fused.ply", "open_state_drawer_nksr_mesh.ply"),
            ("open_state_static_surface_points.ply", "open_state_static_nksr_mesh.ply"),
            ("open_state_full_surface_points.ply", "open_state_full_nksr_mesh.ply"),
        ]
    reports = []
    for input_name, output_name in jobs:
        output_path = args.fusion_dir / output_name
        if args.skip_existing and output_path.exists():
            print(f"[NKSR] skip existing {output_name}", flush=True)
            continue
        print(f"[NKSR] {input_name}", flush=True)
        report = reconstruct(
            reconstructor,
            device,
            args.fusion_dir / input_name,
            output_path,
            args.max_input_points,
            args.sample_seed,
            args.detail_level,
            args.mise_iter,
        )
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    final = {
        "stage": "extended point-cloud triplet NKSR reconstruction",
        "profile": args.profile,
        "algorithm_parameters": (
            f"detail_level={args.detail_level}, mise_iter={args.mise_iter}; "
            "iTACO official-compatible values are 1.0 and 2"
        ),
        "max_input_points": args.max_input_points,
        "sample_seed": args.sample_seed,
        "jobs": reports,
    }
    report_path = args.fusion_dir / args.report_name
    report_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
