#!/usr/bin/env python3
"""Recanonicalize extended drawer geometry with the GT-plane reference axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def make_pcd(points: np.ndarray, colors: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def masked_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    pose: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    valid = mask & np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
    camera = np.stack(
        [
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    )
    return camera[valid] @ pose[:3, :3].T + pose[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    parser.add_argument("--static-moving-distance-m", type=float, default=0.03)
    args = parser.parse_args()

    source_dir = args.run_root / "extended/prismatic_open_fusion"
    source_report = json.loads((source_dir / "fusion_report.json").read_text())
    axis_report = json.loads(
        (args.run_root / "validation/axis_official_vs_gt_plane_reference.json").read_text()
    )
    axis = np.asarray(axis_report["gt_plane_normal_reference_world"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    open_indices = list(
        range(
            int(source_report["open_local_frame_range"][0]),
            int(source_report["open_local_frame_range"][1]) + 1,
        )
    )
    frame_record = {int(record["local_index"]): record for record in source_report["per_frame"]}

    metadata = json.loads((args.run_root / "official/view/metadata.json").read_text())
    intrinsic = np.asarray(metadata["K"], dtype=np.float64).reshape(3, 3).T
    poses = np.load(args.run_root / "official/preprocess/gt_cam2world_all.npy")
    masks = np.load(args.run_root / "official/preprocess/gt_dynamic_masks.npz")["a"].astype(bool)
    depth_paths = sorted(
        (args.run_root / "preprocess/gt_depth_interaction_reprojected/npy").glob("*.npy")
    )
    centers = []
    seeds = []
    for index, depth_path in enumerate(depth_paths):
        points = masked_world_points(np.load(depth_path), masks[index], poses[index], intrinsic)
        seeds.append(points)
        centers.append(np.median(points, axis=0))
    centers = np.stack(centers)
    reference_state = float(np.median(centers[open_indices[-3:]] @ axis))

    output_dir = args.run_root / "extended/prismatic_open_fusion_gt_plane_axis"
    per_frame_dir = output_dir / "per_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    all_points = []
    all_colors = []
    all_normals = []
    records = []
    for index in open_indices:
        old_path = next((source_dir / "per_frame").glob(f"{index:06d}_source*.ply"))
        old_pcd = o3d.io.read_point_cloud(str(old_path))
        old_points = np.asarray(old_pcd.points)
        old_colors = np.asarray(old_pcd.colors)
        old_normals = np.asarray(old_pcd.normals)
        old_translation = np.asarray(frame_record[index]["canonical_translation_m"])
        raw_points = old_points - old_translation
        state = float(np.dot(centers[index], axis))
        translation = axis * (reference_state - state)
        points = raw_points + translation
        all_points.append(points)
        all_colors.append(old_colors)
        all_normals.append(old_normals)
        new_path = per_frame_dir / old_path.name
        o3d.io.write_point_cloud(str(new_path), make_pcd(points, old_colors, old_normals))
        records.append(
            {
                "local_index": index,
                "source_index": int(frame_record[index]["source_index"]),
                "input_expanded_points": int(len(points)),
                "front_state_m": state,
                "canonical_translation_m": translation.tolist(),
            }
        )

    raw = make_pcd(
        np.concatenate(all_points),
        np.concatenate(all_colors),
        np.concatenate(all_normals),
    )
    fused = raw.voxel_down_sample(args.voxel_size_m)
    fused, _ = fused.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    fused.orient_normals_consistent_tangent_plane(30)
    moving_path = output_dir / "open_state_drawer_fused.ply"
    o3d.io.write_point_cloud(str(moving_path), fused)

    surface = o3d.io.read_point_cloud(str(args.run_root / "official/view/surface/surface.ply"))
    closed_seed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seeds[0]))
    distances = np.asarray(surface.compute_point_cloud_distance(closed_seed))
    static = surface.select_by_index(
        np.flatnonzero(distances >= args.static_moving_distance_m).tolist()
    )
    static_path = output_dir / "open_state_static_surface_points.ply"
    o3d.io.write_point_cloud(str(static_path), static)
    combined = (static + fused).voxel_down_sample(args.voxel_size_m)
    combined_path = output_dir / "open_state_full_surface_points.ply"
    o3d.io.write_point_cloud(str(combined_path), combined)

    report = {
        "output_kind": "extended_non_official_gt_plane_axis_corrected_open_drawer_fusion",
        "official_refinement_modified": False,
        "axis_source": axis_report["reference_kind"],
        "axis_world": axis.tolist(),
        "angle_from_official_refined_axis_deg": axis_report["absolute_axis_angle_deg"],
        "canonical_state": "median of final three interaction frames (open)",
        "open_local_frame_range": source_report["open_local_frame_range"],
        "open_source_frame_range": source_report["open_source_frame_range"],
        "source_expansion_parameters": source_report["selection"],
        "voxel_size_m": args.voxel_size_m,
        "per_frame": records,
        "raw_fused_points": int(sum(len(points) for points in all_points)),
        "fused_points_after_filter": int(len(fused.points)),
        "static_points_after_closed_drawer_removal": int(len(static.points)),
        "combined_open_state_points": int(len(combined.points)),
        "outputs": {
            "moving_point_cloud": str(moving_path),
            "static_point_cloud": str(static_path),
            "combined_point_cloud": str(combined_path),
        },
    }
    report_path = output_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
