#!/usr/bin/env python3
"""Fuse visible open-drawer geometry into one GT-world canonical state.

This is the extended (non-official) output.  It deliberately keeps the
official iTACO refinement untouched and uses its selected prismatic axis only.
Geometry and per-frame camera poses come from the HoloLens Long Throw GT data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def depth_timestamps(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def camera_xyz_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    return np.stack(
        [
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    )


def world_to_pixels(
    world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (world - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    u = np.rint(intrinsic[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + intrinsic[0, 2])
    v = np.rint(intrinsic[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + intrinsic[1, 2])
    return u.astype(np.int64), v.astype(np.int64), z


def robust_plane_basis(seed_points: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = seed_points - np.median(seed_points, axis=0)
    projected = centered - np.outer(centered @ axis, axis)
    covariance = projected.T @ projected / max(len(projected), 1)
    values, vectors = np.linalg.eigh(covariance)
    first = vectors[:, np.argmax(values)]
    first -= axis * np.dot(first, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def make_point_cloud(points: np.ndarray, colors: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--open-start-local", type=int, default=9)
    parser.add_argument("--open-end-local", type=int, default=18)
    parser.add_argument("--mask-dilate-pixels", type=int, default=26)
    parser.add_argument("--body-behind-front-m", type=float, default=0.45)
    parser.add_argument("--front-margin-m", type=float, default=0.07)
    parser.add_argument("--cross-margin-m", type=float, default=0.06)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    parser.add_argument("--static-moving-distance-m", type=float, default=0.03)
    args = parser.parse_args()

    interaction_dir = args.run_root / "inputs/interaction_forward_097_115"
    mapping = json.loads((interaction_dir / "frame_mapping.json").read_text())
    source_pinhole = args.source_root / "pinhole_projection"
    all_poses = load_odometry(source_pinhole / "odometry.log")
    timestamps = depth_timestamps(source_pinhole / "depth.txt")
    fx, fy, cx, cy = np.loadtxt(source_pinhole / "calibration.txt").reshape(-1)[:4]
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    corrected_depth_paths = sorted(
        (args.run_root / "preprocess/gt_depth_interaction_reprojected/npy").glob("*.npy")
    )
    dynamic_masks = np.load(args.run_root / "official/preprocess/gt_dynamic_masks.npz")["a"].astype(bool)
    hand_paths = sorted((args.run_root / "official/preprocess/hand_mask").glob("*.npy"))
    axis = np.load(
        args.run_root
        / "official/prediction/refinement/monst3r/chamfer/0/prismatic/joint_axis.npy"
    ).astype(np.float64)
    axis /= np.linalg.norm(axis)

    count = len(mapping)
    if not (len(corrected_depth_paths) == len(dynamic_masks) == len(hand_paths) == count):
        raise ValueError("Interaction depth/mask/hand/frame counts do not match")

    seed_world_per_frame = []
    seed_centers = []
    for local_index, item in enumerate(mapping):
        depth = np.load(corrected_depth_paths[local_index]).astype(np.float64)
        valid = dynamic_masks[local_index] & np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
        camera_xyz = camera_xyz_from_depth(depth, intrinsic)
        pose = all_poses[int(item["source_index"])]
        seed_world = camera_xyz[valid] @ pose[:3, :3].T + pose[:3, 3]
        if len(seed_world) < 100:
            raise RuntimeError(f"Too few moving seed points in local frame {local_index}")
        seed_world_per_frame.append(seed_world)
        seed_centers.append(np.median(seed_world, axis=0))
    seed_centers = np.stack(seed_centers)

    endpoint_displacement = seed_centers[-3:].mean(axis=0) - seed_centers[:3].mean(axis=0)
    opening_axis = axis * (1.0 if np.dot(endpoint_displacement, axis) >= 0 else -1.0)
    open_indices = list(range(args.open_start_local, args.open_end_local + 1))
    reference_center = np.median(seed_centers[open_indices[-3:]], axis=0)
    reference_state = float(np.dot(reference_center, opening_axis))

    canonical_seed = []
    for index in open_indices:
        state = float(np.dot(seed_centers[index], opening_axis))
        canonical_seed.append(
            seed_world_per_frame[index] + opening_axis * (reference_state - state)
        )
    canonical_seed_all = np.concatenate(canonical_seed)
    basis_u, basis_v = robust_plane_basis(canonical_seed_all, opening_axis)

    output_dir = args.run_root / "extended/prismatic_open_fusion"
    per_frame_dir = output_dir / "per_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((args.mask_dilate_pixels * 2 + 1,) * 2, dtype=np.uint8)
    all_points = []
    all_colors = []
    all_normals = []
    frame_records = []

    for local_index in open_indices:
        item = mapping[local_index]
        source_index = int(item["source_index"])
        timestamp = timestamps[source_index]
        source_ply = args.source_root / "Depth Long Throw" / f"{timestamp}.ply"
        source_pcd = o3d.io.read_point_cloud(str(source_ply))
        points = np.asarray(source_pcd.points)
        colors = np.asarray(source_pcd.colors)
        normals = np.asarray(source_pcd.normals)
        if not len(points):
            raise RuntimeError(f"Empty source PLY: {source_ply}")

        hand = np.load(hand_paths[local_index]).squeeze().astype(bool)
        roi = cv2.dilate(dynamic_masks[local_index].astype(np.uint8), kernel) > 0
        u, v, camera_z = world_to_pixels(points, all_poses[source_index], intrinsic)
        height, width = roi.shape
        inside = (
            np.isfinite(points).all(axis=1)
            & (camera_z > 0.2)
            & (camera_z < 4.0)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        projected_roi = np.zeros(len(points), dtype=bool)
        projected_not_hand = np.zeros(len(points), dtype=bool)
        projected_roi[inside] = roi[v[inside], u[inside]]
        projected_not_hand[inside] = ~hand[v[inside], u[inside]]

        center = seed_centers[local_index]
        seed_rel = seed_world_per_frame[local_index] - center
        seed_u = seed_rel @ basis_u
        seed_v = seed_rel @ basis_v
        u_lo, u_hi = np.quantile(seed_u, [0.01, 0.99]) + np.array(
            [-args.cross_margin_m, args.cross_margin_m]
        )
        v_lo, v_hi = np.quantile(seed_v, [0.01, 0.99]) + np.array(
            [-args.cross_margin_m, args.cross_margin_m]
        )
        rel = points - center
        along = rel @ opening_axis
        across_u = rel @ basis_u
        across_v = rel @ basis_v
        geometric_body = (
            (along >= -args.body_behind_front_m)
            & (along <= args.front_margin_m)
            & (across_u >= u_lo)
            & (across_u <= u_hi)
            & (across_v >= v_lo)
            & (across_v <= v_hi)
        )
        selected = projected_roi & projected_not_hand & geometric_body
        if selected.sum() < 100:
            raise RuntimeError(f"Too few expanded drawer points in local frame {local_index}")

        state = float(np.dot(center, opening_axis))
        canonical_translation = opening_axis * (reference_state - state)
        frame_points = points[selected] + canonical_translation
        frame_colors = colors[selected]
        frame_normals = normals[selected]
        all_points.append(frame_points)
        all_colors.append(frame_colors)
        all_normals.append(frame_normals)
        o3d.io.write_point_cloud(
            str(per_frame_dir / f"{local_index:06d}_source{source_index:03d}.ply"),
            make_point_cloud(frame_points, frame_colors, frame_normals),
        )
        frame_records.append(
            {
                "local_index": local_index,
                "source_index": source_index,
                "source_ply": str(source_ply.resolve()),
                "selected_points": int(selected.sum()),
                "seed_points": int(len(seed_world_per_frame[local_index])),
                "front_state_m": state,
                "canonical_translation_m": canonical_translation.tolist(),
                "front_seed_bounds_u_m": [float(u_lo), float(u_hi)],
                "front_seed_bounds_v_m": [float(v_lo), float(v_hi)],
            }
        )

    fused_raw = make_point_cloud(
        np.concatenate(all_points),
        np.concatenate(all_colors),
        np.concatenate(all_normals),
    )
    fused = fused_raw.voxel_down_sample(args.voxel_size_m)
    fused, kept = fused.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    fused.orient_normals_consistent_tangent_plane(30)
    fused_path = output_dir / "open_state_drawer_fused.ply"
    o3d.io.write_point_cloud(str(fused_path), fused)

    # Build an open-state full point cloud by removing the closed drawer region
    # from the official static surface and inserting the fused open drawer.
    surface = o3d.io.read_point_cloud(str(args.run_root / "official/view/surface/surface.ply"))
    closed_seed = make_point_cloud(
        seed_world_per_frame[0],
        np.zeros((len(seed_world_per_frame[0]), 3)),
        np.zeros((len(seed_world_per_frame[0]), 3)),
    )
    distances = np.asarray(surface.compute_point_cloud_distance(closed_seed))
    static_indices = np.flatnonzero(distances >= args.static_moving_distance_m)
    static_open = surface.select_by_index(static_indices.tolist())
    combined = static_open + fused
    combined = combined.voxel_down_sample(args.voxel_size_m)
    combined_path = output_dir / "open_state_full_surface_points.ply"
    o3d.io.write_point_cloud(str(combined_path), combined)
    static_path = output_dir / "open_state_static_surface_points.ply"
    o3d.io.write_point_cloud(str(static_path), static_open)

    report = {
        "output_kind": "extended_non_official_visible_open_drawer_fusion",
        "official_refinement_modified": False,
        "selected_joint_type": "prismatic",
        "refined_axis_raw": axis.tolist(),
        "opening_axis_world": opening_axis.tolist(),
        "endpoint_seed_displacement_world_m": endpoint_displacement.tolist(),
        "endpoint_displacement_along_opening_axis_m": float(
            np.dot(endpoint_displacement, opening_axis)
        ),
        "canonical_state": "median of final three interaction frames (open)",
        "open_local_frame_range": [args.open_start_local, args.open_end_local],
        "open_source_frame_range": [
            int(mapping[args.open_start_local]["source_index"]),
            int(mapping[args.open_end_local]["source_index"]),
        ],
        "selection": {
            "source": "GT-depth/GT-camera moving seeds from AutoSeg UIDs 3 and 10",
            "mask_dilate_pixels": args.mask_dilate_pixels,
            "body_behind_front_m": args.body_behind_front_m,
            "front_margin_m": args.front_margin_m,
            "cross_margin_m": args.cross_margin_m,
        },
        "voxel_size_m": args.voxel_size_m,
        "per_frame": frame_records,
        "raw_fused_points": int(sum(len(points) for points in all_points)),
        "fused_points_after_filter": int(len(fused.points)),
        "static_points_after_closed_drawer_removal": int(len(static_open.points)),
        "combined_open_state_points": int(len(combined.points)),
        "outputs": {
            "moving_point_cloud": str(fused_path),
            "static_point_cloud": str(static_path),
            "combined_point_cloud": str(combined_path),
        },
    }
    report_path = output_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
