#!/usr/bin/env python3
"""Fit a revolute axis from HoloLens-world moving-mask centroids.

This is a validation/calibration extension around iTACO, not an official
iTACO stage.  It uses only the selected AutoSeg moving mask, reprojected
HoloLens depth, and recorded camera-to-world poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def camera_xyz(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
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


def fit_circle_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:2]
    radius = float(np.sqrt(max(solution[2] + center @ center, 0.0)))
    return center, radius


def line_distance(point: np.ndarray, line_point: np.ndarray, axis: np.ndarray) -> float:
    delta = point - line_point
    return float(np.linalg.norm(delta - axis * np.dot(delta, axis)))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trim-ends", type=int, default=2)
    args = parser.parse_args()

    preprocess = args.run_root / "official/preprocess"
    depths = sorted((args.run_root / "preprocess/gt_depth_interaction_reprojected/npy").glob("*.npy"))
    masks = np.load(preprocess / "gt_dynamic_masks.npz")["a"].astype(bool)
    poses = np.load(preprocess / "gt_cam2world_all.npy")
    metadata = json.loads((args.run_root / "official/view/metadata.json").read_text())
    intrinsic = np.asarray(metadata["K"], dtype=np.float64).reshape(3, 3).T
    if not (len(depths) == len(masks) == len(poses)):
        raise ValueError("depth/mask/pose counts differ")

    centers = []
    plane_normals = []
    seed_clouds = []
    records = []
    for index, (depth_path, mask, pose) in enumerate(zip(depths, masks, poses)):
        depth = np.load(depth_path).astype(np.float64)
        valid = mask & np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
        xyz = camera_xyz(depth, intrinsic)
        world = xyz[valid] @ pose[:3, :3].T + pose[:3, 3]
        if len(world) < 100:
            raise RuntimeError(f"frame {index}: only {len(world)} moving points")
        center = np.median(world, axis=0)
        covariance = np.cov((world - center).T)
        _, frame_vectors = np.linalg.eigh(covariance)
        normal = frame_vectors[:, 0]
        if plane_normals and np.dot(normal, plane_normals[-1]) < 0:
            normal = -normal
        centers.append(center)
        plane_normals.append(normal)
        seed_clouds.append(world)
        records.append({"local_index": index, "points": int(len(world)), "centroid_world_m": center.tolist()})
    centers = np.stack(centers)
    plane_normals = np.stack(plane_normals)

    trim = args.trim_ends
    fit_centers = centers[trim : len(centers) - trim] if trim else centers
    fit_normals = plane_normals[trim : len(plane_normals) - trim] if trim else plane_normals
    normal_covariance = fit_normals.T @ fit_normals
    _, normal_vectors = np.linalg.eigh(normal_covariance)
    axis = normal_vectors[:, 0]
    origin = fit_centers.mean(axis=0)
    # HoloLens world is gravity-aligned in this sequence; choose positive y for
    # stable reporting, without constraining the fit to gravity.
    if axis[1] < 0:
        axis = -axis
    basis_u = fit_normals[0] - axis * np.dot(fit_normals[0], axis)
    basis_u -= axis * np.dot(basis_u, axis)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    projected = np.column_stack([(fit_centers - origin) @ basis_u, (fit_centers - origin) @ basis_v])
    center_2d, radius = fit_circle_2d(projected)
    circle_axis_point = origin + basis_u * center_2d[0] + basis_v * center_2d[1]
    plane_offsets = np.sum(fit_normals * fit_centers, axis=1)
    hinge_design = np.vstack([fit_normals, axis])
    hinge_target = np.concatenate([plane_offsets, [np.median(fit_centers @ axis)]])
    axis_point, *_ = np.linalg.lstsq(hinge_design, hinge_target, rcond=None)

    all_projected = np.column_stack([(centers - axis_point) @ basis_u, (centers - axis_point) @ basis_v])
    normal_projected = np.column_stack([plane_normals @ basis_u, plane_normals @ basis_v])
    raw_angles = np.arctan2(normal_projected[:, 1], normal_projected[:, 0])
    angles = np.unwrap(raw_angles)
    if angles[-1] < angles[0]:
        axis = -axis
        basis_v = -basis_v
        angles = np.unwrap(np.arctan2(-normal_projected[:, 1], normal_projected[:, 0]))
    angle_from_closed = angles - np.median(angles[:3])
    circle_projected = np.column_stack([(centers - circle_axis_point) @ basis_u, (centers - circle_axis_point) @ basis_v])
    radial = np.linalg.norm(circle_projected, axis=1)
    residual = radial - radius
    plane_residual = (centers - axis_point) @ axis

    comparisons = {}
    candidates = {
        "official_joint_camera": args.run_root
        / "official/prediction/refinement/monst3r/chamfer/0/revolute",
        "gt_fixed_camera": args.run_root
        / "official/prediction/refinement_gt_fixed_camera/monst3r/chamfer/0/revolute",
    }
    for name, directory in candidates.items():
        axis_path = directory / "joint_axis.npy"
        pos_path = directory / "joint_pos.npy"
        if axis_path.exists() and pos_path.exists():
            candidate_axis = np.load(axis_path).astype(np.float64)
            candidate_axis /= np.linalg.norm(candidate_axis)
            candidate_pos = np.load(pos_path).astype(np.float64)
            comparisons[name] = {
                "axis": candidate_axis.tolist(),
                "position": candidate_pos.tolist(),
                "axis_acute_angle_deg": angle_deg(axis, candidate_axis),
                "axis_line_distance_m": line_distance(candidate_pos, axis_point, axis),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "geometric_joint_axis.npy", axis)
    np.save(args.output_dir / "geometric_joint_pos.npy", axis_point)
    np.save(args.output_dir / "interaction_angles_rad.npy", angle_from_closed)
    np.save(args.output_dir / "moving_centroids_world.npy", centers)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(seed_clouds))
    o3d.io.write_point_cloud(str(args.output_dir / "interaction_moving_seed_points.ply"), cloud)
    report = {
        "method": "per-frame door-plane intersection for axis, plus centroid-circle metric check",
        "role": "non-official geometric validation and canonicalization extension",
        "camera_source": "recorded HoloLens camera-to-world, held fixed",
        "trimmed_endpoint_frames_for_fit": trim,
        "joint_axis_world": axis.tolist(),
        "joint_position_world_m": axis_point.tolist(),
        "centroid_circle_axis_position_world_m": circle_axis_point.tolist(),
        "circle_radius_m": radius,
        "opening_angle_rad": float(angle_from_closed[-1] - angle_from_closed[0]),
        "opening_angle_deg": float(np.degrees(angle_from_closed[-1] - angle_from_closed[0])),
        "circle_residual_rmse_m": float(np.sqrt(np.mean(residual * residual))),
        "plane_residual_rmse_m": float(np.sqrt(np.mean(plane_residual * plane_residual))),
        "comparisons": comparisons,
        "frames": [
            {
                **record,
                "angle_from_closed_rad": float(angle),
                "radial_residual_m": float(radial_value - radius),
                "plane_residual_m": float(plane_value),
            }
            for record, angle, radial_value, plane_value in zip(
                records, angle_from_closed, radial, plane_residual
            )
        ],
    }
    (args.output_dir / "geometric_hinge_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
