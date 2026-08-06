#!/usr/bin/env python3
"""Fuse static cabinet and moving drawer into separate canonical point volumes.

This is an extended, non-official reconstruction.  AutoSeg masks are used as
motion seeds, not final ownership.  Drawer observations are transformed into a
closed-state canonical coordinate system before fusion; static observations
stay in ARKit world coordinates.  Ambiguous mask boundaries and hands are
excluded from both volumes.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def camera_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
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


def to_world(camera_xyz: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    return camera_xyz.reshape(-1, 3) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return cloud


def finalize_cloud(
    points: list[np.ndarray],
    colors: list[np.ndarray],
    voxel_size: float,
    normal_radius: float,
) -> o3d.geometry.PointCloud:
    cloud = point_cloud(np.concatenate(points), np.concatenate(colors))
    cloud = cloud.voxel_down_sample(voxel_size)
    if len(cloud.points) >= 100:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=80
            )
        )
        cloud.normalize_normals()
    return cloud


def region_grow(
    candidate: np.ndarray,
    seed: np.ndarray,
    depth: np.ndarray,
    max_depth_step: float,
) -> np.ndarray:
    height, width = candidate.shape
    output = np.zeros_like(candidate)
    queue: deque[tuple[int, int]] = deque()
    for y, x in np.argwhere(seed & candidate):
        output[y, x] = True
        queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        source_depth = depth[y, x]
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= yy < height
                and 0 <= xx < width
                and candidate[yy, xx]
                and not output[yy, xx]
                and abs(float(depth[yy, xx] - source_depth)) <= max_depth_step
            ):
                output[yy, xx] = True
                queue.append((yy, xx))
    return output


def fit_front_basis(front_points: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(front_points, axis=0)
    centered = front_points - center
    planar = centered - np.outer(centered @ axis, axis)
    covariance = planar.T @ planar / len(planar)
    values, vectors = np.linalg.eigh(covariance)
    u = vectors[:, np.argmax(values)]
    u -= axis * np.dot(u, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return u, v


def coordinates(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    relative = points - origin
    return np.stack([relative @ axis, relative @ u, relative @ v], axis=1)


def write_label_visualization(
    rgb: np.ndarray,
    drawer: np.ndarray,
    unknown: np.ndarray,
    output: Path,
    source_index: int,
) -> np.ndarray:
    overlay = rgb.copy()
    overlay[drawer] = (0, 0, 255)
    overlay[unknown] = (0, 255, 255)
    rendered = cv2.addWeighted(rgb, 0.58, overlay, 0.42, 0)
    cv2.putText(
        rendered,
        f"source {source_index}: red=drawer yellow=unknown",
        (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), rendered)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--closed-stride", type=int, default=5)
    parser.add_argument("--interaction-stride", type=int, default=2)
    parser.add_argument("--open-stride", type=int, default=5)
    parser.add_argument("--confidence-min", type=int, default=1)
    parser.add_argument("--seed-distance-m", type=float, default=0.025)
    parser.add_argument("--growth-distance-m", type=float, default=0.04)
    parser.add_argument("--growth-depth-step-m", type=float, default=0.02)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    args = parser.parse_args()

    motion = np.load(args.run_root / "extended/motion_seed/motion_seed.npz")
    masks_forward = motion["masks_forward"].astype(bool)
    primary_ids = motion["selected_track_ids"].astype(int).tolist()
    recovered_ids = motion["recovered_track_ids"].astype(int).tolist()
    drawer_track_ids = motion["all_drawer_track_ids"].astype(int).tolist()
    axis = motion["opening_axis_world"].astype(np.float64)
    axis /= np.linalg.norm(axis)
    q_interaction = motion["q_per_interaction_frame_m"].astype(np.float64)
    travel = float(np.median(q_interaction[95:116]))
    hand_interaction = motion["hand_masks"].astype(bool)
    poses = np.load(args.run_root / "normalized/camera_to_world.npy")
    intrinsics = np.load(args.run_root / "normalized/intrinsics_depth.npy")

    canonical_seed_points = []
    canonical_seed_colors = []
    canonical_front_points = []
    for local_index, source_index in enumerate(range(405, 541)):
        depth_mm = cv2.imread(
            str(args.raw_root / "depth" / f"{source_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        confidence = cv2.imread(
            str(args.raw_root / "confidence" / f"{source_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = cv2.imread(str(args.rgb_dir / f"{source_index:06d}.jpg"), cv2.IMREAD_COLOR)
        depth = depth_mm.astype(np.float64) / 1000.0
        valid = (
            (depth > 0.25)
            & (depth < 4.5)
            & (confidence >= args.confidence_min)
            & (~hand_interaction[local_index])
        )
        world = to_world(camera_points(depth, intrinsics[source_index]), poses[source_index])
        world = world.reshape(depth.shape + (3,))
        canonical = world - q_interaction[local_index] * axis
        drawer_mask = np.any(masks_forward[local_index, drawer_track_ids], axis=0) & valid
        front_mask = np.any(masks_forward[local_index, primary_ids], axis=0) & valid
        if drawer_mask.sum() >= 50:
            canonical_seed_points.append(canonical[drawer_mask])
            canonical_seed_colors.append(rgb[drawer_mask][:, ::-1].astype(np.float64) / 255.0)
        if front_mask.sum() >= 50:
            canonical_front_points.append(canonical[front_mask])

    seed_points_raw = np.concatenate(canonical_seed_points)
    seed_colors_raw = np.concatenate(canonical_seed_colors)
    front_points_raw = np.concatenate(canonical_front_points)
    front_axis_values = front_points_raw @ axis
    front_axis_median = np.median(front_axis_values)
    front_plane_points = front_points_raw[
        np.abs(front_axis_values - front_axis_median) <= 0.04
    ]
    front_center = np.median(front_plane_points, axis=0)
    basis_u, basis_v = fit_front_basis(front_plane_points, axis)
    front_coords = coordinates(front_plane_points, front_center, axis, basis_u, basis_v)
    u_lo, u_hi = np.quantile(front_coords[:, 1], [0.02, 0.98]) + np.array([-0.02, 0.02])
    v_lo, v_hi = np.quantile(front_coords[:, 2], [0.02, 0.98]) + np.array([-0.02, 0.02])
    raw_seed_coords = coordinates(seed_points_raw, front_center, axis, basis_u, basis_v)
    provisional = (
        (raw_seed_coords[:, 0] >= -0.55)
        & (raw_seed_coords[:, 0] <= 0.08)
        & (raw_seed_coords[:, 1] >= u_lo)
        & (raw_seed_coords[:, 1] <= u_hi)
        & (raw_seed_coords[:, 2] >= v_lo)
        & (raw_seed_coords[:, 2] <= v_hi)
    )
    axis_lo, axis_hi = np.quantile(raw_seed_coords[provisional, 0], [0.01, 0.99])
    drawer_lo = np.array([axis_lo - 0.02, u_lo, v_lo])
    drawer_hi = np.array([axis_hi + 0.02, u_hi, v_hi])
    supported = np.all(
        (raw_seed_coords >= drawer_lo) & (raw_seed_coords <= drawer_hi), axis=1
    )
    seed_cloud = point_cloud(
        seed_points_raw[supported], seed_colors_raw[supported]
    ).voxel_down_sample(0.005)
    seed_cloud, _ = seed_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    seed_points = np.asarray(seed_cloud.points)
    seed_tree = cKDTree(seed_points)
    cabinet_lo = np.array([-0.90, drawer_lo[1] - 0.45, drawer_lo[2] - 0.45])
    cabinet_hi = np.array([0.20, drawer_hi[1] + 0.45, drawer_hi[2] + 0.45])

    frame_q = np.full(1102, travel, dtype=np.float64)
    frame_q[:405] = 0.0
    frame_q[405:541] = q_interaction
    closed_frames = list(range(0, 401, args.closed_stride))
    interaction_frames = list(range(405, 541, args.interaction_stride))
    if interaction_frames[-1] != 540:
        interaction_frames.append(540)
    open_frames = list(range(545, 1102, args.open_stride))
    selected_frames = closed_frames + interaction_frames + open_frames
    visual_frames = {0, 100, 200, 300, 400, 420, 450, 480, 520, 540, 650, 800, 950, 1100}

    static_points: list[np.ndarray] = []
    static_colors: list[np.ndarray] = []
    drawer_points: list[np.ndarray] = []
    drawer_colors: list[np.ndarray] = []
    frame_records = []
    visual_tiles = []
    kernel_hand = np.ones((11, 11), np.uint8)
    kernel_boundary = np.ones((5, 5), np.uint8)

    for iteration, source_index in enumerate(selected_frames):
        depth_mm = cv2.imread(
            str(args.raw_root / "depth" / f"{source_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        confidence = cv2.imread(
            str(args.raw_root / "confidence" / f"{source_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = cv2.imread(str(args.rgb_dir / f"{source_index:06d}.jpg"), cv2.IMREAD_COLOR)
        depth = depth_mm.astype(np.float64) / 1000.0
        valid = (
            (depth > 0.25)
            & (depth < 4.5)
            & (confidence >= args.confidence_min)
        )
        world = to_world(camera_points(depth, intrinsics[source_index]), poses[source_index])
        world = world.reshape(depth.shape + (3,))
        canonical = world - frame_q[source_index] * axis
        flat_canonical = canonical.reshape(-1, 3)
        coords = coordinates(flat_canonical, front_center, axis, basis_u, basis_v).reshape(
            depth.shape + (3,)
        )
        world_coords = coordinates(
            world.reshape(-1, 3), front_center, axis, basis_u, basis_v
        ).reshape(depth.shape + (3,))
        inside_drawer_box = np.all((coords >= drawer_lo) & (coords <= drawer_hi), axis=2)
        inside_cabinet_crop = np.all(
            (world_coords >= cabinet_lo) & (world_coords <= cabinet_hi), axis=2
        )

        distances = np.full(depth.shape, np.inf, dtype=np.float64)
        query_mask = valid & inside_drawer_box
        if query_mask.any():
            distances[query_mask] = seed_tree.query(canonical[query_mask], workers=-1)[0]
        seed = query_mask & (distances <= args.seed_distance_m)
        candidate = query_mask & (distances <= args.growth_distance_m)
        if 405 <= source_index <= 540:
            local_index = source_index - 405
            exact = np.any(masks_forward[local_index, drawer_track_ids], axis=0) & valid
            seed |= exact
            candidate |= exact
            hand = hand_interaction[local_index]
        else:
            exact = np.zeros_like(valid)
            hand = np.zeros_like(valid)
        if source_index <= 400:
            drawer_mask = seed
        elif source_index <= 540:
            drawer_mask = exact | seed
        else:
            drawer_mask = region_grow(candidate, seed, depth, args.growth_depth_step_m)
        hand_unknown = cv2.dilate(hand.astype(np.uint8), kernel_hand) > 0
        dilated = cv2.dilate(drawer_mask.astype(np.uint8), kernel_boundary) > 0
        eroded = cv2.erode(drawer_mask.astype(np.uint8), kernel_boundary) > 0
        boundary_unknown = dilated & (~eroded) & (~drawer_mask)
        proximity_unknown = query_mask & (distances <= 0.05) & (~drawer_mask)
        unknown = hand_unknown | boundary_unknown | proximity_unknown
        drawer_mask &= valid & (~unknown)
        static_mask = valid & inside_cabinet_crop & (~drawer_mask) & (~unknown)

        drawer_points.append(canonical[drawer_mask])
        drawer_colors.append(rgb[drawer_mask][:, ::-1].astype(np.float64) / 255.0)
        static_points.append(world[static_mask])
        static_colors.append(rgb[static_mask][:, ::-1].astype(np.float64) / 255.0)
        frame_records.append(
            {
                "source_index": source_index,
                "q_m": float(frame_q[source_index]),
                "valid_pixels": int(valid.sum()),
                "drawer_pixels": int(drawer_mask.sum()),
                "static_pixels": int(static_mask.sum()),
                "unknown_pixels": int(unknown.sum()),
                "exact_autoseg_pixels": int(exact.sum()),
            }
        )
        if source_index in visual_frames:
            tile = write_label_visualization(
                rgb,
                drawer_mask,
                unknown,
                args.run_root / "validation/tri_state_masks" / f"source_{source_index:04d}.jpg",
                source_index,
            )
            visual_tiles.append(tile)
        print(
            f"[{iteration + 1:03d}/{len(selected_frames):03d}] src={source_index:04d} "
            f"q={frame_q[source_index]:.3f} drawer={drawer_mask.sum():5d} "
            f"static={static_mask.sum():5d} unknown={unknown.sum():5d}",
            flush=True,
        )

    output_dir = args.run_root / "extended/dual_volume_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    drawer_closed = finalize_cloud(
        drawer_points, drawer_colors, args.voxel_size_m, normal_radius=0.025
    )
    static = finalize_cloud(
        static_points, static_colors, args.voxel_size_m, normal_radius=0.035
    )
    drawer_open = o3d.geometry.PointCloud(drawer_closed)
    drawer_open.translate(axis * travel)
    combined_open = static + drawer_open
    combined_open = combined_open.voxel_down_sample(args.voxel_size_m)
    drawer_closed_path = output_dir / "drawer_canonical_closed_points.ply"
    drawer_open_path = output_dir / "drawer_canonical_open_points.ply"
    static_path = output_dir / "cabinet_static_points.ply"
    combined_path = output_dir / "combined_open_points.ply"
    o3d.io.write_point_cloud(str(drawer_closed_path), drawer_closed)
    o3d.io.write_point_cloud(str(drawer_open_path), drawer_open)
    o3d.io.write_point_cloud(str(static_path), static)
    o3d.io.write_point_cloud(str(combined_path), combined_open)

    if visual_tiles:
        rows = []
        for start in range(0, len(visual_tiles), 3):
            row = visual_tiles[start : start + 3]
            while len(row) < 3:
                row.append(np.zeros_like(visual_tiles[0]))
            rows.append(np.concatenate(row, axis=1))
        cv2.imwrite(
            str(args.run_root / "validation/tri_state_masks_contact_sheet.jpg"),
            np.concatenate(rows, axis=0),
        )

    report = {
        "output_kind": "extended_non_official_articulation_aware_dual_point_volume",
        "representation": (
            "mask/occlusion/pose-corrected voxel point fusion; TSDF intentionally not used "
            "as a semantic or deghosting substitute"
        ),
        "frame_sampling": {
            "closed": [0, 400, args.closed_stride, len(closed_frames)],
            "interaction": [405, 540, args.interaction_stride, len(interaction_frames)],
            "open": [545, 1101, args.open_stride, len(open_frames)],
        },
        "motion": {
            "primary_track_ids": primary_ids,
            "recovered_new_surface_track_ids": recovered_ids,
            "all_drawer_track_ids": drawer_track_ids,
            "opening_axis_world": axis.tolist(),
            "travel_m": travel,
            "canonical_fusion_state": "closed",
            "exported_visual_state": "open",
        },
        "tri_state_policy": {
            "drawer": (
                "AutoSeg rigid-motion seeds plus depth-continuous growth constrained by the "
                f"canonical drawer box and <={args.growth_distance_m * 100:.3g}cm distance "
                "from observed drawer geometry"
            ),
            "static": "valid cabinet-crop observations outside drawer and unknown masks",
            "unknown": "11x11 dilated hand masks plus 5x5 drawer boundary uncertainty band",
            "clutter": "not separately segmented in this trial, per user instruction",
        },
        "depth": {
            "confidence_min": args.confidence_min,
            "range_m": [0.25, 4.5],
        },
        "geometry_bounds_drawer_axis_u_v_m": {
            "lower": drawer_lo.tolist(),
            "upper": drawer_hi.tolist(),
            "origin_world": front_center.tolist(),
            "basis_u_world": basis_u.tolist(),
            "basis_v_world": basis_v.tolist(),
        },
        "parameters": {
            "seed_distance_m": args.seed_distance_m,
            "growth_distance_m": args.growth_distance_m,
            "growth_depth_step_m": args.growth_depth_step_m,
            "voxel_size_m": args.voxel_size_m,
        },
        "points": {
            "drawer_closed": int(len(drawer_closed.points)),
            "drawer_open": int(len(drawer_open.points)),
            "static": int(len(static.points)),
            "combined_open": int(len(combined_open.points)),
        },
        "outputs": {
            "drawer_closed": str(drawer_closed_path.resolve()),
            "drawer_open": str(drawer_open_path.resolve()),
            "static": str(static_path.resolve()),
            "combined_open": str(combined_path.resolve()),
        },
        "frames": frame_records,
    }
    (output_dir / "fusion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"points": report["points"], "outputs": report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
