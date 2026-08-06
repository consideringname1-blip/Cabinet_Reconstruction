#!/usr/bin/env python3
"""Score AutoSeg tracks with ARKit-compensated metric RGB-D motion.

The AutoSeg masks are proposals only.  Tracks are selected as drawer-motion
seeds when their world-space centroids follow a common prismatic trajectory.
Large static cabinet masks therefore remain static even when their 2D semantic
segmentation covers the whole front face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter


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


def robust_center(points: np.ndarray) -> np.ndarray:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    cutoff = np.quantile(distances, 0.8)
    return np.median(points[distances <= cutoff], axis=0)


def endpoint(values: np.ndarray, valid: np.ndarray, indices: np.ndarray) -> np.ndarray | None:
    chosen = indices[valid[indices]]
    if len(chosen) < 3:
        return None
    return np.median(values[chosen], axis=0)


def orient_axis(axis: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    if np.median(displacements @ axis) < 0:
        return -axis
    return axis


def contact_sheet(
    rgb_dir: Path,
    masks_forward: np.ndarray,
    selected_ids: list[int],
    source_indices: list[int],
    output: Path,
) -> None:
    tiles = []
    for source_index in source_indices:
        local_index = source_index - 405
        bgr = cv2.imread(str(rgb_dir / f"{source_index:06d}.jpg"), cv2.IMREAD_COLOR)
        combined = np.any(masks_forward[local_index, selected_ids], axis=0)
        overlay = bgr.copy()
        overlay[combined] = (0, 0, 255)
        rendered = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
        cv2.putText(
            rendered,
            f"src {source_index} tracks {selected_ids}",
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(rendered)
    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start : start + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument("--min-endpoint-displacement-m", type=float, default=0.07)
    parser.add_argument("--max-endpoint-displacement-m", type=float, default=0.55)
    parser.add_argument("--max-axis-angle-deg", type=float, default=18.0)
    parser.add_argument("--max-orthogonal-range-m", type=float, default=0.10)
    args = parser.parse_args()

    poses = np.load(args.run_root / "normalized/camera_to_world.npy")
    intrinsics = np.load(args.run_root / "normalized/intrinsics_depth.npy")
    auto_paths = sorted(
        (args.run_root / "preprocess/autoseg_reverse/small/final-output").glob("mask_*.npz")
    )
    if len(auto_paths) != 136:
        raise ValueError(f"Expected 136 AutoSeg files, got {len(auto_paths)}")
    masks_reverse = np.stack([np.load(path)["a"][:, 0].astype(bool) for path in auto_paths])
    masks_forward = masks_reverse[::-1]
    frame_count, track_count, height, width = masks_forward.shape
    source_frames = np.arange(405, 541)

    hand_paths = sorted((args.run_root / "preprocess/hand_per_frame/mask").glob("*.npy"))
    if len(hand_paths) == frame_count:
        hands = np.stack([np.load(path).squeeze().astype(bool) for path in hand_paths])
        hand_policy = "GroundingDINO+SAM2 per-frame, prompt hand and arm"
    else:
        hands = np.zeros((frame_count, height, width), dtype=bool)
        hand_policy = "not available; no hand exclusion"

    centers = np.full((track_count, frame_count, 3), np.nan, dtype=np.float64)
    areas = np.zeros((track_count, frame_count), dtype=np.int64)
    hand_overlap = np.zeros((track_count, frame_count), dtype=np.float64)
    for local_index, source_index in enumerate(source_frames):
        depth_mm = cv2.imread(
            str(args.raw_root / "depth" / f"{source_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        confidence = cv2.imread(
            str(args.raw_root / "confidence" / f"{source_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        depth = depth_mm.astype(np.float64) / 1000.0
        valid_depth = (
            np.isfinite(depth)
            & (depth > 0.25)
            & (depth < 4.5)
            & (confidence >= 1)
        )
        xyz_camera = camera_points(depth, intrinsics[source_index])
        pose = poses[source_index]
        xyz_world = xyz_camera.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        xyz_world = xyz_world.reshape(height, width, 3)
        for track_id in range(track_count):
            raw_mask = masks_forward[local_index, track_id]
            raw_area = int(raw_mask.sum())
            hand_overlap[track_id, local_index] = (
                float((raw_mask & hands[local_index]).sum()) / max(raw_area, 1)
            )
            mask = raw_mask & (~hands[local_index]) & valid_depth
            areas[track_id, local_index] = int(mask.sum())
            if areas[track_id, local_index] >= args.min_pixels:
                centers[track_id, local_index] = robust_center(xyz_world[mask])

    closed_indices = np.flatnonzero((source_frames >= 405) & (source_frames <= 420))
    open_indices = np.flatnonzero((source_frames >= 500) & (source_frames <= 520))
    first_pass = []
    metrics = []
    displacement_vectors = []
    displacement_ids = []
    for track_id in range(track_count):
        valid = np.isfinite(centers[track_id]).all(axis=1)
        closed = endpoint(centers[track_id], valid, closed_indices)
        opened = endpoint(centers[track_id], valid, open_indices)
        if closed is not None and opened is not None:
            displacement = opened - closed
            displacement_m = float(np.linalg.norm(displacement))
        else:
            displacement = np.zeros(3)
            displacement_m = 0.0
        valid_values = centers[track_id, valid]
        if len(valid_values) >= 5:
            centered = valid_values - np.median(valid_values, axis=0)
            covariance = centered.T @ centered / len(centered)
            eigenvalues = np.linalg.eigvalsh(covariance)
            linearity = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
        else:
            linearity = 0.0
        eligible = bool(
            valid.sum() >= 50
            and args.min_endpoint_displacement_m
            <= displacement_m
            <= args.max_endpoint_displacement_m
            and linearity >= 0.65
            and np.median(hand_overlap[track_id]) < 0.6
        )
        if eligible:
            displacement_vectors.append(displacement)
            displacement_ids.append(track_id)
        metrics.append(
            {
                "track_id": track_id,
                "valid_frames": int(valid.sum()),
                "median_valid_pixels": float(np.median(areas[track_id, valid])) if valid.any() else 0.0,
                "median_hand_overlap": float(np.median(hand_overlap[track_id])),
                "closed_center_world_m": closed.tolist() if closed is not None else None,
                "open_center_world_m": opened.tolist() if opened is not None else None,
                "endpoint_displacement_world_m": displacement.tolist(),
                "endpoint_displacement_m": displacement_m,
                "trajectory_linearity": linearity,
                "first_pass_eligible": eligible,
            }
        )

    if not displacement_vectors:
        raise RuntimeError("No AutoSeg track passed the first metric-motion gate")
    anchor_candidates = [
        metric
        for metric in metrics
        if metric["first_pass_eligible"]
        and metric["valid_frames"] >= int(0.9 * frame_count)
        and metric["median_valid_pixels"] >= 200
        and metric["endpoint_displacement_m"] >= 0.15
    ]
    if not anchor_candidates:
        raise RuntimeError("No long-lived, sufficiently large moving track can anchor the drawer")
    anchor = max(
        anchor_candidates,
        key=lambda item: item["median_valid_pixels"] * item["endpoint_displacement_m"],
    )
    anchor_vector = np.asarray(anchor["endpoint_displacement_world_m"], dtype=np.float64)
    anchor_axis = anchor_vector / np.linalg.norm(anchor_vector)
    aligned_vectors = []
    aligned_weights = []
    for metric in anchor_candidates:
        vector = np.asarray(metric["endpoint_displacement_world_m"], dtype=np.float64)
        direction = vector / np.linalg.norm(vector)
        angle = np.degrees(np.arccos(np.clip(np.dot(direction, anchor_axis), -1.0, 1.0)))
        if angle <= 12.0:
            aligned_vectors.append(direction)
            aligned_weights.append(
                metric["median_valid_pixels"] * metric["endpoint_displacement_m"]
            )
    opening_axis = np.average(
        np.stack(aligned_vectors), axis=0, weights=np.asarray(aligned_weights)
    )
    opening_axis /= np.linalg.norm(opening_axis)

    selected_ids = []
    seed_q_curves = []
    for track_id, metric in enumerate(metrics):
        displacement = np.asarray(metric["endpoint_displacement_world_m"])
        displacement_m = metric["endpoint_displacement_m"]
        if displacement_m > 0:
            direction = displacement / displacement_m
            angle_deg = float(
                np.degrees(np.arccos(np.clip(np.dot(direction, opening_axis), -1.0, 1.0)))
            )
        else:
            angle_deg = 180.0
        valid = np.isfinite(centers[track_id]).all(axis=1)
        if valid.any():
            projected = centers[track_id, valid] @ opening_axis
            orthogonal = centers[track_id, valid] - np.outer(projected, opening_axis)
            orthogonal_center = np.median(orthogonal, axis=0)
            orthogonal_range = float(
                np.quantile(np.linalg.norm(orthogonal - orthogonal_center, axis=1), 0.9)
            )
        else:
            orthogonal_range = float("inf")
        selected = bool(
            metric["first_pass_eligible"]
            and displacement_m >= 0.15
            and metric["valid_frames"] >= int(0.8 * frame_count)
            and metric["median_valid_pixels"] >= 200
            and angle_deg <= 12.0
            and orthogonal_range <= args.max_orthogonal_range_m
        )
        metric["axis_angle_deg"] = angle_deg
        metric["orthogonal_p90_range_m"] = orthogonal_range
        metric["selected_motion_seed"] = selected
        if selected:
            selected_ids.append(track_id)
            curve = np.full(frame_count, np.nan)
            curve[valid] = centers[track_id, valid] @ opening_axis
            closed_baseline = np.nanmedian(curve[closed_indices])
            curve -= closed_baseline
            seed_q_curves.append(curve)

    if not selected_ids:
        raise RuntimeError("No AutoSeg track agrees with the consensus prismatic axis")
    q_raw = np.nanmedian(np.stack(seed_q_curves), axis=0)
    q_valid = np.isfinite(q_raw)
    q_interpolated = np.interp(np.arange(frame_count), np.flatnonzero(q_valid), q_raw[q_valid])
    q_smoothed = median_filter(q_interpolated, size=7, mode="nearest")
    q_monotonic = np.maximum.accumulate(q_smoothed)
    q_monotonic -= np.median(q_monotonic[closed_indices])
    travel = float(np.median(q_monotonic[open_indices]))
    if travel < 0:
        opening_axis *= -1
        q_monotonic *= -1
        travel *= -1

    recovered_track_ids = []
    for track_id, metric in enumerate(metrics):
        valid = np.isfinite(centers[track_id]).all(axis=1)
        moving_valid = valid & (q_monotonic >= 0.02) & (q_monotonic <= max(travel * 1.05, 0.08))
        if moving_valid.sum() >= 15:
            q_values = q_monotonic[moving_valid]
            axis_values = centers[track_id, moving_valid] @ opening_axis
            design = np.stack([q_values, np.ones_like(q_values)], axis=1)
            coefficient, *_ = np.linalg.lstsq(design, axis_values, rcond=None)
            prediction = design @ coefficient
            residual = float(np.sum((axis_values - prediction) ** 2))
            total = float(np.sum((axis_values - axis_values.mean()) ** 2))
            q_r2 = 1.0 - residual / max(total, 1e-12)
            q_span = float(np.quantile(q_values, 0.9) - np.quantile(q_values, 0.1))
            q_slope = float(coefficient[0])
        else:
            q_r2 = 0.0
            q_span = 0.0
            q_slope = 0.0
        recovered = bool(
            track_id not in selected_ids
            and moving_valid.sum() >= 15
            and q_span >= 0.08
            and 0.55 <= q_slope <= 1.45
            and q_r2 >= 0.70
            and metric["orthogonal_p90_range_m"] <= 0.08
            and metric["median_hand_overlap"] < 0.6
        )
        metric["joint_q_overlap_frames"] = int(moving_valid.sum())
        metric["joint_q_span_m"] = q_span
        metric["joint_q_axis_slope"] = q_slope
        metric["joint_q_axis_r2"] = q_r2
        metric["recovered_new_surface_candidate"] = recovered
        if recovered:
            recovered_track_ids.append(track_id)

    all_drawer_track_ids = selected_ids + recovered_track_ids

    output_dir = args.run_root / "extended/motion_seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "motion_seed.npz",
        masks_forward=masks_forward,
        selected_track_ids=np.asarray(selected_ids, dtype=np.int64),
        recovered_track_ids=np.asarray(recovered_track_ids, dtype=np.int64),
        all_drawer_track_ids=np.asarray(all_drawer_track_ids, dtype=np.int64),
        opening_axis_world=opening_axis,
        q_per_interaction_frame_m=q_monotonic,
        source_frames=source_frames,
        centers_world_m=centers,
        areas_pixels=areas,
        hand_masks=hands,
    )
    contact_sheet(
        args.rgb_dir,
        masks_forward,
        selected_ids,
        [405, 420, 435, 450, 465, 480, 500, 520, 540],
        output_dir / "selected_motion_tracks_contact_sheet.jpg",
    )
    contact_sheet(
        args.rgb_dir,
        masks_forward,
        all_drawer_track_ids,
        [405, 420, 435, 450, 465, 480, 500, 520, 540],
        output_dir / "all_drawer_tracks_contact_sheet.jpg",
    )
    for metric in sorted(metrics, key=lambda item: item["endpoint_displacement_m"], reverse=True)[:12]:
        contact_sheet(
            args.rgb_dir,
            masks_forward,
            [int(metric["track_id"])],
            [405, 420, 435, 450, 465, 480, 500, 520, 540],
            output_dir / "top_track_sheets" / f"track_{int(metric['track_id']):02d}.jpg",
        )

    report = {
        "output_kind": "extended_non_official_metric_motion_gated_autoseg",
        "autoseg_input_order": "source 540 down to 405",
        "autoseg_parameters": {
            "level": "small",
            "batch_size": 10,
            "detect_stride": 5,
            "pred_iou_thresh": 0.9,
            "stability_score_thresh": 0.95,
        },
        "hand_policy": hand_policy,
        "depth_policy": "LiDAR depth with confidence >= 1, 0.25m < depth < 4.5m",
        "track_count": track_count,
        "first_pass_eligible_ids": displacement_ids,
        "anchor_track_id": int(anchor["track_id"]),
        "anchor_policy": (
            "maximum median_pixels * displacement among tracks visible in >=90% frames, "
            "area >=200px, displacement >=0.15m; axis averaged with anchors within 12deg"
        ),
        "selected_track_ids": selected_ids,
        "recovered_new_surface_track_ids": recovered_track_ids,
        "all_drawer_track_ids": all_drawer_track_ids,
        "opening_axis_world": opening_axis.tolist(),
        "estimated_travel_m": travel,
        "thresholds": {
            "min_pixels": args.min_pixels,
            "min_endpoint_displacement_m": args.min_endpoint_displacement_m,
            "max_endpoint_displacement_m": args.max_endpoint_displacement_m,
            "min_valid_frames": 50,
            "min_trajectory_linearity": 0.65,
            "max_axis_angle_deg": args.max_axis_angle_deg,
            "max_orthogonal_p90_range_m": args.max_orthogonal_range_m,
        },
        "tracks": metrics,
    }
    (output_dir / "motion_seed_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_track_ids": selected_ids,
                "recovered_new_surface_track_ids": recovered_track_ids,
                "all_drawer_track_ids": all_drawer_track_ids,
                "opening_axis_world": opening_axis.tolist(),
                "estimated_travel_m": travel,
            },
            indent=2,
        )
    )
    for metric in sorted(metrics, key=lambda item: item["endpoint_displacement_m"], reverse=True):
        print(
            f"track={metric['track_id']:02d} disp={metric['endpoint_displacement_m']:.3f} "
            f"angle={metric['axis_angle_deg']:.1f} ortho={metric['orthogonal_p90_range_m']:.3f} "
            f"linearity={metric['trajectory_linearity']:.3f} "
            f"valid={metric['valid_frames']} selected={metric['selected_motion_seed']}"
        )


if __name__ == "__main__":
    main()
