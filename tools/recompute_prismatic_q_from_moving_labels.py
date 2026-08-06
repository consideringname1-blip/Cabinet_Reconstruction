#!/usr/bin/env python3
"""Recompute a monotonic prismatic q_t from repaired moving observations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pava(values: np.ndarray) -> np.ndarray:
    """Equal-weight nondecreasing isotonic regression."""
    levels: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    for index, value in enumerate(values):
        levels.append(float(value))
        weights.append(1)
        starts.append(index)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total
            levels[-2:] = [merged]
            weights[-2:] = [total]
            starts.pop()
    result = np.empty(len(values), dtype=np.float64)
    for block, start in enumerate(starts):
        stop = starts[block + 1] if block + 1 < len(starts) else len(values)
        result[start:stop] = levels[block]
    return result


def suppress_short_terminal_block(values: np.ndarray, minimum_support: int) -> np.ndarray:
    """Winsorize a terminal isotonic block supported by too few frames."""
    result = values.copy()
    starts = np.r_[0, np.flatnonzero(np.diff(result) > 1e-12) + 1]
    if len(starts) < 2:
        return result
    terminal_start = int(starts[-1])
    previous_start = int(starts[-2])
    terminal_support = len(result) - terminal_start
    previous_support = terminal_start - previous_start
    if terminal_support < minimum_support and previous_support >= minimum_support:
        result[terminal_start:] = result[terminal_start - 1]
    return result


def moving_center(depth: np.ndarray, label: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray, moving_label: int):
    yy, xx = np.indices(depth.shape)
    valid = (label == moving_label) & np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
    z = depth[valid]
    if len(z) == 0:
        raise RuntimeError("Moving observation has no valid depth")
    camera = np.stack(
        (
            (xx[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (yy[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ),
        axis=1,
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    return np.median(world, axis=0), int(valid.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-key", default="a")
    parser.add_argument("--moving-label", type=int, default=2)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--frame-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-terminal-plateau-frames", type=int, default=3)
    args = parser.parse_args()

    labels = np.load(args.labels)[args.label_key]
    depths = sorted(args.depth_dir.glob("*.npy"))
    poses = np.load(args.poses)
    axis = np.load(args.axis).astype(np.float64)
    axis /= np.linalg.norm(axis)
    fx, fy, cx, cy = np.loadtxt(args.calibration).reshape(-1)[:4]
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    mapping = json.loads(args.frame_mapping.read_text())
    if not (len(labels) == len(depths) == len(poses) == len(mapping)):
        raise ValueError("labels, depths, poses, and frame mapping counts disagree")

    centers, counts = [], []
    for index, depth_path in enumerate(depths):
        center, count = moving_center(
            np.load(depth_path), labels[index], poses[index], intrinsics, args.moving_label
        )
        centers.append(center)
        counts.append(count)
    centers = np.asarray(centers)
    projection = centers @ axis
    q_raw = projection - projection.min()
    q_monotonic = suppress_short_terminal_block(pava(q_raw), args.minimum_terminal_plateau_frames)
    q_monotonic -= q_monotonic.min()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "moving_centers_world.npy", centers)
    np.save(args.output_dir / "q_interaction_raw_m.npy", q_raw)
    np.save(args.output_dir / "q_interaction_monotonic_m.npy", q_monotonic)
    with (args.output_dir / "q_interaction.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["processing_index", "original_frame_id", "q_raw_m", "q_monotonic_m", "moving_valid_points"])
        for index, (raw, monotonic, count) in enumerate(zip(q_raw, q_monotonic, counts)):
            writer.writerow([index, mapping[index]["source_index"], raw, monotonic, count])

    minimum = float(q_monotonic.min())
    report = {
        "method": "project repaired moving-label 3D centroid onto the fixed moving-centroid axis, followed by equal-weight isotonic regression for the single opening interaction",
        "camera": "fixed HoloLens T_world_camera",
        "axis_world": axis.tolist(),
        "reference_policy": "q=0 is the minimum isotonic state; no original frame ID is hard-coded as reference",
        "zero_state_processing_indices": np.flatnonzero(np.isclose(q_monotonic, minimum, atol=1e-12)).tolist(),
        "raw_min_processing_index": int(np.argmin(q_raw)),
        "raw_max_processing_index": int(np.argmax(q_raw)),
        "q_raw_range_m": float(np.ptp(q_raw)),
        "q_monotonic_range_m": float(np.ptp(q_monotonic)),
        "open_plateau_m": float(q_monotonic.max()),
        "raw_negative_steps": int((np.diff(q_raw) < 0).sum()),
        "monotonic_negative_steps": int((np.diff(q_monotonic) < 0).sum()),
        "mean_absolute_isotonic_correction_m": float(np.mean(np.abs(q_monotonic - q_raw))),
        "frame_count": len(q_raw),
        "mapping_source": str(args.frame_mapping.resolve()),
        "terminal_plateau_minimum_support_frames": args.minimum_terminal_plateau_frames,
    }
    (args.output_dir / "q_recomputation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
