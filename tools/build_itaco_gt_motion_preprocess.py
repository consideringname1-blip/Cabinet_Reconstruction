#!/usr/bin/env python3
"""Create iTACO-compatible GT depth, camera, and camera-compensated motion masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def robust_mean(values: list[np.ndarray]) -> np.ndarray | None:
    if not values:
        return None
    array = np.stack(values)
    return np.median(array, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pinhole-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--corrected-depth-dir", type=Path, required=True)
    parser.add_argument("--autoseg-dir", type=Path, required=True)
    parser.add_argument("--hand-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-uids", default="")
    parser.add_argument("--min-track-pixels", type=int, default=100)
    parser.add_argument("--min-endpoint-displacement", type=float, default=0.04)
    args = parser.parse_args()

    mapping = json.loads((args.sequence_dir / "frame_mapping.json").read_text())
    all_poses = load_odometry(args.source_pinhole_dir / "odometry.log")
    poses = np.stack([all_poses[int(item["source_index"])] for item in mapping])
    fx, fy, cx, cy = np.loadtxt(args.source_pinhole_dir / "calibration.txt").reshape(-1)[:4]
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    rgb_files = sorted((args.sequence_dir / "jpg").glob("*.jpg"))
    depth_files = sorted(args.corrected_depth_dir.glob("*.npy"))
    hand_files = sorted(args.hand_mask_dir.glob("*.npy"))
    segment_files = sorted(args.autoseg_dir.glob("mask_*.npz"))
    count = len(mapping)
    assert len(rgb_files) == len(depth_files) == len(hand_files) == len(segment_files) == count

    depths = [np.load(path).astype(np.float32) for path in depth_files]
    hands = [np.load(path).squeeze().astype(bool) for path in hand_files]
    # AutoSeg was run source 115->97. Official data.py reverse-sorts these
    # files, restoring interaction order 97->115.
    segments = [np.load(segment_files[count - 1 - index])["a"][:, 0] for index in range(count)]
    track_count = segments[0].shape[0]

    centroids: list[list[np.ndarray | None]] = [[] for _ in range(track_count)]
    areas: list[list[int]] = [[] for _ in range(track_count)]
    for frame_index, (depth, hand, frame_segments, pose) in enumerate(
        zip(depths, hands, segments, poses)
    ):
        height, width = depth.shape
        yy, xx = np.indices((height, width))
        camera_xyz = np.stack(
            [
                (xx - cx) * depth / fx,
                (yy - cy) * depth / fy,
                depth,
            ],
            axis=-1,
        )
        world_xyz = camera_xyz.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        world_xyz = world_xyz.reshape(height, width, 3)
        valid_depth = np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
        for uid in range(track_count):
            mask = frame_segments[uid].astype(bool) & (~hand) & valid_depth
            area = int(mask.sum())
            areas[uid].append(area)
            if area >= args.min_track_pixels:
                centroids[uid].append(np.median(world_xyz[mask], axis=0))
            else:
                centroids[uid].append(None)

    metrics = []
    auto_selected = []
    for uid in range(track_count):
        start = robust_mean([value for value in centroids[uid][:4] if value is not None])
        end = robust_mean([value for value in centroids[uid][-4:] if value is not None])
        valid_ids = [index for index, value in enumerate(centroids[uid]) if value is not None]
        values = (
            np.stack([centroids[uid][index] for index in valid_ids])
            if valid_ids
            else np.zeros((0, 3))
        )
        displacement = float(np.linalg.norm(end - start)) if start is not None and end is not None else 0.0
        if len(valid_ids) >= 3:
            time = np.array(valid_ids, dtype=np.float64)
            design = np.stack([time, np.ones_like(time)], axis=1)
            coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
            prediction = design @ coeff
            residual = float(np.sum((values - prediction) ** 2))
            total = float(np.sum((values - values.mean(axis=0)) ** 2))
            linear_r2 = 1.0 - residual / max(total, 1e-12)
            linear_range = float(np.linalg.norm(coeff[0]) * (count - 1))
            direction = coeff[0] / max(np.linalg.norm(coeff[0]), 1e-12)
        else:
            linear_r2 = 0.0
            linear_range = 0.0
            direction = np.zeros(3)
        selected = (
            len(valid_ids) >= 12
            and np.median(areas[uid]) >= args.min_track_pixels
            and displacement >= args.min_endpoint_displacement
            and linear_range >= args.min_endpoint_displacement
            and linear_r2 >= 0.35
        )
        if selected:
            auto_selected.append(uid)
        metrics.append(
            {
                "uid": uid,
                "valid_frames": len(valid_ids),
                "area_pixels": areas[uid],
                "median_area_pixels": float(np.median(areas[uid])),
                "start_centroid_world_m": start.tolist() if start is not None else None,
                "end_centroid_world_m": end.tolist() if end is not None else None,
                "endpoint_displacement_m": displacement,
                "linear_range_m": linear_range,
                "linear_r2": linear_r2,
                "linear_direction_world": direction.tolist(),
                "auto_selected": bool(selected),
            }
        )

    if args.selected_uids.strip():
        selected_uids = [int(value) for value in args.selected_uids.split(",") if value.strip()]
        selection_mode = "explicit"
    else:
        selected_uids = auto_selected
        selection_mode = "automatic_threshold"
    if not selected_uids:
        raise RuntimeError("No moving AutoSeg tracks passed GT world-centroid motion checks")

    prompt_depth_dir = args.output_dir / "prompt_depth_video"
    dynamic_dir = args.output_dir / "monst3r"
    validation_dir = args.output_dir.parent / "validation" / "gt_motion_masks"
    prompt_depth_dir.mkdir(parents=True, exist_ok=True)
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    dynamic_masks = []
    for frame_index, (rgb_path, depth, hand, frame_segments) in enumerate(
        zip(rgb_files, depths, hands, segments)
    ):
        np.save(prompt_depth_dir / f"{frame_index:06d}.npy", depth)
        moving = np.any(frame_segments[selected_uids].astype(bool), axis=0)
        moving &= ~hand
        moving &= (depth > 0.2) & (depth < 4.0)
        dynamic_masks.append(moving)
        cv2.imwrite(
            str(dynamic_dir / f"dynamic_mask_{frame_index}.png"),
            moving.astype(np.uint8) * 255,
        )
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        overlay = bgr.copy()
        overlay[moving] = (0, 0, 255)
        overlay[hand] = (255, 0, 255)
        annotated = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
        cv2.putText(
            annotated,
            f"local {frame_index:02d} source {int(mapping[frame_index]['source_index'])} moving={selected_uids}",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(validation_dir / f"{frame_index:06d}.jpg"), annotated)

    np.save(args.output_dir / "gt_cam2world_all.npy", poses)
    np.save(args.output_dir / "cam2world.npy", poses[0])
    np.savez_compressed(
        args.output_dir / "gt_dynamic_masks.npz",
        a=np.stack(dynamic_masks),
        selected_uids=np.array(selected_uids, dtype=np.int64),
    )
    report = {
        "method": "AutoSeg tracks classified by GT-depth world-centroid motion after GT camera compensation",
        "frame_count": count,
        "intrinsic": intrinsic.tolist(),
        "selection_mode": selection_mode,
        "selected_uids": selected_uids,
        "automatic_selected_uids": auto_selected,
        "thresholds": {
            "min_track_pixels": args.min_track_pixels,
            "min_endpoint_displacement_m": args.min_endpoint_displacement,
            "min_linear_r2": 0.35,
            "min_valid_frames": 12,
        },
        "tracks": metrics,
    }
    (args.output_dir / "gt_motion_track_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected_uids": selected_uids, "automatic_selected_uids": auto_selected}, indent=2))
    for metric in sorted(metrics, key=lambda item: item["endpoint_displacement_m"], reverse=True):
        print(
            f"uid={metric['uid']:02d} disp={metric['endpoint_displacement_m']:.4f}m "
            f"range={metric['linear_range_m']:.4f}m r2={metric['linear_r2']:.3f} "
            f"valid={metric['valid_frames']} selected={metric['auto_selected']}"
        )


if __name__ == "__main__":
    main()
