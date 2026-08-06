#!/usr/bin/env python3
"""Prepare a Stray Scanner RGB-D recording for articulation-aware reconstruction.

This is an extended, non-official preprocessing stage.  It keeps the original
archive untouched, stores the per-frame ARKit camera trajectory and intrinsics,
and creates the exact forward/reverse interaction sequences used by AutoSeg.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(link_path)
    link_path.symlink_to(target_path.resolve())


def make_sequence(
    output_dir: Path,
    source_rgb_dir: Path,
    source_indices: list[int],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for local_index, source_index in enumerate(source_indices):
        source = source_rgb_dir / f"{source_index:06d}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / f"{local_index:06d}.jpg"
        replace_symlink(target, source)
        mapping.append(
            {
                "local_index": local_index,
                "source_index": source_index,
                "source_rgb": str(source.resolve()),
            }
        )
    (output_dir.parent / f"{output_dir.name}_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interaction-start", type=int, default=405)
    parser.add_argument("--interaction-end", type=int, default=540)
    parser.add_argument("--closed-start", type=int, default=0)
    parser.add_argument("--closed-end", type=int, default=400)
    parser.add_argument("--open-start", type=int, default=540)
    parser.add_argument("--open-end", type=int, default=1101)
    args = parser.parse_args()

    odometry_path = args.raw_root / "odometry.csv"
    depth_dir = args.raw_root / "depth"
    confidence_dir = args.raw_root / "confidence"
    with odometry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rgb_paths = sorted(args.rgb_dir.glob("*.jpg"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    confidence_paths = sorted(confidence_dir.glob("*.png"))
    usable_count = min(len(rgb_paths), len(depth_paths), len(confidence_paths), len(rows))
    if usable_count <= args.open_end:
        raise ValueError(f"Only {usable_count} synchronized frames; need index {args.open_end}")

    poses = []
    intrinsics_depth = []
    timestamps = []
    for expected_index, row in enumerate(rows[:usable_count]):
        frame_index = int(row[" frame"])
        if frame_index != expected_index:
            raise ValueError(f"Odometry discontinuity: expected {expected_index}, got {frame_index}")
        quaternion_xyzw = [float(row[key]) for key in (" qx", " qy", " qz", " qw")]
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
        camera_to_world[:3, 3] = [float(row[key]) for key in (" x", " y", " z")]
        poses.append(camera_to_world)
        rgb_intrinsic = np.array(
            [
                [float(row[" fx"]), 0.0, float(row[" cx"])],
                [0.0, float(row[" fy"]), float(row[" cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        intrinsics_depth.append(np.diag([1.0 / 7.5, 1.0 / 7.5, 1.0]) @ rgb_intrinsic)
        timestamps.append(float(row["timestamp"]))

    first_depth = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
    first_confidence = cv2.imread(str(confidence_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_depth.shape != (192, 256) or first_depth.dtype != np.uint16:
        raise ValueError(f"Unexpected depth format: {first_depth.shape} {first_depth.dtype}")
    if first_confidence.shape != (192, 256) or first_confidence.dtype != np.uint8:
        raise ValueError(
            f"Unexpected confidence format: {first_confidence.shape} {first_confidence.dtype}"
        )

    normalized_dir = args.run_root / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    poses_array = np.stack(poses)
    intrinsics_array = np.stack(intrinsics_depth)
    np.save(normalized_dir / "camera_to_world.npy", poses_array)
    np.save(normalized_dir / "intrinsics_depth.npy", intrinsics_array)
    np.save(normalized_dir / "timestamps.npy", np.asarray(timestamps))

    sequence_root = args.run_root / "sequences"
    forward_indices = list(range(args.interaction_start, args.interaction_end + 1))
    reverse_indices = forward_indices[::-1]
    make_sequence(sequence_root / "interaction_forward", args.rgb_dir, forward_indices)
    make_sequence(sequence_root / "interaction_reverse", args.rgb_dir, reverse_indices)

    positions = poses_array[:, :3, 3]
    manifest = {
        "output_kind": "extended_non_official_articulation_aware_trial_input",
        "archive": str(args.archive.resolve()),
        "archive_sha256": sha256(args.archive),
        "raw_root": str(args.raw_root.resolve()),
        "usable_frame_range": [0, usable_count - 1],
        "source_counts": {
            "decoded_rgb": len(rgb_paths),
            "depth": len(depth_paths),
            "confidence": len(confidence_paths),
            "odometry": len(rows),
            "usable": usable_count,
        },
        "dropped_sources": {
            "depth_confidence_odometry_tail": list(range(usable_count, len(rows))),
            "reason": "MP4 declares 1103 frames but only 1102 frames decode",
        },
        "frame_ranges": {
            "closed_surface": [args.closed_start, args.closed_end],
            "interaction_forward": [args.interaction_start, args.interaction_end],
            "interaction_autoseg_reverse": [args.interaction_end, args.interaction_start],
            "open_surface": [args.open_start, args.open_end],
        },
        "rgb_resolution_wh": [256, 192],
        "depth_resolution_wh": [256, 192],
        "depth_unit": "uint16 millimeters",
        "confidence_values": {"0": "low", "1": "medium", "2": "high"},
        "pose_convention": "ARKit camera-to-world from translation and quaternion xyzw",
        "intrinsics": {
            "policy": "per-frame RGB intrinsics scaled by 1/7.5 to 256x192",
            "fx_range_depth": [
                float(intrinsics_array[:, 0, 0].min()),
                float(intrinsics_array[:, 0, 0].max()),
            ],
            "fy_range_depth": [
                float(intrinsics_array[:, 1, 1].min()),
                float(intrinsics_array[:, 1, 1].max()),
            ],
        },
        "trajectory": {
            "net_translation_m": float(np.linalg.norm(positions[-1] - positions[0])),
            "bbox_extent_m": (positions.max(axis=0) - positions.min(axis=0)).tolist(),
        },
        "sequences": {
            "interaction_forward": str((sequence_root / "interaction_forward").resolve()),
            "interaction_reverse": str((sequence_root / "interaction_reverse").resolve()),
        },
    }
    manifest_path = args.run_root / "INPUT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
