#!/usr/bin/env python3
"""Project the selected axis through each frame's GT moving-seed center."""

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


def project(point: np.ndarray, camera_to_world: np.ndarray, intrinsic: np.ndarray) -> tuple[int, int]:
    camera = (point - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    pixel = intrinsic @ camera
    pixel = pixel[:2] / pixel[2]
    return int(round(pixel[0])), int(round(pixel[1]))


def moving_center_world(
    depth: np.ndarray,
    moving_mask: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    valid = moving_mask & np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
    camera = np.stack(
        [
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    )
    world = camera[valid] @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    return np.median(world, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    interaction = args.run_root / "inputs/interaction_forward_097_115"
    mapping = json.loads((interaction / "frame_mapping.json").read_text())
    poses = load_odometry(args.source_root / "pinhole_projection/odometry.log")
    fx, fy, cx, cy = np.loadtxt(
        args.source_root / "pinhole_projection/calibration.txt"
    ).reshape(-1)[:4]
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    fusion_report = json.loads(
        (
            args.run_root
            / "extended/prismatic_open_fusion/fusion_report.json"
        ).read_text()
    )
    axis = np.asarray(fusion_report["opening_axis_world"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    depths = sorted(
        (args.run_root / "preprocess/gt_depth_interaction_reprojected/npy").glob("*.npy")
    )
    moving_masks = np.load(
        args.run_root / "official/preprocess/gt_dynamic_masks.npz"
    )["a"].astype(bool)

    panels = []
    for local_index in [0, 6, 12, 18]:
        item = mapping[local_index]
        image = cv2.imread(item["rgb"], cv2.IMREAD_COLOR)
        pose = poses[int(item["source_index"])]
        center = moving_center_world(
            np.load(depths[local_index]), moving_masks[local_index], pose, intrinsic
        )
        start_pixel = project(center - axis * 0.30, pose, intrinsic)
        end_pixel = project(center + axis * 0.30, pose, intrinsic)
        cv2.arrowedLine(
            image,
            start_pixel,
            end_pixel,
            (0, 255, 80),
            4,
            cv2.LINE_AA,
            tipLength=0.08,
        )
        center_pixel = project(center, pose, intrinsic)
        cv2.circle(image, center_pixel, 6, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"source {item['source_index']}  axis through moving center",
            (7, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 80),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    output = np.concatenate(panels, axis=1)
    output_path = args.run_root / "validation/refined_prismatic_axis_rgb_keyframes.jpg"
    cv2.imwrite(str(output_path), output)
    print(output_path)


if __name__ == "__main__":
    main()
