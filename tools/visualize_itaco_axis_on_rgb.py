#!/usr/bin/env python3
"""Project the selected prismatic axis onto key interaction RGB frames."""

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
    if camera[2] <= 0:
        raise ValueError("Axis point is behind camera")
    pixel = intrinsic @ camera
    pixel = pixel[:2] / pixel[2]
    return int(round(pixel[0])), int(round(pixel[1]))


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

    fusion_dir = args.run_root / "extended/prismatic_open_fusion"
    fusion_report = json.loads((fusion_dir / "fusion_report.json").read_text())
    axis = np.asarray(fusion_report["opening_axis_world"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    import open3d as o3d

    moving = np.asarray(
        o3d.io.read_point_cloud(str(fusion_dir / "open_state_drawer_fused.ply")).points
    )
    axis_center = np.median(moving, axis=0)
    line_start = axis_center - axis * 0.38
    line_end = axis_center + axis * 0.38
    key_indices = [0, 6, 12, 18]
    panels = []
    for local_index in key_indices:
        item = mapping[local_index]
        image = cv2.imread(item["rgb"], cv2.IMREAD_COLOR)
        pose = poses[int(item["source_index"])]
        start_pixel = project(line_start, pose, intrinsic)
        end_pixel = project(line_end, pose, intrinsic)
        cv2.arrowedLine(
            image,
            start_pixel,
            end_pixel,
            (0, 255, 80),
            4,
            cv2.LINE_AA,
            tipLength=0.08,
        )
        cv2.putText(
            image,
            f"source {item['source_index']}  prismatic opening axis",
            (7, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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
