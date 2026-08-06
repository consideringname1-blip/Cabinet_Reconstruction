#!/usr/bin/env python3
"""Compare the iTACO axis with a GT-depth drawer-front plane normal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def project(point: np.ndarray, camera_to_world: np.ndarray, intrinsic: np.ndarray) -> tuple[int, int]:
    camera = (point - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    pixel = intrinsic @ camera
    pixel = pixel[:2] / pixel[2]
    return int(round(pixel[0])), int(round(pixel[1]))


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
    args = parser.parse_args()

    mapping = json.loads(
        (args.run_root / "inputs/interaction_forward_097_115/frame_mapping.json").read_text()
    )
    rgb_paths = [Path(item["rgb"]) for item in mapping]
    depths = [
        np.load(path)
        for path in sorted(
            (args.run_root / "preprocess/gt_depth_interaction_reprojected/npy").glob("*.npy")
        )
    ]
    masks = np.load(args.run_root / "official/preprocess/gt_dynamic_masks.npz")["a"].astype(bool)
    poses = np.load(args.run_root / "official/preprocess/gt_cam2world_all.npy")
    metadata = json.loads((args.run_root / "official/view/metadata.json").read_text())
    intrinsic = np.asarray(metadata["K"], dtype=np.float64).reshape(3, 3).T
    refined = np.load(
        args.run_root
        / "official/prediction/refinement/monst3r/chamfer/0/prismatic/joint_axis.npy"
    )
    refined /= np.linalg.norm(refined)

    o3d.utility.random.seed(0)
    centers = []
    normals = []
    plane_records = []
    for index in range(len(depths)):
        points = masked_world_points(depths[index], masks[index], poses[index], intrinsic)
        centers.append(np.median(points, axis=0))
        if index < 4:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            plane, inliers = pcd.segment_plane(
                distance_threshold=0.008,
                ransac_n=3,
                num_iterations=3000,
            )
            normal = np.asarray(plane[:3], dtype=np.float64)
            normal /= np.linalg.norm(normal)
            if normals and np.dot(normal, normals[0]) < 0:
                normal *= -1
            normals.append(normal)
            plane_records.append(
                {
                    "local_index": index,
                    "source_index": int(mapping[index]["source_index"]),
                    "normal_world": normal.tolist(),
                    "inliers": int(len(inliers)),
                    "moving_seed_points": int(len(points)),
                    "inlier_fraction": float(len(inliers) / len(points)),
                }
            )
    centers = np.stack(centers)
    endpoint_displacement = centers[-3:].mean(axis=0) - centers[:3].mean(axis=0)
    reference = np.mean(normals, axis=0)
    reference /= np.linalg.norm(reference)
    if np.dot(reference, endpoint_displacement) < 0:
        reference *= -1
    refined_opening = refined * (1.0 if np.dot(refined, endpoint_displacement) >= 0 else -1.0)
    angle_deg = float(
        np.degrees(
            np.arccos(np.clip(np.abs(np.dot(refined_opening, reference)), -1.0, 1.0))
        )
    )
    normal_spread_deg = [
        float(np.degrees(np.arccos(np.clip(np.abs(np.dot(normal, reference)), -1.0, 1.0))))
        for normal in normals
    ]

    panels = []
    for local_index in [0, 6, 12, 18]:
        image = cv2.imread(str(rgb_paths[local_index]), cv2.IMREAD_COLOR)
        center = centers[local_index]
        for direction, color in [
            (refined_opening, (0, 255, 80)),
            (reference, (255, 200, 0)),
        ]:
            p0 = project(center - direction * 0.27, poses[local_index], intrinsic)
            p1 = project(center + direction * 0.27, poses[local_index], intrinsic)
            cv2.arrowedLine(image, p0, p1, color, 4, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(image, project(center, poses[local_index], intrinsic), 5, (0, 160, 255), -1)
        cv2.putText(
            image,
            f"source {mapping[local_index]['source_index']}  angle={angle_deg:.1f} deg",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    legend = np.zeros((42, sum(panel.shape[1] for panel in panels), 3), dtype=np.uint8)
    cv2.putText(
        legend,
        "GREEN: official iTACO refined prismatic axis",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        legend,
        "CYAN: GT-depth closed drawer-front plane-normal reference",
        (8, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )
    contact = np.concatenate(panels, axis=1)
    visualization = np.concatenate([legend, contact], axis=0)
    validation_dir = args.run_root / "validation"
    image_path = validation_dir / "axis_official_vs_gt_plane_reference.jpg"
    cv2.imwrite(str(image_path), visualization)
    report = {
        "reference_kind": "GT-depth closed drawer-front plane normal; geometric reference, not labeled joint GT",
        "official_refined_opening_axis_world": refined_opening.tolist(),
        "gt_plane_normal_reference_world": reference.tolist(),
        "absolute_axis_angle_deg": angle_deg,
        "gt_plane_normal_frame_spread_deg": normal_spread_deg,
        "endpoint_moving_seed_displacement_world_m": endpoint_displacement.tolist(),
        "endpoint_displacement_norm_m": float(np.linalg.norm(endpoint_displacement)),
        "projection_on_official_axis_m": float(np.dot(endpoint_displacement, refined_opening)),
        "projection_on_gt_plane_normal_m": float(np.dot(endpoint_displacement, reference)),
        "closed_frame_plane_fits": plane_records,
        "visualization": str(image_path),
    }
    report_path = validation_dir / "axis_official_vs_gt_plane_reference.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
