#!/usr/bin/env python3
"""Estimate closed-to-open door motion from masked RGB-D feature matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def camera_to_world(pixel: np.ndarray, depth: np.ndarray, pose: np.ndarray, k: np.ndarray) -> np.ndarray:
    u = np.rint(pixel[:, 0]).astype(int)
    v = np.rint(pixel[:, 1]).astype(int)
    z = depth[v, u]
    camera = np.column_stack(
        [
            (pixel[:, 0] - k[0, 2]) * z / k[0, 0],
            (pixel[:, 1] - k[1, 2]) * z / k[1, 1],
            z,
        ]
    )
    return camera @ pose[:3, :3].T + pose[:3, 3]


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def ransac_rigid(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        rotation, translation = rigid_fit(source[sample], target[sample])
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        inliers = residual < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        raise RuntimeError("Rigid RANSAC found fewer than three inliers")
    rotation, translation = rigid_fit(source[best], target[best])
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    best = residual < threshold
    rotation, translation = rigid_fit(source[best], target[best])
    return rotation, translation, best


def axis_from_transform(rotation: np.ndarray, translation: np.ndarray, longitudinal: float) -> tuple[np.ndarray, np.ndarray, float]:
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    values, vectors = np.linalg.eig(rotation)
    axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    axis /= np.linalg.norm(axis)
    if axis[1] < 0:
        axis = -axis
        angle = -angle
    design = np.vstack([np.eye(3) - rotation, axis])
    target = np.concatenate([translation, [longitudinal]])
    position, *_ = np.linalg.lstsq(design, target, rcond=None)
    return axis, position, angle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--closed-indices", default="0,1,2,3,4")
    parser.add_argument("--open-indices", default="31,32,33,34,35,36")
    parser.add_argument("--ratio-test", type=float, default=0.78)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.018)
    parser.add_argument("--ransac-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    view = args.run_root / "official/view"
    preprocess = args.run_root / "official/preprocess"
    rgb_paths = sorted((view / "rgb").glob("*.jpg"))
    depth_paths = sorted((preprocess / "prompt_depth_video").glob("*.npy"))
    poses = np.load(preprocess / "gt_cam2world_all.npy")
    masks = np.load(preprocess / "gt_dynamic_masks.npz")["a"].astype(np.uint8)
    metadata = json.loads((view / "metadata.json").read_text())
    k = np.asarray(metadata["K"], dtype=np.float64).reshape(3, 3).T
    closed_ids = [int(x) for x in args.closed_indices.split(",")]
    open_ids = [int(x) for x in args.open_indices.split(",")]
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=20)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    all_source = []
    all_target = []
    pair_records = []
    kernel = np.ones((7, 7), np.uint8)
    for closed_index in closed_ids:
        closed_rgb = cv2.imread(str(rgb_paths[closed_index]), cv2.IMREAD_GRAYSCALE)
        closed_mask = cv2.erode(masks[closed_index], kernel)
        closed_kp, closed_desc = sift.detectAndCompute(closed_rgb, closed_mask * 255)
        if closed_desc is None:
            continue
        for open_index in open_ids:
            open_rgb = cv2.imread(str(rgb_paths[open_index]), cv2.IMREAD_GRAYSCALE)
            open_mask = cv2.erode(masks[open_index], kernel)
            open_kp, open_desc = sift.detectAndCompute(open_rgb, open_mask * 255)
            if open_desc is None:
                continue
            candidates = matcher.knnMatch(closed_desc, open_desc, k=2)
            good = [a for a, b in candidates if a.distance < args.ratio_test * b.distance]
            if not good:
                continue
            closed_pixels = np.array([closed_kp[m.queryIdx].pt for m in good])
            open_pixels = np.array([open_kp[m.trainIdx].pt for m in good])
            closed_uv = np.rint(closed_pixels).astype(int)
            open_uv = np.rint(open_pixels).astype(int)
            closed_depth = np.load(depth_paths[closed_index])
            open_depth = np.load(depth_paths[open_index])
            valid = (
                (closed_uv[:, 0] >= 0)
                & (closed_uv[:, 0] < closed_depth.shape[1])
                & (closed_uv[:, 1] >= 0)
                & (closed_uv[:, 1] < closed_depth.shape[0])
                & (open_uv[:, 0] >= 0)
                & (open_uv[:, 0] < open_depth.shape[1])
                & (open_uv[:, 1] >= 0)
                & (open_uv[:, 1] < open_depth.shape[0])
            )
            closed_z = np.zeros(len(good))
            open_z = np.zeros(len(good))
            closed_z[valid] = closed_depth[closed_uv[valid, 1], closed_uv[valid, 0]]
            open_z[valid] = open_depth[open_uv[valid, 1], open_uv[valid, 0]]
            valid &= (closed_z > 0.2) & (closed_z < 4.0) & (open_z > 0.2) & (open_z < 4.0)
            valid &= masks[closed_index, closed_uv[:, 1].clip(0, masks.shape[1] - 1), closed_uv[:, 0].clip(0, masks.shape[2] - 1)].astype(bool)
            valid &= masks[open_index, open_uv[:, 1].clip(0, masks.shape[1] - 1), open_uv[:, 0].clip(0, masks.shape[2] - 1)].astype(bool)
            if valid.sum():
                source = camera_to_world(closed_pixels[valid], closed_depth, poses[closed_index], k)
                target = camera_to_world(open_pixels[valid], open_depth, poses[open_index], k)
                all_source.append(source)
                all_target.append(target)
            pair_records.append(
                {
                    "closed_index": closed_index,
                    "open_index": open_index,
                    "sift_closed": len(closed_kp),
                    "sift_open": len(open_kp),
                    "ratio_matches": len(good),
                    "valid_rgbd_matches": int(valid.sum()),
                }
            )
    if not all_source:
        raise RuntimeError("No valid masked RGB-D feature correspondences")
    source = np.concatenate(all_source)
    target = np.concatenate(all_target)
    rotation, translation, inliers = ransac_rigid(
        source, target, args.ransac_threshold_m, args.ransac_iterations, args.seed
    )
    axis, position, angle = axis_from_transform(
        rotation, translation, float(np.median(np.concatenate([source, target])[:, 1]))
    )
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "closed_to_open_rotation.npy", rotation)
    np.save(args.output_dir / "closed_to_open_translation.npy", translation)
    np.save(args.output_dir / "feature_joint_axis.npy", axis)
    np.save(args.output_dir / "feature_joint_pos.npy", position)
    report = {
        "method": "masked SIFT RGB correspondences lifted with HoloLens depth and fixed poses, rigid RANSAC",
        "role": "non-official GT geometric validation",
        "closed_indices": closed_ids,
        "open_indices": open_ids,
        "ratio_test": args.ratio_test,
        "ransac_threshold_m": args.ransac_threshold_m,
        "candidate_rgbd_matches": int(len(source)),
        "inliers": int(inliers.sum()),
        "inlier_fraction": float(inliers.mean()),
        "inlier_rmse_m": float(np.sqrt(np.mean(residual[inliers] ** 2))),
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "axis_world": axis.tolist(),
        "axis_position_world_m": position.tolist(),
        "angle_rad": angle,
        "angle_deg": float(np.degrees(angle)),
        "pairs": pair_records,
    }
    (args.output_dir / "feature_motion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
