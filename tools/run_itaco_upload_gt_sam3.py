#!/usr/bin/env python3
"""Run an iTACO-style coarse joint prediction on a HoloLens upload.

The upstream iTACO RealDataLoader is iPhone/Polycam-shaped and hard-codes a
960x720 image size. This adapter keeps the core coarse-prediction geometry:

  - LoFTR image correspondences
  - GT depth backprojection
  - GT camera poses from pinhole_projection/odometry.log
  - SAM3 moving/object/hand masks for selecting dynamic correspondences
  - iTACO's revolute-vs-prismatic transform scoring

Outputs include a concise JSON summary, per-pair CSV diagnostics, and an
iTACO-like coarse_prediction tree containing joint_axis / joint_pos /
joint_value arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR
from scipy.spatial.transform import Rotation as R


DEFAULT_INPUT = Path("/workspace_whz/data/upload/2026-07-27-175228/pinhole_projection")
DEFAULT_MASK_DIR = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_sam3_formal_masks")
DEFAULT_OUTPUT = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_itaco_gt_sam3")
ITACO_ROOT = Path("/workspace_whz/code/reconstruction/video2articulation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intervals", default="1,2,3")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--match-conf", type=float, default=0.85)
    parser.add_argument("--min-dynamic", type=int, default=40)
    parser.add_argument("--max-pair-motion", type=float, default=0.50, help="meters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_intrinsics(input_dir: Path) -> np.ndarray:
    vals = np.loadtxt(input_dir / "calibration.txt", dtype=np.float64).reshape(-1)
    if vals.size < 4:
        raise ValueError(f"Expected fx fy cx cy in {input_dir / 'calibration.txt'}")
    fx, fy, cx, cy = vals[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_odometry(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    poses: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        # A header line such as "17 17 17", followed by 4 matrix rows.
        if i + 4 >= len(lines):
            break
        mat = []
        for row in lines[i + 1 : i + 5]:
            mat.append([float(x) for x in row.split()])
        poses.append(np.asarray(mat, dtype=np.float64))
        i += 5
    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return poses


def sorted_files(input_dir: Path) -> tuple[list[Path], list[Path]]:
    rgb_files = sorted((input_dir / "rgb").glob("*.png"))
    if not rgb_files:
        rgb_files = sorted((input_dir / "rgb").glob("*.jpg"))
    depth_files = sorted((input_dir / "depth").glob("*.png"))
    if not rgb_files or not depth_files:
        raise FileNotFoundError(f"Missing RGB/depth frames under {input_dir}")
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"RGB/depth count mismatch: {len(rgb_files)} vs {len(depth_files)}")
    return rgb_files, depth_files


def depth_png_to_m(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read {depth_path}")
    depth = depth.astype(np.float32)
    # HoloLens exported depth PNG is uint16 in millimeters.
    if depth.max(initial=0) > 20:
        depth = depth / 1000.0
    return depth


def depth_to_xyz(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    z = depth_m.astype(np.float32)
    x = (xs - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys - float(K[1, 2])) * z / float(K[1, 1])
    return np.stack([x, y, z], axis=-1)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def transform_points(points: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    return points @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def estimate_se3_transformation(target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    """Fallback Kabsch implementation matching iTACO utils semantics.

    Returns source-to-target SE(3): target ~= source @ R.T + t
    """
    target_centroid = np.mean(target_xyz, axis=0)
    source_centroid = np.mean(source_xyz, axis=0)
    target_centered = target_xyz - target_centroid
    source_centered = source_xyz - source_centroid
    h = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    trans = target_centroid - rot @ source_centroid
    se3 = np.eye(4, dtype=np.float64)
    se3[:3, :3] = rot
    se3[:3, 3] = trans
    return se3


def load_itaco_estimator_if_possible() -> None:
    """Prefer upstream helper if importable, otherwise keep fallback above."""
    global estimate_se3_transformation
    if ITACO_ROOT.exists():
        sys.path.insert(0, str(ITACO_ROOT))
        try:
            from utils import estimate_se3_transformation as itaco_estimate

            estimate_se3_transformation = itaco_estimate
            print(f"[itaco] using upstream estimate_se3_transformation from {ITACO_ROOT}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[itaco] warning: using fallback SE3 estimator because import failed: {exc}")


def compute_match(matcher: LoFTR, img0: np.ndarray, img1: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    t0 = torch.from_numpy(gray0).float().to(device)[None, None] / 255.0
    t1 = torch.from_numpy(gray1).float().to(device)[None, None] / 255.0
    out = matcher({"image0": t0, "image1": t1})
    return (
        out["keypoints0"].detach().cpu().numpy(),
        out["keypoints1"].detach().cpu().numpy(),
        out["confidence"].detach().cpu().numpy(),
    )


def filter_correspondence_motion(base_kp: np.ndarray, curr_kp: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = np.linalg.norm(base_kp - curr_kp, axis=1)
    keep = np.isfinite(dist) & (dist < thresh)
    return base_kp[keep], curr_kp[keep], dist[keep]


def estimate_joint_transformation(
    base_kp: np.ndarray,
    curr_kp: np.ndarray,
    joint_type: str,
    *,
    rng: np.random.Generator,
    ransac: bool = True,
    iters: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    n = base_kp.shape[0]
    if n == 0:
        return np.eye(4), np.zeros((0,), dtype=np.int64)
    if not ransac or n < 12:
        inlier = np.arange(n)
        if joint_type == "revolute":
            return estimate_se3_transformation(base_kp, curr_kp), inlier
        trans = np.mean(base_kp - curr_kp, axis=0)
        se3 = np.eye(4, dtype=np.float64)
        se3[:3, 3] = trans
        return se3, inlier

    sample_n = min(10, n)
    inlier_thresh = 1e-2
    min_consensus = max(4, int(n * 0.4))
    best_se3: np.ndarray | None = None
    best_error = math.inf
    best_inlier: np.ndarray | None = None
    fallback_se3: np.ndarray | None = None
    fallback_inlier: np.ndarray | None = None

    for _ in range(iters):
        sample = rng.choice(n, sample_n, replace=False)
        if joint_type == "revolute":
            se3 = estimate_se3_transformation(base_kp[sample], curr_kp[sample])
        else:
            trans = np.mean(base_kp[sample] - curr_kp[sample], axis=0)
            se3 = np.eye(4, dtype=np.float64)
            se3[:3, 3] = trans
        rot = se3[:3, :3]
        trans = se3[:3, 3]
        transformed = curr_kp @ rot.T + trans
        dist = np.linalg.norm(base_kp - transformed, axis=1)
        inlier = np.nonzero(dist < inlier_thresh)[0]

        if fallback_inlier is None or inlier.shape[0] > fallback_inlier.shape[0]:
            fallback_se3 = se3
            fallback_inlier = inlier

        if inlier.shape[0] >= min_consensus:
            if joint_type == "revolute":
                se3 = estimate_se3_transformation(base_kp[inlier], curr_kp[inlier])
            else:
                trans = np.mean(base_kp[inlier] - curr_kp[inlier], axis=0)
                se3 = np.eye(4, dtype=np.float64)
                se3[:3, 3] = trans
            rot = se3[:3, :3]
            trans = se3[:3, 3]
            transformed = curr_kp[inlier] @ rot.T + trans
            error = float(np.mean((base_kp[inlier] - transformed) ** 2))
            if error < best_error:
                best_se3 = se3
                best_error = error
                best_inlier = inlier

    if best_se3 is not None and best_inlier is not None:
        return best_se3, best_inlier
    if fallback_se3 is not None and fallback_inlier is not None and fallback_inlier.shape[0] > 0:
        return fallback_se3, fallback_inlier
    # Last fallback: all points.
    if joint_type == "revolute":
        return estimate_se3_transformation(base_kp, curr_kp), np.arange(n)
    trans = np.mean(base_kp - curr_kp, axis=0)
    se3 = np.eye(4, dtype=np.float64)
    se3[:3, 3] = trans
    return se3, np.arange(n)


def estimate_joint_single(base_kp: np.ndarray, curr_kp: np.ndarray, rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    curr2base, revolute_inlier = estimate_joint_transformation(base_kp, curr_kp, "revolute", rng=rng, ransac=True)
    rotation = curr2base[:3, :3]
    translation = curr2base[:3, 3]
    rotvec = R.from_matrix(rotation.T).as_rotvec()
    rot_norm = float(np.linalg.norm(rotvec))
    det = float(np.linalg.det(np.eye(3) - rotation))
    revolute_valid = bool(rot_norm > 1e-8 and abs(det) >= 1e-17 and revolute_inlier.shape[0] > 0)
    if rot_norm > 1e-8:
        revolute_axis = rotvec / rot_norm
    else:
        revolute_axis = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    try:
        revolute_pos = np.linalg.inv(np.eye(3) - rotation) @ translation
        if np.all(np.isfinite(revolute_axis)):
            revolute_pos = revolute_pos - np.dot(revolute_pos, revolute_axis) * revolute_axis
    except Exception:
        revolute_pos = np.zeros(3, dtype=np.float64)
        revolute_valid = False
    if revolute_inlier.shape[0] > 0:
        rotate_curr = (curr_kp[revolute_inlier] - revolute_pos) @ rotation.T + revolute_pos
        rotation_error = float(np.mean((base_kp[revolute_inlier] - rotate_curr) ** 2))
    else:
        rotation_error = math.inf
        revolute_valid = False
    result["revolute"] = {
        "X": curr_kp[revolute_inlier],
        "Y": base_kp[revolute_inlier],
        "axis": revolute_axis,
        "pos": revolute_pos,
        "error": rotation_error,
        "det": det,
        "valid": revolute_valid,
        "inliers": int(revolute_inlier.shape[0]),
    }

    prismatic_se3, prismatic_inlier = estimate_joint_transformation(base_kp, curr_kp, "prismatic", rng=rng, ransac=True)
    only_translation = prismatic_se3[:3, 3]
    trans_norm = float(np.linalg.norm(only_translation))
    prismatic_valid = bool(trans_norm > 1e-8 and prismatic_inlier.shape[0] > 0)
    prismatic_axis = only_translation / trans_norm if trans_norm > 1e-8 else np.array([np.nan, np.nan, np.nan])
    prismatic_pos = base_kp[0]
    if prismatic_inlier.shape[0] > 0:
        translate_curr = curr_kp[prismatic_inlier] + only_translation
        translation_error = float(np.mean((base_kp[prismatic_inlier] - translate_curr) ** 2))
    else:
        translation_error = math.inf
        prismatic_valid = False
    result["prismatic"] = {
        "X": curr_kp[prismatic_inlier],
        "Y": base_kp[prismatic_inlier],
        "axis": prismatic_axis,
        "pos": prismatic_pos,
        "error": translation_error,
        "valid": prismatic_valid,
        "inliers": int(prismatic_inlier.shape[0]),
    }
    return result


def average_unit_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    valid = [v for v in vectors if np.all(np.isfinite(v)) and np.linalg.norm(v) > 1e-8]
    if not valid:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    acc = valid[0].copy()
    for v in valid[1:]:
        # Axes are sign-ambiguous; align signs before averaging.
        if np.dot(acc, v) < 0:
            v = -v
        acc += v
    norm = np.linalg.norm(acc)
    return acc / norm if norm > 1e-8 else np.array([np.nan, np.nan, np.nan], dtype=np.float64)


def estimate_joint_all(results: list[dict[str, dict[str, Any]]]) -> tuple[dict[str, dict[str, Any]], str]:
    rev_valid = [r for r in results if r["revolute"]["valid"]]
    pri_valid = [r for r in results if r["prismatic"]["valid"]]

    rev_error = float(np.mean([r["revolute"]["error"] for r in rev_valid])) if rev_valid else math.inf
    pri_error = float(np.mean([r["prismatic"]["error"] for r in pri_valid])) if pri_valid else math.inf

    vote = 0
    for r in results:
        if r["revolute"]["valid"] and r["prismatic"]["valid"]:
            vote += 1 if r["revolute"]["error"] < r["prismatic"]["error"] else -1
    if vote > 0:
        pred_type = "revolute"
    elif vote < 0:
        pred_type = "prismatic"
    else:
        pred_type = "revolute" if rev_error < pri_error else "prismatic"

    rev_axis = average_unit_vectors([r["revolute"]["axis"] for r in rev_valid])
    pri_axis = average_unit_vectors([r["prismatic"]["axis"] for r in pri_valid])
    rev_pos_list = [r["revolute"]["pos"] for r in rev_valid if np.all(np.isfinite(r["revolute"]["pos"]))]
    rev_pos = np.mean(rev_pos_list, axis=0) if rev_pos_list else np.zeros(3, dtype=np.float64)

    metrics = {
        "revolute": {"axis": rev_axis, "pos": rev_pos, "error": rev_error, "valid_count": len(rev_valid)},
        "prismatic": {"axis": pri_axis, "pos": np.zeros(3, dtype=np.float64), "error": pri_error, "valid_count": len(pri_valid)},
    }
    return metrics, pred_type


def compute_average_rotation_angle(x: np.ndarray, y: np.ndarray, axis: np.ndarray, pos: np.ndarray) -> float:
    axis = axis / np.linalg.norm(axis)
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for px, py in zip(x, y):
        px = px - pos
        py = py - pos
        px_perp = px - np.dot(px, axis) * axis
        py_perp = py - np.dot(py, axis) * axis
        nx = np.linalg.norm(px_perp)
        ny = np.linalg.norm(py_perp)
        if nx <= 1e-8 or ny <= 1e-8:
            continue
        px_perp = px_perp / nx
        py_perp = py_perp / ny
        cos_sum += float(np.dot(px_perp, py_perp))
        sin_sum += float(np.dot(axis, np.cross(px_perp, py_perp)))
        count += 1
    if count == 0:
        return float("nan")
    return float(np.arctan2(sin_sum, cos_sum))


def compute_average_translation_distance(x: np.ndarray, y: np.ndarray, axis: np.ndarray) -> float:
    axis = axis / np.linalg.norm(axis)
    return float(np.mean(np.dot(y - x, axis)))


@dataclass
class PreparedData:
    rgb: list[np.ndarray]
    xyz_cam: list[np.ndarray]
    object_masks: list[np.ndarray]
    moving_masks: list[np.ndarray]
    hand_masks: list[np.ndarray]
    poses_as_read: list[np.ndarray]
    K: np.ndarray
    rgb_files: list[Path]
    depth_files: list[Path]


def prepare_data(args: argparse.Namespace) -> PreparedData:
    rgb_files, depth_files = sorted_files(args.input_dir)
    poses = load_odometry(args.input_dir / "odometry.log")
    n = min(len(rgb_files), len(depth_files), len(poses))
    if args.max_frames and args.max_frames > 0:
        n = min(n, args.max_frames)
    rgb_files = rgb_files[:n]
    depth_files = depth_files[:n]
    poses = poses[:n]

    K = load_intrinsics(args.input_dir)
    rgb: list[np.ndarray] = []
    xyz_cam: list[np.ndarray] = []
    object_masks: list[np.ndarray] = []
    moving_masks: list[np.ndarray] = []
    hand_masks: list[np.ndarray] = []
    for idx, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        image = read_rgb(rgb_path)
        depth_m = depth_png_to_m(depth_path)
        if depth_m.shape != image.shape[:2]:
            raise ValueError(f"Depth/RGB shape mismatch at {idx}: {depth_m.shape} vs {image.shape[:2]}")
        shape = depth_m.shape
        rgb.append(image)
        xyz_cam.append(depth_to_xyz(depth_m, K))
        object_masks.append(read_mask(args.mask_dir / "object" / f"{idx:06d}.png", shape))
        moving_masks.append(read_mask(args.mask_dir / "moving" / f"{idx:06d}.png", shape))
        hand_masks.append(read_mask(args.mask_dir / "hand" / f"{idx:06d}.png", shape))
    return PreparedData(rgb, xyz_cam, object_masks, moving_masks, hand_masks, poses, K, rgb_files, depth_files)


def keypoints_to_int(mkpts: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.rint(mkpts).astype(np.int64)
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    return pts, valid


def choose_pose_convention(data: PreparedData, matcher: LoFTR, device: torch.device, match_conf: float) -> tuple[str, list[np.ndarray], dict[str, float]]:
    candidates = {
        "cam_to_world": data.poses_as_read,
        "world_to_cam_inverted": [np.linalg.inv(p) for p in data.poses_as_read],
    }
    scores: dict[str, float] = {}
    sample_pairs = min(12, len(data.rgb) - 1)
    for name, poses in candidates.items():
        dists: list[float] = []
        for i in range(sample_pairs):
            mk0, mk1, conf = compute_match(matcher, data.rgb[i], data.rgb[i + 1], device)
            keep = conf > match_conf
            mk0 = mk0[keep]
            mk1 = mk1[keep]
            if mk0.shape[0] == 0:
                continue
            h, w = data.xyz_cam[i].shape[:2]
            p0, v0 = keypoints_to_int(mk0, h, w)
            p1, v1 = keypoints_to_int(mk1, h, w)
            valid = v0 & v1
            if not valid.any():
                continue
            p0 = p0[valid]
            p1 = p1[valid]
            z0 = data.xyz_cam[i][p0[:, 1], p0[:, 0], 2]
            z1 = data.xyz_cam[i + 1][p1[:, 1], p1[:, 0], 2]
            valid_z = (z0 > 0) & (z1 > 0) & np.isfinite(z0) & np.isfinite(z1)
            if not valid_z.any():
                continue
            p0 = p0[valid_z]
            p1 = p1[valid_z]
            x0 = transform_points(data.xyz_cam[i][p0[:, 1], p0[:, 0]], poses[i])
            x1 = transform_points(data.xyz_cam[i + 1][p1[:, 1], p1[:, 0]], poses[i + 1])
            dist = np.linalg.norm(x0 - x1, axis=1)
            if dist.size:
                dists.append(float(np.median(dist)))
        scores[name] = float(np.median(dists)) if dists else math.inf
    selected = min(scores, key=scores.get)
    return selected, candidates[selected], scores


def save_prediction_tree(output_dir: Path, mask_name: str, seed: int, metrics: dict[str, dict[str, Any]], pred_type: str, n_frames: int, cam_poses: list[np.ndarray]) -> None:
    result_dir = output_dir / "coarse_prediction" / mask_name / str(seed)
    for joint_type in ["revolute", "prismatic"]:
        (result_dir / joint_type).mkdir(parents=True, exist_ok=True)
        np.save(result_dir / joint_type / "joint_axis.npy", metrics[joint_type]["axis"])
        np.save(result_dir / joint_type / "joint_pos.npy", metrics[joint_type]["pos"])
        avg_value = float(metrics[joint_type].get("average_value", 0.0))
        np.save(result_dir / joint_type / "joint_value.npy", np.arange(n_frames, dtype=np.float64) * avg_value)
    (result_dir / "cam_pose").mkdir(parents=True, exist_ok=True)
    for idx, pose in enumerate(cam_poses):
        np.save(result_dir / "cam_pose" / f"cam2label_{idx}.npy", pose)
    (result_dir / "pred_joint_type.txt").write_text(pred_type + "\n")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("KORNIA_CHECK_SHAPE", "0")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    load_itaco_estimator_if_possible()

    data = prepare_data(args)
    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[itaco] frames={len(data.rgb)} intervals={intervals} device={device}")
    print(f"[itaco] match_conf={args.match_conf} min_dynamic={args.min_dynamic}")
    matcher = LoFTR(pretrained="indoor").to(device).eval()

    convention, cam_poses, convention_scores = choose_pose_convention(data, matcher, device, args.match_conf)
    print(f"[itaco] pose_convention={convention} scores={convention_scores}")

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, dict[str, Any]]] = []
    result_pairs: list[tuple[int, int]] = []
    pair_rows: list[dict[str, Any]] = []

    n = len(data.rgb)
    h, w = data.xyz_cam[0].shape[:2]
    for interval in intervals:
        for i in range(0, n - interval):
            j = i + interval
            mk0, mk1, conf = compute_match(matcher, data.rgb[i], data.rgb[j], device)
            keep_conf = conf > args.match_conf
            mk0 = mk0[keep_conf]
            mk1 = mk1[keep_conf]
            conf_kept = conf[keep_conf]
            if mk0.shape[0] == 0:
                pair_rows.append({"i": i, "j": j, "status": "no_match", "matches": 0})
                continue
            p0, v0 = keypoints_to_int(mk0, h, w)
            p1, v1 = keypoints_to_int(mk1, h, w)
            valid = v0 & v1
            p0 = p0[valid]
            p1 = p1[valid]
            conf_valid = conf_kept[valid]
            if p0.shape[0] == 0:
                pair_rows.append({"i": i, "j": j, "status": "oob", "matches": int(mk0.shape[0])})
                continue

            xyz0_cam = data.xyz_cam[i][p0[:, 1], p0[:, 0]]
            xyz1_cam = data.xyz_cam[j][p1[:, 1], p1[:, 0]]
            valid_depth = (
                (xyz0_cam[:, 2] > 0)
                & (xyz1_cam[:, 2] > 0)
                & np.all(np.isfinite(xyz0_cam), axis=1)
                & np.all(np.isfinite(xyz1_cam), axis=1)
            )
            p0 = p0[valid_depth]
            p1 = p1[valid_depth]
            xyz0_cam = xyz0_cam[valid_depth]
            xyz1_cam = xyz1_cam[valid_depth]
            conf_valid = conf_valid[valid_depth]
            if p0.shape[0] == 0:
                pair_rows.append({"i": i, "j": j, "status": "no_depth", "matches": int(mk0.shape[0]), "valid": 0})
                continue

            obj_i = data.object_masks[i][p0[:, 1], p0[:, 0]]
            obj_j = data.object_masks[j][p1[:, 1], p1[:, 0]]
            mov_i = data.moving_masks[i][p0[:, 1], p0[:, 0]]
            mov_j = data.moving_masks[j][p1[:, 1], p1[:, 0]]
            hand_i = data.hand_masks[i][p0[:, 1], p0[:, 0]]
            hand_j = data.hand_masks[j][p1[:, 1], p1[:, 0]]

            object_sel = (obj_i | mov_i) & (obj_j | mov_j)
            dynamic_sel = (mov_i | mov_j) & object_sel & (~hand_i) & (~hand_j)
            static_sel = object_sel & (~mov_i) & (~mov_j) & (~hand_i) & (~hand_j)

            world0 = transform_points(xyz0_cam, cam_poses[i])
            world1 = transform_points(xyz1_cam, cam_poses[j])
            dyn0 = world0[dynamic_sel]
            dyn1 = world1[dynamic_sel]
            dyn0, dyn1, dyn_dist = filter_correspondence_motion(dyn0, dyn1, args.max_pair_motion)

            row: dict[str, Any] = {
                "i": i,
                "j": j,
                "interval": interval,
                "status": "skip",
                "matches": int(mk0.shape[0]),
                "valid": int(p0.shape[0]),
                "dynamic": int(dynamic_sel.sum()),
                "dynamic_after_filter": int(dyn0.shape[0]),
                "static": int(static_sel.sum()),
                "conf_median": float(np.median(conf_valid)) if conf_valid.size else math.nan,
                "dyn_world_dist_median": float(np.median(dyn_dist)) if dyn_dist.size else math.nan,
                "dyn_world_dist_mean": float(np.mean(dyn_dist)) if dyn_dist.size else math.nan,
            }

            if dyn0.shape[0] >= args.min_dynamic:
                try:
                    result = estimate_joint_single(dyn0, dyn1, rng)
                    row.update(
                        {
                            "status": "used",
                            "revolute_error": float(result["revolute"]["error"]),
                            "revolute_valid": bool(result["revolute"]["valid"]),
                            "revolute_inliers": int(result["revolute"]["inliers"]),
                            "prismatic_error": float(result["prismatic"]["error"]),
                            "prismatic_valid": bool(result["prismatic"]["valid"]),
                            "prismatic_inliers": int(result["prismatic"]["inliers"]),
                        }
                    )
                    results.append(result)
                    result_pairs.append((i, j))
                except Exception as exc:
                    row.update({"status": "estimate_error", "error": repr(exc)})
            pair_rows.append(row)

        print(
            f"[itaco] interval={interval} done used={sum(1 for r in pair_rows if r.get('status') == 'used')} "
            f"pairs_seen={len(pair_rows)}",
            flush=True,
        )

    if results:
        metrics, pred_type = estimate_joint_all(results)
        for joint_type in ["revolute", "prismatic"]:
            per_frame_values: list[float] = []
            axis = metrics[joint_type]["axis"]
            pos = metrics[joint_type]["pos"]
            if np.all(np.isfinite(axis)) and np.linalg.norm(axis) > 1e-8:
                for pair, result in zip(result_pairs, results):
                    interval = pair[1] - pair[0]
                    if joint_type == "revolute" and result[joint_type]["valid"]:
                        val = compute_average_rotation_angle(result[joint_type]["X"], result[joint_type]["Y"], axis, pos) / interval
                    elif joint_type == "prismatic" and result[joint_type]["valid"]:
                        val = compute_average_translation_distance(result[joint_type]["X"], result[joint_type]["Y"], axis) / interval
                    else:
                        val = math.nan
                    if np.isfinite(val):
                        per_frame_values.append(float(val))
            metrics[joint_type]["average_value"] = float(np.mean(per_frame_values)) if per_frame_values else math.nan
            metrics[joint_type]["average_value_count"] = len(per_frame_values)
    else:
        metrics = {
            "revolute": {"axis": np.array([np.nan, np.nan, np.nan]), "pos": np.zeros(3), "error": math.inf, "valid_count": 0, "average_value": math.nan},
            "prismatic": {"axis": np.array([np.nan, np.nan, np.nan]), "pos": np.zeros(3), "error": math.inf, "valid_count": 0, "average_value": math.nan},
        }
        pred_type = "none"

    # Save diagnostics.
    csv_path = args.output_dir / "pair_metrics.csv"
    fieldnames = sorted({k for row in pair_rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    np.savez(
        args.output_dir / "joint_prediction.npz",
        pred_joint_type=np.array(pred_type),
        revolute_axis=metrics["revolute"]["axis"],
        revolute_pos=metrics["revolute"]["pos"],
        revolute_error=np.array(metrics["revolute"]["error"]),
        revolute_average_value=np.array(metrics["revolute"].get("average_value", math.nan)),
        prismatic_axis=metrics["prismatic"]["axis"],
        prismatic_pos=metrics["prismatic"]["pos"],
        prismatic_error=np.array(metrics["prismatic"]["error"]),
        prismatic_average_value=np.array(metrics["prismatic"].get("average_value", math.nan)),
        used_pairs=np.asarray(result_pairs, dtype=np.int64) if result_pairs else np.zeros((0, 2), dtype=np.int64),
        K=data.K,
    )
    save_prediction_tree(args.output_dir, "sam3_gt_camera", args.seed, metrics, pred_type, n, cam_poses)

    mask_area = {
        "object_mean": float(np.mean([m.sum() for m in data.object_masks])),
        "object_zero_frames": int(sum(m.sum() == 0 for m in data.object_masks)),
        "moving_mean": float(np.mean([m.sum() for m in data.moving_masks])),
        "moving_zero_frames": int(sum(m.sum() == 0 for m in data.moving_masks)),
        "hand_mean": float(np.mean([m.sum() for m in data.hand_masks])),
    }
    used_rows = [row for row in pair_rows if row.get("status") == "used"]
    summary = {
        "input_dir": str(args.input_dir),
        "mask_dir": str(args.mask_dir),
        "output_dir": str(args.output_dir),
        "frames": n,
        "intervals": intervals,
        "match_conf": args.match_conf,
        "min_dynamic": args.min_dynamic,
        "max_pair_motion": args.max_pair_motion,
        "pose_convention": convention,
        "pose_convention_scores": convention_scores,
        "mask_area": mask_area,
        "pairs_total": len(pair_rows),
        "pairs_used": len(used_rows),
        "pairs_by_status": {status: sum(1 for row in pair_rows if row.get("status") == status) for status in sorted({row.get("status") for row in pair_rows})},
        "dynamic_after_filter_mean": float(np.mean([row.get("dynamic_after_filter", 0) for row in pair_rows])) if pair_rows else 0.0,
        "dynamic_after_filter_median": float(np.median([row.get("dynamic_after_filter", 0) for row in pair_rows])) if pair_rows else 0.0,
        "pred_joint_type": pred_type,
        "metrics": {
            joint_type: {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in joint_metrics.items()
                if key not in {"X", "Y"}
            }
            for joint_type, joint_metrics in metrics.items()
        },
        "outputs": {
            "pair_metrics_csv": str(csv_path),
            "joint_prediction_npz": str(args.output_dir / "joint_prediction.npz"),
            "coarse_prediction_dir": str(args.output_dir / "coarse_prediction" / "sam3_gt_camera" / str(args.seed)),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))

    print(f"[itaco] used_pairs={len(used_rows)}/{len(pair_rows)} pred_joint_type={pred_type}")
    print(
        "[itaco] errors "
        f"revolute={metrics['revolute']['error']} prismatic={metrics['prismatic']['error']}"
    )
    print(f"[itaco] summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
