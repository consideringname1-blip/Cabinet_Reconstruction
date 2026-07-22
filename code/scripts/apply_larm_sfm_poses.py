from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
LARM_ROOT = CODE_ROOT / "LARM"
AXIS_EST_ROOT = LARM_ROOT / "axis_est"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(AXIS_EST_ROOT) not in sys.path:
    sys.path.insert(0, str(AXIS_EST_ROOT))

from task_json import save_task_json  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_records(meta: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for qtok in sorted(meta["inputs"].keys(), key=lambda x: float(x)):
        for frame_key in sorted(meta["inputs"][qtok].keys(), key=lambda x: int(x.rsplit("_", 1)[-1])):
            rec = dict(meta["inputs"][qtok][frame_key])
            rec["qtok"] = qtok
            rec["frame_key"] = frame_key
            records.append(rec)
    if len(records) != 6:
        raise ValueError(f"Expected 6 LARM input frames, got {len(records)}")
    return records


def _target_camera_radius(k: np.ndarray, resolution: int = 512) -> float:
    fy = float(k[1, 1])
    fovy = 2.0 * np.arctan(resolution / (2.0 * fy))
    return float((0.5 / np.tan(fovy / 2.0)) * 1.2)


def _load_images(records: list[dict[str, Any]]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for rec in records:
        path = Path(rec["image_path"])
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images


def _recover_pair_pose(
    runner: Any,
    image0: np.ndarray,
    image1: np.ndarray,
    k: np.ndarray,
    conf_threshold: float,
    ransac_threshold: float,
    min_pose_inliers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cor = runner.predict(np.stack([image0]), np.stack([image1]))[0]
    raw_matches = int(len(cor))
    if len(cor):
        cor = cor[cor[:, 4] >= conf_threshold]
    if len(cor) < 8:
        raise RuntimeError(f"LoFTR produced only {len(cor)} matches after filtering")

    pts0 = cor[:, :2].astype(np.float64)
    pts1 = cor[:, 2:4].astype(np.float64)
    essential, mask_e = cv2.findEssentialMat(
        pts0,
        pts1,
        k,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )
    if essential is None or mask_e is None:
        raise RuntimeError("findEssentialMat failed")
    if essential.shape[0] > 3:
        essential = essential[:3]

    pose_inliers, r, t, mask_pose = cv2.recoverPose(essential, pts0, pts1, k, mask=mask_e)
    pose_inliers = int(pose_inliers)
    if pose_inliers < min_pose_inliers:
        raise RuntimeError(f"recoverPose had only {pose_inliers} inliers")

    keep = mask_pose.ravel() > 0
    pts0_in = pts0[keep]
    pts1_in = pts1[keep]
    p0 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p1 = k @ np.hstack([r, t.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(p0, p1, pts0_in.T, pts1_in.T)
    denom = pts4[3:4].copy()
    denom[np.abs(denom) < 1e-8] = 1e-8
    pts3 = (pts4[:3] / denom).T
    pts3_cam1 = (r @ pts3.T + t.reshape(3, 1)).T
    good3d = (
        np.isfinite(pts3).all(axis=1)
        & (pts3[:, 2] > 1e-5)
        & (pts3_cam1[:, 2] > 1e-5)
        & (np.linalg.norm(pts3, axis=1) < 100.0)
    )
    pts3 = pts3[good3d]
    if len(pts3) == 0:
        raise RuntimeError("Triangulation produced no finite points")

    diagnostics = {
        "raw_matches": raw_matches,
        "filtered_matches": int(len(cor)),
        "essential_inliers": int(mask_e.ravel().sum()),
        "pose_inliers": pose_inliers,
        "triangulated_points": int(len(pts3)),
    }
    return r.astype(np.float64), t.reshape(3).astype(np.float64), pts3.astype(np.float64), diagnostics


def _estimate_sfm_poses(
    images: list[np.ndarray],
    k: np.ndarray,
    conf_threshold: float,
    ransac_threshold: float,
    min_pose_inliers: int,
    target_radius: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    old_cwd = Path.cwd()
    os.chdir(AXIS_EST_ROOT)
    try:
        from loftr import LoftrRunner

        runner = LoftrRunner()
    finally:
        os.chdir(old_cwd)

    w2c: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    points_all: list[np.ndarray] = []
    pair_diagnostics: dict[str, Any] = {}
    for idx in range(1, len(images)):
        r, t, pts3, diag = _recover_pair_pose(
            runner,
            images[0],
            images[idx],
            k,
            conf_threshold,
            ransac_threshold,
            min_pose_inliers,
        )
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = r
        mat[:3, 3] = t
        w2c.append(mat)
        points_all.append(pts3[:: max(1, len(pts3) // 5000)])
        pair_diagnostics[f"0-{idx}"] = diag

    sparse = np.concatenate(points_all, axis=0)
    lo = np.percentile(sparse, 5.0, axis=0)
    hi = np.percentile(sparse, 95.0, axis=0)
    sparse = sparse[np.all((sparse >= lo) & (sparse <= hi), axis=1)]
    object_center = np.median(sparse, axis=0)

    c2w_cv: list[np.ndarray] = []
    centers = []
    for mat in w2c:
        r = mat[:3, :3]
        t = mat[:3, 3]
        center = -r.T @ t
        centers.append(center)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = r.T
        pose[:3, 3] = center
        c2w_cv.append(pose)
    centers_arr = np.stack(centers, axis=0)
    radii_raw = np.linalg.norm(centers_arr - object_center[None, :], axis=1)
    median_radius = float(np.median(radii_raw[radii_raw > 1e-6]))
    if not np.isfinite(median_radius) or median_radius <= 1e-6:
        raise RuntimeError(f"Invalid SfM camera radius: {median_radius}")
    scale = float(target_radius / median_radius)

    normalized: list[np.ndarray] = []
    for pose in c2w_cv:
        out = pose.copy()
        out[:3, 3] = (out[:3, 3] - object_center) * scale
        normalized.append(out)

    diagnostics = {
        "reference_frame": 0,
        "pair_diagnostics": pair_diagnostics,
        "object_center_sfm": [float(v) for v in object_center.tolist()],
        "raw_camera_radii": [float(v) for v in radii_raw.tolist()],
        "raw_median_camera_radius": median_radius,
        "target_camera_radius": float(target_radius),
        "scale": scale,
        "normalized_camera_radii": [float(np.linalg.norm(p[:3, 3])) for p in normalized],
        "conf_threshold": float(conf_threshold),
        "ransac_threshold": float(ransac_threshold),
    }
    return normalized, diagnostics


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.2)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--min-pose-inliers", type=int, default=40)
    parser.add_argument("--target-camera-radius", type=float, default=0.0)
    args = parser.parse_args()

    input_json = args.input_json.resolve()
    meta = _load_json(input_json)
    records = _ordered_records(meta)
    images = _load_images(records)
    k = np.asarray(meta["intrinsics"], dtype=np.float64)
    target_radius = float(args.target_camera_radius) if args.target_camera_radius > 0 else _target_camera_radius(k)
    c2w_cv, diagnostics = _estimate_sfm_poses(
        images,
        k,
        args.conf_threshold,
        args.ransac_threshold,
        args.min_pose_inliers,
        target_radius,
    )

    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    output_root = (CODE_ROOT.parent / "data" / "upload" / "larm" / args.output_name).resolve()
    images_root = output_root / "images"
    sam3_root = output_root / "sam3"
    images_root.mkdir(parents=True, exist_ok=True)
    sam3_root.mkdir(parents=True, exist_ok=True)

    out_meta = dict(meta)
    out_meta["inputs"] = {qtok: {} for qtok in sorted(meta["inputs"].keys(), key=lambda x: float(x))}
    out_meta["pose_source"] = "loftr_essential_sfm_from_black_segmented_images"
    out_meta["sfm_pose_diagnostics"] = diagnostics

    for rec, pose_cv in zip(records, c2w_cv):
        src_img = Path(rec["image_path"])
        dst_img = images_root / src_img.name
        _copy_or_link(src_img, dst_img)
        pose_blender = pose_cv @ blender2opencv
        out_meta["inputs"][rec["qtok"]][rec["frame_key"]] = {
            "transform_matrix": [[float(v) for v in row] for row in pose_blender.tolist()],
            "image_path": str(dst_img),
            "qpos": rec["qpos"],
        }

    grouped = []
    for item in meta.get("grouped_captures", []):
        item = dict(item)
        if item.get("image_path"):
            item["image_path"] = str(images_root / Path(item["image_path"]).name)
        if item.get("mask_path"):
            src = Path(item["mask_path"])
            dst = sam3_root / src.name
            _copy_or_link(src, dst)
            item["mask_path"] = str(dst)
        if item.get("overlay_path"):
            src = Path(item["overlay_path"])
            dst = sam3_root / src.name
            _copy_or_link(src, dst)
            item["overlay_path"] = str(dst)
        grouped.append(item)
    out_meta["grouped_captures"] = grouped

    out_json = output_root / f"{args.output_name}.json"
    save_task_json(out_json, out_meta)
    data_txt = output_root / "data.txt"
    data_txt.write_text(str(out_json) + "\n", encoding="utf-8")
    diag_path = output_root / "sfm_pose_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"metadata_json": str(out_json), "datalist_path": str(data_txt), "diagnostics": str(diag_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
