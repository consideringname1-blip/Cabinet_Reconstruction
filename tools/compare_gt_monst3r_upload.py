#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


GT_ROOT = Path("/workspace_whz/data/upload/2026-07-27-175228/pinhole_projection")
RUN_ROOT = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_official_monst3r_autoseg")
PRED_ROOT = RUN_ROOT / "preprocess" / "monst3r"
MANIFEST = RUN_ROOT / "input_rgb_jpg_manifest.json"
OUT_ROOT = RUN_ROOT / "comparison_gt_vs_monst3r"


def read_calibration(path: Path) -> dict[str, float]:
    vals = [float(x) for x in path.read_text().split()]
    if len(vals) < 4:
        raise ValueError(f"bad calibration: {path}")
    return {"fx": vals[0], "fy": vals[1], "cx": vals[2], "cy": vals[3]}


def read_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())


def gt_depth_path_from_rgb_source(source: str) -> Path:
    p = Path(source)
    return p.parent.parent / "depth" / p.name


def monst3r_224_crop_depth(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    # Matches monst3r/dust3r/utils/image.py::crop_img(size=224) for 320x288 inputs:
    # resize short side to 224 by setting long edge to round(224 * max(W/H, H/W)),
    # then center-crop a square.
    h1, w1 = depth.shape[:2]
    size = 224
    long_edge_size = round(size * max(w1 / h1, h1 / w1))
    s = max(w1, h1)
    w2 = int(round(w1 * long_edge_size / s))
    h2 = int(round(h1 * long_edge_size / s))
    resized = cv2.resize(depth, (w2, h2), interpolation=cv2.INTER_NEAREST)
    cx, cy = w2 // 2, h2 // 2
    half = min(cx, cy)
    cropped = resized[cy - half : cy + half, cx - half : cx + half]
    meta = {
        "input_wh": [w1, h1],
        "resized_wh": [w2, h2],
        "crop_xyxy": [cx - half, cy - half, cx + half, cy + half],
        "scale_x": w2 / w1,
        "scale_y": h2 / h1,
    }
    return cropped, meta


def transform_intrinsics_to_monst3r_224(k: dict[str, float], crop_meta: dict) -> np.ndarray:
    sx, sy = crop_meta["scale_x"], crop_meta["scale_y"]
    left, top, _, _ = crop_meta["crop_xyxy"]
    out = np.array(
        [
            [k["fx"] * sx, 0.0, k["cx"] * sx - left],
            [0.0, k["fy"] * sy, k["cy"] * sy - top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return out


def load_gt_poses(path: Path) -> np.ndarray:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    poses = []
    i = 0
    while i < len(lines):
        # header "idx idx idx"
        _ = lines[i]
        mat = []
        for j in range(1, 5):
            mat.append([float(x) for x in lines[i + j].split()])
        poses.append(np.array(mat, dtype=np.float64))
        i += 5
    return np.stack(poses, axis=0)


def qxyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q.astype(np.float64)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_pred_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    ts = arr[:, 0]
    xyz = arr[:, 1:4]
    R = np.stack([qxyzw_to_R(q) for q in arr[:, 4:8]], axis=0)
    return ts, xyz, R


def umeyama_similarity(x: np.ndarray, y: np.ndarray, allow_reflection: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, R, t for y ~= scale * R @ x + t."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    xc = x - mx
    yc = y - my
    cov = (yc.T @ xc) / n
    u, d, vt = np.linalg.svd(cov)
    sgn = np.ones(3)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        sgn[-1] = -1
    R = u @ np.diag(sgn) @ vt
    var_x = np.sum(xc * xc) / n
    scale = float(np.sum(d * sgn) / var_x)
    t = my - scale * (R @ mx)
    return scale, R, t


def path_length(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())


def rot_angle_deg(R: np.ndarray) -> float:
    val = (np.trace(R) - 1.0) * 0.5
    val = min(1.0, max(-1.0, float(val)))
    return math.degrees(math.acos(val))


def summarize(vals: np.ndarray, prefix: str) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(vals.size),
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_median": float(np.median(vals)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(vals * vals))),
        f"{prefix}_p90": float(np.percentile(vals, 90)),
        f"{prefix}_p95": float(np.percentile(vals, 95)),
        f"{prefix}_max": float(np.max(vals)),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    n = len(manifest)
    if n == 0:
        raise RuntimeError("empty manifest")

    k_gt = read_calibration(GT_ROOT / "calibration.txt")
    first_depth = cv2.imread(str(gt_depth_path_from_rgb_source(manifest[0]["source"])), cv2.IMREAD_UNCHANGED)
    first_crop, crop_meta = monst3r_224_crop_depth(first_depth)
    k_gt_224 = transform_intrinsics_to_monst3r_224(k_gt, crop_meta)
    k_pred_all = np.loadtxt(PRED_ROOT / "pred_intrinsics.txt").reshape(-1, 3, 3)
    k_pred_mean = k_pred_all.mean(axis=0)
    k_pred_std = k_pred_all.std(axis=0)

    depth_pairs = []
    per_frame_rows = []
    sum_pd_gt = 0.0
    sum_pd2 = 0.0
    total_valid = 0
    total_pixels = 0
    sample_ratios = []
    rng = np.random.default_rng(7)

    for item in manifest:
        idx = int(item["index"])
        gt_path = gt_depth_path_from_rgb_source(item["source"])
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            raise FileNotFoundError(gt_path)
        gt_crop, _ = monst3r_224_crop_depth(gt_raw)
        gt = gt_crop.astype(np.float32)
        if np.nanmax(gt) > 20:
            gt *= 0.001  # uint16 mm -> meters
        pred = np.load(PRED_ROOT / f"frame_{idx:04d}.npy").astype(np.float32)
        if pred.shape != gt.shape:
            raise ValueError((idx, pred.shape, gt.shape))
        valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.05) & (gt < 8.0) & (pred > 1e-6)
        total_pixels += int(valid.size)
        nvalid = int(valid.sum())
        total_valid += nvalid
        if nvalid:
            p = pred[valid].astype(np.float64)
            g = gt[valid].astype(np.float64)
            sum_pd_gt += float(np.dot(p, g))
            sum_pd2 += float(np.dot(p, p))
            ratios = g / p
            if ratios.size > 5000:
                ratios = rng.choice(ratios, size=5000, replace=False)
            sample_ratios.append(ratios.astype(np.float32))
            s_frame = float(np.median(g / p))
            e_direct = p - g
            e_scaled = s_frame * p - g
            per_frame_rows.append(
                {
                    "index": idx,
                    "valid_pixels": nvalid,
                    "valid_fraction": nvalid / valid.size,
                    "gt_median_m": float(np.median(g)),
                    "pred_median_raw": float(np.median(p)),
                    "frame_median_scale_gt_over_pred": s_frame,
                    "direct_mae_m": float(np.mean(np.abs(e_direct))),
                    "direct_rmse_m": float(np.sqrt(np.mean(e_direct * e_direct))),
                    "frame_scaled_mae_m": float(np.mean(np.abs(e_scaled))),
                    "frame_scaled_rmse_m": float(np.sqrt(np.mean(e_scaled * e_scaled))),
                    "frame_scaled_absrel": float(np.mean(np.abs(e_scaled) / g)),
                }
            )
            # Store full arrays for global metrics. This is ~9M pixels, acceptable here.
            depth_pairs.append((p.astype(np.float32), g.astype(np.float32)))

    all_pred = np.concatenate([p for p, _ in depth_pairs]).astype(np.float64)
    all_gt = np.concatenate([g for _, g in depth_pairs]).astype(np.float64)
    scale_l2 = float(sum_pd_gt / sum_pd2)
    scale_median = float(np.median(np.concatenate(sample_ratios)))

    direct_err = all_pred - all_gt
    scaled_err = scale_l2 * all_pred - all_gt
    scaled_absrel = np.abs(scaled_err) / all_gt
    direct_absrel = np.abs(direct_err) / all_gt

    gt_poses = load_gt_poses(GT_ROOT / "odometry.log")
    gt_xyz = gt_poses[:, :3, 3]
    gt_R = gt_poses[:, :3, :3]
    _, pred_xyz, pred_R = load_pred_tum(PRED_ROOT / "pred_traj.txt")
    m = min(len(gt_xyz), len(pred_xyz), n)
    gt_xyz = gt_xyz[:m]
    gt_R = gt_R[:m]
    pred_xyz = pred_xyz[:m]
    pred_R = pred_R[:m]

    sim_scale, sim_R, sim_t = umeyama_similarity(pred_xyz, gt_xyz, allow_reflection=False)
    pred_xyz_aligned = (sim_scale * (sim_R @ pred_xyz.T)).T + sim_t
    ate = np.linalg.norm(pred_xyz_aligned - gt_xyz, axis=1)
    sim_scale_reflect, sim_R_reflect, sim_t_reflect = umeyama_similarity(pred_xyz, gt_xyz, allow_reflection=True)
    pred_reflect_aligned = (sim_scale_reflect * (sim_R_reflect @ pred_xyz.T)).T + sim_t_reflect
    ate_reflect = np.linalg.norm(pred_reflect_aligned - gt_xyz, axis=1)

    # Orientation after applying the global rotation from position alignment.
    ori_err = np.array([rot_angle_deg((sim_R @ pred_R[i]) @ gt_R[i].T) for i in range(m)], dtype=np.float64)
    gt_delta = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    pred_delta_scaled = sim_scale * np.linalg.norm(np.diff(pred_xyz, axis=0), axis=1)
    step_err = pred_delta_scaled - gt_delta
    gt_rel_rot = np.array([rot_angle_deg(gt_R[i].T @ gt_R[i + 1]) for i in range(m - 1)], dtype=np.float64)
    pred_rel_rot = np.array([rot_angle_deg(pred_R[i].T @ pred_R[i + 1]) for i in range(m - 1)], dtype=np.float64)
    rel_rot_angle_absdiff = np.abs(pred_rel_rot - gt_rel_rot)

    summary = {
        "paths": {
            "gt_root": str(GT_ROOT),
            "pred_root": str(PRED_ROOT),
            "manifest": str(MANIFEST),
            "out_root": str(OUT_ROOT),
        },
        "frame_count": n,
        "monst3r_image_size": [224, 224],
        "crop_meta": crop_meta,
        "gt_depth_units_assumed": "meters after dividing GT uint16 depth by 1000 when max>20",
        "intrinsics": {
            "gt_original_320x288": k_gt,
            "gt_transformed_to_monst3r_224": k_gt_224.tolist(),
            "pred_mean_224": k_pred_mean.tolist(),
            "pred_std_224": k_pred_std.tolist(),
            "fx_ratio_pred_over_gt224": float(k_pred_mean[0, 0] / k_gt_224[0, 0]),
            "fy_ratio_pred_over_gt224": float(k_pred_mean[1, 1] / k_gt_224[1, 1]),
            "cx_diff_px_pred_minus_gt224": float(k_pred_mean[0, 2] - k_gt_224[0, 2]),
            "cy_diff_px_pred_minus_gt224": float(k_pred_mean[1, 2] - k_gt_224[1, 2]),
            "hfov_deg_gt224": float(math.degrees(2 * math.atan(224 / (2 * k_gt_224[0, 0])))),
            "hfov_deg_pred": float(math.degrees(2 * math.atan(224 / (2 * k_pred_mean[0, 0])))),
            "vfov_deg_gt224": float(math.degrees(2 * math.atan(224 / (2 * k_gt_224[1, 1])))),
            "vfov_deg_pred": float(math.degrees(2 * math.atan(224 / (2 * k_pred_mean[1, 1])))),
        },
        "depth": {
            "valid_pixels": total_valid,
            "valid_fraction": total_valid / total_pixels,
            "global_l2_scale_gt_over_pred": scale_l2,
            "sample_median_scale_gt_over_pred": scale_median,
            "gt_median_m": float(np.median(all_gt)),
            "pred_median_raw": float(np.median(all_pred)),
            "direct": {
                "mae_m": float(np.mean(np.abs(direct_err))),
                "rmse_m": float(np.sqrt(np.mean(direct_err * direct_err))),
                "median_abs_m": float(np.median(np.abs(direct_err))),
                "p90_abs_m": float(np.percentile(np.abs(direct_err), 90)),
                "absrel_mean": float(np.mean(direct_absrel)),
                "absrel_median": float(np.median(direct_absrel)),
            },
            "global_scale_aligned": {
                "mae_m": float(np.mean(np.abs(scaled_err))),
                "rmse_m": float(np.sqrt(np.mean(scaled_err * scaled_err))),
                "median_abs_m": float(np.median(np.abs(scaled_err))),
                "p90_abs_m": float(np.percentile(np.abs(scaled_err), 90)),
                "absrel_mean": float(np.mean(scaled_absrel)),
                "absrel_median": float(np.median(scaled_absrel)),
            },
            "per_frame_frame_scale_aligned": {
                "mae_m_mean": float(np.mean([r["frame_scaled_mae_m"] for r in per_frame_rows])),
                "rmse_m_mean": float(np.mean([r["frame_scaled_rmse_m"] for r in per_frame_rows])),
                "absrel_mean": float(np.mean([r["frame_scaled_absrel"] for r in per_frame_rows])),
                "median_scale_gt_over_pred_mean": float(np.mean([r["frame_median_scale_gt_over_pred"] for r in per_frame_rows])),
                "median_scale_gt_over_pred_std": float(np.std([r["frame_median_scale_gt_over_pred"] for r in per_frame_rows])),
            },
        },
        "trajectory": {
            "frame_count": m,
            "gt_path_length_m": path_length(gt_xyz),
            "pred_path_length_raw": path_length(pred_xyz),
            "path_length_ratio_gt_over_pred": path_length(gt_xyz) / path_length(pred_xyz),
            "sim3_scale_gt_over_pred": sim_scale,
            "sim3_det_rotation": float(np.linalg.det(sim_R)),
            "ate_sim3_m": summarize(ate, "ate_m"),
            "ate_allow_reflection_m": summarize(ate_reflect, "ate_m"),
            "step_length_after_sim3": {
                "gt_step_mean_m": float(np.mean(gt_delta)),
                "pred_scaled_step_mean_m": float(np.mean(pred_delta_scaled)),
                "step_mae_m": float(np.mean(np.abs(step_err))),
                "step_rmse_m": float(np.sqrt(np.mean(step_err * step_err))),
                "step_median_abs_m": float(np.median(np.abs(step_err))),
                "step_p90_abs_m": float(np.percentile(np.abs(step_err), 90)),
            },
            "orientation": {
                "absolute_after_position_global_rotation_deg": summarize(ori_err, "ori_deg"),
                "relative_step_angle_absdiff_deg": summarize(rel_rot_angle_absdiff, "rel_rot_absdiff_deg"),
                "gt_rel_rot_mean_deg": float(np.mean(gt_rel_rot)),
                "pred_rel_rot_mean_deg": float(np.mean(pred_rel_rot)),
            },
        },
    }

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT_ROOT / "per_frame_depth.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
