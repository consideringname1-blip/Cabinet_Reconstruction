#!/usr/bin/env python3
"""Compare Video2Articulation RGBD ground truth with VGGT predictions.

The comparison is performed in VGGT's preprocessed image plane. Ground-truth
depth and intrinsics are resized/cropped/padded the same way as VGGT input
images, then both ground truth and VGGT depth maps are back-projected into
OpenCV camera coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = 518
PATCH_SIZE = 14
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PreprocessGeometry:
    mode: str
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    final_width: int
    final_height: int
    scale_x: float
    scale_y: float
    crop_left: int
    crop_top: int
    pad_left: int
    pad_top: int


@dataclass(frozen=True)
class FrameSample:
    frame_id: int
    image_path: Path
    depth_path: Path


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else 0), path.name


def sorted_images(image_dir: Path) -> list[Path]:
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=numeric_key)


def resolve_depth_path(depth_dir: Path, frame_id: int, stem: str) -> Path:
    candidates = [
        depth_dir / f"{frame_id:06d}.npz",
        depth_dir / f"{frame_id}.npz",
        depth_dir / f"{stem}.npz",
        depth_dir / f"{frame_id:06d}.npy",
        depth_dir / f"{frame_id}.npy",
        depth_dir / f"{stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    depth_files = sorted(
        [p for p in depth_dir.iterdir() if p.is_file() and p.suffix.lower() in {".npz", ".npy"}],
        key=numeric_key,
    )
    if 0 <= frame_id < len(depth_files):
        return depth_files[frame_id]

    raise FileNotFoundError(f"No depth file found for frame {frame_id} in {depth_dir}")


def discover_frames(view_dir: Path, image_source: str, max_frames: int | None, stride: int) -> list[FrameSample]:
    image_dir = view_dir / image_source
    depth_dir = view_dir / "depth"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    frames: list[FrameSample] = []
    for image_path in sorted_images(image_dir):
        frame_id = numeric_key(image_path)[0]
        frames.append(FrameSample(frame_id, image_path, resolve_depth_path(depth_dir, frame_id, image_path.stem)))

    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No image frames found in {image_dir}")
    return frames


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        loaded = np.load(path)
        key = "a" if "a" in loaded.files else loaded.files[0]
        depth = loaded[key]
    else:
        depth = np.load(path)

    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map at {path}, got shape {depth.shape}")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)
    return depth


def vggt_preprocess_geometry(width: int, height: int, mode: str) -> PreprocessGeometry:
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be 'crop' or 'pad'")

    if mode == "pad":
        if width >= height:
            resized_width = TARGET_SIZE
            resized_height = round(height * (resized_width / width) / PATCH_SIZE) * PATCH_SIZE
        else:
            resized_height = TARGET_SIZE
            resized_width = round(width * (resized_height / height) / PATCH_SIZE) * PATCH_SIZE
        final_width = TARGET_SIZE
        final_height = TARGET_SIZE
        pad_left = (final_width - resized_width) // 2
        pad_top = (final_height - resized_height) // 2
        crop_left = 0
        crop_top = 0
    else:
        resized_width = TARGET_SIZE
        resized_height = round(height * (resized_width / width) / PATCH_SIZE) * PATCH_SIZE
        crop_top = (resized_height - TARGET_SIZE) // 2 if resized_height > TARGET_SIZE else 0
        crop_left = 0
        pad_left = 0
        pad_top = 0
        final_width = resized_width
        final_height = min(resized_height, TARGET_SIZE)

    return PreprocessGeometry(
        mode=mode,
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        final_width=final_width,
        final_height=final_height,
        scale_x=resized_width / width,
        scale_y=resized_height / height,
        crop_left=crop_left,
        crop_top=crop_top,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def transform_intrinsics(K: np.ndarray, geom: PreprocessGeometry) -> np.ndarray:
    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 0] *= geom.scale_x
    K_out[1, 1] *= geom.scale_y
    K_out[0, 2] = K_out[0, 2] * geom.scale_x - geom.crop_left + geom.pad_left
    K_out[1, 2] = K_out[1, 2] * geom.scale_y - geom.crop_top + geom.pad_top
    return K_out


def preprocess_depth(depth: np.ndarray, geom: PreprocessGeometry, interpolation: str) -> np.ndarray:
    interp = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    resized = cv2.resize(depth, (geom.resized_width, geom.resized_height), interpolation=interp)
    if geom.mode == "crop":
        if geom.crop_top:
            resized = resized[geom.crop_top : geom.crop_top + geom.final_height, :]
        return resized.astype(np.float32)

    out = np.zeros((geom.final_height, geom.final_width), dtype=np.float32)
    y0 = geom.pad_top
    x0 = geom.pad_left
    out[y0 : y0 + geom.resized_height, x0 : x0 + geom.resized_width] = resized
    return out


def backproject_opencv(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def finite_positive_mask(gt_depth: np.ndarray, pred_depth: np.ndarray, pred_conf: np.ndarray | None, conf_percentile: float) -> np.ndarray:
    mask = np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 1e-6) & (pred_depth > 1e-6)
    if pred_conf is not None and conf_percentile > 0 and np.any(mask):
        conf_values = pred_conf[mask]
        threshold = np.percentile(conf_values, conf_percentile)
        mask &= pred_conf >= threshold
    return mask


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if not np.any(mask):
        return {
            "valid_pixels": 0,
            "mae": math.nan,
            "median_abs": math.nan,
            "rmse": math.nan,
            "abs_rel": math.nan,
            "mean_signed": math.nan,
            "delta_1_25": math.nan,
        }
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    err = p - g
    abs_err = np.abs(err)
    ratio = np.maximum(p / np.maximum(g, 1e-12), g / np.maximum(p, 1e-12))
    return {
        "valid_pixels": int(mask.sum()),
        "mae": float(abs_err.mean()),
        "median_abs": float(np.median(abs_err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "abs_rel": float(np.mean(abs_err / np.maximum(g, 1e-12))),
        "mean_signed": float(err.mean()),
        "delta_1_25": float(np.mean(ratio < 1.25)),
    }


def point_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if not np.any(mask):
        return {
            "valid_pixels": 0,
            "mean_l2": math.nan,
            "median_l2": math.nan,
            "rmse_l2": math.nan,
            "abs_rel_l2": math.nan,
        }
    diff = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    l2 = np.linalg.norm(diff, axis=-1)
    gt_norm = np.linalg.norm(gt[mask].astype(np.float64), axis=-1)
    return {
        "valid_pixels": int(mask.sum()),
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "rmse_l2": float(np.sqrt(np.mean(l2 * l2))),
        "abs_rel_l2": float(np.mean(l2 / np.maximum(gt_norm, 1e-12))),
    }


def aggregate_metric_dicts(items: Iterable[dict[str, float | int]], prefix: str) -> dict[str, float]:
    rows = list(items)
    out: dict[str, float] = {}
    if not rows:
        return out
    keys = [k for k in rows[0].keys() if k != "valid_pixels"]
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{prefix}_{key}_mean"] = float(np.nanmean(values))
        out[f"{prefix}_{key}_median"] = float(np.nanmedian(values))
    out[f"{prefix}_valid_pixels_total"] = float(sum(int(row["valid_pixels"]) for row in rows))
    return out


def median_scale(pred_depth: np.ndarray, gt_depth: np.ndarray, masks: np.ndarray) -> float:
    if not np.any(masks):
        return math.nan
    ratios = gt_depth[masks].astype(np.float64) / np.maximum(pred_depth[masks].astype(np.float64), 1e-12)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    return float(np.median(ratios)) if ratios.size else math.nan


def colorize(values: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    safe = np.nan_to_num(values, nan=vmin, posinf=vmax, neginf=vmin)
    scaled = np.clip((safe - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    img = (scaled * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    bgr[~valid] = (0, 0, 0)
    return bgr


def add_label(bgr: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((28, bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, bgr])


def save_visual_panels(
    output_dir: Path,
    frames: list[FrameSample],
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    pred_depth_scaled: np.ndarray,
    valid_masks: np.ndarray,
    max_visuals: int,
) -> list[str]:
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    limit = min(max_visuals, len(frames))
    if limit <= 0:
        return paths

    valid_values = gt_depth[valid_masks]
    if valid_values.size:
        vmin = float(np.percentile(valid_values, 2))
        vmax = float(np.percentile(valid_values, 98))
    else:
        vmin, vmax = 0.0, 1.0

    err = np.abs(pred_depth_scaled - gt_depth)
    valid_err = err[valid_masks]
    err_max = float(np.percentile(valid_err, 95)) if valid_err.size else 1.0

    for idx in range(limit):
        mask = valid_masks[idx]
        panels = [
            add_label(colorize(gt_depth[idx], mask, vmin, vmax), "GT depth"),
            add_label(colorize(pred_depth[idx], mask, vmin, vmax), "VGGT depth raw"),
            add_label(colorize(pred_depth_scaled[idx], mask, vmin, vmax), "VGGT depth scaled"),
            add_label(colorize(err[idx], mask, 0.0, err_max), "abs error scaled"),
        ]
        panel = cv2.hconcat(panels)
        out_path = maps_dir / f"frame_{frames[idx].frame_id:06d}_depth_compare.png"
        cv2.imwrite(str(out_path), panel)
        paths.append(str(out_path))
    return paths


def run_vggt(image_paths: list[Path], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    import torch

    sys.path.insert(0, str(args.vggt_repo))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    model = VGGT.from_pretrained(str(args.model_dir))
    model.eval().to(device)
    timings["model_load_seconds"] = time.perf_counter() - t0

    images = load_and_preprocess_images([str(p) for p in image_paths], mode=args.preprocess_mode).to(device)
    dtype = torch.float32
    amp_ctx = nullcontext()
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_ctx = torch.cuda.amp.autocast(dtype=dtype)

    t0 = time.perf_counter()
    with torch.no_grad():
        with amp_ctx:
            predictions = model(images)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    if device == "cuda":
        torch.cuda.synchronize()
    timings["inference_seconds"] = time.perf_counter() - t0
    timings["seconds_per_frame"] = timings["inference_seconds"] / len(image_paths)

    out: dict[str, np.ndarray] = {
        "depth": predictions["depth"].detach().cpu().numpy().squeeze(0).squeeze(-1).astype(np.float32),
        "depth_conf": predictions["depth_conf"].detach().cpu().numpy().squeeze(0).astype(np.float32),
        "world_points": predictions["world_points"].detach().cpu().numpy().squeeze(0).astype(np.float32),
        "world_points_conf": predictions["world_points_conf"].detach().cpu().numpy().squeeze(0).astype(np.float32),
        "extrinsic": extrinsic.detach().cpu().numpy().squeeze(0).astype(np.float32),
        "intrinsic": intrinsic.detach().cpu().numpy().squeeze(0).astype(np.float32),
        "input_shape_hw": np.array(images.shape[-2:], dtype=np.int32),
    }
    return out, timings


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view_dir", required=True, type=Path, help="Video2Articulation view directory, e.g. .../view_0")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for metrics, npz, and visual panels")
    parser.add_argument("--vggt_repo", type=Path, default=Path("/workspace_whz/code/reconstruction/vggt"))
    parser.add_argument("--model_dir", type=Path, default=Path(os.environ.get("VGGT_MODEL_DIR", "/workspace_whz/models/vggt/VGGT-1B")))
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 1")
    parser.add_argument("--image_source", choices=["sample_rgb", "rgb"], default="sample_rgb")
    parser.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--depth_resize", choices=["nearest", "linear"], default="nearest")
    parser.add_argument("--max_frames", type=int, default=8, help="Limit frames for memory/runtime. Use 0 for all frames.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf_percentile", type=float, default=0.0, help="Drop lower VGGT depth confidence percentile before scoring.")
    parser.add_argument("--max_visuals", type=int, default=4)
    parser.add_argument("--save_npz", action="store_true", help="Save dense GT and VGGT arrays as compressed npz.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    t_start = time.perf_counter()
    view_dir = args.view_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_frames = None if args.max_frames == 0 else args.max_frames
    frames = discover_frames(view_dir, args.image_source, max_frames, args.stride)
    with Image.open(frames[0].image_path) as img:
        original_width, original_height = img.size
    geom = vggt_preprocess_geometry(original_width, original_height, args.preprocess_mode)

    K = np.load(view_dir / "intrinsics.npy").astype(np.float32)
    K_gt = transform_intrinsics(K, geom)
    K_gt_stack = np.repeat(K_gt[None], len(frames), axis=0)

    gt_depths: list[np.ndarray] = []
    gt_cam_points: list[np.ndarray] = []
    warnings: list[str] = []
    for sample in frames:
        depth = load_depth(sample.depth_path)
        if depth.shape != (original_height, original_width):
            warnings.append(
                f"Depth shape {depth.shape} for frame {sample.frame_id} does not match image {(original_height, original_width)}; resized first."
            )
            depth = cv2.resize(depth, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        depth_pre = preprocess_depth(depth, geom, args.depth_resize)
        gt_depths.append(depth_pre)
        gt_cam_points.append(backproject_opencv(depth_pre, K_gt))

    gt_depth = np.stack(gt_depths)
    gt_cam = np.stack(gt_cam_points)

    pred, timings = run_vggt([f.image_path for f in frames], args)
    pred_depth = pred["depth"]
    pred_K = pred["intrinsic"]
    pred_cam = np.stack([backproject_opencv(pred_depth[i], pred_K[i]) for i in range(len(frames))])

    if tuple(pred["input_shape_hw"]) != (geom.final_height, geom.final_width):
        raise RuntimeError(
            f"Preprocess mismatch: expected {(geom.final_height, geom.final_width)}, VGGT saw {tuple(pred['input_shape_hw'])}"
        )

    valid_masks = np.stack(
        [
            finite_positive_mask(gt_depth[i], pred_depth[i], pred["depth_conf"][i], args.conf_percentile)
            for i in range(len(frames))
        ]
    )
    global_scale = median_scale(pred_depth, gt_depth, valid_masks)
    pred_depth_scaled = pred_depth * global_scale
    pred_cam_scaled = pred_cam * global_scale

    rows: list[dict[str, object]] = []
    raw_depth_items = []
    scaled_depth_items = []
    raw_point_items = []
    scaled_point_items = []
    for i, sample in enumerate(frames):
        raw_depth = depth_metrics(pred_depth[i], gt_depth[i], valid_masks[i])
        scaled_depth = depth_metrics(pred_depth_scaled[i], gt_depth[i], valid_masks[i])
        raw_point = point_metrics(pred_cam[i], gt_cam[i], valid_masks[i])
        scaled_point = point_metrics(pred_cam_scaled[i], gt_cam[i], valid_masks[i])
        raw_depth_items.append(raw_depth)
        scaled_depth_items.append(scaled_depth)
        raw_point_items.append(raw_point)
        scaled_point_items.append(scaled_point)

        rows.append(
            {
                "frame_id": sample.frame_id,
                "image": str(sample.image_path),
                "depth": str(sample.depth_path),
                "valid_pixels": raw_depth["valid_pixels"],
                "depth_mae_raw": raw_depth["mae"],
                "depth_rmse_raw": raw_depth["rmse"],
                "depth_abs_rel_raw": raw_depth["abs_rel"],
                "depth_mae_scaled": scaled_depth["mae"],
                "depth_rmse_scaled": scaled_depth["rmse"],
                "depth_abs_rel_scaled": scaled_depth["abs_rel"],
                "cam_l2_mean_raw": raw_point["mean_l2"],
                "cam_l2_rmse_raw": raw_point["rmse_l2"],
                "cam_l2_mean_scaled": scaled_point["mean_l2"],
                "cam_l2_rmse_scaled": scaled_point["rmse_l2"],
                "gt_fx": K_gt[0, 0],
                "gt_fy": K_gt[1, 1],
                "gt_cx": K_gt[0, 2],
                "gt_cy": K_gt[1, 2],
                "vggt_fx": pred_K[i, 0, 0],
                "vggt_fy": pred_K[i, 1, 1],
                "vggt_cx": pred_K[i, 0, 2],
                "vggt_cy": pred_K[i, 1, 2],
                "fx_delta": pred_K[i, 0, 0] - K_gt[0, 0],
                "fy_delta": pred_K[i, 1, 1] - K_gt[1, 1],
                "cx_delta": pred_K[i, 0, 2] - K_gt[0, 2],
                "cy_delta": pred_K[i, 1, 2] - K_gt[1, 2],
            }
        )

    csv_path = output_dir / "per_frame_metrics.csv"
    write_csv(csv_path, rows)
    visual_paths = save_visual_panels(output_dir, frames, gt_depth, pred_depth, pred_depth_scaled, valid_masks, args.max_visuals)

    K_delta = pred_K - K_gt_stack
    focal_rel = np.stack([K_delta[:, 0, 0] / K_gt[0, 0], K_delta[:, 1, 1] / K_gt[1, 1]], axis=1)
    summary = {
        "view_dir": str(view_dir),
        "frames": [sample.frame_id for sample in frames],
        "image_source": args.image_source,
        "preprocess_geometry": asdict(geom),
        "coordinate_convention": {
            "comparison": "OpenCV camera coordinates: x right, y down, z forward",
            "note": "Video2Articulation synthetic loader often converts points to OpenGL with [x, -y, -z]; this script compares OpenCV back-projections for consistency with VGGT depth/intrinsics.",
        },
        "scale_alignment": {
            "method": "global median(gt_depth / vggt_depth) over valid pixels",
            "scale": global_scale,
        },
        "aggregate_metrics": {
            **aggregate_metric_dicts(raw_depth_items, "depth_raw"),
            **aggregate_metric_dicts(scaled_depth_items, "depth_scaled"),
            **aggregate_metric_dicts(raw_point_items, "camcoord_raw"),
            **aggregate_metric_dicts(scaled_point_items, "camcoord_scaled"),
        },
        "intrinsics": {
            "gt_preprocessed": K_gt.tolist(),
            "vggt_mean": np.mean(pred_K, axis=0).tolist(),
            "mean_abs_delta_pixels": np.mean(np.abs(K_delta), axis=0).tolist(),
            "mean_focal_relative_delta": float(np.mean(focal_rel)),
            "mean_abs_focal_relative_delta": float(np.mean(np.abs(focal_rel))),
        },
        "timing": {
            **timings,
            "total_seconds": time.perf_counter() - t_start,
        },
        "warnings": warnings,
        "outputs": {
            "per_frame_csv": str(csv_path),
            "visual_panels": visual_paths,
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_npz:
        npz_path = output_dir / "dense_compare_arrays.npz"
        np.savez_compressed(
            npz_path,
            frame_ids=np.array([sample.frame_id for sample in frames], dtype=np.int32),
            gt_depth=gt_depth,
            gt_cam_points_opencv=gt_cam,
            gt_intrinsic=K_gt_stack,
            vggt_depth=pred_depth,
            vggt_depth_scaled=pred_depth_scaled,
            vggt_cam_points_opencv=pred_cam,
            vggt_cam_points_scaled_opencv=pred_cam_scaled,
            vggt_intrinsic=pred_K,
            vggt_extrinsic=pred["extrinsic"],
            vggt_depth_conf=pred["depth_conf"],
            valid_masks=valid_masks,
        )
        summary["outputs"]["dense_npz"] = str(npz_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
