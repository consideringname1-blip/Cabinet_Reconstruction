#!/usr/bin/env python3
"""Visualize initial and refined iTACO moving maps frame by frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def initial_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    color = np.zeros_like(result)
    color[..., 2] = 255
    alpha = mask.astype(np.float32)[..., None] * 0.58
    result = np.clip(result * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    label(result, f"AutoSeg UID25 seed  area={mask.mean() * 100:.1f}%")
    return result


def refined_overlay(rgb: np.ndarray, moving: np.ndarray, threshold: float) -> np.ndarray:
    moving = np.clip(moving.astype(np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap(np.rint(moving * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    result = cv2.addWeighted(rgb, 0.43, heat, 0.57, 0)
    binary = (moving > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    label(
        result,
        f"fixed-GT prismatic refined  >{threshold:g}={binary.mean() * 100:.1f}%",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    rgb_paths = sorted((args.run_root / "official/view/rgb").glob("*.jpg"))
    initial = np.load(args.run_root / "official/preprocess/gt_dynamic_masks.npz")["a"].astype(bool)
    refined = np.load(args.refinement_dir / "moving_map.npz")["a"].astype(np.float32)
    if not (len(rgb_paths) == len(initial) == len(refined)):
        raise ValueError("RGB, initial mask, and refined map frame counts differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    cells = []
    per_frame = []
    for index, (rgb_path, seed, moving) in enumerate(zip(rgb_paths, initial, refined)):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Could not read {rgb_path}")
        original = rgb.copy()
        label(original, f"RGB  local={index:02d} source={177 + index}")
        seed_view = initial_overlay(rgb, seed)
        refined_view = refined_overlay(rgb, moving, args.threshold)
        triptych = np.concatenate([original, seed_view, refined_view], axis=1)
        frames.append(triptych)
        thumbnail = cv2.resize(
            triptych,
            (480, 144),
            interpolation=cv2.INTER_AREA,
        )
        cells.append(thumbnail)
        frame_path = args.output_dir / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), triptych)
        per_frame.append(
            {
                "local_index": index,
                "source_index": 177 + index,
                "initial_fraction": float(seed.mean()),
                "refined_above_threshold_fraction": float((moving > args.threshold).mean()),
                "refined_mean": float(moving.mean()),
                "output": str(frame_path),
            }
        )

    cols = 4
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * 144, cols * 480, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, col = divmod(index, cols)
        sheet[row * 144 : (row + 1) * 144, col * 480 : (col + 1) * 480] = cell
    sheet_path = args.output_dir / "moving_map_comparison_all37.jpg"
    cv2.imwrite(str(sheet_path), sheet)

    height, width = frames[0].shape[:2]
    video_path = args.output_dir / "moving_map_comparison_all37.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer")
    for frame in frames:
        writer.write(frame)
    writer.release()

    report = {
        "purpose": "validation visualization; no reconstruction input was modified",
        "panels": [
            "original RGB",
            "explicit AutoSeg UID25 initial moving seed",
            "fixed-HoloLens-camera prismatic refined moving map with threshold contour",
        ],
        "threshold": args.threshold,
        "frame_order": "interaction forward, source 177 through 213",
        "frame_count": len(frames),
        "global_initial_fraction": float(initial.mean()),
        "global_refined_above_threshold_fraction": float((refined > args.threshold).mean()),
        "outputs": {
            "contact_sheet": str(sheet_path),
            "video": str(video_path),
            "per_frame_directory": str(args.output_dir),
        },
        "per_frame": per_frame,
    }
    (args.output_dir / "moving_map_visualization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
