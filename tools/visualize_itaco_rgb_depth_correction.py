#!/usr/bin/env python3
"""Visualize projected-RGB footprint and RGB/depth-gated iTACO moving maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def add_label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 23), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    color = np.empty_like(rgb)
    color[:] = (210, 0, 210)
    color[mask] = (0, 210, 0)
    output = cv2.addWeighted(rgb, 0.45, color, 0.55, 0)
    add_label(output, f"{name}: green=used  {mask.mean() * 100:.1f}%")
    return output


def heat_overlay(
    rgb: np.ndarray, moving: np.ndarray, threshold: float, name: str
) -> np.ndarray:
    moving = np.clip(moving.astype(np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap(
        np.rint(moving * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    output = cv2.addWeighted(rgb, 0.42, heat, 0.58, 0)
    binary = (moving > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    add_label(output, f"{name}: >{threshold:g}={binary.mean() * 100:.1f}%")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--original-map", type=Path, required=True)
    parser.add_argument("--corrected-map", type=Path, required=True)
    parser.add_argument("--rgb-footprint", type=Path, required=True)
    parser.add_argument("--combined-validity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    rgb_paths = sorted((args.run_root / "official/view/rgb").glob("*.jpg"))
    original = np.load(args.original_map)["a"].astype(np.float32)
    corrected = np.load(args.corrected_map)["a"].astype(np.float32)
    rgb_footprint = np.load(args.rgb_footprint)["a"].astype(bool)
    combined = np.load(args.combined_validity)["a"].astype(bool)
    lengths = {
        len(rgb_paths),
        len(original),
        len(corrected),
        len(rgb_footprint),
        len(combined),
    }
    if len(lengths) != 1:
        raise ValueError(f"Frame counts differ: {lengths}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    thumbnails: list[np.ndarray] = []
    records = []
    for index, (rgb_path, old, new, footprint, valid) in enumerate(
        zip(rgb_paths, original, corrected, rgb_footprint, combined)
    ):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(rgb_path)
        rgb_panel = rgb.copy()
        add_label(rgb_panel, f"RGB local={index:02d} source={177 + index}")
        panels = [
            rgb_panel,
            mask_overlay(rgb, footprint, "projected RGB footprint"),
            mask_overlay(rgb, valid, "RGB & depth support"),
            heat_overlay(rgb, old, args.threshold, "original"),
            heat_overlay(rgb, new, args.threshold, "corrected"),
        ]
        frame = np.concatenate(panels, axis=1)
        frames.append(frame)
        thumbnails.append(
            cv2.resize(frame, (800, 144), interpolation=cv2.INTER_AREA)
        )
        cv2.imwrite(str(args.output_dir / f"frame_{index:06d}.jpg"), frame)
        records.append(
            {
                "local_index": index,
                "source_index": 177 + index,
                "rgb_footprint_fraction": float(footprint.mean()),
                "combined_support_fraction": float(valid.mean()),
                "original_above_threshold_fraction": float(
                    (old > args.threshold).mean()
                ),
                "corrected_above_threshold_fraction": float(
                    (new > args.threshold).mean()
                ),
                "original_high_outside_rgb_fraction_of_high": float(
                    ((old > args.threshold) & ~footprint).sum()
                    / max(1, (old > args.threshold).sum())
                ),
                "corrected_high_outside_combined_pixels": int(
                    ((new > args.threshold) & ~valid).sum()
                ),
            }
        )

    cols = 2
    rows = (len(thumbnails) + cols - 1) // cols
    sheet = np.zeros((rows * 144, cols * 800, 3), dtype=np.uint8)
    for index, cell in enumerate(thumbnails):
        row, col = divmod(index, cols)
        sheet[row * 144 : (row + 1) * 144, col * 800 : (col + 1) * 800] = cell
    sheet_path = args.output_dir / "rgb_depth_correction_all37.jpg"
    cv2.imwrite(str(sheet_path), sheet)

    height, width = frames[0].shape[:2]
    video_path = args.output_dir / "rgb_depth_correction_all37.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")
    for frame in frames:
        writer.write(frame)
    writer.release()

    report = {
        "purpose": "corrected-extension validation; official outputs unchanged",
        "panels": [
            "projected RGB",
            "eroded projected-RGB footprint",
            "intersection of eroded RGB footprint and eroded valid depth",
            "original fixed-camera prismatic moving map",
            "RGB/depth-gated corrected moving map",
        ],
        "threshold": args.threshold,
        "frame_order": "source 177 through 213, forward",
        "outputs": {
            "contact_sheet": str(sheet_path),
            "video": str(video_path),
        },
        "records": records,
    }
    report_path = args.output_dir / "rgb_depth_visualization_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
