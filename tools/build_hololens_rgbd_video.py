#!/usr/bin/env python3
"""Build a side-by-side video from every unique projected HoloLens RGBD pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--depth-min-mm", type=int, default=200)
    parser.add_argument("--depth-max-mm", type=int, default=4000)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_timestamp(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def read_association(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                rows.append((int(fields[0]), fields[1].replace("\\", "/")))
    return rows


def validate_inputs(
    input_root: Path,
) -> tuple[list[Path], list[Path], dict[str, object]]:
    rgb_dir = input_root / "rgb"
    depth_dir = input_root / "depth"
    rgb_paths = sorted(rgb_dir.glob("*_proj.png"), key=numeric_timestamp)
    depth_paths = sorted(depth_dir.glob("*_proj.png"), key=numeric_timestamp)
    rgb_names = [path.name for path in rgb_paths]
    depth_names = [path.name for path in depth_paths]
    if not rgb_paths or not depth_paths:
        raise RuntimeError("No projected RGBD files found")
    if rgb_names != depth_names:
        missing_depth = sorted(set(rgb_names) - set(depth_names))
        missing_rgb = sorted(set(depth_names) - set(rgb_names))
        raise RuntimeError(
            f"RGB/depth names differ: missing_depth={missing_depth[:10]}, "
            f"missing_rgb={missing_rgb[:10]}"
        )

    rgb_assoc = read_association(input_root / "rgb.txt")
    depth_assoc = read_association(input_root / "depth.txt")
    if len(rgb_assoc) != len(depth_assoc):
        raise RuntimeError("RGB/depth association lengths differ")
    rgb_assoc_names = [Path(item[1]).name for item in rgb_assoc]
    depth_assoc_names = [Path(item[1]).name for item in depth_assoc]
    if rgb_assoc_names != depth_assoc_names:
        raise RuntimeError("RGB/depth association filenames differ")
    association_timestamps = np.asarray([item[0] for item in rgb_assoc], dtype=np.int64)

    first_rgb = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    first_depth = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_rgb is None or first_depth is None:
        raise RuntimeError("Failed to decode first RGBD pair")
    height, width = first_rgb.shape[:2]
    if first_rgb.shape != (height, width, 3):
        raise RuntimeError(f"Unexpected first RGB shape: {first_rgb.shape}")
    if first_depth.shape != (height, width) or first_depth.dtype != np.uint16:
        raise RuntimeError(
            f"Unexpected first depth: shape={first_depth.shape}, dtype={first_depth.dtype}"
        )

    metadata = {
        "unique_pair_count": len(rgb_paths),
        "association_row_count": len(rgb_assoc),
        "association_unique_file_count": len(set(rgb_assoc_names)),
        "association_duplicate_rows": len(rgb_assoc) - len(set(rgb_assoc_names)),
        "rgb_depth_unique_names_identical": True,
        "rgb_depth_association_names_identical": True,
        "width": width,
        "height": height,
        "rgb_dtype": str(first_rgb.dtype),
        "depth_dtype": str(first_depth.dtype),
        "first_file": rgb_paths[0].name,
        "last_file": rgb_paths[-1].name,
        "association_median_interval_seconds": float(
            np.median(np.diff(association_timestamps)) / 1e7
        ),
        "association_duration_seconds": float(
            (association_timestamps[-1] - association_timestamps[0]) / 1e7
        ),
    }
    return rgb_paths, depth_paths, metadata


def colorize_depth(
    depth_mm: np.ndarray, minimum_mm: int, maximum_mm: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = (depth_mm >= minimum_mm) & (depth_mm <= maximum_mm)
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    clipped = np.clip(depth_mm.astype(np.float32), minimum_mm, maximum_mm)
    normalized[valid] = np.round(
        255.0
        * (1.0 - (clipped[valid] - minimum_mm) / (maximum_mm - minimum_mm))
    ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored, valid


def overlay_frame_labels(
    canvas: np.ndarray,
    index: int,
    total: int,
    source_timestamp: int,
    valid_ratio: float,
    fps: float,
) -> None:
    height, width = canvas.shape[:2]
    half = width // 2
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0, canvas)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "RGB", (10, 23), font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "DEPTH 0.2-4.0m",
        (half + 10, 23),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    footer = (
        f"{index + 1:04d}/{total:04d}  t={index / fps:06.1f}s  "
        f"src={source_timestamp}  valid={valid_ratio * 100:05.1f}%"
    )
    (text_width, _), _ = cv2.getTextSize(footer, font, 0.45, 1)
    footer_x = max(6, (width - text_width) // 2)
    cv2.rectangle(canvas, (0, height - 25), (width, height), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        footer,
        (footer_x, height - 8),
        font,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (half - 1, 0), (half - 1, height), (255, 255, 255), 2)


def ffprobe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,"
            "avg_frame_rate,nb_frames,nb_read_frames,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def main() -> None:
    args = parse_args()
    started = time.time()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hololens_all_rgbd_381pairs_5fps.mp4"
    partial_path = output_dir / "hololens_all_rgbd_381pairs_5fps.partial.mp4"
    mapping_path = output_dir / "frame_mapping.csv"
    summary_path = output_dir / "summary.json"

    print(f"[start] input={input_root}", flush=True)
    rgb_paths, depth_paths, input_metadata = validate_inputs(input_root)
    frame_count = len(rgb_paths)
    height = int(input_metadata["height"])
    width = int(input_metadata["width"])
    output_width = width * 2

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{output_width}x{height}",
        "-r",
        f"{args.fps:g}",
        "-i",
        "-",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial_path),
    ]
    print("[ffmpeg] " + " ".join(command), flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    mapping_rows: list[dict[str, object]] = []
    valid_ratios: list[float] = []
    try:
        for index, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                raise RuntimeError(f"Decode failure at frame {index}: {rgb_path.name}")
            if rgb.shape != (height, width, 3):
                raise RuntimeError(f"RGB shape changed at frame {index}: {rgb.shape}")
            if depth.shape != (height, width) or depth.dtype != np.uint16:
                raise RuntimeError(
                    f"Depth format changed at frame {index}: {depth.shape}, {depth.dtype}"
                )
            depth_color, valid = colorize_depth(
                depth, args.depth_min_mm, args.depth_max_mm
            )
            valid_ratio = float(np.mean(valid))
            valid_ratios.append(valid_ratio)
            canvas = np.hstack((rgb, depth_color))
            source_timestamp = numeric_timestamp(rgb_path)
            overlay_frame_labels(
                canvas,
                index,
                frame_count,
                source_timestamp,
                valid_ratio,
                args.fps,
            )
            process.stdin.write(canvas.tobytes())
            mapping_rows.append(
                {
                    "video_frame": index,
                    "time_seconds": index / args.fps,
                    "source_timestamp": source_timestamp,
                    "filename": rgb_path.name,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "valid_depth_ratio_0p2m_to_4m": valid_ratio,
                }
            )
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == frame_count:
                print(f"[progress] {index + 1}/{frame_count}", flush=True)
    except Exception:
        process.stdin.close()
        process.wait()
        raise
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")

    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    probe = ffprobe_video(partial_path)
    if int(probe["nb_read_frames"]) != frame_count:
        raise RuntimeError(f"Output frame count mismatch: {probe}")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(partial_path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    os.replace(partial_path, output_path)
    summary = {
        "status": "complete",
        "input_root": str(input_root),
        "input": input_metadata,
        "selection_policy": (
            "Every unique same-name RGB/depth projected PNG pair exactly once, "
            "sorted by numeric filename timestamp. The association table's final "
            "duplicate reference does not add a second copy of the same photo."
        ),
        "depth_visualization": {
            "minimum_mm": args.depth_min_mm,
            "maximum_mm": args.depth_max_mm,
            "invalid_color": "black",
            "colormap": "OpenCV TURBO, reversed metric normalization so near is warm",
            "valid_ratio_mean": float(np.mean(valid_ratios)),
            "valid_ratio_min": float(np.min(valid_ratios)),
            "valid_ratio_max": float(np.max(valid_ratios)),
        },
        "video": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "fps": args.fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / args.fps,
            "ffprobe": probe,
        },
        "frame_mapping_csv": str(mapping_path),
        "command": sys.argv,
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[complete] video={output_path}", flush=True)
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
