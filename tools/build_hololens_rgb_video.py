#!/usr/bin/env python3
"""Encode every unique projected HoloLens RGB frame without overlays."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def timestamp(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_association_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                names.append(Path(fields[1].replace("\\", "/")).name)
    return names


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def main() -> None:
    args = parse_args()
    started = time.time()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hololens_all_rgb_381frames_5fps.mp4"
    partial_path = output_dir / "hololens_all_rgb_381frames_5fps.partial.mp4"
    mapping_path = output_dir / "frame_mapping.csv"
    summary_path = output_dir / "summary.json"

    rgb_paths = sorted((input_root / "rgb").glob("*_proj.png"), key=timestamp)
    depth_names = {
        path.name for path in (input_root / "depth").glob("*_proj.png")
    }
    if len(rgb_paths) != 381:
        raise RuntimeError(f"Expected 381 unique RGB files, found {len(rgb_paths)}")
    if len({path.name for path in rgb_paths}) != len(rgb_paths):
        raise RuntimeError("RGB filename list contains duplicates")
    if {path.name for path in rgb_paths} != depth_names:
        raise RuntimeError("Unique RGB and depth filenames are not one-to-one")

    rgb_assoc_names = read_association_names(input_root / "rgb.txt")
    depth_assoc_names = read_association_names(input_root / "depth.txt")
    if rgb_assoc_names != depth_assoc_names:
        raise RuntimeError("RGB/depth association filename sequences differ")
    if len(rgb_assoc_names) != 382 or len(set(rgb_assoc_names)) != 381:
        raise RuntimeError(
            "Expected 382 association rows referring to 381 unique files"
        )

    first = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot decode {rgb_paths[0]}")
    height, width = first.shape[:2]
    if (width, height) != (320, 288):
        raise RuntimeError(f"Expected 320x288, found {width}x{height}")

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
        f"{width}x{height}",
        "-r",
        f"{args.fps:g}",
        "-i",
        "-",
        "-frames:v",
        str(len(rgb_paths)),
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
    print(f"[start] {len(rgb_paths)} unique RGB frames", flush=True)
    print("[ffmpeg] " + " ".join(command), flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    mapping: list[dict[str, object]] = []
    try:
        for index, path in enumerate(rgb_paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Decode failure at {index}: {path}")
            if image.shape != (height, width, 3):
                raise RuntimeError(f"Shape change at {index}: {image.shape}")
            process.stdin.write(image.tobytes())
            mapping.append(
                {
                    "video_frame": index,
                    "time_seconds": index / args.fps,
                    "source_timestamp": timestamp(path),
                    "filename": path.name,
                    "rgb_path": str(path),
                    "paired_depth_path": str(input_root / "depth" / path.name),
                }
            )
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(rgb_paths):
                print(f"[progress] {index + 1}/{len(rgb_paths)}", flush=True)
    except Exception:
        process.stdin.close()
        process.wait()
        raise
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")

    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)

    video_probe = probe(partial_path)
    if int(video_probe["nb_read_frames"]) != len(rgb_paths):
        raise RuntimeError(f"Output frame count mismatch: {video_probe}")
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
        "layout": "rgb_only_no_overlay",
        "input_root": str(input_root),
        "selection_policy": (
            "Every unique projected RGB file exactly once, sorted by numeric "
            "filename timestamp. All 381 names have a same-name depth partner. "
            "The association table's final duplicate reference is not repeated."
        ),
        "input": {
            "unique_rgb_frames": len(rgb_paths),
            "unique_paired_depth_frames": len(depth_names),
            "association_rows": len(rgb_assoc_names),
            "association_unique_files": len(set(rgb_assoc_names)),
            "first_file": rgb_paths[0].name,
            "last_file": rgb_paths[-1].name,
            "width": width,
            "height": height,
        },
        "video": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "fps": args.fps,
            "frame_count": len(rgb_paths),
            "duration_seconds": len(rgb_paths) / args.fps,
            "ffprobe": video_probe,
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
