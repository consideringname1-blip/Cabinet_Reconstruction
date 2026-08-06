#!/usr/bin/env python3
"""Create native-resolution official iTACO view for arbitrary HoloLens ranges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {target}")
    os.symlink(source.resolve(), target)


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-pinhole-dir", type=Path, required=True)
    parser.add_argument("--interaction-dir", type=Path, required=True)
    parser.add_argument("--surface-sequence-dir", type=Path, required=True)
    parser.add_argument("--surface-depth-dir", type=Path, required=True)
    parser.add_argument("--hand-mask-dir", type=Path, required=True)
    parser.add_argument("--autoseg-root", type=Path, required=True)
    parser.add_argument("--surface-keyframe-count", type=int, default=8)
    args = parser.parse_args()

    view = args.run_root / "official/view"
    preprocess = args.run_root / "official/preprocess"
    forward_rgb = sorted((args.interaction_dir / "jpg").glob("*.jpg"))
    first = cv2.imread(str(forward_rgb[0]), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    fx, fy, cx, cy = np.loadtxt(args.source_pinhole_dir / "calibration.txt").reshape(-1)[:4]
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    for index, source in enumerate(forward_rgb):
        link(source, view / "rgb" / f"{index:06d}.jpg")
        link(source, view / "sample_rgb" / f"{index:06d}.jpg")
    link(args.hand_mask_dir, preprocess / "hand_mask")
    link(args.autoseg_root, preprocess / "video_segment_reverse")

    metadata = {
        "K": intrinsic.T.reshape(-1).tolist(),
        "width": width,
        "height": height,
        "native_resolution": True,
        "source": "HoloLens PV pinhole projection",
    }
    (view / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (view / "surface/mesh_info.json").write_text(
        json.dumps(
            {
                "alignmentTransform": np.eye(4).T.reshape(-1).tolist(),
                "note": "Surface and interaction share the HoloLens world coordinate",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    static_mapping = json.loads(
        (args.surface_sequence_dir / "frame_mapping.json").read_text()
    )
    all_poses = load_odometry(args.source_pinhole_dir / "odometry.log")
    keyframe_local_ids = (
        np.linspace(0, len(static_mapping) - 1, args.surface_keyframe_count)
        .round()
        .astype(int)
        .tolist()
    )
    prompt_surface = preprocess / "prompt_depth_surface"
    prompt_surface.mkdir(parents=True, exist_ok=True)
    for key_id, local_index in enumerate(keyframe_local_ids):
        source_index = int(static_mapping[local_index]["source_index"])
        image_source = args.surface_sequence_dir / "jpg" / f"{local_index:06d}.jpg"
        name = f"{key_id:06d}"
        link(image_source, view / "surface/keyframes/corrected_images" / f"{name}.jpg")
        link(args.surface_depth_dir / f"{local_index:06d}.npy", prompt_surface / f"{name}.npy")
        pose = all_poses[source_index]
        camera = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": width,
            "height": height,
            "source_local_index": local_index,
            "source_index": source_index,
        }
        for row in range(3):
            for column in range(4):
                camera[f"t_{row}{column}"] = float(pose[row, column])
        camera_path = view / "surface/keyframes/corrected_cameras" / f"{name}.json"
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")

    report = {
        "view_dir": str(view.resolve()),
        "preprocess_dir": str(preprocess.resolve()),
        "frame_count": len(forward_rgb),
        "image_size_wh": [width, height],
        "intrinsic": intrinsic.tolist(),
        "surface_keyframe_local_indices": keyframe_local_ids,
        "surface_coordinate": "HoloLens world",
        "interaction_camera_coordinate_to_surface": "per-frame HoloLens PV camera-to-world",
        "explicit_substitutions": {
            "MonST3R_depth_camera": "HoloLens metric depth and odometry",
            "PromptDA_depth": "HoloLens metric depth",
        },
    }
    (args.run_root / "official/native_gt_view_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
