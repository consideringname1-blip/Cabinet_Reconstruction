#!/usr/bin/env python3
"""Prepare a 32-frame ARKit-initialized adapter for official iTACO BA core.

The interaction is ordered open-to-closed so frame zero matches the open
surface target. Native LiDAR depth is retained; PromptDA is not claimed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=32)
    args = parser.parse_args()

    view = args.output_root / "view"
    preprocess = args.output_root / "preprocess"
    prediction = args.output_root / "prediction"
    rgb_dir = args.source_run_root / "normalized/rgb_256x192"
    autoseg_dir = (
        args.source_run_root / "preprocess/autoseg_reverse/small/final-output"
    )
    hand_dir = args.source_run_root / "preprocess/hand_per_frame/mask"
    poses = np.load(args.source_run_root / "normalized/camera_to_world.npy")
    intrinsics = np.load(args.source_run_root / "normalized/intrinsics_depth.npy")
    motion = np.load(args.source_run_root / "extended/motion_seed/motion_seed.npz")
    axis = motion["opening_axis_world"].astype(np.float64)
    axis /= np.linalg.norm(axis)
    q_forward = motion["q_per_interaction_frame_m"].astype(np.float64)
    travel = float(np.median(q_forward[95:116]))

    reverse_local = np.unique(
        np.round(np.linspace(0, 135, args.frame_count)).astype(int)
    )
    if len(reverse_local) != args.frame_count:
        raise ValueError(reverse_local)
    source_indices = 540 - reverse_local
    selected_poses = poses[source_indices]
    selected_joint_state = np.clip(
        travel - q_forward[source_indices - 405], 0.0, travel
    )

    first_rgb = cv2.imread(str(rgb_dir / f"{source_indices[0]:06d}.jpg"))
    height, width = first_rgb.shape[:2]
    if (height, width) != (192, 256):
        raise ValueError((height, width))
    intrinsic = intrinsics[source_indices[0]]

    for local, source in enumerate(source_indices.tolist()):
        rgb_source = rgb_dir / f"{source:06d}.jpg"
        link(rgb_source, view / "rgb" / f"{local:06d}.jpg")
        link(rgb_source, view / "sample_rgb" / f"{local:06d}.jpg")

        depth_mm = cv2.imread(
            str(args.raw_root / "depth" / f"{source:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        confidence = cv2.imread(
            str(args.raw_root / "confidence" / f"{source:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        depth = depth_mm.astype(np.float64) / 1000.0
        depth_valid = (
            (depth > 0.25) & (depth < 4.5) & (confidence >= 1)
        )
        depth[~depth_valid] = 0.0
        depth_out = preprocess / "prompt_depth_video" / f"{local:06d}.npy"
        valid_out = preprocess / "depth_valid" / f"{local:06d}.npy"
        depth_out.parent.mkdir(parents=True, exist_ok=True)
        valid_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_out, depth)
        np.save(valid_out, depth_valid)

        hand_source = hand_dir / f"{source - 405:06d}.npy"
        hand = np.load(hand_source).astype(bool)
        hand_out = preprocess / "hand_mask" / f"{local:06d}.npy"
        hand_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(hand_out, hand)

        # RealDataLoader sorts AutoSeg results in reverse filename order.
        # Rename the desired open-to-closed sequence so that reverse sorting
        # returns local frame 0, 1, ..., 31.
        auto_local = 540 - source
        destination_index = args.frame_count - 1 - local
        link(
            autoseg_dir / f"mask_{auto_local:03d}.npz",
            preprocess
            / "video_segment_reverse/small/final-output"
            / f"mask_{destination_index:03d}.npz",
        )

    metadata = {
        "K": intrinsic.T.reshape(-1).tolist(),
        "width": width,
        "height": height,
        "native_resolution": True,
        "image_orientation": (
            "native Stray Scanner 256x192 landscape raster; not display-rotated "
            "so RGB, depth, masks and intrinsics remain pixel-aligned"
        ),
    }
    (view / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (view / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    surface_source = (
        args.source_run_root
        / "extended/dual_volume_fusion_final_crop/combined_open_points.ply"
    )
    link(surface_source, view / "surface/surface.ply")
    (view / "surface/mesh_info.json").write_text(
        json.dumps(
            {
                "alignmentTransform": np.eye(4).T.reshape(-1).tolist(),
                "note": "Open-state point target and ARKit poses share the same world frame",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    camera = {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "width": width,
        "height": height,
    }
    for row in range(3):
        for column in range(4):
            camera[f"t_{row}{column}"] = float(selected_poses[0, row, column])
    camera_path = view / "surface/keyframes/corrected_cameras/000000.json"
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    camera_path.write_text(json.dumps(camera, indent=2) + "\n", encoding="utf-8")

    np.save(preprocess / "cam2world.npy", selected_poses[0])
    np.save(preprocess / "arkit_cam2world_all.npy", selected_poses)

    coarse = prediction / "coarse_prediction/monst3r/0"
    (coarse / "prismatic").mkdir(parents=True, exist_ok=True)
    np.save(coarse / "prismatic/joint_axis.npy", axis)
    np.save(coarse / "prismatic/joint_pos.npy", np.zeros(3, dtype=np.float64))
    np.save(coarse / "prismatic/joint_value.npy", selected_joint_state)
    for local, pose in enumerate(selected_poses):
        camera_path = coarse / "cam_pose" / f"cam2label_{local}.npy"
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(camera_path, pose)

    surface = o3d.io.read_point_cloud(str(surface_source))
    manifest = {
        "output_kind": "official_itaco_bundle_adjustment_core_with_iphone_adapter",
        "official_loss_implementation": "joint_refinement.BundleAdjustment unchanged",
        "not_official_preprocessing": [
            "ARKit camera-to-world used as coarse initialization",
            "native measured LiDAR depth used instead of PromptDA",
            "open-state surface target built from the same recording",
            "known prismatic hypothesis initialized from metric motion tracks",
        ],
        "frame_order": "open-to-closed",
        "frame_count": args.frame_count,
        "source_indices": source_indices.tolist(),
        "autoseg_reverse_local_indices": reverse_local.tolist(),
        "image_size_wh": [width, height],
        "intrinsic": intrinsic.tolist(),
        "surface_target": str(surface_source.resolve()),
        "surface_target_points": len(surface.points),
        "axis_initial_world": axis.tolist(),
        "joint_state_initial_m": selected_joint_state.tolist(),
        "travel_m": travel,
        "view_dir": str(view.resolve()),
        "preprocess_dir": str(preprocess.resolve()),
        "prediction_dir": str(prediction.resolve()),
    }
    (args.output_root / "ADAPTER_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
