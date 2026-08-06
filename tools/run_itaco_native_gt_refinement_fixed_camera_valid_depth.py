#!/usr/bin/env python3
"""Run a depth-validity-gated, fixed-camera iTACO refinement extension.

The official joint-camera and fixed-camera outputs are not modified.  This
extension changes RealDataLoader.load_obj_mask so zero/unsupported depth pixels
cannot enter either the static or dynamic Chamfer terms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb


def build_validity_masks(
    depth_paths: list[Path],
    depth_min: float,
    depth_max: float,
    erosion_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    eroded = []
    size = erosion_radius * 2 + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    for path in depth_paths:
        depth = np.load(path)
        valid = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
        valid_eroded = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
        raw.append(valid)
        eroded.append(valid_eroded)
    return np.stack(raw), np.stack(eroded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itaco-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--coarse-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joint", choices=["revolute", "prismatic"], default="prismatic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--erosion-radius", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.itaco_dir.resolve()))

    from data import RealDataLoader
    from joint_refinement import BundleAdjustment

    depth_paths = sorted((args.preprocess_dir / "prompt_depth_video").glob("*.npy"))
    raw_validity, eroded_validity = build_validity_masks(
        depth_paths, args.depth_min, args.depth_max, args.erosion_radius
    )

    class ValidDepthNativeGTRealDataLoader(RealDataLoader):
        def __init__(self, video_dir: str, preprocess_dir: str):
            super().__init__(video_dir, preprocess_dir)
            first = cv2.imread(
                str(sorted((Path(video_dir) / "rgb").glob("*.jpg"))[0])
            )
            self.H, self.W = first.shape[:2]

        def load_gt_camera_pose_se3(self) -> np.ndarray:
            return np.load(Path(self.preprocess_dir) / "gt_cam2world_all.npy")

        def load_obj_mask(self) -> np.ndarray:
            masks = []
            for local_index, source_index in enumerate(self.sample_rgb_index):
                hand = np.load(
                    f"{self.hand_segment_dir}/{source_index:06d}.npy"
                ).squeeze().astype(bool)
                masks.append((~hand) & eroded_validity[local_index])
            return np.stack(masks)

    loader = ValidDepthNativeGTRealDataLoader(
        str(args.view_dir), str(args.preprocess_dir)
    )
    if len(loader.sample_rgb_index) != len(depth_paths):
        raise ValueError("Sample frame and depth counts differ")

    loss = "chamfer"
    mask_type = "monst3r"
    exp_name = "refinement_gt_fixed_camera_valid_depth"
    log_dir = args.output_root / exp_name / mask_type
    output_dir = log_dir / loss / str(args.seed) / args.joint
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")

    run_config = {
        "mask_type": mask_type,
        "loss_function": loss,
        "epochs": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "camera_mode": "fixed HoloLens camera-to-world",
        "depth_validity": {
            "strict_range_m": [args.depth_min, args.depth_max],
            "erosion_radius_pixels": args.erosion_radius,
            "erosion_kernel": [
                args.erosion_radius * 2 + 1,
                args.erosion_radius * 2 + 1,
            ],
        },
    }
    with wandb.init(
        project=f"video_articulation_{exp_name}_{mask_type}",
        config=run_config,
        dir=str(output_dir),
        name=f"{args.view_dir}/{args.joint}/fixed_camera_valid_depth/seed{args.seed}/",
    ):
        adjustment = BundleAdjustment(
            loader,
            str(args.coarse_prediction_dir),
            mask_type,
            args.joint,
            args.lr,
            loss,
            args.steps,
            str(log_dir),
            torch.device(args.device),
            args.seed,
            False,
        )
        if not adjustment.valid:
            raise RuntimeError("No valid official coarse initialization")
        adjustment.camera_pose.requires_grad_(False)
        adjustment.optimizer = torch.optim.Adam(
            [
                adjustment.joint_axis,
                adjustment.joint_pos,
                adjustment.joint_state,
                adjustment.moving_map_vec,
            ],
            lr=args.lr,
        )
        adjustment.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            adjustment.optimizer, args.steps
        )
        adjustment.dump_configuration()
        adjustment.optimize_adam(None, None, None, None)
        adjustment.save_estimation_results()

    raw_moving = np.load(output_dir / "moving_map.npz")["a"]
    validity_for_sampled_frames = eroded_validity[loader.sample_rgb_index]
    masked_moving = raw_moving * validity_for_sampled_frames.astype(raw_moving.dtype)
    np.savez_compressed(output_dir / "depth_validity_raw.npz", a=raw_validity)
    np.savez_compressed(output_dir / "depth_validity_eroded3.npz", a=eroded_validity)
    np.savez_compressed(
        output_dir / "moving_map_validity_masked.npz", a=masked_moving
    )

    report = {
        "stage": "iTACO BundleAdjustment corrected extension",
        "output_kind": "depth-validity-gated fixed-HoloLens-camera extension",
        "official_outputs_modified": False,
        "joint_hypothesis": args.joint,
        "steps": args.steps,
        "learning_rate": args.lr,
        "best_loss": float(adjustment.best_loss),
        "joint_axis_raw": adjustment.best_joint_axis.tolist(),
        "joint_axis_unit": (
            adjustment.best_joint_axis / np.linalg.norm(adjustment.best_joint_axis)
        ).tolist(),
        "joint_value": adjustment.best_joint_state.tolist(),
        "moving_vector": adjustment.best_moving_vectors.tolist(),
        "camera_mode": "requires_grad=False; optimizer excludes camera_pose",
        "depth_validity": {
            "strict_range_m": [args.depth_min, args.depth_max],
            "raw_valid_fraction": float(raw_validity.mean()),
            "erosion_radius_pixels": args.erosion_radius,
            "erosion_kernel_pixels": [
                args.erosion_radius * 2 + 1,
                args.erosion_radius * 2 + 1,
            ],
            "eroded_valid_fraction": float(eroded_validity.mean()),
            "loss_object_mask": "(~hand_mask) & eroded_depth_validity",
        },
        "moving_map": {
            "raw_above_0_7_fraction_full_image": float((raw_moving > 0.7).mean()),
            "validity_masked_above_0_7_fraction_full_image": float(
                (masked_moving > 0.7).mean()
            ),
            "validity_masked_above_0_7_fraction_within_eroded_validity": float(
                (masked_moving[validity_for_sampled_frames] > 0.7).mean()
            ),
        },
        "source_revisions_and_inputs": {
            "itaco_dir": str(args.itaco_dir.resolve()),
            "coarse_prediction_dir": str(args.coarse_prediction_dir.resolve()),
            "view_dir": str(args.view_dir.resolve()),
            "preprocess_dir": str(args.preprocess_dir.resolve()),
            "selected_autoseg_uid": 25,
            "interaction_frame_order": "source 177 through 213, forward",
        },
        "outputs": {
            "raw_official_style_moving_map": str(output_dir / "moving_map.npz"),
            "validity_masked_moving_map": str(
                output_dir / "moving_map_validity_masked.npz"
            ),
            "raw_validity": str(output_dir / "depth_validity_raw.npz"),
            "eroded_validity": str(output_dir / "depth_validity_eroded3.npz"),
        },
    }
    (output_dir / "corrected_refinement_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"joint_value", "moving_vector"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
