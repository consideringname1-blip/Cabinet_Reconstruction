#!/usr/bin/env python3
"""Run iTACO refinement while keeping HoloLens camera poses fixed as GT."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itaco-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--joint", choices=["revolute", "prismatic"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.itaco_dir.resolve()))

    from data import RealDataLoader
    from joint_refinement import BundleAdjustment

    class NativeGTRealDataLoader(RealDataLoader):
        def __init__(self, video_dir: str, preprocess_dir: str):
            super().__init__(video_dir, preprocess_dir)
            first = cv2.imread(
                str(sorted((Path(video_dir) / "rgb").glob("*.jpg"))[0])
            )
            self.H, self.W = first.shape[:2]

        def load_gt_camera_pose_se3(self) -> np.ndarray:
            return np.load(Path(self.preprocess_dir) / "gt_cam2world_all.npy")

    loader = NativeGTRealDataLoader(str(args.view_dir), str(args.preprocess_dir))
    loss = "chamfer"
    mask_type = "monst3r"
    exp_name = "refinement_gt_fixed_camera"
    log_dir = args.prediction_dir / exp_name / mask_type
    output_dir = log_dir / loss / str(args.seed) / args.joint
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")

    with wandb.init(
        project=f"video_articulation_{exp_name}_{mask_type}",
        config={
            "mask_type": mask_type,
            "loss_function": loss,
            "epochs": args.steps,
            "learning_rate": args.lr,
            "seed": args.seed,
            "camera_mode": "fixed HoloLens GT",
        },
        dir=str(output_dir),
        name=f"{args.view_dir}/{args.joint}/fixed_camera/seed{args.seed}/",
    ):
        adjustment = BundleAdjustment(
            loader,
            str(args.prediction_dir),
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
            raise RuntimeError("No valid coarse initialization")
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

    gt_poses = loader.load_gt_camera_pose_se3()
    best_camera_quat_t = adjustment.best_camera_poses
    report = {
        "stage": "iTACO BundleAdjustment with HoloLens camera fixed",
        "output_kind": "GT-fixed extension; official joint-camera refinement is preserved separately",
        "joint_hypothesis": args.joint,
        "steps": args.steps,
        "learning_rate": args.lr,
        "best_loss": float(adjustment.best_loss),
        "joint_axis": adjustment.best_joint_axis.tolist(),
        "joint_pos": adjustment.best_joint_pos.tolist(),
        "joint_value": adjustment.best_joint_state.tolist(),
        "moving_vector": adjustment.best_moving_vectors.tolist(),
        "camera_mode": "requires_grad=False; optimizer excludes camera_pose",
        "gt_camera_pose_count": len(gt_poses),
        "saved_camera_parameter_count": len(best_camera_quat_t),
    }
    (output_dir / "gt_fixed_camera_refinement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"joint_value", "moving_vector"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
