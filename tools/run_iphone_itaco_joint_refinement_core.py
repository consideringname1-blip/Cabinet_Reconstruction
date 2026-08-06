#!/usr/bin/env python3
"""Run the official iTACO BundleAdjustment core with the iPhone/ARKit adapter."""

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

    class NativeIPhoneRealDataLoader(RealDataLoader):
        def __init__(self, video_dir: str, preprocess_dir: str):
            super().__init__(video_dir, preprocess_dir)
            first_rgb = cv2.imread(str(sorted((Path(video_dir) / "rgb").glob("*.jpg"))[0]))
            self.H, self.W = first_rgb.shape[:2]

        def load_obj_mask(self) -> np.ndarray:
            masks = []
            for frame_index in self.sample_rgb_index:
                hand = np.load(Path(self.hand_segment_dir) / f"{frame_index:06d}.npy").astype(bool)
                depth_valid = np.load(Path(self.preprocess_dir) / "depth_valid" / f"{frame_index:06d}.npy").astype(bool)
                masks.append((~hand) & depth_valid)
            return np.stack(masks)

    data_loader = NativeIPhoneRealDataLoader(
        str(args.view_dir), str(args.preprocess_dir)
    )
    loss = "chamfer"
    mask_type = "monst3r"
    exp_name = "refinement_iphone_adapter"
    log_dir = args.prediction_dir / exp_name / mask_type
    output_dir = log_dir / loss / str(args.seed) / args.joint
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    run_config = {
        "mask_type": mask_type,
        "loss_function": loss,
        "epochs": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "native_resolution_hw": [data_loader.H, data_loader.W],
        "camera_initialization": "ARKit per-frame camera-to-world",
    }

    with wandb.init(
        project=f"video_articulation_{exp_name}_{mask_type}",
        config=run_config,
        dir=str(output_dir),
        name=f"{args.view_dir}/{args.joint}/seed{args.seed}/",
    ):
        adjustment = BundleAdjustment(
            data_loader,
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
            raise RuntimeError("Official BundleAdjustment has no valid coarse initialization")
        initial_loss = float(adjustment.eval().detach().cpu().item())
        initial_joint_axis = adjustment.joint_axis.detach().cpu().numpy().copy()
        initial_joint_state = adjustment.joint_state.detach().cpu().numpy().copy()
        initial_camera_poses = adjustment.camera_pose.detach().cpu().numpy().copy()
        initial_moving_vectors = adjustment.moving_map_vec.detach().cpu().numpy().copy()
        np.save(output_dir / "initial_joint_axis.npy", initial_joint_axis)
        np.save(output_dir / "initial_joint_value.npy", initial_joint_state)
        np.save(output_dir / "initial_camera_poses.npy", initial_camera_poses)
        np.save(output_dir / "initial_moving_vectors.npy", initial_moving_vectors)
        wandb.watch(adjustment, log="all", log_freq=1)
        adjustment.dump_configuration()
        adjustment.optimize_adam(None, None, None, None)
        adjustment.save_estimation_results()

    report = {
        "stage": "official iTACO BundleAdjustment core with iPhone adapter",
        "joint_hypothesis": args.joint,
        "native_resolution_hw": [data_loader.H, data_loader.W],
        "steps": args.steps,
        "learning_rate": args.lr,
        "best_loss": float(adjustment.best_loss),
        "initial_sampled_loss": initial_loss,
        "initial_joint_axis": initial_joint_axis.tolist(),
        "initial_joint_value": initial_joint_state.tolist(),
        "initial_camera_poses_wxyz_t": initial_camera_poses.tolist(),
        "best_camera_poses_wxyz_t": adjustment.best_camera_poses.tolist(),
        "adapter_deviations": [
            "native measured LiDAR depth instead of PromptDA",
            "ARKit camera-to-world coarse initialization",
            "open-state target surface from the same recording",
            "known prismatic metric motion seed",
        ],
        "joint_axis": adjustment.best_joint_axis.tolist(),
        "joint_pos": adjustment.best_joint_pos.tolist(),
        "joint_value": adjustment.best_joint_state.tolist(),
        "moving_vector": adjustment.best_moving_vectors.tolist(),
        "camera_initialization": "ARKit per-frame camera-to-world; camera remains jointly optimized by the unchanged official refinement core",
    }
    (output_dir / "iphone_adapter_refinement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key not in {"joint_value", "moving_vector"}}, indent=2))


if __name__ == "__main__":
    main()
