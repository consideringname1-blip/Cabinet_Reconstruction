#!/usr/bin/env python3
"""Run prismatic iTACO with both projected-RGB and depth support masks."""

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


def depth_masks(
    paths: list[Path], minimum: float, maximum: float, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((radius * 2 + 1,) * 2, dtype=np.uint8)
    raw, eroded = [], []
    for path in paths:
        depth = np.load(path)
        valid = np.isfinite(depth) & (depth > minimum) & (depth < maximum)
        raw.append(valid)
        eroded.append(cv2.erode(valid.astype(np.uint8), kernel).astype(bool))
    return np.stack(raw), np.stack(eroded)


def rgb_masks(
    paths: list[Path], close_radius: int, erosion_radius: int, min_area: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_kernel = np.ones((close_radius * 2 + 1,) * 2, dtype=np.uint8)
    erosion_kernel = np.ones((erosion_radius * 2 + 1,) * 2, dtype=np.uint8)
    exact, filled, eroded = [], [], []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read projected RGB: {path}")
        nonzero = np.any(image != 0, axis=2).astype(np.uint8)
        closed = cv2.morphologyEx(nonzero, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        footprint = np.zeros_like(nonzero)
        retained = [c for c in contours if cv2.contourArea(c) >= min_area]
        cv2.drawContours(footprint, retained, -1, 1, thickness=-1)
        exact.append(nonzero.astype(bool))
        filled.append(footprint.astype(bool))
        eroded.append(
            cv2.erode(footprint, erosion_kernel, iterations=1).astype(bool)
        )
    return np.stack(exact), np.stack(filled), np.stack(eroded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itaco-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--coarse-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--depth-erosion-radius", type=int, default=3)
    parser.add_argument("--rgb-close-radius", type=int, default=3)
    parser.add_argument("--rgb-erosion-radius", type=int, default=3)
    parser.add_argument("--rgb-min-contour-area", type=float, default=100.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.itaco_dir.resolve()))

    from data import RealDataLoader
    from joint_refinement import BundleAdjustment

    depth_paths = sorted((args.preprocess_dir / "prompt_depth_video").glob("*.npy"))
    rgb_paths = sorted((args.view_dir / "rgb").glob("*.jpg"))
    depth_raw, depth_eroded = depth_masks(
        depth_paths, args.depth_min, args.depth_max, args.depth_erosion_radius
    )
    rgb_exact, rgb_filled, rgb_eroded = rgb_masks(
        rgb_paths,
        args.rgb_close_radius,
        args.rgb_erosion_radius,
        args.rgb_min_contour_area,
    )
    if not (len(depth_raw) == len(rgb_exact)):
        raise ValueError("RGB and depth frame counts differ")
    combined = depth_eroded & rgb_eroded

    class RGBDepthValidLoader(RealDataLoader):
        def __init__(self, video_dir: str, preprocess_dir: str):
            super().__init__(video_dir, preprocess_dir)
            first = cv2.imread(str(rgb_paths[0]))
            self.H, self.W = first.shape[:2]

        def load_gt_camera_pose_se3(self) -> np.ndarray:
            return np.load(Path(self.preprocess_dir) / "gt_cam2world_all.npy")

        def load_obj_mask(self) -> np.ndarray:
            masks = []
            for local_index, source_index in enumerate(self.sample_rgb_index):
                hand = np.load(
                    f"{self.hand_segment_dir}/{source_index:06d}.npy"
                ).squeeze().astype(bool)
                masks.append((~hand) & combined[local_index])
            return np.stack(masks)

    loader = RGBDepthValidLoader(str(args.view_dir), str(args.preprocess_dir))
    output_dir = (
        args.output_root
        / "refinement_gt_fixed_camera_rgb_depth_valid"
        / "monst3r/chamfer"
        / str(args.seed)
        / "prismatic"
    )
    log_dir = (
        args.output_root
        / "refinement_gt_fixed_camera_rgb_depth_valid"
        / "monst3r"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    config = {
        "joint": "prismatic",
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "camera": "fixed HoloLens camera-to-world",
        "loss_mask": "(~hand) & eroded_depth_support & eroded_projected_rgb_footprint",
        "depth_range_m": [args.depth_min, args.depth_max],
        "depth_erosion_radius_pixels": args.depth_erosion_radius,
        "rgb_footprint": {
            "source": "nonzero pixels in each original projected RGB",
            "close_radius_pixels": args.rgb_close_radius,
            "external_contour_min_area_pixels": args.rgb_min_contour_area,
            "fill_external_contours": True,
            "erosion_radius_pixels": args.rgb_erosion_radius,
        },
    }
    with wandb.init(
        project="video_articulation_refinement_gt_fixed_camera_rgb_depth_valid",
        config=config,
        dir=str(output_dir),
        name=f"{args.view_dir}/prismatic/fixed_rgb_depth_valid/seed{args.seed}/",
    ):
        adjustment = BundleAdjustment(
            loader,
            str(args.coarse_prediction_dir),
            "monst3r",
            "prismatic",
            args.lr,
            "chamfer",
            args.steps,
            str(log_dir),
            torch.device(args.device),
            args.seed,
            False,
        )
        if not adjustment.valid:
            raise RuntimeError("No valid official coarse prismatic initialization")
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
    combined_sampled = combined[loader.sample_rgb_index]
    masked_moving = raw_moving * combined_sampled.astype(raw_moving.dtype)
    np.savez_compressed(output_dir / "depth_validity_raw.npz", a=depth_raw)
    np.savez_compressed(output_dir / "depth_validity_eroded3.npz", a=depth_eroded)
    np.savez_compressed(output_dir / "rgb_exact_nonzero.npz", a=rgb_exact)
    np.savez_compressed(output_dir / "rgb_footprint_filled.npz", a=rgb_filled)
    np.savez_compressed(output_dir / "rgb_footprint_eroded3.npz", a=rgb_eroded)
    np.savez_compressed(output_dir / "combined_validity.npz", a=combined)
    np.savez_compressed(
        output_dir / "moving_map_rgb_depth_masked.npz", a=masked_moving
    )

    axis = adjustment.best_joint_axis
    axis_unit = axis / np.linalg.norm(axis)
    values = adjustment.best_joint_state
    report = {
        "stage": "iTACO BundleAdjustment corrected extension",
        "output_kind": "projected-RGB-footprint and depth-validity gated, fixed camera",
        "official_outputs_modified": False,
        "joint_hypothesis": "prismatic",
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "best_loss": float(adjustment.best_loss),
        "joint_axis_raw": axis.tolist(),
        "joint_axis_unit": axis_unit.tolist(),
        "joint_value": values.tolist(),
        "joint_range_m": float(values.max() - values.min()),
        "camera_mode": "requires_grad=False; optimizer excludes camera_pose",
        "support": {
            "depth_raw_fraction": float(depth_raw.mean()),
            "depth_eroded_fraction": float(depth_eroded.mean()),
            "rgb_exact_nonzero_fraction": float(rgb_exact.mean()),
            "rgb_filled_footprint_fraction": float(rgb_filled.mean()),
            "rgb_eroded_footprint_fraction": float(rgb_eroded.mean()),
            "combined_rgb_depth_fraction": float(combined.mean()),
            "loss_object_mask": "(~hand) & combined_rgb_depth",
        },
        "moving_map": {
            "raw_above_0_7_fraction_full_canvas": float((raw_moving > 0.7).mean()),
            "masked_above_0_7_fraction_full_canvas": float(
                (masked_moving > 0.7).mean()
            ),
            "masked_above_0_7_fraction_within_combined_support": float(
                (masked_moving[combined_sampled] > 0.7).mean()
            ),
        },
        "source": {
            "itaco_revision": "74903b413f3ef3b7f64df166aac4adcd3ac8ebe5",
            "interaction_order": "source 177 through 213, forward",
            "autoseg_selected_uid": 25,
            "coarse_initialization": str(args.coarse_prediction_dir.resolve()),
            "monst3r_ran": False,
            "promptda_ran": False,
        },
        "outputs": {
            "raw_moving_map": str(output_dir / "moving_map.npz"),
            "rgb_depth_masked_moving_map": str(
                output_dir / "moving_map_rgb_depth_masked.npz"
            ),
            "combined_validity": str(output_dir / "combined_validity.npz"),
        },
    }
    (output_dir / "corrected_refinement_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "joint_value"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
