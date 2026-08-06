#!/usr/bin/env python3
"""Run RGB/depth-gated prismatic iTACO with a GT-track-constrained axis."""

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

from run_itaco_fixed_camera_rgb_depth_valid import depth_masks, rgb_masks


def load_track_direction(path: Path, uid: int) -> np.ndarray:
    report = json.loads(path.read_text(encoding="utf-8"))
    track = next((item for item in report["tracks"] if item["uid"] == uid), None)
    if track is None:
        raise KeyError(f"UID {uid} not present in {path}")
    direction = np.asarray(track["linear_direction_world"], dtype=np.float64)
    return direction / np.linalg.norm(direction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itaco-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--coarse-prediction-dir", type=Path, required=True)
    parser.add_argument("--gt-track-metrics", type=Path, required=True)
    parser.add_argument("--gt-track-uid", type=int, default=25)
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
    if len(depth_raw) != len(rgb_exact):
        raise ValueError("RGB and depth frame counts differ")
    combined = depth_eroded & rgb_eroded
    gt_axis = load_track_direction(args.gt_track_metrics, args.gt_track_uid)

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
        / "refinement_gt_fixed_camera_gt_axis_rgb_depth_valid"
        / "monst3r/chamfer"
        / str(args.seed)
        / "prismatic"
    )
    log_dir = (
        args.output_root
        / "refinement_gt_fixed_camera_gt_axis_rgb_depth_valid"
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
        "axis": f"fixed GT-depth centroid trajectory, AutoSeg UID {args.gt_track_uid}",
        "loss_mask": "(~hand) & eroded_depth_support & eroded_projected_rgb_footprint",
        "depth_range_m": [args.depth_min, args.depth_max],
        "depth_erosion_radius_pixels": args.depth_erosion_radius,
        "rgb_close_radius_pixels": args.rgb_close_radius,
        "rgb_erosion_radius_pixels": args.rgb_erosion_radius,
        "rgb_min_contour_area_pixels": args.rgb_min_contour_area,
    }
    with wandb.init(
        project="video_articulation_refinement_gt_fixed_camera_gt_axis_rgb_depth_valid",
        config=config,
        dir=str(output_dir),
        name=f"{args.view_dir}/prismatic/fixed_gt_axis_rgb_depth/seed{args.seed}/",
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

        coarse_axis = adjustment.joint_axis.detach().cpu().numpy()
        if float(np.dot(coarse_axis, gt_axis)) < 0:
            gt_axis = -gt_axis
        with torch.no_grad():
            adjustment.joint_axis.copy_(
                torch.from_numpy(gt_axis).to(
                    device=adjustment.device, dtype=adjustment.joint_axis.dtype
                )
            )
        adjustment.camera_pose.requires_grad_(False)
        adjustment.joint_axis.requires_grad_(False)
        adjustment.optimizer = torch.optim.Adam(
            [
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
        "output_kind": "fixed GT-track axis, projected-RGB and depth gated",
        "official_outputs_modified": False,
        "joint_hypothesis": "prismatic",
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "best_loss": float(adjustment.best_loss),
        "joint_axis_unit": axis_unit.tolist(),
        "joint_value": values.tolist(),
        "joint_range_m": float(np.ptp(values)),
        "joint_end_delta_m": float(abs(values[-1] - values[0])),
        "joint_positive_steps": int((np.diff(values) > 0).sum()),
        "joint_negative_steps": int((np.diff(values) < 0).sum()),
        "camera_mode": "fixed HoloLens camera-to-world",
        "axis_mode": (
            f"fixed world-centroid trajectory direction for AutoSeg UID "
            f"{args.gt_track_uid}; sign matched to official coarse axis"
        ),
        "support": {
            "rgb_eroded_footprint_fraction": float(rgb_eroded.mean()),
            "depth_eroded_fraction": float(depth_eroded.mean()),
            "combined_rgb_depth_fraction": float(combined.mean()),
            "loss_object_mask": "(~hand) & combined_rgb_depth",
        },
        "moving_map": {
            "masked_above_0_7_fraction_full_canvas": float(
                (masked_moving > 0.7).mean()
            ),
            "masked_above_0_7_fraction_within_combined_support": float(
                (masked_moving[combined_sampled] > 0.7).mean()
            ),
            "above_0_7_outside_combined_pixels": int(
                ((masked_moving > 0.7) & ~combined_sampled).sum()
            ),
        },
        "source": {
            "itaco_revision": "74903b413f3ef3b7f64df166aac4adcd3ac8ebe5",
            "interaction_order": "source 177 through 213, forward",
            "autoseg_selected_uid": args.gt_track_uid,
            "gt_track_metrics": str(args.gt_track_metrics.resolve()),
            "coarse_initialization": str(args.coarse_prediction_dir.resolve()),
            "monst3r_ran": False,
            "promptda_ran": False,
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
