#!/usr/bin/env python3
"""Run official iTACO coarse prediction with GT cameras and scaled match count."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itaco-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-min-matches", type=int, default=80)
    parser.add_argument("--reference-height", type=int, default=480)
    parser.add_argument("--reference-width", type=int, default=640)
    args = parser.parse_args()
    sys.path.insert(0, str(args.itaco_dir.resolve()))

    from data import RealDataLoader
    from joint_coarse_prediction import CoarsePrediction

    class NativeGTRealDataLoader(RealDataLoader):
        def __init__(self, video_dir: str, preprocess_dir: str):
            super().__init__(video_dir, preprocess_dir)
            first_rgb = cv2.imread(str(sorted((Path(video_dir) / "rgb").glob("*.jpg"))[0]))
            self.H, self.W = first_rgb.shape[:2]

        def load_gt_camera_pose_se3(self) -> np.ndarray:
            return np.load(Path(self.preprocess_dir) / "gt_cam2world_all.npy")

    class ScaledGTCoarsePrediction(CoarsePrediction):
        def __init__(self, *positional, min_dynamic_matches: int, **keywords):
            super().__init__(*positional, **keywords)
            self.min_dynamic_matches = min_dynamic_matches
            self.pair_diagnostics = []

        def align_view(self):
            poses = self.data_loader.load_gt_camera_pose_se3()
            self.camera2label = [pose.copy() for pose in poses]
            return [
                (
                    xyz.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
                ).reshape(self.H, self.W, 3)
                for xyz, pose in zip(self.xyz_list, poses)
            ]

        def estimate_joint(self):
            pc_list = self.align_view()
            result_list = []
            pair_list = []
            for interval in [1, 2, 3]:
                for frame in range(0, len(self.rgb_list) - interval):
                    mkpts0, mkpts1, confidence = self.compute_match(
                        self.rgb_list[frame], self.rgb_list[frame + interval]
                    )
                    confident = confidence > 0.9
                    mkpts0 = mkpts0[confident].astype(np.uint32)
                    mkpts1 = mkpts1[confident].astype(np.uint32)
                    dynamic_mask = self.dynamic_mask_list[frame]
                    if self.obj_mask_list is not None:
                        dynamic_mask = dynamic_mask & self.obj_mask_list[frame]
                    dynamic_index = np.nonzero(
                        dynamic_mask[mkpts0[:, 1], mkpts0[:, 0]]
                    )[0]
                    dynamic_pts0 = mkpts0[dynamic_index]
                    dynamic_pts1 = mkpts1[dynamic_index]
                    dynamic_kp0 = pc_list[frame][
                        dynamic_pts0[:, 1], dynamic_pts0[:, 0]
                    ]
                    dynamic_kp1 = pc_list[frame + interval][
                        dynamic_pts1[:, 1], dynamic_pts1[:, 0]
                    ]
                    before_depth_filter = len(dynamic_kp0)
                    dynamic_kp0, dynamic_kp1 = self.filter_match(
                        dynamic_kp0, dynamic_kp1
                    )
                    after_depth_filter = len(dynamic_kp0)
                    accepted = after_depth_filter >= self.min_dynamic_matches
                    self.pair_diagnostics.append(
                        {
                            "frame0": frame,
                            "frame1": frame + interval,
                            "confident_matches": int(confident.sum()),
                            "dynamic_matches_before_depth_filter": before_depth_filter,
                            "dynamic_matches_after_depth_filter": after_depth_filter,
                            "accepted": accepted,
                        }
                    )
                    if accepted:
                        result = self.estimate_joint_single(
                            dynamic_kp0, dynamic_kp1, RANSAC=True
                        )
                        result_list.append(result)
                        pair_list.append((frame, frame + interval))

            if result_list:
                metrics, predicted_type = self.estimate_joint_all(result_list)
                for joint_type in metrics:
                    value_per_frame = 0.0
                    for pair, result in zip(pair_list, result_list):
                        if joint_type == "revolute":
                            value = self.compute_average_rotation_angle(
                                result[joint_type]["X"],
                                result[joint_type]["Y"],
                                metrics[joint_type]["axis"],
                                metrics[joint_type]["pos"],
                            )
                        else:
                            value = self.compute_average_translation_distance(
                                result[joint_type]["X"],
                                result[joint_type]["Y"],
                                metrics[joint_type]["axis"],
                            )
                        value_per_frame += value / (pair[1] - pair[0])
                    metrics[joint_type]["average_value"] = value_per_frame / len(
                        result_list
                    )
            else:
                metrics, predicted_type = None, None
            self.prediction_joint_metrics = metrics
            self.prediction_joint_type = predicted_type
            return metrics, predicted_type

    loader = NativeGTRealDataLoader(str(args.view_dir), str(args.preprocess_dir))
    scaled_matches = max(
        10,
        round(
            args.reference_min_matches
            * loader.H
            * loader.W
            / (args.reference_height * args.reference_width)
        ),
    )
    predictor = ScaledGTCoarsePrediction(
        loader,
        str(args.prediction_dir),
        "monst3r",
        torch.device("cuda"),
        args.seed,
        min_dynamic_matches=scaled_matches,
    )
    metrics, predicted_type = predictor.estimate_joint()
    predictor.save_prediction_results()
    report = {
        "stage": "official iTACO CoarsePrediction with native-resolution threshold scaling",
        "native_resolution_hw": [loader.H, loader.W],
        "camera_alignment": "GT per-frame PV camera-to-world",
        "reference_dynamic_match_threshold": args.reference_min_matches,
        "reference_resolution_hw": [args.reference_height, args.reference_width],
        "scaled_dynamic_match_threshold": scaled_matches,
        "predicted_joint_type": predicted_type,
        "metrics": {
            joint_type: {
                key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
                for key, value in joint_metrics.items()
            }
            for joint_type, joint_metrics in (metrics or {}).items()
        },
        "accepted_pair_count": int(
            sum(item["accepted"] for item in predictor.pair_diagnostics)
        ),
        "pair_diagnostics": predictor.pair_diagnostics,
    }
    output = (
        args.prediction_dir
        / "coarse_prediction"
        / "monst3r"
        / str(args.seed)
        / "gt_native_coarse_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pair_diagnostics"}, indent=2))
    print(
        "pair match counts:",
        [
            item["dynamic_matches_after_depth_filter"]
            for item in predictor.pair_diagnostics
        ],
    )


if __name__ == "__main__":
    main()
