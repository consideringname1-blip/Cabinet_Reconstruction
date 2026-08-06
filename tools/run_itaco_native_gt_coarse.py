#!/usr/bin/env python3
"""Run official iTACO coarse prediction with native resolution and GT cameras."""

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

    class GTCoarsePrediction(CoarsePrediction):
        def align_view(self):
            poses = self.data_loader.load_gt_camera_pose_se3()
            if len(poses) != len(self.xyz_list):
                raise ValueError((poses.shape, len(self.xyz_list)))
            self.camera2label = [pose.copy() for pose in poses]
            aligned = []
            for xyz, pose in zip(self.xyz_list, poses):
                world = xyz.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
                aligned.append(world.reshape(self.H, self.W, 3))
            return aligned

    loader = NativeGTRealDataLoader(str(args.view_dir), str(args.preprocess_dir))
    predictor = GTCoarsePrediction(
        loader,
        str(args.prediction_dir),
        "monst3r",
        torch.device("cuda"),
        args.seed,
    )
    metrics, predicted_type = predictor.estimate_joint()
    predictor.save_prediction_results()
    report = {
        "stage": "official iTACO CoarsePrediction",
        "adapter": {
            "native_resolution_hw": [loader.H, loader.W],
            "camera_alignment": "GT per-frame PV camera-to-world replacing MonST3R/LoFTR camera estimation",
            "moving_mask_path_semantics": "camera-compensated GT seed stored in official monst3r filename convention",
        },
        "predicted_joint_type": predicted_type,
        "metrics": {
            joint_type: {
                key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
                for key, value in joint_metrics.items()
            }
            for joint_type, joint_metrics in (metrics or {}).items()
        },
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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
