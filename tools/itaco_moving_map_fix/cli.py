#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.itaco_moving_map_fix.bundle_adjustment_adapter import (
    make_bundle_adjustment_adapter,
)
from tools.itaco_moving_map_fix.config import load_config
from tools.itaco_moving_map_fix.diagnostics import (
    compare_modes,
    comparison_metrics,
    environment_manifest,
    label_array,
    proposal_evidence_table,
    sha256_file,
    write_csv,
)
from tools.itaco_moving_map_fix.valid_support import (
    build_sensor_support,
    load_depth_validity,
    load_original_hand_masks,
    load_projected_rgb_footprints,
)


def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def core_hash(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "moving_map_gated.npz",
        "joint_axis.npy",
        "joint_pos.npy",
        "joint_value.npy",
        "camera_poses.npy",
        "proposal_scores_final.npy",
    ):
        digest.update(name.encode())
        loaded = np.load(output_dir / name)
        array = loaded["a"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_mode(config_path: Path, mode: str) -> None:
    config = load_config(config_path)
    output_root = Path(config["experiment"]["output_root"])
    output_dir = output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["mode"] = mode
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    inputs = config["inputs"]
    itaco_dir = Path(inputs["itaco_dir"])
    view_dir = Path(inputs["view_dir"])
    preprocess_dir = Path(inputs["preprocess_dir"])
    prediction_dir = Path(inputs["prediction_dir"])
    sys.path.insert(0, str(itaco_dir))
    import wandb
    from data import RealDataLoader
    from joint_refinement import (
        BundleAdjustment,
        axis_angle_to_matrix,
        chamfer_distance,
        quaternion_to_matrix,
    )

    baseline_paths = [itaco_dir / "joint_refinement.py", itaco_dir / "data.py"]
    before_hash = {str(path): sha256_file(path) for path in baseline_paths}

    rgb_paths = sorted((view_dir / "rgb").glob("*.jpg"))
    depth_paths = sorted((preprocess_dir / "prompt_depth_video").glob("*.npy"))
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise RuntimeError("RGB/depth inputs are absent or have unequal frame counts")
    first = cv2.imread(str(rgb_paths[0]))
    height, width = first.shape[:2]

    class NativeResolutionLoader(RealDataLoader):
        def __init__(self, video_dir: str, preprocess: str):
            super().__init__(video_dir, preprocess)
            self.H, self.W = height, width

    loader = NativeResolutionLoader(str(view_dir), str(preprocess_dir))
    support_cfg = config["valid_support"]
    rgb_exact, rgb_footprint = load_projected_rgb_footprints(
        rgb_paths,
        close_radius=support_cfg["rgb_close_radius_pixels"],
        erosion_radius=support_cfg["rgb_erosion_radius_pixels"],
        min_contour_area=support_cfg["rgb_min_contour_area_pixels"],
    )
    depth_raw, depth_valid = load_depth_validity(
        depth_paths,
        minimum_m=support_cfg["depth_min_m"],
        maximum_m=support_cfg["depth_max_m"],
        erosion_radius=support_cfg["depth_erosion_radius_pixels"],
    )
    hand = load_original_hand_masks(
        preprocess_dir / "hand_mask", loader.sample_rgb_index
    )
    sampled = np.asarray(loader.sample_rgb_index)
    rgb_exact = rgb_exact[sampled]
    rgb_footprint = rgb_footprint[sampled]
    depth_raw = depth_raw[sampled]
    depth_valid = depth_valid[sampled]
    sensor_support = build_sensor_support(rgb_footprint, depth_valid, hand)
    official_obj_mask = loader.load_obj_mask().astype(bool)
    if not np.array_equal(official_obj_mask, ~hand):
        raise AssertionError("Official obj_mask is not exactly inverse original hand mask")

    manifest = environment_manifest(itaco_dir, baseline_paths)
    manifest["inputs"] = {
        "rgb_frame_count": len(rgb_paths),
        "depth_frame_count": len(depth_paths),
        "sample_rgb_index": loader.sample_rgb_index,
        "resolution_hw": [height, width],
        "original_obj_mask_verified_as_inverse_hand": True,
    }
    (output_dir / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    save_npz(
        output_dir / "valid_support.npz",
        sensor_support=sensor_support,
        projected_rgb_footprint=rgb_footprint,
        rgb_exact_nonzero=rgb_exact,
        depth_valid=depth_valid,
        depth_valid_raw=depth_raw,
        original_hand_mask=hand,
    )

    opt = config["optimization"]
    mm = config["moving_map"]
    adapter_class = make_bundle_adjustment_adapter(
        BundleAdjustment,
        chamfer_distance=chamfer_distance,
        axis_angle_to_matrix=axis_angle_to_matrix,
        quaternion_to_matrix=quaternion_to_matrix,
        wandb=wandb,
    )
    internal_log = output_dir / "official_layout" / "monst3r"
    os.environ.setdefault("WANDB_MODE", "offline")
    with wandb.init(
        project="itaco_proposal_moving_map_fix",
        config=resolved,
        dir=str(output_dir),
        name=f"{mode}/seed{opt['seed']}",
    ):
        adjustment = adapter_class(
            loader,
            str(prediction_dir),
            "monst3r",
            opt["joint_type"],
            opt["learning_rate"],
            opt["loss"],
            opt["steps"],
            str(internal_log),
            torch.device(opt["device"]),
            opt["seed"],
            False,
            mode=mode,
            valid_support=sensor_support,
            static_threshold=mm["static_threshold"],
            moving_threshold=mm["moving_threshold"],
            overlap_aggregation=mm["overlap_aggregation"],
            logit_epsilon=mm["logit_epsilon"],
        )
        if not adjustment.valid:
            raise RuntimeError("Official coarse initialization is missing")
        moving_parameter_mode = mm.get("parameter_mode", "optimize")
        moving_source_report = {"mode": moving_parameter_mode}
        if moving_parameter_mode == "fixed_from_file":
            if mode != "gate_no_minmax":
                raise ValueError(
                    "fixed_from_file currently requires gate_no_minmax logits"
                )
            source_path = Path(mm["fixed_parameter_path"])
            fixed_parameter = np.load(source_path)
            if fixed_parameter.shape != tuple(adjustment._moving_parameter().shape):
                raise ValueError(
                    f"fixed moving parameter shape {fixed_parameter.shape} != "
                    f"{tuple(adjustment._moving_parameter().shape)}"
                )
            with torch.no_grad():
                adjustment._moving_parameter().copy_(
                    torch.as_tensor(
                        fixed_parameter,
                        dtype=adjustment._moving_parameter().dtype,
                        device=adjustment.device,
                    )
                )
            adjustment._moving_parameter().requires_grad_(False)
            adjustment.best_parameter = fixed_parameter.copy()
            moving_source_report.update(
                {
                    "source": str(source_path.resolve()),
                    "sha256": sha256_file(source_path),
                    "parameter_count": int(fixed_parameter.size),
                }
            )
            np.save(output_dir / "proposal_parameter_frozen.npy", fixed_parameter)
        elif moving_parameter_mode != "optimize":
            raise ValueError(
                f"Unsupported moving_map.parameter_mode: {moving_parameter_mode}"
            )
        (output_dir / "moving_parameter_source_report.json").write_text(
            json.dumps(moving_source_report, indent=2) + "\n", encoding="utf-8"
        )

        camera_mode = opt.get("camera_mode", "optimize")
        camera_source_report = {"mode": camera_mode}
        if camera_mode == "fixed_hololens":
            hololens_path = preprocess_dir / "gt_cam2world_all.npy"
            hololens_se3 = np.load(hololens_path)
            if hololens_se3.shape != (len(loader.sample_rgb_index), 4, 4):
                raise ValueError("gt_cam2world_all.npy shape does not match sampled video")
            hololens_quat_t = np.concatenate(
                [Rotation.from_matrix(hololens_se3[:, :3, :3]).as_quat(scalar_first=True), hololens_se3[:, :3, 3]], axis=1
            )
            coarse_quat_t = adjustment.camera_pose.detach().cpu().numpy()
            coarse_rotation = Rotation.from_quat(coarse_quat_t[:, :4], scalar_first=True)
            hololens_rotation = Rotation.from_quat(hololens_quat_t[:, :4], scalar_first=True)
            camera_source_report.update({
                "source": str(hololens_path.resolve()),
                "coarse_translation_mean_difference_m": float(np.linalg.norm(coarse_quat_t[:, 4:] - hololens_quat_t[:, 4:], axis=1).mean()),
                "coarse_rotation_mean_difference_degrees": float(np.degrees((coarse_rotation.inv() * hololens_rotation).magnitude()).mean()),
            })
            with torch.no_grad():
                adjustment.camera_pose.copy_(torch.as_tensor(hololens_quat_t, dtype=adjustment.camera_pose.dtype, device=adjustment.device))
            adjustment.camera_pose.requires_grad_(False)
            adjustment.best_camera_poses = hololens_quat_t.copy()
        elif camera_mode != "optimize":
            raise ValueError(f"Unsupported camera_mode: {camera_mode}")
        (output_dir / "camera_source_report.json").write_text(json.dumps(camera_source_report, indent=2) + "\n", encoding="utf-8")
        trainable_parameters = [
            adjustment.joint_axis,
            adjustment.joint_pos,
            adjustment.joint_state,
        ]
        trainable_names = ["joint_axis", "joint_pos", "joint_state"]
        if camera_mode == "optimize":
            trainable_parameters.insert(0, adjustment.camera_pose)
            trainable_names.insert(0, "camera_pose")
        if moving_parameter_mode == "optimize":
            trainable_parameters.append(adjustment._moving_parameter())
            trainable_names.append("moving_parameter")
        adjustment.optimizer = torch.optim.Adam(
            [{"params": tuple(trainable_parameters)}], lr=adjustment.lr
        )
        adjustment.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            adjustment.optimizer, adjustment.steps
        )
        optimizer_report = {
            "trainable_parameters": trainable_names,
            "camera_frozen": camera_mode != "optimize",
            "moving_parameter_frozen": moving_parameter_mode != "optimize",
        }
        (output_dir / "optimizer_parameter_report.json").write_text(
            json.dumps(optimizer_report, indent=2) + "\n", encoding="utf-8"
        )
        wandb.watch(adjustment, log="all", log_freq=1)
        adjustment.dump_configuration()
        initial_occupation = adjustment.initial_occupation.detach().cpu().numpy()
        np.save(output_dir / "proposal_occupation_initial.npy", initial_occupation)
        initial_scores = (
            np.clip(initial_occupation, mm["logit_epsilon"], 1.0 - mm["logit_epsilon"])
            if mode == "gate_no_minmax"
            else initial_occupation
        )
        np.save(output_dir / "proposal_scores_initial.npy", initial_scores)
        initial_state = {
            "camera": adjustment.camera_pose.detach().cpu().numpy(),
            "joint_axis": adjustment.joint_axis.detach().cpu().numpy(),
            "joint_pos": adjustment.joint_pos.detach().cpu().numpy(),
            "joint_value": adjustment.joint_state.detach().cpu().numpy(),
            "proposal_parameter": adjustment._moving_parameter().detach().cpu().numpy(),
        }
        save_npz(output_dir / "initial_parameters.npz", **initial_state)
        adjustment.optimize_adam(None, None, None, None)

    best_parameter = torch.as_tensor(
        adjustment.best_parameter, dtype=torch.float64, device=adjustment.device
    )
    result = adjustment.moving_map_result(best_parameter)
    if mode == "gate_no_minmax":
        proposal_scores = torch.sigmoid(best_parameter).detach().cpu().numpy()
        proposal_logits = adjustment.best_parameter
    else:
        proposal_scores = adjustment.best_parameter.copy()
        proposal_logits = np.full_like(proposal_scores, np.nan)
    np.save(output_dir / "proposal_logits_final.npy", proposal_logits)
    np.save(output_dir / "proposal_scores_final.npy", proposal_scores)
    save_npz(output_dir / "moving_map_raw.npz", a=result["raw_score"].detach().cpu().numpy())
    save_npz(output_dir / "moving_map_gated.npz", a=result["gated_score"].detach().cpu().numpy())
    save_npz(output_dir / "moving_map_labels.npz", a=label_array(result))
    save_npz(
        output_dir / "proposal_coverage.npz",
        a=result["proposal_coverage"].detach().cpu().numpy(),
    )
    np.save(output_dir / "joint_axis.npy", adjustment.best_joint_axis)
    np.save(output_dir / "joint_pos.npy", adjustment.best_joint_pos)
    np.save(output_dir / "joint_value.npy", adjustment.best_joint_state)
    np.save(output_dir / "camera_poses.npy", adjustment.best_camera_poses)
    (output_dir / "best_loss.txt").write_text(f"{adjustment.best_loss}\n")
    np.save(output_dir / "loss_history.npy", np.asarray(adjustment.loss_history))

    evidence_support = (
        np.ones_like(sensor_support) if mode == "official" else sensor_support
    )
    rows = proposal_evidence_table(adjustment, evidence_support, proposal_scores)
    write_csv(output_dir / "proposal_score_table.csv", rows)
    metrics = comparison_metrics(
        result, sensor_support, proposal_scores, adjustment.best_loss, mm["moving_threshold"]
    )
    metrics["mode"] = mode
    metrics["overlap"] = {
        "aggregation": mm["overlap_aggregation"],
        "maximum_coverage_count": int(
            adjustment.part_segments_full.sum(dim=1).max().detach().cpu()
        ),
        "raw_score_over_one_pixels": int(
            (result["raw_score"] > 1).sum().detach().cpu()
        ),
    }
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    after_hash = {str(path): sha256_file(path) for path in baseline_paths}
    audit = {
        "official_files_modified_by_run": before_hash != after_hash,
        "before": before_hash,
        "after": after_hash,
        "core_output_hash": core_hash(output_dir),
    }
    (output_dir / "baseline_integrity.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if audit["official_files_modified_by_run"]:
        raise AssertionError("Official baseline source changed during adapter run")
    print(json.dumps({"mode": mode, "output": str(output_dir), **metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--mode", choices=["official", "gate_only", "gate_no_minmax"], required=True
    )
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_mode(args.config, args.mode)
    else:
        config = load_config(args.config)
        report = compare_modes(Path(config["experiment"]["output_root"]), config)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
