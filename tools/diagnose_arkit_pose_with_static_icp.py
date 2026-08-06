#!/usr/bin/env python3
"""Measure residual static-scene alignment after applying recorded ARKit poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def camera_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    return np.stack(
        [
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    )


def percentile(values: list[float], probabilities: list[float]) -> dict:
    if not values:
        return {}
    result = np.quantile(np.asarray(values), probabilities)
    return {f"p{int(p * 100):02d}": float(v) for p, v in zip(probabilities, result)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size-m", type=float, default=0.01)
    parser.add_argument("--icp-distance-m", type=float, default=0.04)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.fusion_report.read_text())
    geometry = report["geometry_bounds_drawer_axis_u_v_m"]
    origin = np.asarray(geometry["origin_world"])
    axis = np.asarray(report["motion"]["opening_axis_world"])
    basis = np.stack(
        [axis, np.asarray(geometry["basis_u_world"]), np.asarray(geometry["basis_v_world"])],
        axis=1,
    )
    static_crop = report["post_static_crop"]
    static_lower = np.asarray(static_crop["static_crop_lower_m"])
    static_upper = np.asarray(static_crop["static_crop_upper_m"])
    drawer_lower_closed = np.asarray(geometry["lower"]) - 0.015
    drawer_upper_closed = np.asarray(geometry["upper"]) + 0.015
    travel = float(report["motion"]["travel_m"])
    drawer_lower_open = drawer_lower_closed + np.asarray([travel, 0.0, 0.0])
    drawer_upper_open = drawer_upper_closed + np.asarray([travel, 0.0, 0.0])

    poses = np.load(args.run_root / "normalized/camera_to_world.npy")
    intrinsics = np.load(args.run_root / "normalized/intrinsics_depth.npy")
    cache: dict[tuple[int, str], o3d.geometry.PointCloud] = {}

    def cloud(frame: int, state: str) -> o3d.geometry.PointCloud:
        key = (frame, state)
        if key in cache:
            return cache[key]
        depth_mm = cv2.imread(
            str(args.raw_root / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED
        )
        confidence = cv2.imread(
            str(args.raw_root / "confidence" / f"{frame:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        depth = depth_mm.astype(np.float64) / 1000.0
        valid = (depth > 0.25) & (depth < 4.5) & (confidence >= 1)
        camera = camera_points(depth, intrinsics[frame]).reshape(-1, 3)
        world = camera @ poses[frame, :3, :3].T + poses[frame, :3, 3]
        coords = (world - origin) @ basis
        inside_static = np.all(
            (coords >= static_lower) & (coords <= static_upper), axis=1
        )
        if state == "closed":
            drawer_lower, drawer_upper = drawer_lower_closed, drawer_upper_closed
        else:
            drawer_lower, drawer_upper = drawer_lower_open, drawer_upper_open
        inside_drawer = np.all(
            (coords >= drawer_lower) & (coords <= drawer_upper), axis=1
        )
        keep = valid.reshape(-1) & inside_static & (~inside_drawer)
        output = o3d.geometry.PointCloud()
        output.points = o3d.utility.Vector3dVector(world[keep])
        output = output.voxel_down_sample(args.voxel_size_m)
        if len(output.points) >= 30:
            output.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.04, max_nn=60
                )
            )
        cache[key] = output
        return output

    ranges = {
        "closed": list(range(0, 401, args.frame_step)),
        "open": list(range(545, 1102, args.frame_step)),
    }
    results: dict[str, list[dict]] = {}
    for state, frames in ranges.items():
        pairs = []
        for first, second in zip(frames[:-1], frames[1:]):
            target = cloud(first, state)
            source = cloud(second, state)
            if len(target.points) < 100 or len(source.points) < 100:
                continue
            before = o3d.pipelines.registration.evaluate_registration(
                source, target, args.icp_distance_m, np.eye(4)
            )
            refined = o3d.pipelines.registration.registration_icp(
                source,
                target,
                args.icp_distance_m,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
            )
            transform = refined.transformation
            rotation_deg = float(
                np.linalg.norm(Rotation.from_matrix(transform[:3, :3]).as_rotvec())
                * 180.0
                / np.pi
            )
            translation_m = float(np.linalg.norm(transform[:3, 3]))
            accepted = bool(
                refined.fitness >= 0.25
                and rotation_deg <= 5.0
                and translation_m <= 0.10
            )
            pairs.append(
                {
                    "frames": [first, second],
                    "points": [len(target.points), len(source.points)],
                    "before_fitness": float(before.fitness),
                    "before_rmse_m": float(before.inlier_rmse),
                    "after_fitness": float(refined.fitness),
                    "after_rmse_m": float(refined.inlier_rmse),
                    "residual_translation_m": translation_m,
                    "residual_rotation_deg": rotation_deg,
                    "accepted": accepted,
                }
            )
            print(
                f"{state} {first:04d}->{second:04d} fitness={refined.fitness:.3f} "
                f"dt={translation_m * 100:.2f}cm dr={rotation_deg:.2f}deg "
                f"rmse={refined.inlier_rmse * 100:.2f}cm accepted={accepted}",
                flush=True,
            )
        results[state] = pairs

    summary = {}
    for state, pairs in results.items():
        accepted = [item for item in pairs if item["accepted"]]
        summary[state] = {
            "pairs_total": len(pairs),
            "pairs_accepted": len(accepted),
            "residual_translation_cm": percentile(
                [item["residual_translation_m"] * 100 for item in accepted],
                [0.5, 0.75, 0.9, 0.95],
            ),
            "residual_rotation_deg": percentile(
                [item["residual_rotation_deg"] for item in accepted],
                [0.5, 0.75, 0.9, 0.95],
            ),
            "after_rmse_cm": percentile(
                [item["after_rmse_m"] * 100 for item in accepted],
                [0.5, 0.75, 0.9, 0.95],
            ),
        }
    output = {
        "meaning": (
            "Residual rigid correction needed after applying ARKit camera-to-world. "
            "It upper-bounds local ARKit pose error because depth noise, calibration, "
            "mask leakage and ICP ambiguity also contribute."
        ),
        "parameters": {
            "frame_step": args.frame_step,
            "voxel_size_m": args.voxel_size_m,
            "icp_distance_m": args.icp_distance_m,
            "static_crop_axis_u_v_m": [static_lower.tolist(), static_upper.tolist()],
            "drawer_exclusion_margin_m": 0.015,
        },
        "summary": summary,
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
