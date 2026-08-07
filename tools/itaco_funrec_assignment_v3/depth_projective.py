"""Depth-aware projective LoFTR association under frozen static/drawer models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature import LoFTR
from scipy.spatial import cKDTree

from .core import periodic_seed_frames
from .loftr_tracking import _balanced_seed_keypoints, _infer


def world_to_camera(point_world: np.ndarray, pose_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(point_world) - pose_world_camera[:3, 3]) @ pose_world_camera[:3, :3]


def project_world(point_world: np.ndarray, pose_world_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, float]:
    camera = world_to_camera(point_world, pose_world_camera)
    z = float(camera[2])
    if z <= 0:
        return np.asarray([np.nan, np.nan]), z
    return np.asarray([intrinsic[0, 0] * camera[0] / z + intrinsic[0, 2],
                       intrinsic[1, 1] * camera[1] / z + intrinsic[1, 2]]), z


def unproject_world(uv: np.ndarray, depth_m: float, pose_world_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.asarray(uv, dtype=float)
    camera = np.asarray([(u - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
                         (v - intrinsic[1, 2]) * depth_m / intrinsic[1, 1], depth_m])
    return camera, camera @ pose_world_camera[:3, :3].T + pose_world_camera[:3, 3]


def model_prediction(anchor_world: np.ndarray, q_seed: float, q_target: float,
                     axis: np.ndarray, model: str) -> np.ndarray:
    if model == "static":
        return np.asarray(anchor_world, dtype=float)
    if model == "drawer":
        return np.asarray(anchor_world, dtype=float) + (float(q_target) - float(q_seed)) * np.asarray(axis, dtype=float)
    raise ValueError(model)


def _components(actual_world: np.ndarray, predicted_world: np.ndarray, axis: np.ndarray) -> tuple[float, float, float]:
    delta = np.asarray(actual_world) - np.asarray(predicted_world)
    along = float(np.dot(delta, axis))
    perpendicular = float(np.linalg.norm(delta - along * axis))
    return along, perpendicular, float(np.linalg.norm(delta))


def candidate_metrics(anchor_world: np.ndarray, q_seed: float, target_uv: np.ndarray,
                      target_frame: dict, q_target: float, axis: np.ndarray,
                      intrinsic: np.ndarray, cfg: dict) -> dict | None:
    u, v = np.rint(target_uv).astype(int)
    height, width = target_frame["depth"].shape
    if not (0 <= u < width and 0 <= v < height and target_frame["valid"][v, u]):
        return None
    measured_depth = float(target_frame["depth"][v, u])
    point_camera, actual_world = unproject_world(target_uv, measured_depth, target_frame["pose"], intrinsic)
    result = {"target_depth_m": measured_depth, "point_camera": point_camera, "point_world": actual_world}
    costs = {}
    for model in ("static", "drawer"):
        predicted_world = model_prediction(anchor_world, q_seed, q_target, axis, model)
        predicted_uv, predicted_depth = project_world(predicted_world, target_frame["pose"], intrinsic)
        pixel_error = float(np.linalg.norm(np.asarray(target_uv) - predicted_uv)) if np.all(np.isfinite(predicted_uv)) else float("inf")
        signed_depth = measured_depth - predicted_depth
        depth_error = abs(float(signed_depth))
        along, perpendicular, error_3d = _components(actual_world, predicted_world, axis)
        occluded = bool(measured_depth + float(cfg["occlusion_margin_m"]) < predicted_depth)
        cost = float(np.hypot(pixel_error / float(cfg["pixel_sigma_px"]), depth_error / float(cfg["depth_sigma_m"])))
        if occluded or depth_error > float(cfg["maximum_depth_residual_m"]):
            cost = float("inf")
        result.update({
            f"{model}_predicted_uv": predicted_uv,
            f"{model}_predicted_depth_m": predicted_depth,
            f"{model}_projective_error_px": pixel_error,
            f"{model}_signed_depth_residual_m": float(signed_depth),
            f"{model}_depth_residual_m": depth_error,
            f"{model}_axis_error_m": along,
            f"{model}_perpendicular_error_m": perpendicular,
            f"{model}_3d_error_m": error_3d,
            f"{model}_occluded": occluded,
            f"{model}_association_cost": cost,
        })
        costs[model] = cost
    best = min(costs, key=costs.get)
    difference = abs(costs["static"] - costs["drawer"])
    result["association_model"] = "ambiguous" if difference < float(cfg["model_cost_margin"]) else best
    result["association_cost"] = float(costs[best])
    result["association_accepted"] = bool(np.isfinite(costs[best]) and costs[best] <= float(cfg["maximum_association_cost"]))
    return result


def _observation(track: dict, frame: dict, uv: np.ndarray, q_value: float,
                 intrinsic: np.ndarray, confidence: float, fb_error: float,
                 metrics: dict, proposal_ids: list[str]) -> dict:
    payload = {
        "track_id": track["track_id"], "seed_frame_id": track["seed_frame_id"],
        "original_frame_id": int(frame["source"]), "pixel_uv": np.asarray(uv, float).tolist(),
        "depth_m": float(metrics["target_depth_m"]), "point_camera": metrics["point_camera"].tolist(),
        "point_world": metrics["point_world"].tolist(), "q_t": float(q_value),
        "proposal_ids": proposal_ids, "tracking_confidence": float(confidence),
        "forward_backward_error": float(fb_error), "visibility": True, "occlusion": False,
        "depth_edge_distance": float(frame["edge_distance"][int(round(uv[1])), int(round(uv[0]))]),
        "hand_flag": False,
    }
    for key, value in metrics.items():
        if key in ("point_camera", "point_world", "target_depth_m"):
            continue
        payload[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return payload


def _seed_metrics(point_world: np.ndarray, point_camera: np.ndarray, uv: np.ndarray, depth: float) -> dict:
    result = {"target_depth_m": depth, "point_camera": point_camera, "point_world": point_world,
              "association_model": "seed", "association_cost": 0.0, "association_accepted": True}
    for model in ("static", "drawer"):
        result.update({f"{model}_predicted_uv": np.asarray(uv, float), f"{model}_predicted_depth_m": depth,
                       f"{model}_projective_error_px": 0.0, f"{model}_signed_depth_residual_m": 0.0,
                       f"{model}_depth_residual_m": 0.0, f"{model}_axis_error_m": 0.0,
                       f"{model}_perpendicular_error_m": 0.0, f"{model}_3d_error_m": 0.0,
                       f"{model}_occluded": False, f"{model}_association_cost": 0.0})
    return result


def build_depth_projective_tracks(frames: list[dict], axis: np.ndarray, q: np.ndarray,
                                  intrinsic: np.ndarray, cfg: dict) -> tuple[list[dict], list[dict]]:
    tcfg, acfg = cfg["tracking"], cfg["depth_projective_association"]
    torch.manual_seed(int(cfg["determinism"]["seed"])); torch.cuda.manual_seed_all(int(cfg["determinism"]["seed"]))
    model = LoFTR(pretrained="indoor").eval().cuda()
    grays = [cv2.cvtColor(frame["rgb"], cv2.COLOR_BGR2GRAY) for frame in frames]
    tracks, attempts, next_id = [], [], 0
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    for seed in periodic_seed_frames(len(frames), int(tcfg["reseed_interval_frames"])):
        if seed == len(frames) - 1:
            continue
        kp0, _, confidence0 = _infer(model, grays[seed], grays[seed + 1])
        usable = confidence0 >= float(tcfg["loftr_min_confidence"])
        balanced = _balanced_seed_keypoints(kp0[usable], confidence0[usable], frames[seed]["valid"], tcfg)
        batch = []
        for uv in kp0[usable][balanced]:
            u, v = np.rint(uv).astype(int); depth = float(frames[seed]["depth"][v, u])
            point_camera, point_world = unproject_world(uv, depth, frames[seed]["pose"], intrinsic)
            proposal_ids = [p["proposal_id"] for p in frames[seed]["proposals"] if p["mask"][v, u]]
            track = {"track_id": next_id, "seed_frame_id": int(frames[seed]["source"]),
                     "seed_uv": uv.astype(np.float32), "anchor_world": point_world,
                     "q_seed": float(q[seed]), "observations": []}
            next_id += 1
            track["observations"].append(_observation(track, frames[seed], uv, q[seed], intrinsic, 1.0, 0.0,
                                                       _seed_metrics(point_world, point_camera, uv, depth), proposal_ids))
            tracks.append(track); batch.append(track)
        for target_index in range(seed + 1, len(frames)):
            kp_seed, kp_target, confidence = _infer(model, grays[seed], grays[target_index])
            reverse_target, reverse_seed, reverse_confidence = _infer(model, grays[target_index], grays[seed])
            if not len(kp_seed) or not len(reverse_target):
                continue
            forward_tree, reverse_tree = cKDTree(kp_seed), cKDTree(reverse_target)
            for track in batch:
                candidate_indices = forward_tree.query_ball_point(track["seed_uv"], float(tcfg["loftr_seed_association_radius_pixels"]))
                best = None
                for match in candidate_indices:
                    target_uv = kp_target[match]
                    reverse_distance, reverse_match = reverse_tree.query(target_uv)
                    back_uv = reverse_seed[reverse_match]
                    fb_error = float(np.linalg.norm(back_uv - track["seed_uv"]))
                    loftr_score = float(min(confidence[match], reverse_confidence[reverse_match]))
                    metrics = candidate_metrics(track["anchor_world"], track["q_seed"], target_uv,
                                                frames[target_index], q[target_index], axis, intrinsic, acfg)
                    accepted = bool(metrics is not None and reverse_distance <= float(tcfg["loftr_reverse_association_radius_pixels"])
                                    and fb_error <= float(tcfg["forward_backward_threshold_pixels"])
                                    and loftr_score >= float(tcfg["loftr_min_confidence"])
                                    and metrics["association_accepted"])
                    row = {"track_id": track["track_id"], "seed_frame_id": track["seed_frame_id"],
                           "target_frame_id": int(frames[target_index]["source"]), "loftr_confidence": loftr_score,
                           "seed_keypoint_distance_px": float(np.linalg.norm(kp_seed[match] - track["seed_uv"])),
                           "reverse_keypoint_distance_px": float(reverse_distance), "forward_backward_error_px": fb_error,
                           "accepted": accepted, "rejection_reason": None}
                    if metrics is None:
                        row["rejection_reason"] = "invalid_target_depth_or_mask"
                    else:
                        row.update({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()
                                    if key not in ("point_camera", "point_world")})
                        if not accepted:
                            if loftr_score < float(tcfg["loftr_min_confidence"]): row["rejection_reason"] = "low_loftr_confidence"
                            elif fb_error > float(tcfg["forward_backward_threshold_pixels"]): row["rejection_reason"] = "forward_backward_inconsistent"
                            elif reverse_distance > float(tcfg["loftr_reverse_association_radius_pixels"]): row["rejection_reason"] = "reverse_association_too_far"
                            elif metrics["static_occluded"] and metrics["drawer_occluded"]: row["rejection_reason"] = "occluded_under_both_models"
                            elif min(metrics["static_depth_residual_m"], metrics["drawer_depth_residual_m"]) > float(acfg["maximum_depth_residual_m"]): row["rejection_reason"] = "depth_residual_too_large"
                            else: row["rejection_reason"] = "projective_cost_too_large"
                    attempts.append(row)
                    if accepted and (best is None or metrics["association_cost"] < best[0]):
                        best = (metrics["association_cost"], target_uv, loftr_score, fb_error, metrics)
                if best is None:
                    continue
                _, target_uv, score, fb_error, metrics = best
                u, v = np.rint(target_uv).astype(int)
                proposals = [p["proposal_id"] for p in frames[target_index]["proposals"] if p["mask"][v, u]]
                track["observations"].append(_observation(track, frames[target_index], target_uv, q[target_index],
                                                           intrinsic, score * np.exp(-metrics["association_cost"]),
                                                           fb_error, metrics, proposals))
    for track in tracks:
        for key in ("seed_uv", "anchor_world", "q_seed"):
            track.pop(key, None)
    return tracks, attempts


def decompose_track_observations(tracks: list[dict], axis: np.ndarray) -> list[dict]:
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    rows = []
    for track in tracks:
        observations = track["observations"]
        world = np.asarray([o["point_world"] for o in observations], float)
        q = np.asarray([o["q_t"] for o in observations], float)
        static_center = np.median(world, axis=0)
        canonical = world - q[:, None] * axis
        drawer_center = np.median(canonical, axis=0)
        for observation, point, canonical_point in zip(observations, world, canonical):
            static_axis, static_perp, static_total = _components(point, static_center, axis)
            drawer_axis, drawer_perp, drawer_total = _components(canonical_point, drawer_center, axis)
            rows.append({
                "track_id": int(track["track_id"]), "seed_frame_id": int(track["seed_frame_id"]),
                "original_frame_id": int(observation["original_frame_id"]), "q_t": float(observation["q_t"]),
                "u": float(observation["pixel_uv"][0]), "v": float(observation["pixel_uv"][1]),
                "depth_m": float(observation["depth_m"]), "tracking_confidence": float(observation["tracking_confidence"]),
                "forward_backward_error_px": float(observation["forward_backward_error"]),
                "association_model": observation.get("association_model", "unavailable"),
                "association_cost": float(observation.get("association_cost", np.nan)),
                "static_projective_error_px": float(observation.get("static_projective_error_px", np.nan)),
                "drawer_projective_error_px": float(observation.get("drawer_projective_error_px", np.nan)),
                "static_depth_residual_m": float(observation.get("static_depth_residual_m", np.nan)),
                "drawer_depth_residual_m": float(observation.get("drawer_depth_residual_m", np.nan)),
                "static_track_axis_residual_m": static_axis, "static_track_perpendicular_residual_m": static_perp,
                "static_track_total_residual_m": static_total, "drawer_track_axis_residual_m": drawer_axis,
                "drawer_track_perpendicular_residual_m": drawer_perp, "drawer_track_total_residual_m": drawer_total,
            })
    return rows


def write_error_diagnostics(output_dir: Path, tracks: list[dict], attempts: list[dict],
                            evidence: list[dict], axis: np.ndarray, cfg: dict) -> dict:
    rows = decompose_track_observations(tracks, axis)
    def write_csv(path: Path, records: list[dict]) -> None:
        if not records: path.write_text(""); return
        keys = sorted({key for record in records for key in record})
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(records)
    write_csv(output_dir / "observation_error_decomposition.csv", rows)
    write_csv(output_dir / "projective_association_attempts.csv", attempts)
    high_ids = {int(item["track_id"]) for item in evidence if item["reason"] == "unknown_absolute_residual"}
    high_rows = [row for row in rows if row["track_id"] in high_ids]
    write_csv(output_dir / "high_residual_observations.csv", high_rows)
    accepted = [row for row in attempts if row.get("accepted")]
    rejected = [row for row in attempts if not row.get("accepted")]
    def quantiles(records: list[dict], key: str) -> dict | None:
        values = np.asarray([float(r[key]) for r in records if key in r and np.isfinite(float(r[key]))])
        if not len(values): return None
        return {"p10": float(np.percentile(values, 10)), "p50": float(np.median(values)),
                "p90": float(np.percentile(values, 90)), "max": float(values.max())}
    report = {
        "fixed_inputs": ["T_world_camera", "prismatic axis", "q_t"],
        "tracks": len(tracks), "observations": len(rows), "association_attempts": len(attempts),
        "accepted_attempts": len(accepted), "rejected_attempts": len(rejected),
        "high_residual_track_count": len(high_ids), "high_residual_observation_count": len(high_rows),
        "association_model_counts": {name: sum(r.get("association_model") == name for r in accepted)
                                     for name in ("static", "drawer", "ambiguous")},
        "rejection_reasons": {reason: sum(r.get("rejection_reason") == reason for r in rejected)
                              for reason in sorted({r.get("rejection_reason") for r in rejected}) if reason},
        "accepted_static_projective_error_px": quantiles(accepted, "static_projective_error_px"),
        "accepted_drawer_projective_error_px": quantiles(accepted, "drawer_projective_error_px"),
        "accepted_static_depth_residual_m": quantiles(accepted, "static_depth_residual_m"),
        "accepted_drawer_depth_residual_m": quantiles(accepted, "drawer_depth_residual_m"),
        "high_residual_static_axis_m": quantiles(high_rows, "static_track_axis_residual_m"),
        "high_residual_static_perpendicular_m": quantiles(high_rows, "static_track_perpendicular_residual_m"),
        "high_residual_drawer_axis_m": quantiles(high_rows, "drawer_track_axis_residual_m"),
        "high_residual_drawer_perpendicular_m": quantiles(high_rows, "drawer_track_perpendicular_residual_m"),
    }
    (output_dir / "high_residual_track_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        visual = output_dir / "visualization"; visual.mkdir(exist_ok=True)
        static_depth = np.asarray([r["static_depth_residual_m"] for r in rows]); drawer_depth = np.asarray([r["drawer_depth_residual_m"] for r in rows])
        static_perp = np.asarray([r["static_track_perpendicular_residual_m"] for r in rows]); drawer_perp = np.asarray([r["drawer_track_perpendicular_residual_m"] for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(static_depth, drawer_depth, s=3, alpha=.35); axes[0].set(xlabel="static depth residual (m)", ylabel="drawer depth residual (m)", title="Per-observation registered-depth errors")
        axes[1].scatter(static_perp, drawer_perp, s=3, alpha=.35); axes[1].set(xlabel="static perpendicular residual (m)", ylabel="drawer perpendicular residual (m)", title="Track-center perpendicular errors")
        for axis_plot in axes: axis_plot.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(visual / "observation_error_decomposition.png", dpi=180); plt.close(fig)
    return report
