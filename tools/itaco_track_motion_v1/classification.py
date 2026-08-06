"""Track-level static/moving/unknown evidence under one fixed moving model."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import math
import numpy as np

from .geometry import KnownMotion


def _soft_sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def _model_fit(points: np.ndarray, sigma_floor: float) -> dict:
    center = np.median(points, axis=0)
    residual = np.linalg.norm(points - center[None], axis=1)
    rss = float(np.sum(residual ** 2))
    sample_count = max(3 * len(points), 1)
    variance = max(rss / sample_count, sigma_floor ** 2)
    bic = sample_count * math.log(variance) + 3 * math.log(sample_count)
    median = float(np.median(residual))
    mad = float(1.4826 * np.median(np.abs(residual - median)))
    return {"center": center, "residual": residual, "rss": rss, "rmse_m": float(np.sqrt(rss / max(len(points), 1))), "median_residual_m": median, "residual_mad_m": mad, "bic": float(bic)}


def classify_tracks(tracks: list[dict], known_motion: KnownMotion, config: dict) -> tuple[list[dict], list[dict], dict[int, np.ndarray]]:
    residual_records, labels, representative = [], [], {}
    state = known_motion.state
    for track in tracks:
        observations = track["observations"]
        points = np.asarray([obs["point_world"] for obs in observations], dtype=np.float64)
        indices = np.asarray([obs["processing_index"] for obs in observations], dtype=np.int64)
        static_fit = _model_fit(points, float(config["residual_sigma_floor_m"]))
        moving_points = known_motion.canonicalize(points, indices)
        moving_fit = _model_fit(moving_points, float(config["residual_sigma_floor_m"]))
        delta_bic = float(static_fit["bic"] - moving_fit["bic"])
        moving_binary = _soft_sigmoid(delta_bic / float(config["bic_temperature"]))
        static_binary = 1.0 - moving_binary
        span = int(indices.max() - indices.min() + 1)
        coverage = len(indices) / max(span, 1)
        occlusion_rate = track["occlusion_failures"] / max(track["attempted_transitions"], 1)
        boundary_rate = track["boundary_failures"] / max(track["attempted_transitions"], 1)
        depth_confidence = float(np.median([obs["depth_confidence"] for obs in observations]))
        proposal_counts = Counter(obs["proposal_id"] for obs in observations if obs["proposal_id"])
        proposal_support = sum(proposal_counts.values()) / max(len(observations), 1)
        dominant_proposal = proposal_counts.most_common(1)[0][0] if proposal_counts else ""
        depth_uncertainty = float(np.median([obs["depth_uncertainty_m"] for obs in observations]))
        excitation = float(state[indices].max() - state[indices].min())
        reasons = []
        if len(indices) < int(config["min_valid_observations"]): reasons.append("insufficient_valid_observations")
        if coverage < float(config["min_coverage"]): reasons.append("insufficient_temporal_coverage")
        if abs(delta_bic) < float(config["min_abs_delta_bic"]): reasons.append("static_moving_evidence_too_close")
        if static_fit["rmse_m"] > float(config["max_absolute_rmse_m"]) and moving_fit["rmse_m"] > float(config["max_absolute_rmse_m"]): reasons.append("both_models_absolute_residual_too_large")
        if depth_confidence < float(config["min_median_depth_confidence"]): reasons.append("depth_confidence_too_low")
        if depth_uncertainty > float(config["max_median_depth_uncertainty_m"]): reasons.append("depth_uncertainty_too_large")
        if occlusion_rate > float(config["max_occlusion_rate"]): reasons.append("observations_dominated_by_occlusion")
        if moving_binary > 0.5 and moving_fit["rmse_m"] > float(config["max_known_moving_rmse_m"]): reasons.append("known_moving_absolute_residual_too_large")
        if moving_binary > 0.5 and proposal_support < float(config["min_moving_proposal_support"]): reasons.append("moving_proposal_support_too_low")
        if boundary_rate > float(config["max_boundary_failure_rate"]): reasons.append("observations_dominated_by_depth_boundary")
        if excitation < float(config["min_motion_excitation"]): reasons.append("motion_excitation_insufficient")
        if reasons:
            unknown_probability = max(float(config["unknown_probability_floor_on_failure"]), _soft_sigmoid((len(reasons) - 0.5) / float(config["unknown_reason_temperature"])))
        else:
            unknown_probability = float(config["unknown_probability_without_failure"])
        remaining = 1.0 - unknown_probability
        posterior_static, posterior_moving = remaining * static_binary, remaining * moving_binary
        if reasons:
            label = "unknown"
        elif posterior_moving >= float(config["label_posterior_threshold"]):
            label = "moving"
        elif posterior_static >= float(config["label_posterior_threshold"]):
            label = "static"
        else:
            label = "unknown"; reasons.append("posterior_not_decisive")
        residual_record = {
            "track_id": track["track_id"], "valid_observations": len(indices), "temporal_span_frames": span,
            "proposal_support": proposal_support,
            "coverage": coverage, "occlusion_rate": occlusion_rate, "boundary_failure_rate": boundary_rate,
            "median_depth_confidence": depth_confidence, "median_depth_uncertainty_m": depth_uncertainty,
            "motion_excitation": excitation, "static": {key: value for key, value in static_fit.items() if key not in {"center", "residual"}},
            "known_moving": {key: value for key, value in moving_fit.items() if key not in {"center", "residual"}},
            "delta_bic_static_minus_moving": delta_bic, "known_motion_type": known_motion.type,
        }
        label_record = {
            "track_id": track["track_id"], "label": label, "posterior_static": posterior_static,
            "posterior_moving": posterior_moving, "posterior_unknown": unknown_probability,
            "unknown_reasons": reasons, "dominant_proposal_id": dominant_proposal,
            "proposal_support": proposal_support,
            "proposal_observation_counts": dict(proposal_counts),
            "original_frame_ids": [obs["original_frame_id"] for obs in observations],
            "timestamps": [obs["timestamp"] for obs in observations],
        }
        residual_records.append(residual_record); labels.append(label_record)
        representative[track["track_id"]] = static_fit["center"] if label != "moving" else moving_fit["center"]
    return residual_records, labels, representative


def build_clusters(tracks: list[dict], labels: list[dict], representative: dict[int, np.ndarray], config: dict) -> list[dict]:
    label_by_id = {item["track_id"]: item for item in labels}
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    voxel = float(config["voxel_size_m"])
    for track in tracks:
        key = tuple(np.floor(representative[track["track_id"]] / voxel).astype(np.int64).tolist())
        buckets[key].append(track["track_id"])
    clusters, cluster_id = [], 0
    max_size = int(config["max_tracks_per_cluster"])
    for voxel_key, track_ids in sorted(buckets.items()):
        for start in range(0, len(track_ids), max_size):
            subset = track_ids[start:start + max_size]
            records = [label_by_id[track_id] for track_id in subset]
            proposals = sorted({proposal for item in records for proposal in item["proposal_observation_counts"]})
            distribution = Counter(item["label"] for item in records)
            clusters.append({
                "cluster_id": f"cluster_{cluster_id:06d}", "track_ids": subset, "voxel_index": list(voxel_key),
                "track_count": len(subset), "label_distribution": dict(distribution), "proposal_ids": proposals,
                "contains_multiple_proposals": len(proposals) > 1, "proposal_identity_used_as_cluster_constraint": False,
            })
            cluster_id += 1
    return clusters


def proposal_split_merge_diagnostics(labels: list[dict], clusters: list[dict]) -> dict:
    by_proposal: dict[str, set[str]] = defaultdict(set)
    for item in labels:
        for proposal_id in item["proposal_observation_counts"]:
            by_proposal[proposal_id].add(item["label"])
    split = {proposal: sorted(values) for proposal, values in by_proposal.items() if len(values) > 1}
    merged = [cluster["cluster_id"] for cluster in clusters if cluster["contains_multiple_proposals"] and cluster["label_distribution"].get("moving", 0) > 0]
    moving_proposals = sorted({proposal for item in labels if item["label"] == "moving" for proposal in item["proposal_observation_counts"]})
    return {"mixed_proposals_split": split, "clusters_merging_multiple_proposals": merged, "moving_set_proposal_ids": moving_proposals, "moving_set_merges_multiple_proposals": len(moving_proposals) > 1}


def save_posteriors(path: Path, labels: list[dict]) -> None:
    np.savez_compressed(path,
        track_id=np.asarray([item["track_id"] for item in labels], dtype=np.int64),
        posterior_static=np.asarray([item["posterior_static"] for item in labels], dtype=np.float32),
        posterior_moving=np.asarray([item["posterior_moving"] for item in labels], dtype=np.float32),
        posterior_unknown=np.asarray([item["posterior_unknown"] for item in labels], dtype=np.float32),
        label=np.asarray([item["label"] for item in labels], dtype="U8"))


def write_label_ply(path: Path, tracks: list[dict], labels: list[dict], wanted: str) -> None:
    label_by_id = {item["track_id"]: item["label"] for item in labels}
    observations = [obs for track in tracks if label_by_id[track["track_id"]] == wanted for obs in track["observations"]]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(observations)}\nproperty float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int track_id\nproperty int original_frame_id\nend_header\n")
        for obs in observations:
            x, y, z = obs["point_world"]; r, g, b = obs["rgb"]
            handle.write(f"{x:.7g} {y:.7g} {z:.7g} {r} {g} {b} {obs['track_id']} {obs['original_frame_id']}\n")
