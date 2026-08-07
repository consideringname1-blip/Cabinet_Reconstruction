"""Noise-normalized static-vs-drawer multi-baseline model discrimination."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .projected_source_anchor import evaluate_source_indexed_patch


def _spatial_coverage(uv: np.ndarray, selected: np.ndarray, grid_size: int) -> float:
    selected = np.asarray(selected, bool)
    if not np.any(selected):
        return 0.0
    points = np.asarray(uv, float)
    extent = np.maximum(np.ptp(points, axis=0), 1.0)
    cells = np.floor((points - points.min(axis=0)) / extent * grid_size).astype(int)
    cells = np.clip(cells, 0, grid_size - 1)
    all_cells = np.unique(cells[:, 1] * grid_size + cells[:, 0])
    hit = np.unique(cells[selected, 1] * grid_size + cells[selected, 0])
    return float(len(hit) / max(len(all_cells), 1))


def evaluate_multi_baseline_observation(
    patch: dict, source_frame: dict, target_frame: dict, observation,
    runtime: dict, noise_model, cfg: dict
) -> dict:
    evaluations = {}
    for model in ("static", "drawer"):
        evaluations[model] = evaluate_source_indexed_patch(
            patch, target_frame, runtime["context"]["axis"], runtime["context"]["intrinsic"],
            model, cfg["projective_anchor"], runtime["trusted_masks"][int(target_frame["source"])] )
    common = (np.asarray(evaluations["static"]["arrays"]["observable"], bool) &
              np.asarray(evaluations["drawer"]["arrays"]["observable"], bool))
    common_count = int(common.sum())
    coverage = float(common_count / max(int(patch["sample_count"]), 1))
    spatial = _spatial_coverage(
        patch["source_uv"], common, int(cfg["visibility"]["coverage_grid_size"]))
    visibility = (
        common_count >= int(cfg["visibility"]["minimum_common_source_samples"])
        and coverage >= float(cfg["visibility"]["minimum_common_source_coverage"])
        and spatial >= float(cfg["visibility"]["minimum_spatial_coverage"])
    )
    categories = noise_model.category_for_uv(
        source_frame, patch["source_uv"], cfg["plateau_noise"]["edge_definition"])
    model_metrics = {}
    for model, evaluation in evaluations.items():
        residual = np.asarray(evaluation["arrays"]["world_residual_m"], float)
        selected = common & np.isfinite(residual)
        values = residual[selected]
        surprisal = noise_model.surprisal(values, categories[selected]) if len(values) else np.empty(0)
        contradiction = np.asarray(evaluation["arrays"]["contradiction"], bool)
        occluded = np.asarray(evaluation["arrays"]["occluded"], bool)
        model_metrics[model] = {
            "median_residual_m": float(np.median(values)) if len(values) else None,
            "p90_residual_m": float(np.percentile(values, 90)) if len(values) else None,
            "mean_noise_surprisal": float(np.mean(surprisal)) if len(surprisal) else None,
            "median_noise_surprisal": float(np.median(surprisal)) if len(surprisal) else None,
            "contradiction_fraction": float(np.mean(contradiction[common])) if common_count else 0.0,
            "occlusion_fraction": float(np.mean(occluded[common])) if common_count else 0.0,
            "common_sample_count": common_count,
        }
    static_nll = model_metrics["static"]["mean_noise_surprisal"]
    moving_nll = model_metrics["drawer"]["mean_noise_surprisal"]
    evidence = (float(static_nll - moving_nll)
                if static_nll is not None and moving_nll is not None else None)
    alignment = float(abs(source_frame["surface_normal_axis_alignment"]))
    discriminative = alignment * float(observation.abs_delta_q)
    noise_reference = float(noise_model.summary["combined"]["p90_m"])
    discriminative_snr = discriminative / max(noise_reference, 1e-12)
    long_observable = bool(visibility)
    accepted = bool(visibility and (observation.baseline != "long" or long_observable))
    return {
        **observation.to_dict(),
        "normal_axis_alignment_abs": alignment,
        "d_discriminative_m": discriminative,
        "discriminative_to_plateau_p90_ratio": discriminative_snr,
        "common_source_sample_count": common_count,
        "common_source_coverage": coverage,
        "common_source_spatial_coverage": spatial,
        "visibility": "common_source_anchor_observable" if visibility else "insufficient_common_source_visibility",
        "long_anchor_observable": long_observable if observation.baseline == "long" else None,
        "accepted_for_discrimination": accepted,
        "rejection_reason": None if accepted else "insufficient_common_source_visibility",
        "static": model_metrics["static"],
        "moving": model_metrics["drawer"],
        "log_evidence_moving_over_static": evidence,
        "target_proposals_used_for_identity": False,
        "source_anchor_immutable": True,
    }


def _slope(rows: list[dict], model: str) -> float | None:
    x = np.asarray([row["d_discriminative_m"] for row in rows], float)
    y = np.asarray([row[model]["median_residual_m"] for row in rows], float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) <= 1e-12:
        return None
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def discriminate_motion_model(
    observations: list[dict], boundary: dict, noise_model, cfg: dict,
    evaluation_mode: str = "active_motion", causal_static_positive: bool = False,
) -> dict:
    if evaluation_mode == "plateau_only":
        return {
            "label": "UNKNOWN", "formal_ownership_label": "UNKNOWN",
            "reason": "plateau_only_comparisons_cannot_create_ownership",
            "informative_observation_count": 0, "posterior_moving": 0.5,
            "frozen_noise_model_sha256": noise_model.summary["frozen_model_sha256"],
        }
    minimum_snr = float(cfg["minimum_discriminative_to_plateau_p90_ratio"])
    eligible = [row for row in observations
                if row["accepted_for_discrimination"]
                and row["log_evidence_moving_over_static"] is not None
                and row["discriminative_to_plateau_p90_ratio"] >= minimum_snr]
    evidence = np.asarray([row["log_evidence_moving_over_static"] for row in eligible], float)
    total = float(evidence.sum()) if len(evidence) else 0.0
    posterior = float(1.0 / (1.0 + math.exp(-np.clip(total, -700., 700.))))
    baseline_counts = Counter(row["baseline"] for row in eligible)
    required_count = int(cfg["minimum_informative_observations"])
    broad_baseline = any(row["baseline"] in ("medium", "long") for row in eligible)
    moving_surprisal = np.asarray([row["moving"]["mean_noise_surprisal"] for row in eligible], float)
    static_surprisal = np.asarray([row["static"]["mean_noise_surprisal"] for row in eligible], float)
    ceiling = float(noise_model.summary["observation_mean_surprisal_p95"])
    moving_compatible = bool(len(moving_surprisal) and np.median(moving_surprisal) <= ceiling)
    static_compatible = bool(len(static_surprisal) and np.median(static_surprisal) <= ceiling)
    moving_consistency = float(np.mean(evidence > 0)) if len(evidence) else 0.0
    static_consistency = float(np.mean(evidence < 0)) if len(evidence) else 0.0
    static_slope = _slope(eligible, "static")
    moving_slope = _slope(eligible, "moving")
    enough = len(eligible) >= required_count and broad_baseline
    odds = float(cfg["minimum_posterior_probability"])
    tangent_blocked = bool(boundary["tangent_motion"] and not boundary["has_trusted_axis_finite_edge"])
    moving_positive = bool(
        enough and posterior >= odds and moving_compatible
        and moving_consistency >= float(cfg["minimum_direction_consistency_fraction"])
        and static_slope is not None and static_slope > 0 and not tangent_blocked)
    static_positive = bool(
        enough and posterior <= 1.0 - odds and static_compatible
        and static_consistency >= float(cfg["minimum_direction_consistency_fraction"])
        and moving_slope is not None and moving_slope > 0)
    if moving_positive and causal_static_positive:
        label, reason = "CONFLICTING", "moving_likelihood_conflicts_with_independent_causal_static_event"
    elif moving_positive:
        label, reason = "MOVING_LINK", "frozen_noise_multi_baseline_likelihood_favors_drawer_model"
    elif static_positive or causal_static_positive:
        label = "WORLD_STATIC"
        reason = ("frozen_noise_multi_baseline_likelihood_favors_static_model"
                  if static_positive else "independent_causal_disocclusion_world_static")
    else:
        label = "UNKNOWN"
        if tangent_blocked:
            reason = "tangent_or_repeated_plane_without_trusted_finite_physical_edge"
        elif not enough:
            reason = "insufficient_noise_separated_multi_baseline_observations"
        elif moving_compatible and static_compatible:
            reason = "both_models_remain_compatible_with_frozen_plateau_noise"
        elif not moving_compatible and not static_compatible:
            reason = "both_models_incompatible_with_frozen_plateau_noise"
        else:
            reason = "likelihood_or_residual_trend_not_consistent_enough"
    return {
        "label": label,
        "formal_ownership_label": "UNKNOWN" if label == "CONFLICTING" else label,
        "reason": reason,
        "informative_observation_count": len(eligible),
        "informative_baseline_counts": dict(baseline_counts),
        "aggregate_log_evidence_moving_over_static": total,
        "posterior_moving": posterior,
        "moving_model_noise_compatible": moving_compatible,
        "static_model_noise_compatible": static_compatible,
        "moving_evidence_direction_consistency": moving_consistency,
        "static_evidence_direction_consistency": static_consistency,
        "static_residual_vs_discriminative_slope": static_slope,
        "moving_residual_vs_discriminative_slope": moving_slope,
        "tangent_boundary_blocked": tangent_blocked,
        "causal_static_positive": bool(causal_static_positive),
        "frozen_noise_model_sha256": noise_model.summary["frozen_model_sha256"],
        "fixed_residual_margin_used": False,
        "absolute_30mm_compatibility_used_for_final_decision": False,
    }
