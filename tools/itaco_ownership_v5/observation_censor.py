"""Frozen-noise censor for source-indexed v5 motion observations.

The censor never searches target surfaces.  It classifies only the two exact
source-indexed projections already evaluated by the multi-baseline engine.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .plateau_noise import CATEGORIES, FrozenPlateauNoiseModel


class ObservationState(str, Enum):
    WORLD_STATIC_EVIDENCE = "WORLD_STATIC_EVIDENCE"
    MOVING_LINK_EVIDENCE = "MOVING_LINK_EVIDENCE"
    AMBIGUOUS_COMPATIBLE = "AMBIGUOUS_COMPATIBLE"
    IDENTITY_LOST_CENSORED = "IDENTITY_LOST_CENSORED"


def _canonical_noise_hash(summary: dict) -> str:
    payload = {key: value for key, value in summary.items() if key != "frozen_model_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_frozen_noise_model(directory: Path, expected_hash: str) -> tuple[FrozenPlateauNoiseModel, dict]:
    """Load, validate, and reuse the previous plateau model without refitting."""
    summary_path = Path(directory) / "plateau_noise_model.json"
    residual_path = Path(directory) / "plateau_noise_residuals.npz"
    if not summary_path.is_file() or not residual_path.is_file():
        raise FileNotFoundError(json.dumps({
            "code": "missing_frozen_plateau_noise_artifact",
            "summary_path": str(summary_path), "residual_path": str(residual_path),
        }, indent=2))
    summary = json.loads(summary_path.read_text())
    recorded = str(summary.get("frozen_model_sha256", ""))
    recomputed = _canonical_noise_hash(summary)
    if recorded != recomputed or recorded != str(expected_hash):
        raise RuntimeError(json.dumps({
            "code": "frozen_plateau_noise_hash_mismatch",
            "expected": str(expected_hash), "recorded": recorded, "recomputed": recomputed,
        }, indent=2))
    archive = np.load(residual_path)
    if set(archive.files) != set(CATEGORIES):
        raise RuntimeError(f"frozen noise categories mismatch: {archive.files}")
    residuals = {}
    for name in CATEGORIES:
        values = np.asarray(archive[name], float)
        expected_count = int(summary["categories"][name]["sample_count"])
        if len(values) != expected_count or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
            raise RuntimeError(f"invalid frozen residual array for {name}")
        residuals[name] = values
    report = {
        "loaded_without_refit": True,
        "summary_path": str(summary_path), "residual_path": str(residual_path),
        "original_noise_model_hash": recorded,
        "hash_recomputed_and_verified": True,
        "compatibility_statistic": "mean_noise_surprisal",
        "compatibility_threshold_source": "frozen_summary.observation_mean_surprisal_p95",
        "compatibility_threshold_quantile": 0.95,
        "compatibility_threshold_value": float(summary["observation_mean_surprisal_p95"]),
        "compatible_relation": "mean_noise_surprisal <= frozen p95",
        "ood_relation": "mean_noise_surprisal > frozen p95",
    }
    return FrozenPlateauNoiseModel(residuals, summary), report


def censor_observation(row: dict, noise_model: FrozenPlateauNoiseModel) -> dict:
    """Assign one of four observation states using only frozen noise evidence."""
    result = {**row, "static": {**row["static"]}, "moving": {**row["moving"]}}
    threshold = float(noise_model.summary["observation_mean_surprisal_p95"])
    criterion = {
        "statistic": "mean_noise_surprisal",
        "frozen_quantile": 0.95,
        "threshold": threshold,
        "compatible_relation": "<=",
        "ood_relation": ">",
        "original_noise_model_hash": noise_model.summary["frozen_model_sha256"],
    }
    static_score = result["static"].get("mean_noise_surprisal")
    moving_score = result["moving"].get("mean_noise_surprisal")
    visibility_passed = bool(row.get("accepted_for_discrimination", False))
    scores_valid = (
        static_score is not None and moving_score is not None
        and np.isfinite(static_score) and np.isfinite(moving_score))
    if visibility_passed and scores_valid:
        static_ood = bool(float(static_score) > threshold)
        moving_ood = bool(float(moving_score) > threshold)
    else:
        static_ood = moving_ood = None
    for model, score, ood in (
        ("static", static_score, static_ood), ("moving", moving_score, moving_ood)):
        result[model]["mean_noise_log_likelihood"] = (
            -float(score) if score is not None and np.isfinite(score) else None)
        result[model]["noise_ood"] = ood
        result[model]["noise_band_compatible"] = (None if ood is None else not ood)
    raw_evidence = row.get("log_evidence_moving_over_static")
    if not visibility_passed:
        state = ObservationState.IDENTITY_LOST_CENSORED
        reason = "insufficient_common_source_visibility_before_noise_test"
    elif not scores_valid:
        state = ObservationState.IDENTITY_LOST_CENSORED
        reason = "missing_or_nonfinite_frozen_noise_likelihood"
    elif not static_ood and moving_ood:
        state = ObservationState.WORLD_STATIC_EVIDENCE
        reason = "static_within_frozen_noise_band_moving_ood"
    elif static_ood and not moving_ood:
        state = ObservationState.MOVING_LINK_EVIDENCE
        reason = "moving_within_frozen_noise_band_static_ood"
    elif not static_ood and not moving_ood:
        state = ObservationState.AMBIGUOUS_COMPATIBLE
        reason = "both_models_within_frozen_noise_band_neutral"
    else:
        state = ObservationState.IDENTITY_LOST_CENSORED
        reason = "both_models_ood_unrelated_valid_depth_or_identity_lost"
    evidence_state = state in (
        ObservationState.WORLD_STATIC_EVIDENCE, ObservationState.MOVING_LINK_EVIDENCE)
    censored = state == ObservationState.IDENTITY_LOST_CENSORED
    result.update({
        "observation_state": state.value,
        "observation_state_reason": reason,
        "censored": bool(censored),
        "censored_reason": reason if censored else None,
        "neutral_observation": state == ObservationState.AMBIGUOUS_COMPATIBLE,
        "eligible_for_aggregation": bool(evidence_state),
        "accepted_for_discrimination": bool(evidence_state),
        "raw_log_evidence_moving_over_static_diagnostic_only": raw_evidence,
        "aggregation_log_evidence_moving_over_static": (
            float(raw_evidence) if evidence_state and raw_evidence is not None else None),
        "log_evidence_moving_over_static": (
            float(raw_evidence) if evidence_state and raw_evidence is not None else None),
        "noise_ood_criterion": criterion,
        "coverage": {
            "common_source_sample_count": int(row.get("common_source_sample_count", 0)),
            "common_source_coverage": float(row.get("common_source_coverage", 0.0)),
            "common_source_spatial_coverage": float(row.get("common_source_spatial_coverage", 0.0)),
            "visibility": row.get("visibility"),
        },
        "both_ood_smaller_residual_selected": False,
        "target_proposals_used_for_identity": False,
        "source_anchor_immutable": True,
    })
    return result


def _slope(rows: list[dict], model: str) -> float | None:
    x = np.asarray([row["d_discriminative_m"] for row in rows], float)
    y = np.asarray([row[model]["median_residual_m"] for row in rows], float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) <= 1e-12:
        return None
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def discriminate_censored_motion_model(
    observations: list[dict], boundary: dict, noise_model: FrozenPlateauNoiseModel,
    cfg: dict, evaluation_mode: str = "active_motion",
    causal_static_positive: bool = False,
) -> dict:
    """Aggregate only one-model-compatible observations; censored rows vanish."""
    state_counts = Counter(row["observation_state"] for row in observations)
    baseline_state_counts = {
        baseline: dict(Counter(row["observation_state"] for row in observations
                               if row["baseline"] == baseline))
        for baseline in ("short", "medium", "long")
    }
    if evaluation_mode == "plateau_only":
        return {
            "label": "UNKNOWN", "formal_ownership_label": "UNKNOWN",
            "reason": "plateau_only_comparisons_cannot_create_ownership",
            "informative_observation_count": 0, "effective_chain_length": 0,
            "posterior_moving": 0.5, "observation_state_counts": dict(state_counts),
            "baseline_observation_state_counts": baseline_state_counts,
            "original_noise_model_hash": noise_model.summary["frozen_model_sha256"],
        }
    minimum_snr = float(cfg["minimum_discriminative_to_plateau_p90_ratio"])
    retained = [row for row in observations
                if row["eligible_for_aggregation"]
                and row["aggregation_log_evidence_moving_over_static"] is not None
                and row["discriminative_to_plateau_p90_ratio"] >= minimum_snr]
    evidence = np.asarray([
        row["aggregation_log_evidence_moving_over_static"] for row in retained], float)
    total = float(evidence.sum()) if len(evidence) else 0.0
    posterior = float(1.0 / (1.0 + math.exp(-np.clip(total, -700., 700.))))
    baseline_counts = Counter(row["baseline"] for row in retained)
    state_retained = Counter(row["observation_state"] for row in retained)
    required_count = int(cfg["minimum_informative_observations"])
    broad_baseline = any(row["baseline"] in ("medium", "long") for row in retained)
    moving_consistency = float(np.mean([
        row["observation_state"] == ObservationState.MOVING_LINK_EVIDENCE.value
        for row in retained])) if retained else 0.0
    static_consistency = float(np.mean([
        row["observation_state"] == ObservationState.WORLD_STATIC_EVIDENCE.value
        for row in retained])) if retained else 0.0
    static_slope = _slope(retained, "static")
    moving_slope = _slope(retained, "moving")
    enough = len(retained) >= required_count and broad_baseline
    probability = float(cfg["minimum_posterior_probability"])
    tangent_blocked = bool(
        boundary["tangent_motion"] and not boundary["has_trusted_axis_finite_edge"])
    moving_positive = bool(
        enough and posterior >= probability
        and moving_consistency >= float(cfg["minimum_direction_consistency_fraction"])
        and static_slope is not None and static_slope > 0 and not tangent_blocked)
    static_positive = bool(
        enough and posterior <= 1.0 - probability
        and static_consistency >= float(cfg["minimum_direction_consistency_fraction"])
        and moving_slope is not None and moving_slope > 0)
    if moving_positive and causal_static_positive:
        label, reason = "CONFLICTING", "moving_evidence_conflicts_with_independent_causal_static_event"
    elif moving_positive:
        label, reason = "MOVING_LINK", "retained_frozen_noise_observations_favor_drawer_model"
    elif static_positive or causal_static_positive:
        label = "WORLD_STATIC"
        reason = ("retained_frozen_noise_observations_favor_static_model"
                  if static_positive else "independent_causal_disocclusion_world_static")
    else:
        label = "UNKNOWN"
        if tangent_blocked:
            reason = "tangent_or_repeated_plane_without_trusted_finite_physical_edge"
        elif not enough:
            reason = "insufficient_retained_one_model_compatible_observations"
        else:
            reason = "retained_evidence_not_consistent_enough"
    censored = [row for row in observations if row["censored"]]
    ambiguous = [row for row in observations
                 if row["observation_state"] == ObservationState.AMBIGUOUS_COMPATIBLE.value]
    return {
        "label": label,
        "formal_ownership_label": "UNKNOWN" if label == "CONFLICTING" else label,
        "reason": reason,
        "informative_observation_count": len(retained),
        "effective_chain_length": len(retained),
        "retained_baseline_counts": dict(baseline_counts),
        "retained_state_counts": dict(state_retained),
        "censored_observation_count": len(censored),
        "ambiguous_neutral_observation_count": len(ambiguous),
        "observation_state_counts": dict(state_counts),
        "baseline_observation_state_counts": baseline_state_counts,
        "aggregate_log_evidence_moving_over_static": total,
        "posterior_moving": posterior,
        "moving_evidence_direction_consistency": moving_consistency,
        "static_evidence_direction_consistency": static_consistency,
        "static_residual_vs_discriminative_slope": static_slope,
        "moving_residual_vs_discriminative_slope": moving_slope,
        "tangent_boundary_blocked": tangent_blocked,
        "causal_static_positive": bool(causal_static_positive),
        "original_noise_model_hash": noise_model.summary["frozen_model_sha256"],
        "identity_lost_observations_excluded_from_likelihood": True,
        "identity_lost_observations_excluded_from_trend": True,
        "identity_lost_observations_excluded_from_chain_length": True,
        "both_ood_smaller_residual_selected": False,
        "target_proposals_used_for_identity": False,
        "absolute_residual_margin_used_for_ood": False,
    }
