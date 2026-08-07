"""Consecutive source-anchored transition chains and symmetric ownership routes."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .projected_source_anchor import AnchorState


OWNERSHIP_LABELS = ("MOVING_LINK", "WORLD_STATIC", "UNKNOWN", "CONFLICTING")


def _longest_consecutive(values: list[int]) -> list[int]:
    best: list[int] = []
    current: list[int] = []
    for value in sorted(set(values)):
        if current and value != current[-1] + 1:
            if len(current) > len(best):
                best = current
            current = []
        current.append(value)
    if len(current) > len(best):
        best = current
    return best


def summarize_transition_chain(
    rows: list[dict], source_frame: int, source_q: float, timestamps: dict[int, float], cfg: dict
) -> dict:
    """Summarize direct observations while keeping the source anchor immutable."""
    by_frame = {int(row["target_frame_id"]): row for row in rows}
    verified_targets = [frame for frame, row in by_frame.items()
                        if row["identity_state"] == AnchorState.VERIFIED_SOURCE_ANCHOR.value]
    # The source itself bridges backward and forward direct observations but is
    # not counted as a verified target observation.
    full_run = _longest_consecutive(verified_targets + [int(source_frame)])
    target_run = [frame for frame in full_run if frame != int(source_frame) and frame in verified_targets]
    q_values = [float(source_q)] + [float(by_frame[frame]["target_q_m"]) for frame in target_run]
    times = [float(timestamps[source_frame])] + [float(timestamps[frame]) for frame in target_run]
    q_span = float(max(q_values) - min(q_values)) if q_values else 0.0
    elapsed = float(max(times) - min(times)) if times else 0.0
    residuals = [float(by_frame[frame]["median_residual_m"]) for frame in target_run
                 if by_frame[frame]["median_residual_m"] is not None]
    p90s = [float(by_frame[frame]["p90_residual_m"]) for frame in target_run
            if by_frame[frame]["p90_residual_m"] is not None]
    gaps = []
    ordered = sorted(by_frame)
    for left, right in zip(ordered, ordered[1:]):
        if right != left + 1:
            gaps.append({"after_frame": left, "before_frame": right, "missing_frame_count": right - left - 1})
    contradiction_frames = [frame for frame, row in by_frame.items()
                            if row["identity_state"] == AnchorState.FREE_SPACE_CONTRADICTION.value]
    contradiction_run = _longest_consecutive(contradiction_frames)
    positive = (
        len(target_run) >= int(cfg["minimum_consecutive_verified_frames"])
        and q_span >= float(cfg["minimum_verified_q_span_m"])
        and elapsed >= float(cfg["minimum_verified_elapsed_s"])
        and residuals and float(np.median(residuals)) <= float(cfg["maximum_chain_median_residual_m"])
        and p90s and float(np.median(p90s)) <= float(cfg["maximum_chain_p90_residual_m"])
    )
    return {
        "positive_chain": bool(positive),
        "verified_target_count": int(len(verified_targets)),
        "longest_consecutive_verified_run": int(len(target_run)),
        "verified_run_start": int(min(full_run)) if full_run else None,
        "verified_run_end": int(max(full_run)) if full_run else None,
        "verified_target_frames": sorted(verified_targets),
        "verified_q_span_m": q_span,
        "verified_elapsed_time_s": elapsed,
        "median_of_frame_median_residual_m": float(np.median(residuals)) if residuals else None,
        "median_of_frame_p90_residual_m": float(np.median(p90s)) if p90s else None,
        "gaps": gaps,
        "contradiction_intervals": ([{"start_frame": int(min(contradiction_run)),
                                       "end_frame": int(max(contradiction_run)),
                                       "length": len(contradiction_run)}]
                                     if contradiction_run else []),
        "identity_state_counts": dict(Counter(row["identity_state"] for row in rows)),
    }


def moving_edge_chain_confidence(
    patch: dict, rows: list[dict], axis_world: np.ndarray, cfg: dict
) -> dict:
    points = np.asarray(patch["source_world"], float)
    if not len(points):
        return {"confidence": 0.0, "verified_edge_frames": 0, "edge_sample_count": 0}
    coordinate = points @ np.asarray(axis_world, float)
    low, high = np.percentile(coordinate, [20, 80])
    edge = (coordinate <= low) | (coordinate >= high)
    values = []
    frames = []
    for row in rows:
        if row["identity_state"] != AnchorState.VERIFIED_SOURCE_ANCHOR.value:
            continue
        supported = np.asarray(row["arrays"]["supported"], bool)
        values.append(float(np.mean(supported[edge])) if np.any(edge) else 0.0)
        frames.append(int(row["target_frame_id"]))
    confidence = float(np.median(values)) if values else 0.0
    return {
        "confidence": confidence,
        "verified_edge_frames": len(values),
        "edge_sample_count": int(edge.sum()),
        "target_frames": frames,
        "passed": bool(values and confidence >= float(cfg["minimum_moving_edge_support_fraction"])),
    }


def decide_motion_ownership(
    static_chain: dict,
    moving_chain: dict,
    boundary: dict,
    moving_edge: dict,
    causal_static_positive: bool = False,
) -> dict:
    """Apply symmetric routes with explicit both-compatible semantics."""
    static_positive = bool(static_chain["positive_chain"])
    moving_raw = bool(moving_chain["positive_chain"])
    tangent_boundary_ok = (not bool(boundary["tangent_motion"]) or
                           (bool(boundary["has_trusted_axis_finite_edge"]) and bool(moving_edge["passed"])))
    moving_positive = moving_raw and tangent_boundary_ok
    families = {
        "ARTICULATED_MOTION_TRANSITION": moving_positive,
        "WORLD_STATIC_MOTION_TRANSITION": static_positive,
        "CAUSAL_OCCLUSION_DISOCCLUSION": bool(causal_static_positive),
        "FREE_SPACE_CONTRADICTION": bool(
            static_chain["contradiction_intervals"] or moving_chain["contradiction_intervals"]),
        "TRUSTED_IDENTITY_PROPAGATION": False,
    }
    if moving_raw and static_positive:
        label = "UNKNOWN"
        reason = "motion_tangent_or_repeated_surface_ambiguity"
    elif moving_positive and (static_positive or causal_static_positive):
        label = "CONFLICTING"
        reason = "independent_moving_and_world_static_physical_families_conflict"
    elif moving_positive:
        label = "MOVING_LINK"
        reason = "articulated_motion_transition_verified_static_route_not_verified"
    elif static_positive or causal_static_positive:
        label = "WORLD_STATIC"
        reason = ("world_static_motion_transition_verified_moving_route_not_verified"
                  if static_positive else "causal_local_disocclusion_verified_world_static")
    else:
        label = "UNKNOWN"
        if moving_raw and not tangent_boundary_ok:
            reason = "tangent_surface_without_tracked_physical_finite_edge"
        elif not static_chain["verified_target_count"] and not moving_chain["verified_target_count"]:
            reason = "both_models_lost_or_unobservable"
        else:
            reason = "insufficient_consecutive_active_transition_evidence"
    return {
        "label": label,
        "reason": reason,
        "formal_ownership_label": ("UNKNOWN" if label == "CONFLICTING" else label),
        "raw_routes": {
            "moving_chain_before_tangent_boundary_gate": moving_raw,
            "moving_link_route": moving_positive,
            "world_static_motion_route": static_positive,
            "causal_world_static_route": bool(causal_static_positive),
        },
        "evidence_families": families,
        "tangent_boundary_gate_passed": bool(tangent_boundary_ok),
    }
