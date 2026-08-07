"""Finite-surface identity anchored to the source; no target-to-target hopping."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _finite_metrics(predicted: np.ndarray, observed: np.ndarray, threshold: float) -> dict:
    predicted = np.asarray(predicted, float).reshape(-1, 3)
    observed = np.asarray(observed, float).reshape(-1, 3)
    if len(predicted) == 0 or len(observed) == 0:
        return {"predicted_overlap": 0.0, "observed_overlap": 0.0,
                "symmetric_overlap": 0.0, "median_distance": None}
    p_dist = cKDTree(observed).query(predicted)[0]
    o_dist = cKDTree(predicted).query(observed)[0]
    p_overlap = float(np.mean(p_dist <= threshold))
    o_overlap = float(np.mean(o_dist <= threshold))
    return {"predicted_overlap": p_overlap, "observed_overlap": o_overlap,
            "symmetric_overlap": float(np.sqrt(p_overlap * o_overlap)),
            "median_distance": float(np.median(p_dist))}


def verify_anchored_surface_sequence(source_points: np.ndarray, source_q: float,
                                     observations: list[dict], predictor, cfg: dict) -> dict:
    """Verify every observation against the immutable source finite support.

    ``predictor(source_points, source_q, target_q)`` supplies either static or
    articulated prediction. Observed surfaces never become the next anchor.
    Nearby alternatives that fail source-anchored overlap are recorded as lost.
    """
    source = np.asarray(source_points, float).reshape(-1, 3).copy()
    minimum_overlap = float(cfg["minimum_symmetric_overlap"])
    threshold = float(cfg["point_distance_threshold"])
    rows = []
    for observation in observations:
        predicted = predictor(source, float(source_q), float(observation["q"]))
        metrics = _finite_metrics(predicted, observation.get("points", np.empty((0, 3))), threshold)
        verified = metrics["symmetric_overlap"] >= minimum_overlap
        rows.append({
            "frame_id": int(observation["frame_id"]), "q": float(observation["q"]),
            "active_transition": bool(observation.get("active_transition", False)),
            "identity_state": "VERIFIED_SOURCE_ANCHOR" if verified else "IDENTITY_LOST",
            "ownership_eligible": bool(verified and observation.get("active_transition", False)),
            "reason": ("finite_support_matches_immutable_source_anchor" if verified else
                       "source_finite_surface_not_verified_no_substitution"),
            **metrics,
        })
    return {
        "anchor_policy": "immutable_source_finite_surface",
        "target_observations_never_become_anchors": True,
        "rows": rows,
        "verified_active_observations": int(sum(row["ownership_eligible"] for row in rows)),
    }
