"""Local causal disocclusion primitive; large same-ray gaps are rejected."""
from __future__ import annotations


def detect_causal_static_disocclusion(event: dict, cfg: dict) -> dict:
    checks = {
        "active_transition": bool(event.get("active_transition", False)),
        "verified_moving_occluder": bool(event.get("verified_moving_occluder", False)),
        "occluder_motion_matches_articulation": bool(event.get("occluder_motion_matches_articulation", False)),
        "local_depth_gap": abs(float(event.get("depth_gap_m", float("inf")))) <= float(cfg["maximum_local_depth_gap_m"]),
        "near_silhouette_boundary": float(event.get("boundary_distance_px", float("inf"))) <= float(cfg["maximum_boundary_distance_px"]),
        "temporally_adjacent_reveal": int(event.get("reveal_delay_frames", 10**9)) <= int(cfg["maximum_reveal_delay_frames"]),
        "silhouette_leaves_location": bool(event.get("silhouette_leaves_location", False)),
        "new_surface_appears": bool(event.get("new_surface_appears", False)),
        "world_static_persistence": (int(event.get("persistent_frames", 0)) >= int(cfg["minimum_persistent_frames"]) and
                                     float(event.get("world_residual_m", float("inf"))) <= float(cfg["maximum_world_residual_m"])),
    }
    accepted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "positive_static_evidence": accepted,
        "classification": "CAUSAL_STATIC_DISOCCLUSION" if accepted else "NO_IDENTITY_EVIDENCE",
        "checks": checks,
        "failed_checks": failed,
        "reason": "local_causal_reveal_persistent_world_surface" if accepted else "causal_chain_incomplete",
    }
