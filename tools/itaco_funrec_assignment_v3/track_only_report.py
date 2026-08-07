"""Structured safe-stop report for depth-projective track diagnostics."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def write_track_only_report(output_dir: Path, evidence: list[dict], diagnostics: dict, cfg: dict) -> dict:
    labels = Counter(item["label"] for item in evidence)
    reasons = Counter(item["reason"] for item in evidence)
    gate_cfg = cfg["region_propagation_gate"]
    baseline = int(gate_cfg["baseline_reliable_moving_tracks"])
    current = int(labels["moving"])
    required = max(baseline + int(gate_cfg["minimum_absolute_increase"]),
                   int(round(baseline * float(gate_cfg["minimum_increase_factor"]))))
    increased = current >= required
    gate = {
        "baseline_reliable_moving_tracks": baseline,
        "current_reliable_moving_tracks": current,
        "required_reliable_moving_tracks": required,
        "reliable_moving_tracks_clearly_increased": increased,
        "region_voting_ran": False,
        "region_propagation_ran": False,
        "sam2_v3_propagation_ran": False,
        "policy": "always stop this diagnostic before region voting/propagation; a later run requires explicit review",
    }
    (output_dir / "region_propagation_gate_report.json").write_text(json.dumps(gate, indent=2) + "\n")
    comparison = {
        "previous_formal_v3": {"moving": baseline},
        "depth_projective_association": {
            "tracks": len(evidence), "static": int(labels["static"]), "moving": current,
            "unknown": int(labels["unknown"]), "unknown_reasons": dict(reasons),
            "accepted_projective_attempts": diagnostics["accepted_attempts"],
            "rejected_projective_attempts": diagnostics["rejected_attempts"],
            "high_residual_track_count": diagnostics["high_residual_track_count"],
        },
        "region_assignment_comparison": None,
        "reason": "region voting and propagation intentionally did not run",
    }
    (output_dir / "track_association_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    acceptance = {
        "output_kind": "extended_non_official_funrec_inspired_depth_projective_track_diagnostic",
        "fixed_camera_pose": True, "fixed_axis": True, "fixed_q_t": True,
        "camera_optimization_ran": False, "axis_optimization_ran": False, "q_t_optimization_ran": False,
        "region_voting_ran": False, "region_propagation_ran": False, "sam2_v3_propagation_ran": False,
        "tsdf_ran": False, "nksr_ran": False, "mesh_ran": False,
        "reliable_moving_tracks_clearly_increased": increased,
        "ready_for_region_propagation": False,
        "ready_for_dual_tsdf": False,
        "stop_reason": "track diagnostics requested; propagation explicitly prohibited in this run",
    }
    (output_dir / "acceptance_report.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    report = f"""# Depth-aware projective association diagnostic

This run keeps HoloLens poses, the prismatic axis, and q_t fixed. It evaluates
LoFTR candidates against both static and drawer projective-depth predictions.

- Tracks: {len(evidence)}
- Static: {labels['static']}
- Moving: {labels['moving']}
- Unknown: {labels['unknown']}
- High-absolute-residual tracks: {diagnostics['high_residual_track_count']}
- Accepted projective attempts: {diagnostics['accepted_attempts']}
- Rejected projective attempts: {diagnostics['rejected_attempts']}
- Reliable moving baseline / required / current: {baseline} / {required} / {current}
- Clearly increased: {str(increased).lower()}
- Region voting/propagation: **not run**
- SAM2/TSDF/NKSR/Mesh: **not run**

Even if the numeric gate passes, this diagnostic never starts propagation; the
observation-error decomposition must be reviewed first.
"""
    (output_dir / "RUN_REPORT.md").write_text(report)
    return {"labels": dict(labels), "reasons": dict(reasons), "gate": gate}
