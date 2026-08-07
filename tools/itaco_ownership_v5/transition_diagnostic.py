"""Assignment-v5 transition-event audit with control-first hard gating."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from tools.itaco_region_assignment_v4.frame_data import (
    load_and_validate,
    parse_index,
    validate_registered_depth_scale,
)
from tools.itaco_region_assignment_v4.proposals import attach_interaction_proposals

from .causal_reveal import load_trusted_front_masks, measure_causal_reveal, serializable_causal
from .control_evaluation import evaluate_control_gate
from .local_transitions import (
    build_local_active_transitions,
    decompose_local_motion_states,
    local_targets_for_source,
)
from .physical_boundary import analyze_physical_boundary, serializable_boundary
from .projected_source_anchor import (
    evaluate_source_indexed_patch,
    extract_source_patch,
    posthoc_proposal_provenance,
    serializable_evaluation,
)
from .transition_chain import (
    decide_motion_ownership,
    moving_edge_chain_confidence,
    summarize_transition_chain,
)


DECISION_COLORS = {
    "MOVING_LINK": (45, 180, 45),
    "WORLD_STATIC": (220, 120, 20),
    "UNKNOWN": (128, 128, 128),
    "CONFLICTING": (180, 40, 180),
}
BOUNDARY_COLORS = {
    "rgb_footprint_truncation": (255, 80, 80),
    "depth_footprint_truncation": (180, 40, 255),
    "depth_discontinuity": (0, 220, 255),
    "normal_or_plane_discontinuity": (0, 220, 0),
    "invalid_depth_hole": (255, 0, 180),
    "segmentation_only": (150, 150, 150),
}


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _run_synthetic_tests(worktree: Path) -> dict:
    command = [sys.executable, "-m", "unittest", "discover", "-s",
               "tools/itaco_ownership_v5/tests", "-p", "test_*.py", "-v"]
    completed = subprocess.run(command, cwd=worktree, capture_output=True, text=True)
    return {
        "passed": completed.returncode == 0,
        "return_code": int(completed.returncode),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _find_proposal(frame: dict, proposal_id: str) -> dict:
    matches = [row for row in frame["proposals"] if row["proposal_id"] == proposal_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one proposal {proposal_id}, got {len(matches)}")
    return matches[0]


def _validate_annotations(annotations: dict, v4_output: Path, v4_evidence: list[dict],
                          failed_rows: list[dict]) -> dict:
    if annotations.get("evaluation_only") is not True:
        raise ValueError("control annotations must declare evaluation_only=true")
    controls = annotations["controls"]
    manifest = json.loads((v4_output / "sam2_seed_manifest.json").read_text())
    front_expected = {(int(row["original_frame_id"]), row["proposal_id"])
                      for row in manifest["seeds"] if row["seed_kind"] == "front"}
    front_actual = {(int(row["original_frame_id"]), row["proposal_id"])
                    for row in controls if row["control_group"] == "drawer_front"}
    if front_actual != front_expected or len(front_actual) != 4:
        raise RuntimeError("drawer-front controls do not exactly match the four v4 front seeds")
    evidence_by_key = {(int(row["original_frame_id"]), row["proposal_id"]): row for row in v4_evidence}
    box = [row for row in controls if row["control_group"] == "active_comoving_box"]
    box_checks = []
    for row in box:
        key = int(row["original_frame_id"]), row["proposal_id"]
        evidence = evidence_by_key[key]
        passed = evidence["label"] == "drawer" and bool(evidence["high_confidence_drawer"])
        box_checks.append({"key": key, "v4_label": evidence["label"],
                           "v4_high_confidence_drawer": bool(evidence["high_confidence_drawer"]),
                           "passed": passed})
    if len(box) != 5 or not all(row["passed"] for row in box_checks):
        raise RuntimeError("active co-moving box annotations lack the required v4 strong drawer evidence")
    failed_moving = {(int(row["original_frame_id"]), row["proposal_id"])
                     for row in failed_rows if row["v5_decision"] == "MOVING"}
    floor = [row for row in controls if row["control_group"] == "documented_floor_false_positive"]
    floor_keys = {(int(row["original_frame_id"]), row["proposal_id"]) for row in floor}
    layer6 = [row for row in floor if int(row["source_layer"]) == 6]
    layer0 = [row for row in floor if int(row["source_layer"]) == 0]
    if len(layer6) != 19 or len(layer0) != 2 or not floor_keys.issubset(failed_moving):
        raise RuntimeError("floor controls do not match the documented failed_001 subset")
    return {
        "passed": True,
        "evaluation_only": True,
        "classifier_feature_use": False,
        "drawer_front_manifest_match": True,
        "active_comoving_box_v4_evidence": box_checks,
        "documented_floor_layer6_count": len(layer6),
        "documented_floor_early_layer0_count": len(layer0),
    }


def _load_runtime(cfg: dict) -> dict:
    base_cfg = yaml.safe_load(Path(cfg["inputs"]["assignment_v4_config"]).read_text())
    scale_gate = validate_registered_depth_scale(base_cfg)
    if not scale_gate["passed"]:
        raise RuntimeError("registered-depth hard gate failed before real-data processing")
    if float(base_cfg["validity"]["depth_scale_to_m"]) != float(cfg["audit"]["registered_depth_scale_to_m"]):
        raise RuntimeError("registered-depth scale contract conflict")
    context = load_and_validate(base_cfg)
    frames = context["frames"]["interaction"]
    proposal_manifest = attach_interaction_proposals(frames, base_cfg)
    frames_by_id = {int(frame["source"]): frame for frame in frames}
    raw_root = Path(base_cfg["inputs"]["raw_root"])
    timestamp_rows = parse_index(raw_root / "pinhole_projection" / "depth.txt",
                                 raw_root / "pinhole_projection")
    ticks = float(cfg["motion_state"]["timestamp_ticks_per_second"])
    first = timestamp_rows[int(frames[0]["source"])]["association_timestamp"]
    timestamps = {
        int(frame["source"]): (timestamp_rows[int(frame["source"])]["association_timestamp"] - first) / ticks
        for frame in frames
    }
    states = decompose_local_motion_states(
        [int(frame["source"]) for frame in frames],
        [timestamps[int(frame["source"])] for frame in frames],
        [float(frame["q"]) for frame in frames], cfg["motion_state"])
    transitions, rejected = build_local_active_transitions(states, cfg["local_transition"])
    v4_output = Path(cfg["inputs"]["assignment_v4_output"])
    trusted_masks, trusted_report = load_trusted_front_masks(
        v4_output, frames, int(context["phases"]["interaction"][0]), 3)
    return {
        "base_cfg": base_cfg, "context": context, "frames": frames,
        "frames_by_id": frames_by_id, "timestamps": timestamps, "states": states,
        "transitions": transitions, "rejected_transitions": rejected,
        "scale_gate": scale_gate, "proposal_manifest": proposal_manifest,
        "trusted_masks": trusted_masks, "trusted_report": trusted_report,
    }


def _edge_and_causal(
    patch: dict, source_frame: dict, target_rows: list[dict], raw_rows: dict[str, list[dict]],
    boundary: dict, runtime: dict, cfg: dict
) -> tuple[dict, list[dict]]:
    moving_edge = moving_edge_chain_confidence(
        patch, raw_rows["drawer"], runtime["context"]["axis"], cfg["physical_boundary"])
    events = []
    for target_meta in target_rows:
        if not target_meta["causal_eligible"]:
            continue
        transition = target_meta["chronological_transition"]
        target_frame = runtime["frames_by_id"][int(target_meta["target_frame"])]
        future = [frame for frame in runtime["frames"]
                  if int(target_frame["source"]) < int(frame["source"]) <=
                  int(target_frame["source"]) + int(cfg["causal_reveal"]["maximum_future_frames"])]
        event = measure_causal_reveal(
            patch, source_frame, target_frame, future, runtime["trusted_masks"],
            runtime["context"]["axis"], runtime["context"]["intrinsic"], transition,
            cfg["causal_reveal"])
        events.append(event)
    return moving_edge, events


def analyze_region(frame: dict, proposal: dict, runtime: dict, cfg: dict,
                   evaluation_mode: str = "active_transition") -> dict:
    patch_cfg = {**cfg["source_patch"]}
    patch = extract_source_patch(frame, proposal["eroded_mask"],
                                 runtime["context"]["intrinsic"], patch_cfg)
    boundary = analyze_physical_boundary(
        frame, proposal["mask"], patch["source_world"], patch["source_uv"],
        runtime["context"]["axis"], runtime["context"]["intrinsic"], cfg["physical_boundary"])
    target_rows = ([] if evaluation_mode == "plateau_only" else
                   local_targets_for_source(int(frame["source"]), runtime["transitions"], True))
    raw_rows = {"static": [], "drawer": []}
    serial_rows = {"static": [], "drawer": []}
    for target_meta in target_rows:
        target = runtime["frames_by_id"][int(target_meta["target_frame"])]
        for model in ("static", "drawer"):
            evaluation = evaluate_source_indexed_patch(
                patch, target, runtime["context"]["axis"], runtime["context"]["intrinsic"],
                model, cfg["projective_anchor"], runtime["trusted_masks"][int(target["source"])] )
            evaluation["identity_observation_direction"] = target_meta["identity_observation_direction"]
            evaluation["causal_eligible"] = bool(target_meta["causal_eligible"])
            evaluation["chronological_transition"] = target_meta["chronological_transition"]
            # This call is intentionally post-decision and never feeds the state machine.
            evaluation["target_proposal_posthoc_provenance"] = posthoc_proposal_provenance(
                evaluation, target["proposals"])
            raw_rows[model].append(evaluation)
            serial_rows[model].append(serializable_evaluation(evaluation))
    chain = {
        model: summarize_transition_chain(
            raw_rows[model], int(frame["source"]), float(frame["q"]),
            runtime["timestamps"], cfg["transition_chain"])
        for model in ("static", "drawer")
    }
    moving_edge, causal_events = _edge_and_causal(
        patch, frame, target_rows, raw_rows, boundary, runtime, cfg)
    causal_positive = any(event["positive_causal_static_disocclusion"] for event in causal_events)
    decision = decide_motion_ownership(
        chain["static"], chain["drawer"], boundary, moving_edge, causal_positive)
    if evaluation_mode == "plateau_only":
        decision = {
            "label": "UNKNOWN", "formal_ownership_label": "UNKNOWN",
            "reason": "plateau_only_comparisons_cannot_create_ownership",
            "raw_routes": {"moving_chain_before_tangent_boundary_gate": False,
                           "moving_link_route": False, "world_static_motion_route": False,
                           "causal_world_static_route": False},
            "evidence_families": {}, "tangent_boundary_gate_passed": False,
        }
    return {
        "source_metadata": {
            "original_frame_id": int(frame["source"]), "q_m": float(frame["q"]),
            "proposal_id": proposal["proposal_id"], "source_layer": int(proposal["source_layer"]),
            "evaluation_mode": evaluation_mode,
        },
        "source_patch": {key: value for key, value in patch.items()
                         if key not in ("source_sample_indices", "source_uv", "source_depth_m", "source_world")},
        "decision": decision,
        "chains": chain,
        "physical_boundary": serializable_boundary(boundary),
        "moving_edge_chain": moving_edge,
        "causal_reveal_events": [serializable_causal(event) for event in causal_events],
        "per_target": serial_rows,
        "hard_invariants": {
            "assignment_v4_per_target_used": False,
            "target_proposals_used_for_identity": False,
            "target_proposals_posthoc_only": True,
            "directed_transition_keys": True,
            "source_anchor_immutable": True,
            "world_static_is_not_cabinet_membership": True,
            "dense_propagation_ran": False,
            "reconstruction_ran": False,
        },
        "_raw": {"patch": patch, "boundary": boundary, "rows": raw_rows,
                 "causal_events": causal_events, "target_rows": target_rows},
    }


def serializable_region(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_raw"}


def _header(image: np.ndarray, lines: list[str], height: int = 70) -> np.ndarray:
    header = np.zeros((height, image.shape[1], 3), np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(header, line, (5, 18 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .36, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def source_overlay(frame: dict, proposal: dict, result: dict, path: Path) -> None:
    image = frame["rgb"].copy(); mask = np.asarray(proposal["mask"], bool)
    label = result["decision"]["label"]; color = np.asarray(DECISION_COLORS[label], np.uint8)
    blended = cv2.addWeighted(image, .40, np.broadcast_to(color, image.shape).copy(), .60, 0)
    image[mask] = blended[mask]
    static = result["chains"]["static"]; moving = result["chains"]["drawer"]
    lines = [
        f"SOURCE frame={frame['source']} q={frame['q']:.4f} | {proposal['proposal_id']} | {label}",
        f"provenance={result['decision']['reason']}",
        f"static run={static['longest_consecutive_verified_run']} moving run={moving['longest_consecutive_verified_run']} | green=moving orange=world-static gray=unknown purple=conflict",
    ]
    cv2.imwrite(str(path), _header(image, lines, 88))


def transition_panel(frame: dict, result: dict, runtime: dict, path: Path) -> None:
    candidates = []
    raw = result["_raw"]["rows"]
    for index in range(min(len(raw["static"]), len(raw["drawer"]))):
        s, d = raw["static"][index], raw["drawer"][index]
        sm = s["median_residual_m"] if s["median_residual_m"] is not None else 9.0
        dm = d["median_residual_m"] if d["median_residual_m"] is not None else 9.0
        candidates.append((abs(sm - dm), index))
    if not candidates:
        image = np.zeros_like(frame["rgb"])
        cv2.putText(image, "no local ACTIVE transition target", (15, image.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.imwrite(str(path), _header(image, [
            f"SOURCE-ANCHOR frame={frame['source']} q={frame['q']:.4f}",
            f"decision={result['decision']['label']} provenance={result['decision']['reason']}",
        ])); return
    _, index = max(candidates)
    static = raw["static"][index]; moving = raw["drawer"][index]
    target = runtime["frames_by_id"][int(static["target_frame_id"])]
    left = frame["rgb"].copy(); right = target["rgb"].copy()
    legend = [("static", static, (255, 100, 20)), ("moving", moving, (30, 220, 30))]
    for _, evaluation, color in legend:
        arrays = evaluation["arrays"]; ids = np.flatnonzero(arrays["observable"])
        if len(ids) > 500: ids = ids[::max(1, len(ids) // 500)]
        uv = arrays["predicted_uv"][ids]
        residual = arrays["world_residual_m"][ids]
        for (u, v), value in zip(uv, residual):
            intensity = float(np.clip(value / .06, 0, 1)) if np.isfinite(value) else 1.0
            point_color = tuple(int((1 - intensity) * c + intensity * 255) for c in color)
            cv2.circle(right, (int(u), int(v)), 1, point_color, -1)
    canvas = np.hstack((left, right))
    transition = static["chronological_transition"]
    lines = [
        f"SOURCE-ANCHORED PROJECTIVE PATCH {frame['source']} -> obs {target['source']} | chronological={transition['source_frame']}->{transition['target_frame']}",
        f"q={frame['q']:.4f}->{target['q']:.4f} dq={target['q']-frame['q']:+.4f}m direction={static['identity_observation_direction']}",
        f"blue/orange=static predicted pixels green=drawer predicted pixels; brightness=residual | S={static['identity_state']} D={moving['identity_state']}",
    ]
    cv2.imwrite(str(path), _header(canvas, lines))


def boundary_panel(frame: dict, result: dict, path: Path) -> None:
    image = frame["rgb"].copy(); data = result["_raw"]["boundary"]["arrays"]
    uv = data["boundary_uv"]; category = data["category"]
    for (u, v), name in zip(uv, category):
        image[int(v), int(u)] = BOUNDARY_COLORS[str(name)]
    for (u, v) in uv[data["leading"]]: cv2.circle(image, (int(u), int(v)), 2, (0, 255, 255), -1)
    for (u, v) in uv[data["trailing"]]: cv2.circle(image, (int(u), int(v)), 2, (255, 255, 0), -1)
    boundary = result["physical_boundary"]
    lines = [
        f"PHYSICAL BOUNDARY frame={frame['source']} q={frame['q']:.4f} | leading=yellow trailing=cyan",
        "red=RGB clip purple=depth clip orange=depth edge green=normal/plane magenta=hole gray=segmentation-only",
        f"lead={boundary['leading_edge_physical_confidence']:.2f} trail={boundary['trailing_edge_physical_confidence']:.2f} tangent={boundary['tangent_motion']} decision={result['decision']['label']}",
    ]
    cv2.imwrite(str(path), _header(image, lines, 88))


def causal_panel(frame: dict, result: dict, runtime: dict, path: Path) -> None:
    events = result["_raw"]["causal_events"]
    if not events:
        image = np.zeros_like(frame["rgb"])
        cv2.putText(image, "no chronological forward local transition", (8, image.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.imwrite(str(path), _header(image, [f"CAUSAL REVEAL frame={frame['source']} q={frame['q']:.4f}",
                                                  "no event; plateau/backward-only evidence cannot create causality"])); return
    event = max(events, key=lambda row: row["reveal_band_sample_count"])
    target = runtime["frames_by_id"][event["target_frame_id"]]
    arrays = event["arrays"]
    panels = []
    for label, mask, color in (
        ("before trusted", arrays["before_trusted_mask"], (0, 140, 255)),
        ("pred after", arrays["predicted_after_mask"], (255, 100, 20)),
        ("actual after", arrays["actual_after_mask"], (0, 220, 0)),
    ):
        image = (frame["rgb"].copy() if label == "before trusted" else target["rgb"].copy())
        overlay = np.broadcast_to(np.asarray(color, np.uint8), image.shape).copy()
        image[mask] = cv2.addWeighted(image, .4, overlay, .6, 0)[mask]
        cv2.putText(image, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        if label == "actual after":
            reveal_ids = np.asarray(arrays["reveal_band_source_indices"], np.int64)
            supported_ids = np.asarray(arrays["newly_supported_source_indices"], np.int64)
            after_uv = np.asarray(arrays["after_uv"], np.int64)
            for source_index in reveal_ids:
                u, v = after_uv[source_index]
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv2.circle(image, (int(u), int(v)), 2, (0, 255, 255), -1)
            for source_index in supported_ids:
                u, v = after_uv[source_index]
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv2.circle(image, (int(u), int(v)), 2, (0, 255, 0), -1)
        panels.append(image)
    image = np.hstack(panels)
    lines = [
        f"CAUSAL REVEAL {event['source_frame_id']}->{event['target_frame_id']} q={frame['q']:.4f}->{target['q']:.4f} dq={event['delta_q_m']:+.4f}m",
        f"direction=forward elapsed={event['elapsed_s']:.3f}s decision provenance={event['reason']}",
        f"IoU={event['silhouette']['silhouette_iou']:.2f} reveal={event['reveal_band_sample_count']} new={event['newly_supported_sample_count']} persistent residual={event['persistent_world_residual_median']}",
        "yellow=reveal-band samples green=newly supported samples; residual is measured in JSON and line above",
    ]
    cv2.imwrite(str(path), _header(image, lines, 88))


def motion_timeline(states: list, path: Path) -> None:
    color = {"CLOSED_PLATEAU": "#377eb8", "ACTIVE_MOTION": "#e41a1c",
             "OPEN_PLATEAU": "#4daf4a", "INACTIVE_INTERMEDIATE": "#984ea3"}
    figure, axis = plt.subplots(figsize=(12, 4))
    for row in states:
        axis.scatter(row.frame_id, row.q, c=color[row.state.value], s=35)
    for name, value in color.items(): axis.scatter([], [], c=value, label=name)
    axis.set(title="Frozen q motion-state decomposition | raw frames 177-213",
             xlabel="original frame ID (chronological)", ylabel="q (m)")
    axis.legend(ncol=2); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def transition_timeline(states: list, transitions: list, path: Path) -> None:
    by_frame = {row.frame_id: row for row in states}
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot([row.frame_id for row in states], [row.q for row in states], "k.-", label="frozen q")
    for row in transitions:
        axis.plot([row.source_frame, row.target_frame], [row.source_q, row.target_q],
                  color="#ff7f00", alpha=.35, linewidth=1)
    axis.set(title=f"Directed local ACTIVE transitions ({len(transitions)}) | arrows are chronological source->target",
             xlabel="original frame ID", ylabel="q (m)")
    axis.text(.01, .97, "constraints: ACTIVE interval, 0.005<=|dq|<=0.060m, gap<=3, elapsed<=0.8s",
              transform=axis.transAxes, va="top")
    axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def contact_sheet(paths: list[Path], output: Path, columns: int, thumbnail_width: int, title: str) -> None:
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        image = np.zeros((180, 600, 3), np.uint8)
        cv2.putText(image, f"{title}: no cases", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
        cv2.imwrite(str(output), image); return
    resized = []
    for image in images:
        width = thumbnail_width; height = int(round(image.shape[0] * width / image.shape[1]))
        resized.append(cv2.resize(image, (width, height)))
    cell_h = max(image.shape[0] for image in resized); rows = (len(resized) + columns - 1) // columns
    sheet = np.zeros((rows * cell_h + 45, columns * thumbnail_width, 3), np.uint8)
    cv2.putText(sheet, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    for index, image in enumerate(resized):
        y = 45 + (index // columns) * cell_h; x = (index % columns) * thumbnail_width
        sheet[y:y+image.shape[0], x:x+image.shape[1]] = image
    cv2.imwrite(str(output), sheet)


def _write_region_visuals(directory: Path, frame: dict, proposal: dict, result: dict, runtime: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    source_overlay(frame, proposal, result, directory / "source_overlay.jpg")
    transition_panel(frame, result, runtime, directory / "source_anchor_transition.jpg")
    boundary_panel(frame, result, directory / "physical_boundary.jpg")
    causal_panel(frame, result, runtime, directory / "causal_reveal.jpg")


def _summary_row(result: dict, group: str | None = None) -> dict:
    meta = result["source_metadata"]; decision = result["decision"]
    return {
        "original_frame_id": meta["original_frame_id"], "proposal_id": meta["proposal_id"],
        "source_layer": meta["source_layer"], "control_group": group or "ambiguity_116",
        "decision": decision["label"], "formal_ownership_label": decision["formal_ownership_label"],
        "decision_reason": decision["reason"],
        "static_positive_chain": result["chains"]["static"]["positive_chain"],
        "moving_positive_chain": result["chains"]["drawer"]["positive_chain"],
        "static_longest_run": result["chains"]["static"]["longest_consecutive_verified_run"],
        "moving_longest_run": result["chains"]["drawer"]["longest_consecutive_verified_run"],
        "tangent_motion": result["physical_boundary"]["tangent_motion"],
        "physical_boundary_accepted": result["physical_boundary"]["has_trusted_axis_finite_edge"],
        "causal_reveal_positive_count": sum(event["positive_causal_static_disocclusion"]
                                            for event in result["causal_reveal_events"]),
    }


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("ready_for_dual_tsdf") is not False:
        raise RuntimeError("transition diagnostic must preserve ready_for_dual_tsdf=false")
    output = Path(cfg["output_dir"]); review = Path(cfg["review_bundle_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"non-overwrite output exists: {output}")
    if review.exists() and any(review.iterdir()):
        raise FileExistsError(f"non-overwrite review bundle exists: {review}")
    worktree = Path(__file__).resolve().parents[2]
    tests = _run_synthetic_tests(worktree)
    output.mkdir(parents=True); review.mkdir(parents=True)
    _json(output / "synthetic_test_report.json", tests)
    runtime = _load_runtime(cfg)
    _json(output / "scale_consistency_gate_report.json", runtime["scale_gate"])
    _json(output / "trusted_drawer_front_report.json", runtime["trusted_report"])
    _json(output / "motion_states.json", [row.to_dict() for row in runtime["states"]])
    _json(output / "local_active_transitions.json", [row.to_dict() for row in runtime["transitions"]])
    _json(output / "rejected_local_transition_candidates.json",
          [row.to_dict() for row in runtime["rejected_transitions"]])
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    visual = output / "visualization"; visual.mkdir()
    motion_timeline(runtime["states"], visual / "motion_state_timeline.png")
    transition_timeline(runtime["states"], runtime["transitions"], visual / "local_transition_timeline.png")

    v4_output = Path(cfg["inputs"]["assignment_v4_output"])
    v4_evidence = json.loads((v4_output / "region_evidence.json").read_text())
    failed_rows = json.loads(Path(cfg["inputs"]["failed_001_summary"]).read_text())
    annotations = json.loads(Path(cfg["inputs"]["control_annotations"]).read_text())
    annotation_report = _validate_annotations(annotations, v4_output, v4_evidence, failed_rows)
    _json(output / "control_annotation_validation.json", annotation_report)
    control_root = output / "controls"; control_root.mkdir()
    control_results = []; control_rows = []; group_images = defaultdict(list); boundary_rows = []
    for ordinal, annotation in enumerate(annotations["controls"]):
        frame = runtime["frames_by_id"][int(annotation["original_frame_id"])]
        proposal = _find_proposal(frame, annotation["proposal_id"])
        result = analyze_region(frame, proposal, runtime, cfg,
                                annotation.get("evaluation_mode", "active_transition"))
        result["evaluation_annotation"] = annotation
        group = annotation["control_group"]
        directory = control_root / group / f"{ordinal:03d}_{proposal['proposal_id']}"
        _write_region_visuals(directory, frame, proposal, result, runtime)
        _json(directory / "transition_region_diagnostic.json", serializable_region(result))
        group_images[group].append(directory / "source_overlay.jpg")
        row = _summary_row(result, group); row.update({
            "expected_label": annotation["expected_label"],
            "annotation_provenance": annotation["annotation_provenance"],
        })
        control_rows.append(row); control_results.append(serializable_region(result))
        boundary_rows.append({"experiment_set": group, **result["source_metadata"],
                              **result["physical_boundary"]})
        print(f"[control {ordinal + 1:02d}/{len(annotations['controls']):02d}] {proposal['proposal_id']} -> {result['decision']['label']}", flush=True)
    gate = evaluate_control_gate(control_rows, tests["passed"], cfg["controls"])
    _json(output / "control_region_results.json", control_results)
    _csv(output / "control_region_summary.csv", control_rows)
    _json(output / "control_evaluation_report.json", gate)
    contact_root = output / "control_contact_sheets"; contact_root.mkdir()
    for group in ("drawer_front", "active_comoving_box", "documented_floor_false_positive",
                  "plateau_only", "provisional_world_static"):
        contact_sheet(group_images[group], contact_root / f"{group}.jpg",
                      int(cfg["visualization"]["contact_sheet_columns"]),
                      int(cfg["visualization"]["thumbnail_width"]), group)

    full_rows = []; full_results = []; full_ran = False; full_images = defaultdict(list)
    if gate["passed"]:
        ambiguity = json.loads((Path(cfg["inputs"]["ambiguity_audit_output"]) / "ambiguity_regions.json").read_text())
        selected = [row for row in ambiguity if row["audit_category"] == "static_drawer_both_supported"]
        if len(selected) != int(cfg["expected_ambiguity_region_count"]):
            raise RuntimeError("116-region cardinality mismatch")
        evidence_by_key = {(int(row["original_frame_id"]), row["proposal_id"]): row for row in v4_evidence}
        regions_root = output / "regions_116"; regions_root.mkdir(); full_ran = True
        for ordinal, selection in enumerate(selected):
            key = int(selection["original_frame_id"]), selection["proposal_id"]
            if evidence_by_key[key]["label"] != "unknown":
                raise RuntimeError(f"formal v4 label changed for {key}")
            frame = runtime["frames_by_id"][key[0]]; proposal = _find_proposal(frame, key[1])
            result = analyze_region(frame, proposal, runtime, cfg)
            directory = regions_root / f"{ordinal:03d}_{proposal['proposal_id']}"; directory.mkdir()
            _json(directory / "transition_region_diagnostic.json", serializable_region(result))
            source_overlay(frame, proposal, result, directory / "source_overlay.jpg")
            decision = result["decision"]["label"]
            if len(full_images[decision]) < int(cfg["visualization"]["maximum_116_representatives_per_decision"]):
                transition_panel(frame, result, runtime, directory / "source_anchor_transition.jpg")
                boundary_panel(frame, result, directory / "physical_boundary.jpg")
                causal_panel(frame, result, runtime, directory / "causal_reveal.jpg")
                full_images[decision].append(directory / "source_overlay.jpg")
            row = _summary_row(result); full_rows.append(row); full_results.append(serializable_region(result))
            boundary_rows.append({"experiment_set": "ambiguity_116", **result["source_metadata"],
                                  **result["physical_boundary"]})
            print(f"[116 {ordinal + 1:03d}/{len(selected):03d}] {proposal['proposal_id']} -> {decision}", flush=True)
        _csv(output / "per_region_transition_summary.csv", full_rows)
        _json(output / "per_region_transition_summary.json", full_rows)
        counts = Counter(row["decision"] for row in full_rows)
        summary_116 = {
            "ran": True, "input_region_count": len(full_rows),
            "formal_v4_labels_modified": False,
            "decision_counts": {name: counts[name] for name in
                                ("MOVING_LINK", "WORLD_STATIC", "UNKNOWN", "CONFLICTING")},
            "resolved_count": int(counts["MOVING_LINK"] + counts["WORLD_STATIC"]),
            "unknown_count": int(counts["UNKNOWN"] + counts["CONFLICTING"]),
            "raw_moving_route_count": sum(row["moving_positive_chain"] for row in full_rows),
            "raw_world_static_route_count": sum(row["static_positive_chain"] for row in full_rows),
            "tangent_ambiguity_count": sum(row["decision_reason"] == "motion_tangent_or_repeated_surface_ambiguity" for row in full_rows),
            "physical_boundary_accepted_count": sum(row["physical_boundary_accepted"] for row in full_rows),
            "causal_reveal_event_count": sum(row["causal_reveal_positive_count"] for row in full_rows),
            "world_static_motion_event_count": counts["WORLD_STATIC"],
            "moving_transition_event_count": counts["MOVING_LINK"],
            "ready_for_dual_tsdf": False,
        }
        _json(output / "v5_transition_116_summary.json", summary_116)
        full_contact = output / "visualization" / "contact_sheets_116"; full_contact.mkdir()
        for decision in ("MOVING_LINK", "WORLD_STATIC", "UNKNOWN", "CONFLICTING"):
            contact_sheet(full_images[decision], full_contact / f"{decision.lower()}.jpg",
                          int(cfg["visualization"]["contact_sheet_columns"]),
                          int(cfg["visualization"]["thumbnail_width"]), f"116 {decision}")
    else:
        summary_116 = {"ran": False, "reason": "control_hard_gate_failed_stop_before_116",
                       "decision_counts": None, "ready_for_dual_tsdf": False}

    _json(output / "physical_boundary_report.json", boundary_rows)
    summary = {
        "stage": cfg["stage"], "control_gate_passed": gate["passed"],
        "control_metrics": gate["metrics"], "full_116_ran": full_ran,
        "full_116": summary_116,
        "local_transition_count": len(runtime["transitions"]),
        "rejected_local_transition_candidate_count": len(runtime["rejected_transitions"]),
        "assignment_v4_per_target_used": False,
        "target_proposals_used_for_identity": False,
        "world_static_route_implemented": True,
        "causal_fields_measured": True,
        "formal_v4_labels_modified": False,
        "sam2_ran": False, "dense_propagation_ran": False,
        "camera_axis_q_modified": False, "tsdf_ran": False,
        "nksr_ran": False, "mesh_ran": False, "ready_for_dual_tsdf": False,
    }
    _json(output / "transition_event_summary.json", summary)

    # Small tracked bundle; full per-region trees remain local.
    for name in ("control_evaluation_report.json", "control_region_summary.csv",
                 "transition_event_summary.json", "physical_boundary_report.json",
                 "motion_states.json", "local_active_transitions.json",
                 "scale_consistency_gate_report.json", "synthetic_test_report.json"):
        shutil.copy2(output / name, review / name)
    if full_ran:
        for name in ("v5_transition_116_summary.json", "per_region_transition_summary.csv",
                     "per_region_transition_summary.json"):
            shutil.copy2(output / name, review / name)
    shutil.copytree(contact_root, review / "control_contact_sheets")
    shutil.copy2(visual / "motion_state_timeline.png", review / "motion_state_timeline.png")
    shutil.copy2(visual / "local_transition_timeline.png", review / "local_transition_timeline.png")
    representative = review / "representatives"; representative.mkdir()
    for group in ("drawer_front", "active_comoving_box", "documented_floor_false_positive", "plateau_only"):
        paths = group_images[group][:1]
        if paths:
            source_dir = paths[0].parent
            shutil.copytree(source_dir, representative / group)
    print(json.dumps({"output": str(output), "review_bundle": str(review), **summary}, indent=2))
    return summary
