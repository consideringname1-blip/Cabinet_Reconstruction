"""Read-only v5 physical-event diagnostic over the 116 v4 unknown regions."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import yaml

from tools.itaco_region_assignment_v4.frame_data import load_and_validate, parse_index, validate_registered_depth_scale
from tools.itaco_region_assignment_v4.projective_models import CONTRADICTION, OCCLUDED, SUPPORTED, evaluate_model, unproject_pixels
from tools.itaco_region_assignment_v4.proposals import attach_interaction_proposals
from tools.itaco_region_assignment_v4.region_evidence import deterministic_sample
from tools.itaco_region_assignment_v4.surface_identity_visibility import (
    build_trusted_occluders, canonicalize_normal, contradiction_run,
    proposal_memberships, select_controls, spatial_coherence,
)

from .evidence import EvidenceBook, EvidenceFamily
from .motion import build_active_transitions, decompose_motion_states
from .occlusion import detect_causal_static_disocclusion
from .observability import source_finite_surface_observability


DECISION_COLORS = {"MOVING": (40, 180, 40), "STATIC": (220, 120, 20),
                   "UNKNOWN": (120, 120, 120), "CONFLICTING": (180, 40, 180)}


def finite_metrics(predicted: np.ndarray, observed: np.ndarray, threshold: float) -> dict:
    predicted = np.asarray(predicted, float).reshape(-1, 3)
    observed = np.asarray(observed, float).reshape(-1, 3)
    if len(predicted) < 3 or len(observed) < 3:
        return {"predicted_overlap": 0.0, "observed_overlap": 0.0,
                "symmetric_overlap": 0.0, "median_distance_m": None,
                "normal_angle_degrees": None, "plane_offset_m": None,
                "centroid_distance_m": None}
    p_dist = cKDTree(observed).query(predicted)[0]
    o_dist = cKDTree(predicted).query(observed)[0]
    p_overlap = float(np.mean(p_dist <= threshold)); o_overlap = float(np.mean(o_dist <= threshold))
    p_center = np.median(predicted, axis=0); o_center = np.median(observed, axis=0)

    def normal(points: np.ndarray) -> np.ndarray:
        centered = points - np.median(points, axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / max(len(points) - 1, 1))
        return canonicalize_normal(vectors[:, int(np.argmin(values))])

    p_normal = normal(predicted); o_normal = normal(observed)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(p_normal, o_normal)), 0.0, 1.0))))
    return {"predicted_overlap": p_overlap, "observed_overlap": o_overlap,
            "symmetric_overlap": float(np.sqrt(p_overlap * o_overlap)),
            "median_distance_m": float(np.median(p_dist)), "normal_angle_degrees": angle,
            "plane_offset_m": float(abs(np.dot(o_center - p_center, p_normal))),
            "centroid_distance_m": float(np.linalg.norm(o_center - p_center))}


def anchor_compatible(metrics: dict, cfg: dict) -> tuple[bool, list[str]]:
    checks = {
        "predicted_overlap": metrics["predicted_overlap"] >= float(cfg["minimum_predicted_overlap"]),
        "observed_overlap": metrics["observed_overlap"] >= float(cfg["minimum_observed_overlap"]),
        "symmetric_overlap": metrics["symmetric_overlap"] >= float(cfg["minimum_symmetric_overlap"]),
        "normal": metrics["normal_angle_degrees"] is not None and metrics["normal_angle_degrees"] <= float(cfg["maximum_normal_angle_degrees"]),
        "plane_offset": metrics["plane_offset_m"] is not None and metrics["plane_offset_m"] <= float(cfg["maximum_plane_offset_m"]),
        "centroid": metrics["centroid_distance_m"] is not None and metrics["centroid_distance_m"] <= float(cfg["maximum_centroid_distance_m"]),
    }
    return all(checks.values()), [name for name, passed in checks.items() if not passed]


def proposal_points(frame: dict, proposal: dict, context: dict, cfg: dict, model: str) -> np.ndarray:
    mask = np.asarray(proposal["eroded_mask"], bool) & frame["valid"]
    uv = deterministic_sample(mask, int(cfg["maximum_points_per_surface"]))
    if len(uv) < int(cfg["minimum_valid_points"]):
        return np.empty((0, 3))
    world = unproject_pixels(uv, frame["depth"][uv[:, 1], uv[:, 0]], frame["pose"], context["intrinsic"])
    return world if model == "static" else world - float(frame["q"]) * context["axis"]


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    return distance_transform_edt(mask)


def _candidate_rows(evaluation: dict, target: dict, source_model_points: np.ndarray,
                    context: dict, cfg: dict, model: str, cache: dict) -> list[dict]:
    memberships = proposal_memberships(evaluation["uv"], evaluation["status"] == SUPPORTED, target["proposals"])
    rows = []
    for membership in memberships["all_memberships"]:
        fraction = float(membership["hit_count"] / max(memberships["unique_supported_point_count"], 1))
        if membership["hit_count"] < int(cfg["minimum_supported_membership_points"]):
            continue
        if fraction < float(cfg["minimum_supported_membership_fraction"]):
            continue
        proposal = next(item for item in target["proposals"] if item["proposal_id"] == membership["proposal_id"])
        key = (int(target["source"]), proposal["proposal_id"], model)
        if key not in cache:
            cache[key] = proposal_points(target, proposal, context, cfg, model)
        metrics = finite_metrics(source_model_points, cache[key], float(cfg["point_distance_threshold_m"]))
        accepted, failures = anchor_compatible(metrics, cfg)
        rows.append({"proposal_id": proposal["proposal_id"], "source_layer": int(proposal["source_layer"]),
                     "hit_count": int(membership["hit_count"]), "supported_membership_fraction": fraction,
                     "anchor_compatible": bool(accepted), "failed_anchor_checks": failures, **metrics})
    rows.sort(key=lambda row: (-row["symmetric_overlap"], -row["predicted_overlap"], row["proposal_id"]))
    return rows


def _identity_state(candidates: list[dict], cfg: dict) -> tuple[str, dict | None, str]:
    compatible = [row for row in candidates if row["anchor_compatible"]]
    if not compatible:
        return "IDENTITY_LOST", None, "no_target_surface_matches_immutable_source_anchor"
    if bool(cfg["reject_multiple_anchor_compatible_candidates"]) and len(compatible) > 1:
        return "IDENTITY_AMBIGUOUS", None, "multiple_target_surfaces_match_source_anchor_no_substitution"
    return "VERIFIED_SOURCE_ANCHOR", compatible[0], "unique_finite_surface_matches_immutable_source_anchor"


def _active_pair_lookup(states: list, transitions: list) -> tuple[dict[int, object], set[frozenset[int]]]:
    by_frame = {row.frame_id: row for row in states}
    pairs = {frozenset((row.source_frame_id, row.target_frame_id)) for row in transitions}
    return by_frame, pairs


def _model_rows(item: dict, frame: dict, proposal: dict, frames_by_source: dict,
                context: dict, cfg: dict, active_pairs: set, cache: dict, model: str) -> tuple[list[dict], np.ndarray]:
    mask = np.asarray(proposal["eroded_mask"], bool) & frame["valid"]
    uv = deterministic_sample(mask, int(cfg["maximum_points_per_surface"]))
    world = unproject_pixels(uv, frame["depth"][uv[:, 1], uv[:, 0]], frame["pose"], context["intrinsic"])
    source_model = world if model == "static" else world - float(frame["q"]) * context["axis"]
    rows = []
    for target_id in [int(row["target_frame_id"]) for row in item["per_target"]]:
        target = frames_by_source[target_id]
        evaluation = evaluate_model(world, frame["q"], target, target["q"], context["axis"],
                                    context["intrinsic"], model, cfg["projective_evidence"])
        candidates = _candidate_rows(evaluation, target, source_model, context, cfg, model, cache)
        state, selected, reason = _identity_state(candidates, cfg)
        contradiction_fraction = float(np.mean(evaluation["status"] == CONTRADICTION))
        coherence = spatial_coherence(evaluation["uv"], evaluation["status"] == CONTRADICTION,
                                      target["depth"].shape, int(cfg["contradiction"]["spatial_dilation_pixels"]))
        rows.append({"target_frame_id": target_id, "target_q_m": float(target["q"]),
                     "active_transition": frozenset((int(frame["source"]), target_id)) in active_pairs,
                     "identity_state": state, "identity_reason": reason,
                     "selected_anchor_surface": selected, "candidate_count": len(candidates),
                     "anchor_compatible_candidate_count": sum(row["anchor_compatible"] for row in candidates),
                     "candidates": candidates, "support_ratio": float(evaluation["support_ratio"]),
                     "contradiction_fraction": contradiction_fraction,
                     "contradiction_spatial_coherence_fraction": coherence,
                     "status": evaluation["status"], "uv": evaluation["uv"],
                     "predicted_depth_m": evaluation["predicted_depth_m"],
                     "observed_depth_m": evaluation["observed_depth_m"]})
    return rows, world


def _contradiction(rows: list[dict], cfg: dict) -> dict:
    serial = [{key: row[key] for key in ("target_frame_id", "target_q_m", "contradiction_fraction",
                                          "contradiction_spatial_coherence_fraction")} for row in rows]
    return contradiction_run(serial, cfg)


def _verified_active_summary(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["active_transition"] and row["identity_state"] == "VERIFIED_SOURCE_ANCHOR"]
    q = [row["target_q_m"] for row in accepted]
    return {"count": len(accepted), "fraction": float(len(accepted) / max(sum(row["active_transition"] for row in rows), 1)),
            "q_span_m": float(max(q) - min(q)) if len(q) > 1 else 0.0,
            "target_frame_ids": [row["target_frame_id"] for row in accepted]}


def causal_disocclusion_events(static_rows: list[dict], frame: dict, frames_by_source: dict,
                               occluders: dict, cfg: dict, active_pairs: set) -> list[dict]:
    rows = sorted(static_rows, key=lambda row: row["target_frame_id"])
    events = []
    minimum_points = int(cfg["minimum_revealed_point_count"])
    minimum_fraction = float(cfg["minimum_revealed_point_fraction"])
    for index, before in enumerate(rows):
        target_before = frames_by_source[before["target_frame_id"]]
        status = np.asarray(before["status"]); uv = np.asarray(before["uv"], np.int64)
        mask = occluders["drawer"][before["target_frame_id"]]
        boundary = _boundary_distance(mask); h, w = mask.shape
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        ids = np.flatnonzero(inside & (status == OCCLUDED))
        if not len(ids):
            continue
        u, v = uv[ids, 0], uv[ids, 1]
        gap = before["predicted_depth_m"][ids] - before["observed_depth_m"][ids]
        local = (mask[v, u] & (gap > 0) & (gap <= float(cfg["maximum_local_depth_gap_m"])) &
                 (boundary[v, u] <= float(cfg["maximum_boundary_distance_px"])))
        local_ids = ids[local]
        if len(local_ids) < minimum_points or len(local_ids) / max(len(status), 1) < minimum_fraction:
            continue
        for later_index in range(index + 1, min(len(rows), index + int(cfg["maximum_reveal_delay_frames"]) + 2)):
            after = rows[later_index]
            if frozenset((before["target_frame_id"], after["target_frame_id"])) not in active_pairs:
                continue
            after_status = np.asarray(after["status"]); after_uv = np.asarray(after["uv"], np.int64)
            after_mask = occluders["drawer"][after["target_frame_id"]]
            au, av = after_uv[local_ids, 0], after_uv[local_ids, 1]
            valid_after = ((au >= 0) & (au < after_mask.shape[1]) & (av >= 0) & (av < after_mask.shape[0]))
            revealed = valid_after & (after_status[local_ids] == SUPPORTED)
            if np.any(valid_after):
                safe = np.flatnonzero(valid_after)
                revealed[safe] &= ~after_mask[av[safe], au[safe]]
            count = int(revealed.sum())
            if count < minimum_points or count / max(len(local_ids), 1) < minimum_fraction:
                continue
            persistent = 0
            for future in rows[later_index:]:
                if future["identity_state"] == "VERIFIED_SOURCE_ANCHOR" and np.mean(np.asarray(future["status"])[local_ids] == SUPPORTED) >= minimum_fraction:
                    persistent += 1
            payload = {"active_transition": True, "verified_moving_occluder": True,
                       "occluder_motion_matches_articulation": True,
                       "depth_gap_m": float(np.median(gap[local])),
                       "boundary_distance_px": float(np.median(boundary[v[local], u[local]])),
                       "reveal_delay_frames": later_index - index,
                       "silhouette_leaves_location": True, "new_surface_appears": True,
                       "persistent_frames": persistent,
                       "world_residual_m": (0.0 if after["identity_state"] == "VERIFIED_SOURCE_ANCHOR" else float("inf"))}
            result = detect_causal_static_disocclusion(payload, cfg)
            events.append({"occluded_frame_id": before["target_frame_id"],
                           "revealed_frame_id": after["target_frame_id"],
                           "local_occluded_point_count": len(local_ids), "revealed_point_count": count,
                           **payload, **result})
            break
    return events


def _serial_row(row: dict) -> dict:
    return {key: value for key, value in row.items()
            if key not in ("status", "uv", "predicted_depth_m", "observed_depth_m")}


def analyze_region(item: dict, frame: dict, proposal: dict, frames_by_source: dict,
                   context: dict, cfg: dict, active_pairs: set, occluders: dict,
                   cache: dict) -> dict:
    rows = {}; summaries = {}; contradictions = {}; source_world = None
    for model in ("static", "drawer"):
        rows[model], model_world = _model_rows(item, frame, proposal, frames_by_source, context,
                                     cfg["finite_identity_runtime"], active_pairs, cache, model)
        summaries[model] = _verified_active_summary(rows[model])
        contradictions[model] = _contradiction(rows[model], cfg["contradiction"])
        if model == "static":
            source_world = model_world
    observability = source_finite_surface_observability(frame, proposal, source_world, context["axis"], cfg["finite_boundary_observability"])
    events = causal_disocclusion_events(rows["static"], frame, frames_by_source, occluders,
                                        cfg["causal_disocclusion"], active_pairs)
    static_positive = any(event["positive_static_evidence"] for event in events)
    moving_cfg = cfg["motion_evidence"]
    drawer = summaries["drawer"]; static = summaries["static"]
    moving_positive = (drawer["count"] >= int(moving_cfg["minimum_verified_active_observations"]) and
                       drawer["q_span_m"] >= float(moving_cfg["minimum_verified_q_span_m"]) and
                       drawer["fraction"] - static["fraction"] >= float(moving_cfg["minimum_verified_fraction_advantage"]) and
                       not observability["tangent_motion_ambiguity"])
    book = EvidenceBook()
    if moving_positive:
        book.add(EvidenceFamily.ARTICULATED_MOTION_TRANSITION, "moving",
                 "unique_source_anchored_finite_surface_follows_frozen_Tq", summaries, True)
    if static_positive:
        book.add(EvidenceFamily.CAUSAL_OCCLUSION_DISOCCLUSION, "static",
                 "local_verified_drawer_reveal_then_world_persistence",
                 {"accepted_event_count": sum(event["positive_static_evidence"] for event in events)}, True)
    decision = book.decide()
    vetoes = []
    if decision["decision"] == "MOVING" and contradictions["drawer"]["coherent_contradiction_run"]:
        vetoes.append("moving_hypothesis_has_coherent_free_space_contradiction")
    if decision["decision"] == "STATIC" and contradictions["static"]["coherent_contradiction_run"]:
        vetoes.append("static_hypothesis_has_coherent_free_space_contradiction")
    if vetoes:
        decision = {**decision, "decision": "UNKNOWN", "reason": "positive_route_vetoed_by_free_space_contradiction"}
    return {"source_metadata": {"original_frame_id": int(frame["source"]), "q_m": float(frame["q"]),
                                 "proposal_id": item["proposal_id"], "source_layer": int(proposal["source_layer"]),
                                 "formal_v4_label": item["label"]},
            "decision": decision, "vetoes": vetoes, "verified_active": summaries,
            "source_finite_surface_observability": observability,
            "moving_route_blocked_reasons": ([observability["reason"]] if observability["tangent_motion_ambiguity"] else []),
            "contradiction_runs": {model: {key: value for key, value in value.items() if key != "flags"}
                                   for model, value in contradictions.items()},
            "causal_disocclusion_events": events,
            "per_target": {model: [_serial_row(row) for row in model_rows] for model, model_rows in rows.items()},
            "formal_v4_modified": False, "dense_propagation_ran": False, "reconstruction_ran": False}


def source_overlay(frame: dict, proposal: dict, result: dict, path: Path) -> None:
    image = frame["rgb"].copy(); mask = np.asarray(proposal["mask"], bool)
    color = DECISION_COLORS[result["decision"]["decision"]]
    overlay = np.full_like(image, color); image[mask] = cv2.addWeighted(image, .35, overlay, .65, 0)[mask]
    header = np.zeros((62, image.shape[1], 3), np.uint8)
    meta = result["source_metadata"]
    cv2.putText(header, f"frame={meta['original_frame_id']} {meta['proposal_id']} | v5={result['decision']['decision']} v4=unknown",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    text = f"active anchor verified S={result['verified_active']['static']['count']} D={result['verified_active']['drawer']['count']} causal reveals={sum(e['positive_static_evidence'] for e in result['causal_disocclusion_events'])} tangent-amb={result['source_finite_surface_observability']['tangent_motion_ambiguity']}"
    cv2.putText(header, text, (5, 44), cv2.FONT_HERSHEY_SIMPLEX, .32, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack((header, image)))


def timeline(result: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    mapping = {"IDENTITY_LOST": 0, "IDENTITY_AMBIGUOUS": 1, "VERIFIED_SOURCE_ANCHOR": 2}
    colors = {0: "#bdbdbd", 1: "#ff9f1c", 2: "#2ca02c"}
    for axis, model in zip(axes, ("static", "drawer")):
        rows = result["per_target"][model]
        for row in rows:
            value = mapping[row["identity_state"]]
            axis.scatter(row["target_q_m"], value, color=colors[value], marker="s", s=35,
                         edgecolors="black" if row["active_transition"] else "none", linewidths=.6)
        axis.set_yticks([0, 1, 2], ["lost", "ambiguous", "verified"]); axis.set_ylabel(model.upper())
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("frozen q (m); black edge = eligible ACTIVE_MOTION transition")
    meta = result["source_metadata"]
    figure.suptitle(f"v5 immutable-source identity | frame {meta['original_frame_id']} {meta['proposal_id']} | {result['decision']['decision']}")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def contact_sheet(paths: list[Path], output: Path, columns: int, width: int, title: str) -> None:
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None: continue
        images.append(cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1]))))
    if not images:
        canvas = np.zeros((120, width * columns, 3), np.uint8)
        cv2.putText(canvas, title + " | no regions", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 1)
        cv2.imwrite(str(output), canvas); return
    height = max(image.shape[0] for image in images); padded = []
    for image in images:
        canvas = np.zeros((height, width, 3), np.uint8); canvas[:image.shape[0]] = image; padded.append(canvas)
    rows = []
    for start in range(0, len(padded), columns):
        row = padded[start:start + columns]
        row += [np.zeros((height, width, 3), np.uint8)] * (columns - len(row)); rows.append(np.hstack(row))
    body = np.vstack(rows); header = np.zeros((50, body.shape[1], 3), np.uint8)
    cv2.putText(header, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 1)
    cv2.imwrite(str(output), np.vstack((header, body)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text()); output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"non-overwrite output exists: {output}")
    primitives = yaml.safe_load(Path(cfg["inputs"]["primitives_config"]).read_text())
    if primitives.get("ready_for_dual_tsdf") is not False or cfg.get("ready_for_dual_tsdf") is not False:
        raise RuntimeError("v5 diagnostic must keep ready_for_dual_tsdf=false")
    base_cfg = yaml.safe_load(Path(cfg["inputs"]["assignment_v4_config"]).read_text())
    audit_cfg = yaml.safe_load(Path(cfg["inputs"]["v4_surface_audit_config"]).read_text())
    repeated_gate = validate_registered_depth_scale(base_cfg)
    if not repeated_gate["passed"]: raise RuntimeError("verified-depth gate failed")
    context = load_and_validate(base_cfg); frames = context["frames"]["interaction"]
    attach_interaction_proposals(frames, base_cfg); frames_by_source = {int(frame["source"]): frame for frame in frames}
    raw_root = Path(base_cfg["inputs"]["raw_root"]); timestamp_rows = parse_index(raw_root / "pinhole_projection" / "depth.txt", raw_root / "pinhole_projection")
    ticks = float(cfg["motion_state"]["timestamp_ticks_per_second"]); first = timestamp_rows[frames[0]["source"]]["association_timestamp"]
    timestamps = [(timestamp_rows[frame["source"]]["association_timestamp"] - first) / ticks for frame in frames]
    states = decompose_motion_states([frame["source"] for frame in frames], timestamps,
                                     [frame["q"] for frame in frames], cfg["motion_state"])
    transitions = build_active_transitions(states, cfg["active_transition"])
    _, active_pairs = _active_pair_lookup(states, transitions)
    v4 = Path(cfg["inputs"]["assignment_v4_output"])
    evidence = json.loads((v4 / "region_evidence.json").read_text()); evidence_by_key = {(int(row["original_frame_id"]), row["proposal_id"]): row for row in evidence}
    ambiguity = json.loads((Path(cfg["inputs"]["ambiguity_audit_output"]) / "ambiguity_regions.json").read_text())
    selected = [row for row in ambiguity if row["audit_category"] == "static_drawer_both_supported"]
    if len(selected) != int(cfg["expected_region_count"]): raise RuntimeError("116-region cardinality mismatch")
    occluders, occluder_report = build_trusted_occluders(context, base_cfg, audit_cfg, v4, frames)
    output.mkdir(parents=True); (output / "regions").mkdir(); (output / "visualization").mkdir()
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "motion_states.json").write_text(json.dumps([row.to_dict() for row in states], indent=2) + "\n")
    (output / "active_transitions.json").write_text(json.dumps([row.to_dict() for row in transitions], indent=2) + "\n")
    (output / "trusted_occluder_report.json").write_text(json.dumps(occluder_report, indent=2) + "\n")
    (output / "scale_consistency_gate_report.json").write_text(json.dumps(repeated_gate, indent=2) + "\n")
    runtime = {**cfg["finite_identity"], "projective_evidence": base_cfg["region_evidence"],
               "contradiction": cfg["contradiction"]}; cfg["finite_identity_runtime"] = runtime
    cache = {}; results = []; summary_rows = []; images = defaultdict(list)
    for ordinal, selection in enumerate(selected):
        key = (int(selection["original_frame_id"]), selection["proposal_id"]); item = evidence_by_key[key]
        if item["label"] != "unknown": raise RuntimeError(f"v4 formal label is not unknown: {key}")
        frame = frames_by_source[key[0]]; proposal = next(row for row in frame["proposals"] if row["proposal_id"] == key[1])
        result = analyze_region(item, frame, proposal, frames_by_source, context, cfg, active_pairs, occluders, cache)
        directory = output / "regions" / f"{ordinal:03d}_{item['proposal_id']}"; directory.mkdir()
        (directory / "v5_region_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
        source_overlay(frame, proposal, result, directory / "source_region.jpg"); timeline(result, directory / "identity_timeline.png")
        decision = result["decision"]["decision"]; images[decision].append(directory / "source_region.jpg")
        row = {**result["source_metadata"], "v5_decision": decision, "decision_reason": result["decision"]["reason"],
               "static_verified_active": result["verified_active"]["static"]["count"],
               "drawer_verified_active": result["verified_active"]["drawer"]["count"],
               "static_verified_fraction": result["verified_active"]["static"]["fraction"],
               "drawer_verified_fraction": result["verified_active"]["drawer"]["fraction"],
               "accepted_causal_disocclusion_events": sum(event["positive_static_evidence"] for event in result["causal_disocclusion_events"]),
               "static_contradiction_veto": result["contradiction_runs"]["static"]["coherent_contradiction_run"],
               "drawer_contradiction_veto": result["contradiction_runs"]["drawer"]["coherent_contradiction_run"],
               "finite_boundary_confidence": result["source_finite_surface_observability"]["finite_boundary_confidence"],
               "normal_axis_alignment": result["source_finite_surface_observability"]["normal_axis_alignment"],
               "tangent_motion_ambiguity": result["source_finite_surface_observability"]["tangent_motion_ambiguity"],
               "region_directory": str(directory)}
        summary_rows.append(row); results.append(result)
        print(f"[v5 {ordinal + 1:03d}/{len(selected):03d}] {item['proposal_id']} -> {decision}", flush=True)
    _write_csv(output / "per_region_summary.csv", summary_rows)
    (output / "per_region_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    counts = Counter(row["v5_decision"] for row in summary_rows)
    identity_counts = {model: Counter(row["identity_state"] for result in results for row in result["per_target"][model])
                       for model in ("static", "drawer")}
    summary = {"input_regions": len(results), "formal_v4_label": "unknown",
               "v5_decisions": {name: counts[name] for name in ("STATIC", "MOVING", "UNKNOWN", "CONFLICTING")},
               "per_target_identity_states": {model: dict(counter) for model, counter in identity_counts.items()},
               "regions_with_causal_static_disocclusion": sum(any(event["positive_static_evidence"] for event in result["causal_disocclusion_events"]) for result in results),
               "regions_with_moving_transition_evidence": sum(EvidenceFamily.ARTICULATED_MOTION_TRANSITION.value in result["decision"]["moving_families"] for result in results),
               "interpretation": "diagnostic_only", "formal_v4_modified": False,
               "sam2_ran": False, "dense_propagation_ran": False,
               "camera_axis_q_modified": False, "tsdf_ran": False, "nksr_ran": False, "mesh_ran": False,
               "ready_for_dual_tsdf": False}
    (output / "v5_116_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    viz = cfg["visualization"]
    for decision in ("STATIC", "MOVING", "UNKNOWN", "CONFLICTING"):
        contact_sheet(images[decision], output / "visualization" / f"{decision.lower()}_regions.jpg",
                      int(viz["contact_sheet_columns"]), int(viz["contact_sheet_thumbnail_width"]), f"v5 {decision} regions")
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return summary
