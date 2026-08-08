"""Real-data regression and conditional 116 diagnostic for observation censor."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .control_evaluation import evaluate_control_gate
from .motion_discrimination import evaluate_multi_baseline_observation
from .multi_baseline import build_multi_baseline_observations, observations_for_source
from .multi_baseline_diagnostic import _csv, _json, _source_overlay
from .observation_censor import (
    ObservationState,
    censor_observation,
    discriminate_censored_motion_model,
    load_frozen_noise_model,
)
from .transition_diagnostic import (
    _find_proposal,
    _load_runtime,
    _run_synthetic_tests,
    _validate_annotations,
    analyze_region,
)


OBSERVATION_STATES = tuple(item.value for item in ObservationState)
BASELINES = ("short", "medium", "long")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def _before_censor_summary(rows: list[dict], cfg: dict) -> dict:
    minimum_snr = float(cfg["minimum_discriminative_to_plateau_p90_ratio"])
    selected = [row for row in rows
                if row["raw_log_evidence_moving_over_static_diagnostic_only"] is not None
                and row["coverage"]["visibility"] == "common_source_anchor_observable"
                and row["discriminative_to_plateau_p90_ratio"] >= minimum_snr]
    evidence = np.asarray([
        row["raw_log_evidence_moving_over_static_diagnostic_only"] for row in selected], float)
    total = float(evidence.sum()) if len(evidence) else 0.0
    return {
        "role": "diagnostic_reconstruction_of_pre_censor_aggregation",
        "observation_count": len(selected),
        "aggregate_log_evidence_moving_over_static": total,
        "posterior_moving": float(1.0 / (1.0 + math.exp(-np.clip(total, -700., 700.)))),
        "contains_both_ood_observations": any(row["censored_reason"] ==
            "both_models_ood_unrelated_valid_depth_or_identity_lost" for row in selected),
    }


def analyze_censored_region(
    frame: dict, proposal: dict, runtime: dict, noise_model, cfg: dict,
    evaluation_mode: str = "active_transition",
) -> dict:
    # Local causal reveal and finite-boundary logic are deliberately unchanged.
    local = analyze_region(frame, proposal, runtime, cfg, evaluation_mode)
    patch = local["_raw"]["patch"]
    boundary = local["physical_boundary"]
    source_frame = dict(frame)
    source_frame["surface_normal_axis_alignment"] = (
        boundary["normal_axis_alignment"]
        if boundary["normal_axis_alignment"] is not None else 0.0)
    rows = []
    if evaluation_mode != "plateau_only":
        for observation in observations_for_source(
                int(frame["source"]), runtime["multi_baseline_observations"]):
            target = runtime["frames_by_id"][int(observation.target_frame)]
            raw = evaluate_multi_baseline_observation(
                patch, source_frame, target, observation, runtime, noise_model, cfg)
            rows.append(censor_observation(raw, noise_model))
    causal_positive = any(
        event["positive_causal_static_disocclusion"]
        for event in local["_raw"]["causal_events"])
    decision = discriminate_censored_motion_model(
        rows, boundary, noise_model, cfg["model_discrimination"],
        "plateau_only" if evaluation_mode == "plateau_only" else "active_motion",
        causal_static_positive=causal_positive)
    return {
        "source_metadata": {
            "original_frame_id": int(frame["source"]), "q_m": float(frame["q"]),
            "proposal_id": proposal["proposal_id"], "source_layer": int(proposal["source_layer"]),
            "evaluation_mode": evaluation_mode,
        },
        "source_patch": {
            "valid_region_pixel_count": int(patch["valid_region_pixel_count"]),
            "sample_count": int(patch["sample_count"]),
        },
        "physical_boundary": boundary,
        "causal_reveal_events": local["causal_reveal_events"],
        "observations": rows,
        "before_censor_evidence": _before_censor_summary(rows, cfg["model_discrimination"]),
        "decision": decision,
        "hard_invariants": {
            "source_anchor_immutable": True,
            "target_proposals_used_for_identity": False,
            "target_proposal_matching_or_surface_hopping": False,
            "motion_discrimination_allows_forward_and_backward": True,
            "causal_disocclusion_forward_only": True,
            "both_ood_observations_excluded_from_all_aggregation": True,
            "both_ood_smaller_residual_selected": False,
            "plateau_noise_refit": False,
            "controls_are_fresh_heldout_validation": False,
            "camera_pose_axis_q_modified": False,
            "sam2_or_reconstruction_ran": False,
        },
    }


def _summary_row(result: dict, group: str) -> dict:
    meta = result["source_metadata"]; decision = result["decision"]
    row = {
        "original_frame_id": meta["original_frame_id"],
        "proposal_id": meta["proposal_id"], "source_layer": meta["source_layer"],
        "control_group": group, "decision": decision["label"],
        "formal_ownership_label": decision["formal_ownership_label"],
        "decision_reason": decision["reason"],
        "observation_count": len(result["observations"]),
        "informative_observation_count": decision["informative_observation_count"],
        "effective_chain_length": decision["effective_chain_length"],
        "censored_observation_count": decision.get("censored_observation_count", 0),
        "ambiguous_neutral_observation_count": decision.get(
            "ambiguous_neutral_observation_count", 0),
        "posterior_moving_before_censor": result["before_censor_evidence"]["posterior_moving"],
        "posterior_moving_after_censor": decision["posterior_moving"],
        "aggregate_evidence_before_censor": result["before_censor_evidence"][
            "aggregate_log_evidence_moving_over_static"],
        "aggregate_evidence_after_censor": decision.get(
            "aggregate_log_evidence_moving_over_static", 0.0),
        "tangent_motion": result["physical_boundary"]["tangent_motion"],
        "trusted_finite_physical_edge": result["physical_boundary"][
            "has_trusted_axis_finite_edge"],
    }
    for baseline in BASELINES:
        counts = decision.get("baseline_observation_state_counts", {}).get(baseline, {})
        row[f"{baseline}_censored"] = int(counts.get(
            ObservationState.IDENTITY_LOST_CENSORED.value, 0))
        row[f"{baseline}_ambiguous"] = int(counts.get(
            ObservationState.AMBIGUOUS_COMPATIBLE.value, 0))
        row[f"{baseline}_moving_evidence"] = int(counts.get(
            ObservationState.MOVING_LINK_EVIDENCE.value, 0))
        row[f"{baseline}_static_evidence"] = int(counts.get(
            ObservationState.WORLD_STATIC_EVIDENCE.value, 0))
    return row


def aggregate_censor_counts(results_by_group: dict[str, list[dict]]) -> dict:
    report = {}
    for group, results in results_by_group.items():
        group_row = {}
        for baseline in BASELINES:
            counts = Counter(
                observation["observation_state"]
                for result in results for observation in result["observations"]
                if observation["baseline"] == baseline)
            group_row[baseline] = {
                "total": int(sum(counts.values())),
                **{state: int(counts[state]) for state in OBSERVATION_STATES},
            }
        group_row["all_baselines"] = {
            "total": int(sum(item["total"] for item in group_row.values())),
            **{state: int(sum(item[state] for item in group_row.values()))
               for state in OBSERVATION_STATES},
        }
        report[group] = group_row
    return report


def _state_count_plot(counts: dict, path: Path) -> None:
    groups = ("drawer_front", "active_comoving_box",
              "documented_floor_false_positive", "provisional_world_static")
    colors = {
        ObservationState.MOVING_LINK_EVIDENCE.value: "#4daf4a",
        ObservationState.WORLD_STATIC_EVIDENCE.value: "#ff7f00",
        ObservationState.AMBIGUOUS_COMPATIBLE.value: "#999999",
        ObservationState.IDENTITY_LOST_CENSORED.value: "#e41a1c",
    }
    labels = [f"{group}\n{baseline}" for group in groups for baseline in BASELINES]
    figure, axis = plt.subplots(figsize=(15, 6))
    bottom = np.zeros(len(labels), float)
    for state in OBSERVATION_STATES:
        values = np.asarray([
            counts.get(group, {}).get(baseline, {}).get(state, 0)
            for group in groups for baseline in BASELINES], float)
        axis.bar(np.arange(len(labels)), values, bottom=bottom, color=colors[state], label=state)
        bottom += values
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel("source-indexed observations")
    axis.set_title("Observation censor states by regression group and q baseline")
    axis.legend(fontsize=8); axis.grid(axis="y", alpha=.2)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _box_before_after_plot(results: list[dict], path: Path) -> None:
    labels = [str(row["source_metadata"]["original_frame_id"]) for row in results]
    before = [row["before_censor_evidence"]["aggregate_log_evidence_moving_over_static"]
              for row in results]
    after = [row["decision"]["aggregate_log_evidence_moving_over_static"]
             for row in results]
    x = np.arange(len(labels)); width = .36
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(x - width / 2, before, width, label="before censor", color="#999999")
    axis.bar(x + width / 2, after, width, label="after censor", color="#4daf4a")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(x, labels); axis.set_xlabel("box source frame")
    axis.set_ylabel("aggregate log evidence moving/static")
    axis.set_title("Active box evidence before and after both-OOD censor")
    axis.legend(); axis.grid(axis="y", alpha=.2)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _box_state_curve(results: list[dict], path: Path) -> None:
    colors = {
        ObservationState.MOVING_LINK_EVIDENCE.value: "#4daf4a",
        ObservationState.WORLD_STATIC_EVIDENCE.value: "#ff7f00",
        ObservationState.AMBIGUOUS_COMPATIBLE.value: "#999999",
        ObservationState.IDENTITY_LOST_CENSORED.value: "#e41a1c",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for result in results:
        for row in result["observations"]:
            color = colors[row["observation_state"]]
            x = row["d_discriminative_m"] * 1000.
            axes[0].scatter(x, row["static"]["median_residual_m"] * 1000., c=color, s=18, alpha=.65)
            axes[1].scatter(x, row["moving"]["median_residual_m"] * 1000., c=color, s=18, alpha=.65)
    for axis, title in zip(axes, ("static hypothesis", "moving hypothesis")):
        axis.set_title(title); axis.set_xlabel(r"$|n\cdot axis||\Delta q|$ (mm)")
        axis.set_ylabel("median residual (mm)"); axis.grid(alpha=.2)
    handles = [plt.Line2D([], [], color=color, marker="o", linestyle="", label=state)
               for state, color in colors.items()]
    figure.legend(handles=handles, loc="upper center", ncol=2, fontsize=8)
    figure.suptitle("Active box: frozen-noise observation states")
    figure.tight_layout(rect=(0, 0, 1, .88)); figure.savefig(path, dpi=160); plt.close(figure)


def _freeze_manifest(config_path: Path, cfg: dict, noise_report: dict,
                     worktree: Path, phase: str, gate_report: dict | None = None) -> dict:
    modules = [
        "tools/itaco_ownership_v5/observation_censor.py",
        "tools/itaco_ownership_v5/observation_censor_diagnostic.py",
        "tools/itaco_ownership_v5/motion_discrimination.py",
        "tools/itaco_ownership_v5/multi_baseline.py",
        "tools/itaco_ownership_v5/plateau_noise.py",
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree,
        capture_output=True, text=True, check=True).stdout.strip()
    payload = {
        "phase": phase, "frozen": True,
        "git_parent_revision": revision,
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "module_sha256": {name: _sha256(worktree / name) for name in modules},
        "original_noise_model_hash": noise_report["original_noise_model_hash"],
        "noise_model_refit": False,
        "noise_criterion": {
            "statistic": noise_report["compatibility_statistic"],
            "quantile": noise_report["compatibility_threshold_quantile"],
            "threshold": noise_report["compatibility_threshold_value"],
            "compatible_relation": noise_report["compatible_relation"],
            "ood_relation": noise_report["ood_relation"],
        },
        "controls_evaluation_role": "previously_seen_regression_not_fresh_heldout",
        "thresholds_tuned_to_controls": False,
        "target_proposals_used_for_identity": False,
    }
    if gate_report is not None:
        payload["regression_gate_passed_before_116"] = bool(gate_report["passed"])
        payload["regression_gate_report_sha256"] = hashlib.sha256(
            json.dumps(gate_report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("ready_for_dual_tsdf") is not False:
        raise RuntimeError("observation censor must preserve ready_for_dual_tsdf=false")
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
    observations, rejected = build_multi_baseline_observations(
        runtime["states"], cfg["multi_baseline"])
    runtime["multi_baseline_observations"] = observations
    _json(output / "multi_baseline_observation_manifest.json", [row.to_dict() for row in observations])
    _json(output / "rejected_multi_baseline_observations.json", [row.to_dict() for row in rejected])
    noise_model, noise_report = load_frozen_noise_model(
        Path(cfg["inputs"]["frozen_noise_output"]), cfg["inputs"]["frozen_noise_model_hash"])
    if cfg["plateau_noise"]["edge_definition"] != noise_model.summary["edge_definition"]:
        raise RuntimeError("configured edge definition differs from frozen noise model")
    _json(output / "frozen_noise_model_validation.json", noise_report)
    _json(output / "plateau_noise_model_reused.json", noise_model.summary)
    _json(output / "scale_consistency_gate_report.json", runtime["scale_gate"])
    _json(output / "motion_states.json", [row.to_dict() for row in runtime["states"]])
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    pre_regression_freeze = _freeze_manifest(
        config_path, cfg, noise_report, worktree, "before_seen_control_regression")
    _json(output / "implementation_freeze_before_regression.json", pre_regression_freeze)

    v4_output = Path(cfg["inputs"]["assignment_v4_output"])
    v4_evidence = json.loads((v4_output / "region_evidence.json").read_text())
    failed_rows = json.loads(Path(cfg["inputs"]["failed_001_summary"]).read_text())
    annotations = json.loads(Path(cfg["inputs"]["control_annotations"]).read_text())
    annotation_report = _validate_annotations(annotations, v4_output, v4_evidence, failed_rows)
    annotation_report["evaluation_role"] = "previously_seen_regression_not_fresh_heldout"
    _json(output / "control_annotation_validation.json", annotation_report)
    grouped = defaultdict(list); control_results = []; control_rows = []
    controls_root = output / "controls"; controls_root.mkdir()
    for ordinal, annotation in enumerate(annotations["controls"]):
        frame = runtime["frames_by_id"][int(annotation["original_frame_id"])]
        proposal = _find_proposal(frame, annotation["proposal_id"])
        result = analyze_censored_region(
            frame, proposal, runtime, noise_model, cfg,
            annotation.get("evaluation_mode", "active_transition"))
        result["evaluation_annotation"] = annotation
        group = annotation["control_group"]
        directory = controls_root / group / f"{ordinal:03d}_{proposal['proposal_id']}"
        directory.mkdir(parents=True)
        _json(directory / "observation_censor_diagnostic.json", result)
        _source_overlay(frame, proposal, result, directory / "source_overlay.jpg")
        row = _summary_row(result, group); row["expected_label"] = annotation["expected_label"]
        control_rows.append(row); control_results.append(result); grouped[group].append(result)
        print(f"[control {ordinal + 1:02d}/{len(annotations['controls']):02d}] "
              f"{proposal['proposal_id']} -> {result['decision']['label']} "
              f"censored={result['decision'].get('censored_observation_count', 0)}", flush=True)
    gate = evaluate_control_gate(control_rows, tests["passed"], cfg["controls"])
    gate["evaluation_role"] = "previously_seen_regression_not_fresh_heldout"
    gate["noise_model_refit"] = False; gate["threshold_tuning"] = False
    _json(output / "control_regression_gate.json", gate)
    _csv(output / "control_region_summary.csv", control_rows)
    _json(output / "control_region_results.json", control_results)
    censor_counts = aggregate_censor_counts(grouped)
    _json(output / "control_censor_counts.json", censor_counts)
    box_comparison = [{
        "original_frame_id": result["source_metadata"]["original_frame_id"],
        "before": result["before_censor_evidence"], "after": result["decision"],
    } for result in grouped["active_comoving_box"]]
    _json(output / "active_box_before_after.json", box_comparison)
    visual = output / "visualization"; visual.mkdir()
    _state_count_plot(censor_counts, visual / "control_censor_states_by_baseline.png")
    _box_before_after_plot(grouped["active_comoving_box"], visual / "active_box_evidence_before_after.png")
    _box_state_curve(grouped["active_comoving_box"], visual / "active_box_observation_states.png")

    full_rows = []; full_ran = False; full_state_counts = {}
    if gate["passed"]:
        freeze_116 = _freeze_manifest(
            config_path, cfg, noise_report, worktree, "after_regression_before_116", gate)
        _json(output / "diagnostic_116_freeze_manifest.json", freeze_116)
        ambiguity = json.loads((Path(cfg["inputs"]["ambiguity_audit_output"]) /
                                "ambiguity_regions.json").read_text())
        selected = [row for row in ambiguity if row["audit_category"] == "static_drawer_both_supported"]
        if len(selected) != int(cfg["expected_ambiguity_region_count"]):
            raise RuntimeError("116-region cardinality mismatch")
        evidence_by_key = {(int(row["original_frame_id"]), row["proposal_id"]): row
                           for row in v4_evidence}
        region_root = output / "regions_116"; region_root.mkdir(); full_ran = True
        representative_counts = Counter(); full_results = []
        for ordinal, selection in enumerate(selected):
            key = int(selection["original_frame_id"]), selection["proposal_id"]
            if evidence_by_key[key]["label"] != "unknown":
                raise RuntimeError(f"formal v4 label changed for {key}")
            frame = runtime["frames_by_id"][key[0]]; proposal = _find_proposal(frame, key[1])
            result = analyze_censored_region(frame, proposal, runtime, noise_model, cfg)
            directory = region_root / f"{ordinal:03d}_{proposal['proposal_id']}"; directory.mkdir()
            _json(directory / "observation_censor_diagnostic.json", result)
            decision = result["decision"]["label"]
            if representative_counts[decision] < int(
                    cfg["visualization"]["maximum_116_representatives_per_decision"]):
                _source_overlay(frame, proposal, result, directory / "source_overlay.jpg")
                representative_counts[decision] += 1
            full_rows.append(_summary_row(result, "ambiguity_116")); full_results.append(result)
            print(f"[116 {ordinal + 1:03d}/116] {proposal['proposal_id']} -> {decision}", flush=True)
        _csv(output / "per_region_observation_censor_summary.csv", full_rows)
        _json(output / "per_region_observation_censor_summary.json", full_rows)
        full_state_counts = aggregate_censor_counts({"ambiguity_116": full_results})
        _json(output / "regions_116_censor_counts.json", full_state_counts)
        decision_counts = Counter(row["decision"] for row in full_rows)
        full_summary = {
            "ran": True, "evaluation_role": "diagnostic_not_fresh_heldout_validation",
            "input_region_count": len(full_rows),
            "decision_counts": {name: int(decision_counts[name]) for name in
                                ("MOVING_LINK", "WORLD_STATIC", "UNKNOWN", "CONFLICTING")},
            "formal_v4_labels_modified": False, "ready_for_dual_tsdf": False,
        }
    else:
        full_summary = {
            "ran": False, "reason": "seen_control_regression_failed_stop_before_116",
            "decision_counts": None, "ready_for_dual_tsdf": False,
        }
    _json(output / "observation_censor_116_summary.json", full_summary)
    summary = {
        "stage": cfg["stage"],
        "original_noise_model_hash": noise_report["original_noise_model_hash"],
        "noise_model_refit": False,
        "control_evaluation_role": "previously_seen_regression_not_fresh_heldout",
        "control_regression_passed": bool(gate["passed"]),
        "control_metrics": gate["metrics"], "control_censor_counts": censor_counts,
        "full_116_ran": full_ran, "full_116": full_summary,
        "target_proposals_used_for_identity": False,
        "source_anchor_immutable": True, "formal_v4_labels_modified": False,
        "sam2_ran": False, "region_propagation_ran": False,
        "camera_axis_q_modified": False, "tsdf_ran": False, "nksr_ran": False,
        "mesh_ran": False, "ready_for_dual_tsdf": False,
    }
    _json(output / "observation_censor_summary.json", summary)

    for name in ("frozen_noise_model_validation.json", "implementation_freeze_before_regression.json",
                 "control_regression_gate.json", "control_region_summary.csv",
                 "control_censor_counts.json", "active_box_before_after.json",
                 "observation_censor_summary.json", "observation_censor_116_summary.json",
                 "synthetic_test_report.json", "scale_consistency_gate_report.json"):
        shutil.copy2(output / name, review / name)
    if full_ran:
        for name in ("diagnostic_116_freeze_manifest.json",
                     "per_region_observation_censor_summary.csv",
                     "per_region_observation_censor_summary.json",
                     "regions_116_censor_counts.json"):
            shutil.copy2(output / name, review / name)
    shutil.copytree(visual, review / "visualization")
    print(json.dumps({"output": str(output), "review_bundle": str(review), **summary}, indent=2))
    return summary
