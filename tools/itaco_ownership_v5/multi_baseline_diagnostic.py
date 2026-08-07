"""Control-first Assignment-v5 multi-baseline motion discrimination."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
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

from .control_evaluation import evaluate_control_gate
from .motion_discrimination import (
    discriminate_motion_model,
    evaluate_multi_baseline_observation,
)
from .multi_baseline import (
    build_multi_baseline_observations,
    observations_for_source,
)
from .plateau_noise import fit_frozen_plateau_noise_model
from .transition_diagnostic import (
    DECISION_COLORS,
    _find_proposal,
    _load_runtime,
    _run_synthetic_tests,
    _validate_annotations,
    analyze_region,
)


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def analyze_multi_baseline_region(
    frame: dict, proposal: dict, runtime: dict, noise_model, cfg: dict,
    evaluation_mode: str = "active_transition",
) -> dict:
    # The existing local event implementation is retained only for measured,
    # forward-time causal disocclusion and physical finite-edge diagnostics.
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
            rows.append(evaluate_multi_baseline_observation(
                patch, source_frame, target, observation, runtime, noise_model, cfg))
    causal_positive = any(
        event["positive_causal_static_disocclusion"]
        for event in local["_raw"]["causal_events"])
    decision = discriminate_motion_model(
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
        "multi_baseline_observations": rows,
        "decision": decision,
        "legacy_local_absolute_threshold_decision_diagnostic_only": local["decision"],
        "hard_invariants": {
            "source_anchor_immutable": True,
            "target_proposals_used_for_identity": False,
            "target_proposal_matching_or_surface_hopping": False,
            "motion_discrimination_allows_forward_and_backward": True,
            "causal_disocclusion_forward_only": True,
            "absolute_30mm_compatibility_used_for_final_decision": False,
            "plateau_noise_frozen_before_controls": True,
            "control_annotations_used_by_classifier": False,
            "camera_pose_axis_q_modified": False,
            "region_propagation_ran": False,
            "reconstruction_ran": False,
        },
    }


def _summary_row(result: dict, group: str) -> dict:
    meta = result["source_metadata"]; decision = result["decision"]
    observations = result["multi_baseline_observations"]
    accepted = [row for row in observations if row["accepted_for_discrimination"]]
    return {
        "original_frame_id": meta["original_frame_id"],
        "proposal_id": meta["proposal_id"], "source_layer": meta["source_layer"],
        "control_group": group, "decision": decision["label"],
        "formal_ownership_label": decision["formal_ownership_label"],
        "decision_reason": decision["reason"],
        "multi_baseline_observation_count": len(observations),
        "accepted_observation_count": len(accepted),
        "short_count": sum(row["baseline"] == "short" for row in accepted),
        "medium_count": sum(row["baseline"] == "medium" for row in accepted),
        "long_count": sum(row["baseline"] == "long" for row in accepted),
        "informative_observation_count": decision["informative_observation_count"],
        "posterior_moving": decision["posterior_moving"],
        "aggregate_log_evidence_moving_over_static": decision.get(
            "aggregate_log_evidence_moving_over_static", 0.0),
        "static_residual_vs_discriminative_slope": decision.get(
            "static_residual_vs_discriminative_slope"),
        "moving_residual_vs_discriminative_slope": decision.get(
            "moving_residual_vs_discriminative_slope"),
        "tangent_motion": result["physical_boundary"]["tangent_motion"],
        "trusted_finite_physical_edge": result["physical_boundary"]["has_trusted_axis_finite_edge"],
        "causal_static_positive": decision.get("causal_static_positive", False),
    }


def _source_overlay(frame: dict, proposal: dict, result: dict, path: Path) -> None:
    image = frame["rgb"].copy(); mask = np.asarray(proposal["mask"], bool)
    decision = result["decision"]; color = np.asarray(DECISION_COLORS[decision["label"]], np.uint8)
    layer = np.broadcast_to(color, image.shape).copy()
    blended = cv2.addWeighted(image, .4, layer, .6, 0)
    image[mask] = blended[mask]
    header = np.zeros((78, image.shape[1], 3), np.uint8)
    lines = [
        f"frame={frame['source']} {proposal['proposal_id']} -> {decision['label']}",
        decision["reason"],
        f"n={decision['informative_observation_count']} P(moving)={decision['posterior_moving']:.5f}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(header, line[:90], (5, 18 + index * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .38, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack((header, image)))


def _residual_curve(group: str, results: list[dict], noise_summary: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    colors = {"short": "#377eb8", "medium": "#ff7f00", "long": "#e41a1c"}
    count = 0
    for result in results:
        for row in result["multi_baseline_observations"]:
            if not row["accepted_for_discrimination"]:
                continue
            x = row["d_discriminative_m"] * 1000.
            for axis, model in zip(axes, ("static", "moving")):
                metrics = row[model]
                axis.scatter(x, metrics["median_residual_m"] * 1000.,
                             s=17, alpha=.55, c=colors[row["baseline"]], marker="o")
                axis.scatter(x, metrics["p90_residual_m"] * 1000.,
                             s=14, alpha=.25, c=colors[row["baseline"]], marker="^")
            count += 1
    for axis, model in zip(axes, ("static", "drawer/moving")):
        axis.axhspan(0, noise_summary["combined"]["p90_m"] * 1000., color="#4daf4a", alpha=.12)
        axis.axhline(noise_summary["combined"]["median_m"] * 1000., color="#4daf4a", linestyle="--")
        axis.set_title(f"{model} hypothesis")
        axis.set_xlabel(r"$|n\cdot axis||\Delta q|$ (mm)")
        axis.set_ylabel("registered-depth 3D residual (mm)")
        axis.grid(alpha=.2)
    handles = [plt.Line2D([], [], color=value, marker="o", linestyle="", label=key)
               for key, value in colors.items()]
    figure.legend(handles=handles, loc="upper right")
    figure.suptitle(f"{group}: residual vs discriminative displacement ({count} visible observations)\n"
                    "circle=median, triangle=p90, green band=plateau combined p90")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _noise_plot(model, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for name, color in (("surface_interior", "#377eb8"),
                        ("depth_or_geometry_edge", "#e41a1c")):
        values = model.residuals[name] * 1000.
        axis.hist(values, bins=80, density=True, histtype="step", linewidth=1.6,
                  color=color, label=f"{name} n={len(values)}")
    axis.set(title="Frozen plateau registered-depth residual noise",
             xlabel="source-indexed 3D residual (mm)", ylabel="density")
    axis.set_xlim(0, min(80, np.percentile(np.concatenate(list(model.residuals.values())), 99.8) * 1000.))
    axis.legend(); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _freeze_manifest(config_path: Path, cfg: dict, noise_model, worktree: Path) -> dict:
    modules = [
        worktree / "tools/itaco_ownership_v5/multi_baseline.py",
        worktree / "tools/itaco_ownership_v5/plateau_noise.py",
        worktree / "tools/itaco_ownership_v5/motion_discrimination.py",
        worktree / "tools/itaco_ownership_v5/multi_baseline_diagnostic.py",
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "frozen_before_control_evaluation": True,
        "git_parent_revision": revision,
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "noise_model_sha256": noise_model.summary["frozen_model_sha256"],
        "classifier_module_sha256": {str(path.relative_to(worktree)): _sha256(path) for path in modules},
        "frozen_thresholds": {
            "multi_baseline": cfg["multi_baseline"],
            "plateau_noise": cfg["plateau_noise"],
            "visibility": cfg["visibility"],
            "model_discrimination": cfg["model_discrimination"],
            "physical_boundary": cfg["physical_boundary"],
        },
        "control_annotations_read_by_classifier": False,
        "parameter_tuning_after_control_evaluation": False,
    }


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("ready_for_dual_tsdf") is not False:
        raise RuntimeError("Assignment v5 diagnostic must keep ready_for_dual_tsdf=false")
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
    runtime["rejected_multi_baseline_observations"] = rejected
    _json(output / "multi_baseline_observation_manifest.json", [row.to_dict() for row in observations])
    _json(output / "rejected_multi_baseline_observations.json", [row.to_dict() for row in rejected])
    noise_model, noise_pairs = fit_frozen_plateau_noise_model(
        runtime, cfg["plateau_noise"], cfg["projective_anchor"], cfg["source_patch"])
    _json(output / "plateau_noise_model.json", noise_model.summary)
    _json(output / "plateau_noise_calibration_pairs.json", noise_pairs)
    np.savez_compressed(output / "plateau_noise_residuals.npz", **noise_model.residuals)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _json(output / "scale_consistency_gate_report.json", runtime["scale_gate"])
    _json(output / "motion_states.json", [row.to_dict() for row in runtime["states"]])
    freeze = _freeze_manifest(config_path, cfg, noise_model, worktree)
    _json(output / "classifier_freeze_manifest.json", freeze)
    visual = output / "visualization"; visual.mkdir()
    _noise_plot(noise_model, visual / "plateau_noise_floor.png")

    # Held-out annotations are read only after the classifier/noise freeze files exist.
    v4_output = Path(cfg["inputs"]["assignment_v4_output"])
    v4_evidence = json.loads((v4_output / "region_evidence.json").read_text())
    failed_rows = json.loads(Path(cfg["inputs"]["failed_001_summary"]).read_text())
    annotations = json.loads(Path(cfg["inputs"]["control_annotations"]).read_text())
    annotation_report = _validate_annotations(annotations, v4_output, v4_evidence, failed_rows)
    _json(output / "control_annotation_validation.json", annotation_report)
    results = []; rows = []; grouped = defaultdict(list)
    controls_root = output / "controls"; controls_root.mkdir()
    for ordinal, annotation in enumerate(annotations["controls"]):
        frame = runtime["frames_by_id"][int(annotation["original_frame_id"])]
        proposal = _find_proposal(frame, annotation["proposal_id"])
        result = analyze_multi_baseline_region(
            frame, proposal, runtime, noise_model, cfg,
            annotation.get("evaluation_mode", "active_transition"))
        result["evaluation_annotation"] = annotation
        directory = controls_root / annotation["control_group"] / f"{ordinal:03d}_{proposal['proposal_id']}"
        directory.mkdir(parents=True)
        _json(directory / "multi_baseline_diagnostic.json", result)
        _source_overlay(frame, proposal, result, directory / "source_overlay.jpg")
        row = _summary_row(result, annotation["control_group"])
        row["expected_label"] = annotation["expected_label"]
        rows.append(row); results.append(result); grouped[annotation["control_group"]].append(result)
        print(f"[control {ordinal + 1:02d}/{len(annotations['controls']):02d}] "
              f"{proposal['proposal_id']} -> {result['decision']['label']}", flush=True)
    gate = evaluate_control_gate(rows, tests["passed"], cfg["controls"])
    _json(output / "control_evaluation_report.json", gate)
    _csv(output / "control_region_summary.csv", rows)
    _json(output / "control_region_results.json", results)
    for group in ("drawer_front", "active_comoving_box", "documented_floor_false_positive",
                  "provisional_world_static"):
        _residual_curve(group, grouped[group], noise_model.summary,
                        visual / f"residual_vs_discriminative_{group}.png")

    full_rows = []; full_ran = False
    if gate["passed"]:
        ambiguity = json.loads((Path(cfg["inputs"]["ambiguity_audit_output"]) /
                                "ambiguity_regions.json").read_text())
        selected = [row for row in ambiguity if row["audit_category"] == "static_drawer_both_supported"]
        if len(selected) != int(cfg["expected_ambiguity_region_count"]):
            raise RuntimeError("116-region cardinality mismatch")
        evidence_by_key = {(int(row["original_frame_id"]), row["proposal_id"]): row for row in v4_evidence}
        region_root = output / "regions_116"; region_root.mkdir(); full_ran = True
        for ordinal, selection in enumerate(selected):
            key = int(selection["original_frame_id"]), selection["proposal_id"]
            if evidence_by_key[key]["label"] != "unknown":
                raise RuntimeError(f"formal v4 label changed for {key}")
            frame = runtime["frames_by_id"][key[0]]; proposal = _find_proposal(frame, key[1])
            result = analyze_multi_baseline_region(frame, proposal, runtime, noise_model, cfg)
            directory = region_root / f"{ordinal:03d}_{proposal['proposal_id']}"; directory.mkdir()
            _json(directory / "multi_baseline_diagnostic.json", result)
            if ordinal < int(cfg["visualization"]["maximum_116_representatives_per_decision"]):
                _source_overlay(frame, proposal, result, directory / "source_overlay.jpg")
            full_rows.append(_summary_row(result, "ambiguity_116"))
            print(f"[116 {ordinal + 1:03d}/116] {proposal['proposal_id']} -> "
                  f"{result['decision']['label']}", flush=True)
        _csv(output / "per_region_multi_baseline_summary.csv", full_rows)
        _json(output / "per_region_multi_baseline_summary.json", full_rows)
        counts = Counter(row["decision"] for row in full_rows)
        full_summary = {
            "ran": True, "input_region_count": len(full_rows),
            "decision_counts": {name: int(counts[name]) for name in
                                ("MOVING_LINK", "WORLD_STATIC", "UNKNOWN", "CONFLICTING")},
            "formal_v4_labels_modified": False, "ready_for_dual_tsdf": False,
        }
    else:
        full_summary = {
            "ran": False, "reason": "control_hard_gate_failed_stop_before_116",
            "decision_counts": None, "ready_for_dual_tsdf": False,
        }
    _json(output / "multi_baseline_116_summary.json", full_summary)
    summary = {
        "stage": cfg["stage"], "plateau_noise_model": noise_model.summary,
        "control_gate_passed": bool(gate["passed"]), "control_metrics": gate["metrics"],
        "full_116_ran": full_ran, "full_116": full_summary,
        "multi_baseline_observation_count": len(observations),
        "thresholds_frozen_before_control_evaluation": True,
        "parameter_tuning_after_control_evaluation": False,
        "target_proposals_used_for_identity": False,
        "source_anchor_immutable": True, "formal_v4_labels_modified": False,
        "sam2_ran": False, "region_propagation_ran": False,
        "camera_axis_q_modified": False, "tsdf_ran": False, "nksr_ran": False,
        "mesh_ran": False, "ready_for_dual_tsdf": False,
    }
    _json(output / "multi_baseline_summary.json", summary)

    for name in ("plateau_noise_model.json", "classifier_freeze_manifest.json",
                 "control_evaluation_report.json", "control_region_summary.csv",
                 "multi_baseline_summary.json", "multi_baseline_116_summary.json",
                 "synthetic_test_report.json", "scale_consistency_gate_report.json"):
        shutil.copy2(output / name, review / name)
    shutil.copytree(visual, review / "visualization")
    print(json.dumps({"output": str(output), "review_bundle": str(review), **summary}, indent=2))
    return summary
