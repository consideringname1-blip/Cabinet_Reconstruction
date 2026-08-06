"""Strict stage-1 orchestration. No camera/joint optimization or reconstruction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import hashlib
import platform
import subprocess
import sys

import traceback
import numpy as np

from .classification import build_clusters, classify_tracks, proposal_split_merge_diagnostics, save_posteriors, write_label_ply
from .config import load_config, save_resolved_config
from .diagnostics import draw_reference_scores, draw_split_merge_cases, draw_track_label_overlays, draw_validity_overlays
from .errors import Failure, Phase1Error, ValidationErrors
from .geometry import KnownMotion, rebase_poses, save_pose_text
from .manifest import build_manifest, save_manifest
from .proposals import build_explicit_proposals
from .reference import select_reference_frame
import cv2
import scipy
import yaml
from .tracking import build_tracks, save_tracks_npz
from .validity import build_validity


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _usage_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
def _git_state(path: Path) -> dict:
    def command(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(path), *args], check=False, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"
    return {"path": str(path.resolve()), "head": command("rev-parse", "HEAD"), "status_short": command("status", "--short")}


def _source_hashes(package_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            result[str(path.relative_to(package_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _environment_manifest(config: dict) -> dict:
    package_dir = Path(__file__).resolve().parent
    workspace = package_dir.parents[1]
    baseline = workspace / "code/reconstruction/video2articulation"
    return {
        "command": sys.argv,
        "entry_script": str(Path(__file__).resolve()),
        "config_source": config["_config_source"],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_prefix": str(Path(sys.prefix).resolve()),
        "libraries": {"numpy": np.__version__, "opencv": cv2.__version__, "scipy": scipy.__version__, "pyyaml": yaml.__version__},
        "source_revisions": {"workspace": _git_state(workspace), "official_itaco": _git_state(baseline)},
        "stage1_source_sha256": _source_hashes(package_dir),
        "phase_boundary": {"camera_pose_fixed": True, "known_motion_fixed": True, "camera_optimization": False,
                           "joint_estimation": False, "model_selection": False, "free_se3": False,
                           "tsdf": False, "nksr": False, "mesh": False},
    }




def run(config_path: Path, output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Phase1Error(Failure("startup", "output_not_empty", "Stage-1 output directory must be new or empty", details={"output_dir": str(output_dir)}))
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    try:
        config = load_config(config_path)
        save_resolved_config(config, output_dir / "config_resolved.yaml")
        try:
            records, poses_original, validation_report = build_manifest(config)
        except ValidationErrors as exc:
            validation_report = {"valid": False, "errors": [item.to_dict() for item in exc.failures]}
            _json(output_dir / "frame_validation_report.json", validation_report)
            raise
        _json(output_dir / "frame_validation_report.json", validation_report)
        state = np.load(config["known_motion"]["state_path"]).astype(np.float64).reshape(-1)
        reference_index, candidates, selection = select_reference_frame(records, poses_original, config, state)
        poses_rebased = rebase_poses(poses_original, reference_index)
        np.random.seed(int(config["runtime"]["random_seed"]))
        cv2.setRNGSeed(int(config["runtime"]["random_seed"]))
        cv2.setNumThreads(int(config["runtime"]["opencv_num_threads"]))
        _json(output_dir / "environment_manifest.json", _environment_manifest(config))
        identity_error = float(np.linalg.norm(poses_rebased[reference_index] - np.eye(4)))
        selection["reference_identity_error"] = identity_error
        selection["identity_tolerance"] = float(config["reference_selection"]["identity_tolerance"])
        if identity_error > float(config["reference_selection"]["identity_tolerance"]):
            raise Phase1Error(Failure("reference_selection", "reference_not_identity_after_rebase", "C'_r is not identity", records[reference_index]["original_frame_id"], {"error": identity_error}))
        _json(output_dir / "reference_frame_candidates.json", candidates)
        _json(output_dir / "reference_frame_selection.json", selection)
        save_pose_text(output_dir / "camera_pose_original.txt", records, poses_original, "T_world_camera; fixed device initialization")
        save_pose_text(output_dir / "camera_pose_rebased.txt", records, poses_rebased, "T_reference_camera = inverse(T_world_camera(reference)) @ T_world_camera; fixed")

        validity_dir = output_dir / "validity"
        payloads, usage = build_validity(records, config["validity"], validity_dir)
        _usage_csv(output_dir / "frame_usage.csv", usage)
        save_manifest(records, output_dir / "frame_manifest.jsonl")
        proposal_records, proposals_by_frame = build_explicit_proposals(records, config["proposals"], output_dir / "autoseg_masks")
        _json(output_dir / "autoseg_proposals.json", {"identity_policy": "persistent IDs are inferred by mask association; source layer indices are provenance only, never UID semantics", "proposals": proposal_records})

        raw_tracks, filtered_tracks = build_tracks(records, payloads, proposals_by_frame, poses_rebased, config["tracking"])
        save_tracks_npz(output_dir / "tracks_raw.npz", raw_tracks)
        save_tracks_npz(output_dir / "tracks_filtered.npz", filtered_tracks)
        known_motion = KnownMotion(config["known_motion"], poses_original, reference_index, len(records))
        residuals, labels, representative = classify_tracks(filtered_tracks, known_motion, config["classification"])
        clusters = build_clusters(filtered_tracks, labels, representative, config["clustering"])
        split_merge = proposal_split_merge_diagnostics(labels, clusters)
        _json(output_dir / "track_model_residuals.json", {"known_motion": known_motion.source, "records": residuals})
        save_posteriors(output_dir / "track_posteriors.npz", labels)
        _json(output_dir / "track_labels.json", {"labels": labels, "summary": dict(__import__("collections").Counter(item["label"] for item in labels))})
        _json(output_dir / "track_clusters.json", {"clusters": clusters, "split_merge_diagnostics": split_merge})
        for label in ("static", "moving", "unknown"):
            write_label_ply(output_dir / f"{label}_tracks.ply", filtered_tracks, labels, label)

        visualization_dir = output_dir / "visualization"; visualization_dir.mkdir(exist_ok=True)
        draw_validity_overlays(records, payloads, visualization_dir, config["visualization"])
        draw_reference_scores(candidates, reference_index, visualization_dir / "reference_frame_candidates.png", config["visualization"])
        draw_track_label_overlays(records, filtered_tracks, labels, visualization_dir, config["visualization"])
        draw_split_merge_cases(records, filtered_tracks, labels, split_merge, visualization_dir, config["visualization"])

        assertions = {
            "no_hand_observation": all(not obs["hand_mask_flag"] for track in raw_tracks for obs in track["observations"]),
            "reference_can_be_nonzero": True,
            "selected_reference_is_nonzero_for_this_run": reference_index != 0,
            "traceable_original_ids_and_timestamps": all("original_frame_id" in obs and "timestamp" in obs for track in filtered_tracks for obs in track["observations"]),
            "mixed_proposal_split_observed": bool(split_merge["mixed_proposals_split"]),
            "multiple_proposals_merged_into_moving_set": bool(split_merge["moving_set_merges_multiple_proposals"]),
            "unknown_tracks_present": any(item["label"] == "unknown" for item in labels),
            "camera_optimization_run": False, "joint_estimation_run": False, "model_selection_run": False,
            "free_se3_run": False, "tsdf_run": False, "nksr_run": False, "mesh_generated": False,
        }
        summary = {
            "stage": 1, "output_dir": str(output_dir), "frame_count": len(records),
            "reference_frame_id": records[reference_index]["original_frame_id"], "reference_processing_index": reference_index,
            "raw_track_count": len(raw_tracks), "filtered_track_count": len(filtered_tracks),
            "label_counts": dict(__import__("collections").Counter(item["label"] for item in labels)),
            "cluster_count": len(clusters), "known_motion": known_motion.source,
            "assertions": assertions, "forbidden_operations": {"camera_pose_optimization": False, "joint_axis_estimation": False, "joint_type_selection": False, "free_se3": False, "tsdf": False, "nksr": False, "final_mesh": False},
        }
        _json(output_dir / "failure_reasons.json", {"failures": [], "warnings": validation_report.get("warnings", []), "track_unknown_reasons": {str(item["track_id"]): item["unknown_reasons"] for item in labels if item["unknown_reasons"]}})
        _json(output_dir / "stage1_summary.json", summary)
        return summary
    except Exception as exc:
        if isinstance(exc, ValidationErrors):
            failures.extend(item.to_dict() for item in exc.failures)
        elif isinstance(exc, Phase1Error):
            failures.append(exc.failure.to_dict())
        else:
            failures.append({"stage": "unhandled", "code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        _json(output_dir / "failure_reasons.json", {"failures": failures})
        raise
