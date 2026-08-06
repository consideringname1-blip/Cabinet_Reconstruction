"""Independent manual evaluation, coverage, and observability diagnostics."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .classification import classify_tracks


def manual_evaluate(annotation_path: Path, tracks: list[dict], labels: list[dict], config: dict) -> dict:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    label_by_id = {int(item["track_id"]): item["label"] for item in labels}
    by_frame = {}
    for track in tracks:
        for obs in track["observations"]:
            by_frame.setdefault(int(obs["original_frame_id"]), []).append((track["track_id"], obs))
    rows, evaluated, correct, matched_semantic = [], 0, 0, 0
    radius = float(config["max_match_distance_px"])
    for item in annotations["annotations"]:
        uv = np.asarray(item["pixel_uv"], dtype=np.float64)
        candidates = by_frame.get(int(item["original_frame_id"]), [])
        distances = [(float(np.linalg.norm(np.asarray(obs["pixel_uv"])-uv)), track_id, obs) for track_id, obs in candidates]
        distance, track_id, obs = min(distances, default=(float("inf"), None, None))
        predicted = label_by_id.get(track_id) if distance <= radius else None
        expected = item["expected_label"]
        if expected in {"static", "moving", "unknown"}:
            evaluated += 1; matched_semantic += int(predicted is not None); correct += int(predicted == expected)
        if expected == "excluded_hand_or_occlusion":
            evaluated += 1; correct += int(predicted is None)
        rows.append({**item, "matched_track_id": track_id if distance <= radius else None,
                     "match_distance_px": distance if np.isfinite(distance) else None, "predicted_label": predicted,
                     "correct": (predicted == expected) if expected in {"static","moving","unknown"} and predicted is not None else (predicted is None if expected == "excluded_hand_or_occlusion" else None)})
    return {"annotation_file": str(annotation_path.resolve()), "algorithm_used_annotations": False,
            "annotation_count": len(rows), "evaluated_count": evaluated, "correct_count": correct,
            "semantic_point_match_coverage": matched_semantic/max(sum(i["expected_label"] in {"static","moving","unknown"} for i in rows),1),
            "accuracy_on_evaluated": correct/evaluated if evaluated else None, "records": rows}


def audit_gap_associations(annotation_path: Path, associations: list[dict], raw_tracks: list[dict], max_distance_px: float) -> dict:
    annotations=json.loads(annotation_path.read_text(encoding="utf-8"))["annotations"]
    by_frame={}
    for item in annotations:
        if item["expected_label"] in {"static","moving"}: by_frame.setdefault(int(item["original_frame_id"]),[]).append(item)
    tracks={int(track["track_id"]):track for track in raw_tracks}
    audited=[]
    for item in associations:
        if not item["accepted"]: continue
        endpoints=[tracks[item[key]]["observations"][-1 if key=="track_id_before" else 0] for key in ("track_id_before","track_id_after")]
        expected=[]
        for obs in endpoints:
            candidates=by_frame.get(int(obs["original_frame_id"]),[])
            match=min(((np.linalg.norm(np.asarray(a["pixel_uv"])-np.asarray(obs["pixel_uv"])),a) for a in candidates),default=(float("inf"),None),key=lambda x:x[0])
            expected.append(match[1]["expected_label"] if match[0]<=max_distance_px else None)
        if all(expected):
            false=expected[0]!=expected[1]
            audited.append({"track_id_before":item["track_id_before"],"track_id_after":item["track_id_after"],"endpoint_expected_labels":expected,"false_connection":false})
    return {"accepted_connection_count":sum(i["accepted"] for i in associations),"manually_audited_connection_count":len(audited),
            "verified_false_connection_count":sum(i["false_connection"] for i in audited),
            "unaudited_connection_count":sum(i["accepted"] for i in associations)-len(audited),"records":audited,
            "sufficient_to_claim_no_error_increase":len(audited)>=max(3,int(.5*sum(i["accepted"] for i in associations))) and not any(i["false_connection"] for i in audited)}


def _image_coverage(observations: list[dict], width: int, height: int, rows: int, cols: int) -> dict:
    cells = set()
    for obs in observations:
        u, v = obs["pixel_uv"]
        cells.add((min(rows-1, int(v / max(height,1) * rows)), min(cols-1, int(u / max(width,1) * cols))))
    return {"occupied_cells": len(cells), "total_cells": rows*cols, "fraction": len(cells)/(rows*cols), "grid_rows": rows, "grid_cols": cols}


def _information(points: np.ndarray, weights: np.ndarray) -> dict:
    if len(points) == 0:
        return {"eigenvalues": [0.0]*6, "condition_number": None, "rank": 0}
    center = np.average(points, axis=0, weights=np.maximum(weights, 1e-6))
    hessian = np.zeros((6,6), dtype=np.float64)
    for point, weight in zip(points-center, weights):
        x,y,z = point
        skew = np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=np.float64)
        jacobian = np.hstack([np.eye(3), -skew])
        hessian += max(float(weight),1e-6) * jacobian.T @ jacobian
    eigen = np.linalg.eigvalsh(hessian)
    positive = eigen[eigen > 1e-10]
    condition = float(positive.max()/positive.min()) if len(positive) == 6 else None
    return {"eigenvalues": eigen.tolist(), "condition_number": condition, "rank": int(len(positive))}


def _bootstrap(tracks: list[dict], label_by_id: dict[int,str], known_motion, classification_config: dict, config: dict, seed_offset: int) -> dict:
    rng = np.random.default_rng(int(config["random_seed"])+seed_offset)
    stable, tested = 0, 0
    per_track = {}
    for track in tracks:
        observations = track["observations"]
        if len(observations) < int(classification_config["min_valid_observations"]): continue
        outcomes = []
        for _ in range(int(config["bootstrap_samples"])):
            take = max(int(classification_config["min_valid_observations"]), int(np.ceil(len(observations)*float(config["bootstrap_observation_fraction"]))))
            if take > len(observations): continue
            chosen = sorted(rng.choice(len(observations), size=take, replace=False).tolist())
            sample = {**track, "observations": [observations[i] for i in chosen]}
            _, labels, _ = classify_tracks([sample], known_motion, classification_config)
            outcomes.append(labels[0]["label"])
        if outcomes:
            fraction = outcomes.count(label_by_id[track["track_id"]]) / len(outcomes)
            per_track[str(track["track_id"])] = fraction; tested += 1; stable += int(fraction >= float(config["min_track_bootstrap_stability"]))
    return {"tested_tracks": tested, "stable_tracks": stable, "stable_fraction": stable/tested if tested else 0.0,
            "per_track_same_label_fraction": per_track}


def support_report(name: str, tracks: list[dict], labels: list[dict], records: list[dict], known_motion,
                   classification_config: dict, config: dict) -> dict:
    label_by_id = {int(item["track_id"]): item["label"] for item in labels}
    selected = [track for track in tracks if label_by_id[track["track_id"]] == name]
    observations = [obs for track in selected for obs in track["observations"]]
    frame_ids = {obs["original_frame_id"] for obs in observations}
    indices = [obs["processing_index"] for obs in observations]
    points = np.asarray([obs["point_world"] for obs in observations], dtype=np.float64).reshape(-1,3)
    weights = np.asarray([obs.get("derived_depth_quality", obs["depth_confidence"]) for obs in observations], dtype=np.float64)
    import cv2
    image = cv2.imread(records[0]["rgb_path"], cv2.IMREAD_COLOR); height,width=image.shape[:2]
    image_coverage = _image_coverage(observations, width, height, int(config["image_grid_rows"]), int(config["image_grid_cols"]))
    covariance_eigen = np.linalg.eigvalsh(np.cov(points.T)) if len(points) >= 3 else np.zeros(3)
    extents = np.ptp(points, axis=0) if len(points) else np.zeros(3)
    depths = [obs["depth"] for obs in observations]
    representatives = np.asarray([np.median([obs["point_world"] for obs in track["observations"]], axis=0) for track in selected]) if selected else np.empty((0,3))
    baselines = [float(np.linalg.norm(representatives[i]-representatives[j])) for i in range(len(representatives)) for j in range(i)]
    info = _information(points, weights)
    bootstrap = _bootstrap(selected, label_by_id, known_motion, classification_config, config, 0 if name == "static" else 1000)
    reasons = []
    if len(selected) < int(config[f"min_{name}_tracks"]): reasons.append("too_few_tracks")
    if len(frame_ids)/max(len(records),1) < float(config["min_frame_coverage"]): reasons.append("insufficient_frame_coverage")
    if image_coverage["fraction"] < float(config["min_image_grid_coverage"]): reasons.append("concentrated_image_support")
    if len(covariance_eigen) and covariance_eigen[-1] > 0 and covariance_eigen[0]/covariance_eigen[-1] < float(config["min_covariance_eigenvalue_ratio"]): reasons.append("single_plane_or_line_3d_support")
    if info["rank"] < int(config["min_information_rank"]): reasons.append("information_rank_deficient")
    if info["condition_number"] is None or info["condition_number"] > float(config["max_information_condition_number"]): reasons.append("information_ill_conditioned")
    if bootstrap["stable_fraction"] < float(config["min_bootstrap_stable_fraction"]): reasons.append("bootstrap_classification_unstable")
    return {
        "label": name, "track_count": len(selected), "observation_count": len(observations),
        "valid_frame_count": len(frame_ids), "valid_frame_coverage": len(frame_ids)/max(len(records),1),
        "temporal_span_frames": max(indices)-min(indices)+1 if indices else 0, "image_spatial_coverage": image_coverage,
        "world_bbox_extents_m": extents.tolist(), "world_covariance_eigenvalues": covariance_eigen.tolist(),
        "depth_range_m": [min(depths),max(depths)] if depths else None,
        "inter_track_baseline_median_m": float(np.median(baselines)) if baselines else 0.0,
        "inter_track_baseline_max_m": max(baselines) if baselines else 0.0,
        "residual_information_matrix": info, "bootstrap": bootstrap,
        f"{name}_support_degenerate": bool(reasons), "degeneracy_reasons": reasons,
    }


def coverage_report(tracks: list[dict], labels: list[dict], records: list[dict], observability: dict) -> dict:
    counts = Counter(item["label"] for item in labels)
    lengths = [len(track["observations"]) for track in tracks]
    spans = [track["observations"][-1]["processing_index"]-track["observations"][0]["processing_index"]+1 for track in tracks]
    return {"track_count": len(tracks), "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_temporal_span_frames": float(np.mean(spans)) if spans else 0.0, "label_counts": dict(counts),
            "static_spatial_coverage": observability["static"]["image_spatial_coverage"],
            "moving_spatial_coverage": observability["moving"]["image_spatial_coverage"]}


def unknown_distribution(labels: list[dict]) -> dict:
    return dict(Counter(reason for item in labels if item["label"] == "unknown" for reason in item["unknown_reasons"]))


def stage1_metrics(stage1_dir: Path) -> dict:
    raw = np.load(stage1_dir/"tracks_raw.npz"); filtered=np.load(stage1_dir/"tracks_filtered.npz")
    labels=json.loads((stage1_dir/"track_labels.json").read_text())["labels"]
    lengths=np.diff(filtered["track_offsets"]); pidx=filtered["processing_index"]; offsets=filtered["track_offsets"]
    spans=[int(pidx[offsets[i+1]-1]-pidx[offsets[i]]+1) for i in range(len(lengths))]
    return {"raw_track_count": len(raw["track_ids"]), "filtered_track_count": len(filtered["track_ids"]),
            "mean_track_length": float(lengths.mean()) if len(lengths) else 0.0, "mean_temporal_span_frames": float(np.mean(spans)) if spans else 0.0,
            "label_counts": dict(Counter(item["label"] for item in labels)), "unknown_reason_distribution": unknown_distribution(labels)}
