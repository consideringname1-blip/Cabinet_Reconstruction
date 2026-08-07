"""Read-only surface identity, occlusion, and disocclusion audit for Assignment v4."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import yaml

from .diagnostics import save_csv
from .frame_data import load_and_validate, validate_registered_depth_scale
from .projective_models import (CONTRADICTION, OCCLUDED, SUPPORTED, UNOBSERVABLE,
                                evaluate_model, evaluate_prediction, project_world,
                                unproject_pixels)
from .proposals import attach_interaction_proposals
from .region_evidence import deterministic_sample

PREFERENCES = ("static_preferred", "drawer_preferred", "unresolved", "conflicting")
STATE_COLORS = {
    "supported-same-surface": "#2ca02c",
    "supported-surface-switch": "#9467bd",
    "explained-drawer-occlusion": "#ff7f0e",
    "explained-static-occlusion": "#1f77b4",
    "unknown-occlusion": "#8c564b",
    "contradiction": "#d62728",
    "unobservable": "#bdbdbd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, float)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm == 0:
        return np.asarray([0.0, 0.0, 1.0])
    normal = normal / norm
    index = int(np.argmax(np.abs(normal)))
    return -normal if normal[index] < 0 else normal


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    return np.unique(np.floor(np.asarray(points, float) / float(voxel)).astype(np.int64), axis=0)


def surface_descriptor(points: np.ndarray, voxel_sizes: list[float], minimum_points: int = 3) -> dict | None:
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < int(minimum_points):
        return None
    centroid = np.median(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    normal = canonicalize_normal(eigenvectors[:, 0])
    tangent0 = canonicalize_normal(eigenvectors[:, 2])
    tangent1 = np.cross(normal, tangent0)
    tangent1 /= max(np.linalg.norm(tangent1), 1e-12)
    tangent0 = np.cross(tangent1, normal)
    local = np.column_stack((centered @ tangent0, centered @ tangent1, centered @ normal))
    extents = np.percentile(local, 95, axis=0) - np.percentile(local, 5, axis=0)
    voxels = {f"{float(v):.6f}": _voxel_keys(points, float(v)) for v in voxel_sizes}
    return {
        "point_count": int(len(points)), "points": points, "centroid": centroid,
        "eigenvalues": eigenvalues, "normal": normal,
        "tangent_axes": np.stack((tangent0, tangent1)), "extents": extents,
        "bbox_min": points.min(axis=0), "bbox_max": points.max(axis=0),
        "planarity": float(1.0 - eigenvalues[0] / max(eigenvalues.sum(), 1e-12)),
        "voxels": voxels,
    }


def json_descriptor(descriptor: dict | None) -> dict | None:
    if descriptor is None:
        return None
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in descriptor.items() if key not in ("points", "voxels")}


def descriptor_compatibility(source: dict, target: dict, cfg: dict, voxel_m: float) -> dict:
    dot = float(np.clip(abs(np.dot(source["normal"], target["normal"])), 0.0, 1.0))
    normal_angle = float(np.degrees(np.arccos(dot)))
    delta = target["centroid"] - source["centroid"]
    plane_offset = float(abs(np.dot(delta, source["normal"])))
    centroid_distance = float(np.linalg.norm(delta))
    source_extent = np.maximum(source["extents"][:2], 1e-6)
    target_extent = np.maximum(target["extents"][:2], 1e-6)
    extent_compatibility = float(np.mean(np.minimum(source_extent, target_extent) /
                                         np.maximum(source_extent, target_extent)))
    radius = float(cfg["overlap_radius_voxels"]) * float(voxel_m)
    source_points, target_points = source["points"], target["points"]
    source_to_target = float(np.mean(cKDTree(target_points).query(source_points)[0] <= radius))
    target_to_source = float(np.mean(cKDTree(source_points).query(target_points)[0] <= radius))
    symmetric_overlap = float(math.sqrt(source_to_target * target_to_source))
    source_keys = {tuple(row) for row in source["voxels"][f"{float(voxel_m):.6f}"]}
    target_keys = {tuple(row) for row in target["voxels"][f"{float(voxel_m):.6f}"]}
    union = source_keys | target_keys
    jaccard = float(len(source_keys & target_keys) / max(len(union), 1))
    normal_penalty = min(normal_angle / float(cfg["normal_scale_degrees"]), 1.0)
    plane_penalty = min(plane_offset / float(cfg["plane_offset_scale_m"]), 1.0)
    centroid_penalty = min(centroid_distance / float(cfg["centroid_scale_m"]), 1.0)
    extent_penalty = 1.0 - extent_compatibility
    overlap_penalty = 1.0 - symmetric_overlap
    weights = cfg["weights"]
    cost = (float(weights["normal"]) * normal_penalty +
            float(weights["plane_offset"]) * plane_penalty +
            float(weights["centroid"]) * centroid_penalty +
            float(weights["extent"]) * extent_penalty +
            float(weights["overlap"]) * overlap_penalty)
    return {
        "normal_angle_degrees": normal_angle, "plane_offset_m": plane_offset,
        "centroid_distance_m": centroid_distance, "extent_compatibility": extent_compatibility,
        "source_to_target_voxel_overlap": source_to_target,
        "target_to_source_voxel_overlap": target_to_source,
        "symmetric_voxel_overlap": symmetric_overlap, "voxel_jaccard": jaccard,
        "cost": float(cost), "score": float(1.0 - min(cost, 1.0)),
    }


def proposal_memberships(uv: np.ndarray, selected: np.ndarray, proposals: list[dict]) -> dict:
    """Return overlaps and entropy without treating overlapping masks as independent samples."""
    uv = np.asarray(uv, np.int64); selected = np.asarray(selected, bool)
    ids = np.flatnonzero(selected)
    all_rows = []
    fractional = defaultdict(float)
    point_memberships = []
    for point_id in ids:
        u, v = uv[point_id]
        memberships = [p for p in proposals if 0 <= v < p["mask"].shape[0] and 0 <= u < p["mask"].shape[1]
                       and p["mask"][v, u]]
        point_memberships.append((point_id, memberships))
        if memberships:
            weight = 1.0 / len(memberships)
            for proposal in memberships:
                fractional[proposal["proposal_id"]] += weight
    for proposal in proposals:
        count = sum(any(p["proposal_id"] == proposal["proposal_id"] for p in memberships)
                    for _, memberships in point_memberships)
        if count:
            all_rows.append({"proposal_id": proposal["proposal_id"],
                             "source_layer": int(proposal["source_layer"]), "hit_count": int(count),
                             "fractional_hit_count": float(fractional[proposal["proposal_id"]])})
    no_proposal = sum(not memberships for _, memberships in point_memberships)
    weights = np.asarray(list(fractional.values()) + ([float(no_proposal)] if no_proposal else []), float)
    probabilities = weights / max(weights.sum(), 1.0)
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    normalized_entropy = float(entropy / max(np.log(len(probabilities)), 1e-12)) if len(probabilities) > 1 else 0.0
    dominant = max(all_rows, key=lambda row: (row["hit_count"], row["proposal_id"])) if all_rows else None
    return {"all_memberships": all_rows, "dominant": dominant,
            "no_proposal_support_count": int(no_proposal), "support_split_entropy": normalized_entropy,
            "unique_supported_point_count": int(len(ids))}


def select_continuity_chain(candidate_frames: list[list[dict]], cfg: dict) -> list[dict | None]:
    """Deterministic DP over current-frame geometry; proposal IDs are not transition features."""
    selected: list[dict | None] = [None] * len(candidate_frames)
    active = [index for index, candidates in enumerate(candidate_frames) if candidates]
    if not active:
        return selected
    previous_costs = None; previous_candidates = None; back = []
    for frame_index in active:
        candidates = candidate_frames[frame_index]
        emission = np.asarray([c["source_compatibility"]["cost"] +
                               float(cfg["support_emission_weight"]) * (1.0 - c["supported_fraction"])
                               for c in candidates])
        if previous_costs is None:
            costs, parents = emission, np.full(len(candidates), -1, int)
        else:
            costs = np.full(len(candidates), np.inf); parents = np.full(len(candidates), -1, int)
            for j, candidate in enumerate(candidates):
                transition = np.asarray([descriptor_compatibility(prior["descriptor"], candidate["descriptor"], cfg,
                                                                   float(candidate["voxel_m"]))["cost"]
                                         for prior in previous_candidates])
                total = previous_costs + float(cfg["transition_weight"]) * transition
                parents[j] = int(np.argmin(total)); costs[j] = float(total[parents[j]] + emission[j])
        back.append((frame_index, parents, candidates)); previous_costs, previous_candidates = costs, candidates
    choice = int(np.argmin(previous_costs))
    for frame_index, parents, candidates in reversed(back):
        selected[frame_index] = candidates[choice]
        choice = int(parents[choice]) if parents[choice] >= 0 else 0
    return selected


def surface_switches(selected: list[dict | None], cfg: dict) -> dict:
    switches = np.zeros(len(selected), bool); transitions = []
    prior_index = None
    for index, item in enumerate(selected):
        if item is None:
            continue
        if prior_index is not None:
            prior = selected[prior_index]
            metrics = descriptor_compatibility(prior["descriptor"], item["descriptor"], cfg, float(item["voxel_m"]))
            switched = (metrics["normal_angle_degrees"] > float(cfg["switch_normal_degrees"]) or
                        metrics["plane_offset_m"] > float(cfg["switch_plane_offset_m"]) or
                        metrics["centroid_distance_m"] > float(cfg["switch_centroid_m"]))
            switches[index] = switched
            transitions.append({"from_index": prior_index, "to_index": index, "switched": bool(switched), **metrics})
        prior_index = index
    compatible = sum(item is not None for item in selected)
    runs = []; current = 0
    for index, item in enumerate(selected):
        if item is not None and not switches[index]: current += 1
        else:
            if current: runs.append(current)
            current = 1 if item is not None else 0
    if current: runs.append(current)
    return {"flags": switches, "transitions": transitions, "switch_count": int(switches.sum()),
            "switch_fraction": float(switches.sum() / max(compatible - 1, 1)),
            "longest_continuous_surface_run": int(max(runs, default=0))}


def attribute_occlusion(evidence: dict, trusted_drawer: np.ndarray,
                        trusted_static: np.ndarray, unusable: np.ndarray | None = None) -> dict:
    uv = np.asarray(evidence["uv"], np.int64); status = np.asarray(evidence["status"], np.uint8)
    occluded = status == OCCLUDED
    labels = np.full(len(status), "not-occluded", object)
    depth_gaps = evidence["predicted_depth_m"] - evidence["observed_depth_m"]
    h, w = trusted_drawer.shape
    inside = occluded & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(inside)
    for index in ids:
        u, v = uv[index]
        if unusable is not None and unusable[v, u]:
            labels[index] = "unusable_occlusion"
        elif trusted_drawer[v, u]:
            labels[index] = "explained_occluded_by_trusted_drawer"
        elif trusted_static[v, u]:
            labels[index] = "explained_occluded_by_conservative_static"
        else:
            labels[index] = "occluded_by_unknown"
    counts = Counter(labels[occluded])
    total = int(occluded.sum())
    return {"labels": labels, "counts": dict(counts),
            "fractions": {key: float(value / max(total, 1)) for key, value in counts.items()},
            "depth_gap_median": float(np.nanmedian(depth_gaps[occluded])) if total else None,
            "depth_gap_p90": float(np.nanpercentile(depth_gaps[occluded], 90)) if total else None}


def contradiction_run(rows: list[dict], cfg: dict) -> dict:
    flags = np.asarray([float(row["contradiction_fraction"]) >= float(cfg["per_frame_fraction_threshold"])
                        and float(row["contradiction_spatial_coherence_fraction"]) >=
                        float(cfg["minimum_spatial_coherence_fraction"]) for row in rows], bool)
    best = []; current = []
    for index, flag in enumerate(flags):
        if flag and (not current or int(rows[index]["target_frame_id"]) == int(rows[current[-1]]["target_frame_id"]) + 1):
            current.append(index)
        else:
            if len(current) > len(best): best = current
            current = [index] if flag else []
    if len(current) > len(best): best = current
    q_span = float(abs(rows[best[-1]]["target_q_m"] - rows[best[0]]["target_q_m"])) if len(best) > 1 else 0.0
    accepted = len(best) >= int(cfg["minimum_consecutive_frames"]) and q_span >= float(cfg["minimum_q_span_m"])
    interval = None if not best else {
        "start_frame_id": int(rows[best[0]]["target_frame_id"]),
        "end_frame_id": int(rows[best[-1]]["target_frame_id"]),
        "start_q_m": float(rows[best[0]]["target_q_m"]),
        "end_q_m": float(rows[best[-1]]["target_q_m"]),
    }
    return {"flags": flags, "longest_consecutive_contradiction_run": int(len(best)),
            "q_span_of_longest_run_m": q_span, "maximum_contradiction_interval": interval,
            "coherent_contradiction_run": bool(accepted)}


def spatial_coherence(uv: np.ndarray, selected: np.ndarray, shape: tuple[int, int], dilation: int) -> float:
    points = np.asarray(uv, np.int64)[np.asarray(selected, bool)]
    if not len(points): return 0.0
    mask = np.zeros(shape, np.uint8)
    inside = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < shape[0])
    points = points[inside]; mask[points[:, 1], points[:, 0]] = 1
    if dilation:
        size = 2 * int(dilation) + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    count, labels = cv2.connectedComponents(mask)
    if count <= 1: return 0.0
    point_labels = labels[points[:, 1], points[:, 0]]
    largest = max(Counter(point_labels).values(), default=0)
    return float(largest / max(len(points), 1))


def temporal_cues(rows_by_model: dict[str, list[dict]], cfg: dict) -> dict:
    static_rows = sorted(rows_by_model["static"], key=lambda row: float(row["target_q_m"]))
    drawer_rows = sorted(rows_by_model["drawer"], key=lambda row: float(row["target_q_m"]))
    n = len(static_rows); early_n = max(1, int(math.ceil(n * float(cfg["early_q_fraction"]))))
    late_n = max(1, int(math.ceil(n * float(cfg["late_q_fraction"]))))
    early = static_rows[:early_n]; late = static_rows[-late_n:]
    early_drawer = float(np.mean([row["explained_drawer_occlusion_fraction"] for row in early])) if early else 0.0
    late_support = float(np.mean([row["selected_same_surface_supported"] for row in late])) if late else 0.0
    sequence = np.asarray([row["selected_same_surface_supported"] - row["explained_drawer_occlusion_fraction"]
                           for row in static_rows], float)
    monotonicity = float(np.corrcoef(np.arange(len(sequence)), sequence)[0, 1]) if len(sequence) > 1 and np.std(sequence) > 0 else 0.0
    transitions = [row for row in static_rows if row["selected_same_surface_supported"]]
    transition_q = float(transitions[0]["target_q_m"]) if transitions else None
    static_score = float(np.mean([early_drawer, late_support, max(monotonicity, 0.0)]))
    strong_static = (early_drawer >= float(cfg["minimum_early_drawer_occluded_fraction"]) and
                     late_support >= float(cfg["minimum_late_same_surface_support_fraction"]) and
                     monotonicity >= float(cfg["minimum_temporal_monotonicity"]))
    drawer_attachment = float(np.mean([row["selected_same_surface_supported"] for row in drawer_rows])) if drawer_rows else 0.0
    return {"static_disocclusion_score": static_score,
            "early_drawer_occluded_fraction": early_drawer, "transition_q_m": transition_q,
            "late_same_surface_support_fraction": late_support, "temporal_monotonicity": monotonicity,
            "strong_static_disocclusion": bool(strong_static),
            "drawer_attachment_score": drawer_attachment,
            "strong_drawer_attachment": bool(drawer_attachment >= float(cfg["minimum_drawer_attachment_score"]))}


def experimental_preference(static_metrics: dict, drawer_metrics: dict, temporal: dict,
                            contradiction: dict[str, dict], cfg: dict) -> dict:
    advantage = float(static_metrics["continuity_score_median"] - drawer_metrics["continuity_score_median"])
    threshold = float(cfg["continuity_advantage_threshold"])
    cues = {"static": [], "drawer": []}
    if advantage >= threshold: cues["static"].append("surface_continuity_advantage")
    if advantage <= -threshold: cues["drawer"].append("surface_continuity_advantage")
    if temporal["strong_static_disocclusion"]: cues["static"].append("explained_static_disocclusion")
    if temporal["strong_drawer_attachment"]: cues["drawer"].append("drawer_attachment")
    if contradiction["drawer"]["coherent_contradiction_run"]: cues["static"].append("opposite_drawer_contradiction_run")
    if contradiction["static"]["coherent_contradiction_run"]: cues["drawer"].append("opposite_static_contradiction_run")
    minimum = int(cfg["minimum_independent_cues"])
    static_ready, drawer_ready = len(cues["static"]) >= minimum, len(cues["drawer"]) >= minimum
    if static_ready and drawer_ready: preference, reason = "conflicting", "independent cue sets support both hypotheses"
    elif static_ready: preference, reason = "static_preferred", "at least two independent static cues"
    elif drawer_ready: preference, reason = "drawer_preferred", "at least two independent drawer cues"
    else: preference, reason = "unresolved", "fewer than two independent aligned cues"
    return {"experimental_identity_preference": preference, "cues_used": cues,
            "unresolved_reason": reason if preference in ("unresolved", "conflicting") else None,
            "continuity_advantage_static_minus_drawer": advantage}


def _proposal_points(frame: dict, proposal: dict, context: dict, cfg: dict, model: str) -> np.ndarray:
    mask = np.asarray(proposal["eroded_mask"], bool) & frame["valid"]
    uv = deterministic_sample(mask, int(cfg["maximum_points"]))
    if not len(uv): return np.empty((0, 3))
    world = unproject_pixels(uv, frame["depth"][uv[:, 1], uv[:, 0]], frame["pose"], context["intrinsic"])
    return world if model == "static" else world - float(frame["q"]) * context["axis"]


def _descriptor(frame: dict, proposal: dict, context: dict, cfg: dict, model: str,
                cache: dict) -> dict | None:
    key = (int(frame["source"]), proposal["proposal_id"], model)
    if key not in cache:
        cache[key] = surface_descriptor(_proposal_points(frame, proposal, context, cfg, model),
                                        [float(v) for v in cfg["voxel_sizes_m"]],
                                        int(cfg["minimum_valid_points"]))
    return cache[key]


def build_trusted_occluders(context: dict, base_cfg: dict, audit_cfg: dict,
                            v4: Path, frames: list[dict]) -> tuple[dict, dict]:
    seed_manifest = json.loads((v4 / "sam2_seed_manifest.json").read_text())
    front = [row for row in seed_manifest["seeds"] if row["seed_kind"] == "front"]
    if len(front) != int(audit_cfg["trusted_occluders"]["front_seed_strict_agreement"]):
        raise RuntimeError(f"expected audited front seed count missing: {len(front)}")
    interaction_start = int(context["phases"]["interaction"][0])
    drawer_masks = {}; strict_masks = {}
    for frame in frames:
        combined = int(frame["source"]) - interaction_start
        masks = []
        for seed in front:
            path = v4 / "sam2_propagation_per_seed" / seed["seed_id"] / f"{combined:06d}.npy"
            if not path.is_file(): raise FileNotFoundError(path)
            masks.append(np.load(path).astype(bool))
        count = np.stack(masks).sum(axis=0)
        drawer_masks[frame["source"]] = count >= int(audit_cfg["trusted_occluders"]["front_seed_minimum_agreement"])
        strict_masks[frame["source"]] = count >= int(audit_cfg["trusted_occluders"]["front_seed_strict_agreement"])
    import open3d as o3d
    static_core = np.asarray(o3d.io.read_point_cloud(str(Path(base_cfg["inputs"]["v2_dir"]) / "static_core.ply")).points)
    static_cfg = dict(base_cfg["projective_evidence"])
    static_cfg["depth_support_threshold_m"] = float(audit_cfg["trusted_occluders"]["static_core_depth_support_threshold_m"])
    static_masks = {}
    for frame in frames:
        evidence = evaluate_prediction(static_core, frame, context["intrinsic"], static_cfg)
        mask = np.zeros(frame["depth"].shape, bool); ids = np.flatnonzero(evidence["status"] == SUPPORTED)
        if len(ids):
            uv = evidence["uv"][ids]; mask[uv[:, 1], uv[:, 0]] = True
        static_masks[frame["source"]] = mask
    report = {"front_seed_count": len(front), "front_seeds": front,
              "primary_agreement_required": int(audit_cfg["trusted_occluders"]["front_seed_minimum_agreement"]),
              "strict_agreement_required": int(audit_cfg["trusted_occluders"]["front_seed_strict_agreement"]),
              "payload_seeds_used": False, "sam2_rerun": False,
              "per_frame": [{"original_frame_id": int(frame["source"]),
                              "trusted_front_pixels_3of4": int(drawer_masks[frame["source"]].sum()),
                              "trusted_front_pixels_4of4": int(strict_masks[frame["source"]].sum()),
                              "trusted_static_core_pixels": int(static_masks[frame["source"]].sum())}
                             for frame in frames]}
    return {"drawer": drawer_masks, "drawer_strict": strict_masks, "static": static_masks}, report


def select_controls(evidence: list[dict], frames_by_source: dict, v4: Path,
                    context: dict, cfg: dict) -> dict[str, list[dict]]:
    by_key = {(int(item["original_frame_id"]), item["proposal_id"]): item for item in evidence}
    seed_manifest = json.loads((v4 / "sam2_seed_manifest.json").read_text())
    front = [by_key[(int(row["original_frame_id"]), row["proposal_id"])] for row in seed_manifest["seeds"]
             if row["seed_kind"] == "front"]
    front = sorted(front, key=lambda item: -(item["drawer"]["score"] - item["static"]["score"]))[:int(cfg["maximum_drawer_front_controls"])]
    comoving = [by_key[(int(row["original_frame_id"]), row["proposal_id"])] for row in seed_manifest["seeds"]
                if row["seed_kind"] == "revealed"]
    comoving = sorted(comoving, key=lambda item: -(item["drawer"]["score"] - item["static"]["score"]))[:int(cfg["maximum_comoving_controls"])]
    static = []
    for item in evidence:
        if item["label"] != "static" or float(item["seed_overlap_ratio"]) != 0.0: continue
        frame = frames_by_source[int(item["original_frame_id"])]
        proposal = next(p for p in frame["proposals"] if p["proposal_id"] == item["proposal_id"])
        local = int(frame["local_index"]); seed = context["moving_labels"][local] == 2
        distance = distance_transform_edt(~seed)
        values = distance[np.asarray(proposal["eroded_mask"], bool) & frame["valid"]]
        minimum_distance = float(values.min()) if len(values) else 0.0
        if minimum_distance < float(cfg["minimum_static_seed_distance_pixels"]): continue
        copied = dict(item); copied["moving_seed_distance_pixels"] = minimum_distance; static.append(copied)
    static.sort(key=lambda item: (-(item["static"]["score"] - item["drawer"]["score"]),
                                  -item["drawer"]["contradiction_ratio_median"], item["proposal_id"]))
    return {"drawer_front": front, "static": static[:int(cfg["maximum_static_controls"])], "comoving": comoving}


def _metrics(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, float)
    return ((float(np.median(array)), float(np.percentile(array, 25))) if len(array) else (0.0, 0.0))


def analyze_region(item: dict, frame: dict, proposal: dict, frames_by_source: dict,
                   context: dict, base_cfg: dict, audit_cfg: dict, occluders: dict,
                   descriptor_cache: dict) -> dict:
    descriptor_cfg = audit_cfg["surface_descriptor"]; continuity_cfg = {**audit_cfg["continuity"],
        "overlap_radius_voxels": descriptor_cfg["overlap_radius_voxels"]}
    mask = np.asarray(proposal["eroded_mask"], bool) & frame["valid"]
    uv = deterministic_sample(mask, int(base_cfg["region_evidence"]["maximum_sampled_points_per_region"]))
    world = unproject_pixels(uv, frame["depth"][uv[:, 1], uv[:, 0]], frame["pose"], context["intrinsic"])
    target_ids = [int(row["target_frame_id"]) for row in item["per_target"]]
    model_data = {}; rows_by_model = {}; contradiction = {}
    for model in ("static", "drawer"):
        source_descriptor = _descriptor(frame, proposal, context, descriptor_cfg, model, descriptor_cache)
        if source_descriptor is None: raise RuntimeError(f"source descriptor unavailable: {item['proposal_id']}")
        candidates_by_frame = []; evaluations = []; membership_by_frame = []
        for target_id in target_ids:
            target = frames_by_source[target_id]
            evidence_model = evaluate_model(world, float(frame["q"]), target, float(target["q"]), context["axis"],
                                            context["intrinsic"], model, base_cfg["region_evidence"])
            memberships = proposal_memberships(evidence_model["uv"], evidence_model["status"] == SUPPORTED, target["proposals"])
            candidate_rows = []
            minimum_count = int(continuity_cfg["minimum_supported_membership_points"])
            minimum_fraction = float(continuity_cfg["minimum_supported_membership_fraction"])
            for membership in memberships["all_memberships"]:
                fraction = float(membership["hit_count"] / max(memberships["unique_supported_point_count"], 1))
                if membership["hit_count"] < minimum_count or fraction < minimum_fraction: continue
                target_proposal = next(p for p in target["proposals"] if p["proposal_id"] == membership["proposal_id"])
                descriptor = _descriptor(target, target_proposal, context, descriptor_cfg, model, descriptor_cache)
                if descriptor is None: continue
                candidate_rows.append({**membership, "descriptor": descriptor,
                                       "descriptor_json": json_descriptor(descriptor), "supported_fraction": fraction,
                                       "voxel_m": float(descriptor_cfg["primary_voxel_m"]),
                                       "source_compatibility": descriptor_compatibility(source_descriptor, descriptor, continuity_cfg,
                                                                                         float(descriptor_cfg["primary_voxel_m"]))})
            candidate_rows.sort(key=lambda row: (row["source_compatibility"]["cost"], -row["hit_count"], row["proposal_id"]))
            candidates_by_frame.append(candidate_rows); evaluations.append(evidence_model); membership_by_frame.append(memberships)
        selected = select_continuity_chain(candidates_by_frame, continuity_cfg)
        switch = surface_switches(selected, continuity_cfg)
        rows = []
        for index, (target_id, evidence_model, memberships, chosen) in enumerate(zip(target_ids, evaluations, membership_by_frame, selected)):
            target = frames_by_source[target_id]
            attribution = attribute_occlusion(evidence_model, occluders["drawer"][target_id], occluders["static"][target_id], ~target["valid"])
            status = np.asarray(evidence_model["status"])
            contradiction_fraction = float(np.mean(status == CONTRADICTION))
            coherence = spatial_coherence(evidence_model["uv"], status == CONTRADICTION, target["depth"].shape,
                                          int(audit_cfg["contradiction"]["spatial_dilation_pixels"]))
            same = bool(chosen is not None and not switch["flags"][index] and
                        chosen["source_compatibility"]["score"] >= float(continuity_cfg["compatible_score_threshold"]))
            total = max(len(status), 1)
            drawer_occ = attribution["counts"].get("explained_occluded_by_trusted_drawer", 0) / total
            static_occ = attribution["counts"].get("explained_occluded_by_conservative_static", 0) / total
            unknown_occ = attribution["counts"].get("occluded_by_unknown", 0) / total
            if contradiction_fraction >= float(audit_cfg["contradiction"]["per_frame_fraction_threshold"]): state = "contradiction"
            elif same: state = "supported-same-surface"
            elif chosen is not None: state = "supported-surface-switch"
            elif drawer_occ >= max(static_occ, unknown_occ) and drawer_occ > 0: state = "explained-drawer-occlusion"
            elif static_occ >= max(drawer_occ, unknown_occ) and static_occ > 0: state = "explained-static-occlusion"
            elif unknown_occ > 0: state = "unknown-occlusion"
            else: state = "unobservable"
            rows.append({"source_frame_id": int(frame["source"]), "source_q_m": float(frame["q"]),
                         "source_proposal_id": item["proposal_id"], "hypothesis": model,
                         "target_frame_id": target_id, "target_q_m": float(target["q"]), "state": state,
                         "support_ratio": float(evidence_model["support_ratio"]),
                         "contradiction_ratio_testable": float(evidence_model["contradiction_ratio"]),
                         "contradiction_fraction": contradiction_fraction,
                         "contradiction_spatial_coherence_fraction": coherence,
                         "observable_fraction": float(evidence_model["observable_fraction"]),
                         "all_proposal_memberships": memberships["all_memberships"],
                         "dominant_target_proposal": memberships["dominant"],
                         "no_proposal_support_count": memberships["no_proposal_support_count"],
                         "support_split_entropy": memberships["support_split_entropy"],
                         "selected_target_proposal": None if chosen is None else chosen["proposal_id"],
                         "selected_target_descriptor": None if chosen is None else chosen["descriptor_json"],
                         "selected_compatibility": None if chosen is None else chosen["source_compatibility"],
                         "selected_same_surface_supported": same, "surface_switch": bool(switch["flags"][index]),
                         "occlusion_counts": attribution["counts"], "occlusion_fractions": attribution["fractions"],
                         "occlusion_depth_gap_median": attribution["depth_gap_median"],
                         "explained_drawer_occlusion_fraction": float(drawer_occ),
                         "explained_static_occlusion_fraction": float(static_occ),
                         "unknown_occlusion_fraction": float(unknown_occ)})
        scores = [row["selected_compatibility"]["score"] for row in rows if row["selected_compatibility"] is not None]
        overlaps = [row["selected_compatibility"]["symmetric_voxel_overlap"] for row in rows if row["selected_compatibility"] is not None]
        normals = [row["selected_compatibility"]["normal_angle_degrees"] for row in rows if row["selected_compatibility"] is not None]
        offsets = [row["selected_compatibility"]["plane_offset_m"] for row in rows if row["selected_compatibility"] is not None]
        score_median, score_p25 = _metrics(scores); overlap_median, overlap_p25 = _metrics(overlaps)
        model_data[model] = {"source_descriptor": json_descriptor(source_descriptor),
                             "compatible_target_frames": len(scores),
                             "unique_dominant_target_descriptors": (1 + int(switch["switch_count"])) if scores else 0,
                             **{key: value for key, value in switch.items() if key != "flags"},
                             "continuity_score_median": score_median, "continuity_score_p25": score_p25,
                             "voxel_overlap_median": overlap_median, "voxel_overlap_p25": overlap_p25,
                             "normal_error_median": float(np.median(normals)) if normals else None,
                             "plane_offset_median": float(np.median(offsets)) if offsets else None}
        rows_by_model[model] = rows
        contradiction[model] = contradiction_run(rows, audit_cfg["contradiction"])
    temporal = temporal_cues(rows_by_model, audit_cfg["temporal_cues"])
    preference = experimental_preference(model_data["static"], model_data["drawer"], temporal, contradiction,
                                         {**audit_cfg["preference"], **audit_cfg["continuity"]})
    return {"source_metadata": {"original_frame_id": int(frame["source"]), "q_m": float(frame["q"]),
                                 "proposal_id": item["proposal_id"], "source_layer": int(item["source_layer"])},
            "original_v4_evidence": {key: item[key] for key in ("label", "reason", "seed_overlap_ratio", "static", "drawer")},
            "static": model_data["static"], "drawer": model_data["drawer"], "per_target": rows_by_model,
            "contradiction_runs": {model: {key: value for key, value in data.items() if key != "flags"}
                                   for model, data in contradiction.items()},
            "temporal_cues": temporal, **preference}


def _source_panel(frame: dict, proposal: dict, result: dict) -> np.ndarray:
    image = frame["rgb"].copy(); mask = proposal["mask"]
    tint = np.full_like(image, (30, 210, 230)); image[mask] = cv2.addWeighted(image, .35, tint, .65, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (20, 255, 255), 2)
    header = np.zeros((58, image.shape[1], 3), np.uint8)
    lines = [f"source frame={frame['source']} q={frame['q']:.4f}m proposal={proposal['proposal_id']}",
             f"experimental={result['experimental_identity_preference']} (v4 remains unknown)"]
    for index, text in enumerate(lines):
        cv2.putText(header, text, (4, 18 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def q_time_strip(result: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 3.4))
    for row_index, model in enumerate(("STATIC", "DRAWER")):
        rows = result["per_target"][model.lower()]
        for row in rows:
            axis.scatter(float(row["target_q_m"]), row_index, marker="s", s=120,
                         color=STATE_COLORS[row["state"]], edgecolors="black", linewidths=.25)
            axis.annotate(str(row["target_frame_id"]), (float(row["target_q_m"]), row_index),
                          xytext=(0, 8 if row_index else -12), textcoords="offset points", ha="center", fontsize=5, rotation=90)
    axis.set_yticks([0, 1], ["STATIC", "DRAWER"]); axis.set_xlabel("target q (m); number is original frame ID")
    axis.set_ylim(-.55, 1.55); axis.grid(axis="x", alpha=.2)
    source = result["source_metadata"]
    axis.set_title(f"q-time strip | source frame {source['original_frame_id']} q={source['q_m']:.4f} | {source['proposal_id']}")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=name, markersize=7)
               for name, color in STATE_COLORS.items()]
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.24), ncol=4, fontsize=7)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def continuity_plot(result: dict, model: str, path: Path) -> None:
    source = np.asarray(result[model]["source_descriptor"]["centroid"])
    rows = [row for row in result["per_target"][model] if row["selected_target_descriptor"] is not None]
    figure = plt.figure(figsize=(10, 4.8)); axis3d = figure.add_subplot(121, projection="3d"); axis2d = figure.add_subplot(122)
    axis3d.scatter(*source, marker="*", s=150, color="black", label="source surface centroid")
    if rows:
        centroids = np.asarray([row["selected_target_descriptor"]["centroid"] for row in rows]); q = np.asarray([row["target_q_m"] for row in rows])
        shown = axis3d.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=q, cmap="viridis", s=35, label="selected target surfaces")
        figure.colorbar(shown, ax=axis3d, shrink=.65, label="target q (m)")
        axis2d.plot(q, [row["selected_compatibility"]["score"] for row in rows], "o-", label="source compatibility")
        switched = [row for row in rows if row["surface_switch"]]
        if switched: axis2d.scatter([row["target_q_m"] for row in switched], [row["selected_compatibility"]["score"] for row in switched], color="red", marker="x", s=60, label="3D descriptor switch")
    axis3d.set_xlabel("model X"); axis3d.set_ylabel("model Y"); axis3d.set_zlabel("model Z"); axis3d.legend(fontsize=7)
    axis2d.set_xlabel("target q (m)"); axis2d.set_ylabel("continuity score"); axis2d.set_ylim(-.03, 1.03); axis2d.grid(alpha=.25)
    handles, labels = axis2d.get_legend_handles_labels()
    if handles: axis2d.legend(handles, labels, fontsize=8)
    source_meta = result["source_metadata"]
    figure.suptitle(f"{model.upper()} model-frame continuity | source frame {source_meta['original_frame_id']} q={source_meta['q_m']:.4f} | {source_meta['proposal_id']}")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def surface_switch_plot(result: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for axis, model in zip(axes, ("static", "drawer")):
        rows = result["per_target"][model]; q = [row["target_q_m"] for row in rows]
        scores = [np.nan if row["selected_compatibility"] is None else row["selected_compatibility"]["score"] for row in rows]
        axis.plot(q, scores, "o-", color="#2c7fb8", label="selected descriptor continuity")
        switched = [row for row in rows if row["surface_switch"]]
        if switched: axis.scatter([row["target_q_m"] for row in switched], [row["selected_compatibility"]["score"] for row in switched], marker="x", color="red", s=70, label="3D discontinuity switch")
        axis.set_ylabel(f"{model.upper()} score"); axis.set_ylim(-.03, 1.03); axis.grid(alpha=.25); axis.legend(fontsize=8)
    axes[-1].set_xlabel("target q (m)")
    source = result["source_metadata"]
    figure.suptitle(f"Surface switching (UID-independent) | source frame {source['original_frame_id']} q={source['q_m']:.4f} | {source['proposal_id']}")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def occlusion_panel(result: dict, frame: dict, proposal: dict, frames_by_source: dict,
                    context: dict, base_cfg: dict, occluders: dict, path: Path) -> None:
    mask = proposal["eroded_mask"] & frame["valid"]
    uv = deterministic_sample(mask, int(base_cfg["region_evidence"]["maximum_sampled_points_per_region"]))
    world = unproject_pixels(uv, frame["depth"][uv[:, 1], uv[:, 0]], frame["pose"], context["intrinsic"])
    panels = []
    for model in ("static", "drawer"):
        rows = result["per_target"][model]
        selected_row = max(rows, key=lambda row: sum(row["occlusion_counts"].values()))
        target = frames_by_source[int(selected_row["target_frame_id"])]
        evidence = evaluate_model(world, frame["q"], target, target["q"], context["axis"], context["intrinsic"], model, base_cfg["region_evidence"])
        image = target["rgb"].copy(); drawer = occluders["drawer"][target["source"]]; static = occluders["static"][target["source"]]
        overlay = image.copy(); overlay[drawer] = (0, 140, 255); overlay[static] = (255, 120, 20); image = cv2.addWeighted(image, .65, overlay, .35, 0)
        ids = np.flatnonzero(evidence["status"] == OCCLUDED)
        for index in ids:
            u, v = evidence["uv"][index]
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(image, (int(u), int(v)), 2, (20, 20, 240), -1)
                cv2.circle(image, (int(u), int(v)), 4, (255, 255, 255), 1)
        header = np.zeros((54, image.shape[1], 3), np.uint8)
        cv2.putText(header, f"{model.upper()} target frame={target['source']} q={target['q']:.4f}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(header, "red/white=predicted point behind observed front; orange=trusted drawer; blue=static core", (4, 39), cv2.FONT_HERSHEY_SIMPLEX, .27, (220,220,220), 1, cv2.LINE_AA)
        panels.append(np.vstack((header, image)))
    cv2.imwrite(str(path), np.hstack(panels))


def save_region(result: dict, directory: Path, frame: dict, proposal: dict, frames_by_source: dict,
                context: dict, base_cfg: dict, occluders: dict) -> None:
    directory.mkdir(parents=True)
    (directory / "region_identity_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    flat_rows = []
    for model in ("static", "drawer"):
        for row in result["per_target"][model]:
            flat_rows.append({key: (json.dumps(value) if isinstance(value, (dict, list)) else value) for key, value in row.items()})
    save_csv(directory / "per_target_identity_evidence.csv", flat_rows)
    source = _source_panel(frame, proposal, result); cv2.imwrite(str(directory / "source_region.jpg"), source)
    q_time_strip(result, directory / "q_time_strip.png")
    continuity_plot(result, "static", directory / "continuity_world_static.png")
    continuity_plot(result, "drawer", directory / "continuity_drawer_canonical.png")
    surface_switch_plot(result, directory / "surface_switch.png")
    occlusion_panel(result, frame, proposal, frames_by_source, context, base_cfg, occluders, directory / "occlusion_explanation.jpg")
    q_image = cv2.imread(str(directory / "q_time_strip.png")); scaled = cv2.resize(source, (int(source.shape[1] * q_image.shape[0] / source.shape[0]), q_image.shape[0]))
    cv2.imwrite(str(directory / "overview.jpg"), np.hstack((scaled, q_image)))


def contact_sheet(paths: list[Path], output: Path, columns: int, width: int) -> None:
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None: continue
        height = max(1, int(image.shape[0] * width / image.shape[1])); images.append(cv2.resize(image, (width, height)))
    title = output.stem.replace("_", " ")
    if not images:
        canvas = np.zeros((120, max(640, width), 3), np.uint8)
        cv2.putText(canvas, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "No regions in this experimental preference category", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, .55, (190,190,190), 1, cv2.LINE_AA)
        cv2.imwrite(str(output), canvas); return
    height = max(image.shape[0] for image in images); padded = []
    for image in images:
        canvas = np.zeros((height, width, 3), np.uint8); canvas[:image.shape[0]] = image; padded.append(canvas)
    rows = []
    for start in range(0, len(padded), columns):
        row = padded[start:start + columns] + [np.zeros((height, width, 3), np.uint8)] * (columns - len(padded[start:start + columns])); rows.append(np.hstack(row))
    sheet = np.vstack(rows); header = np.zeros((52, sheet.shape[1], 3), np.uint8)
    cv2.putText(header, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), np.vstack((header, sheet)))


def summary_row(result: dict, kind: str, directory: Path) -> dict:
    static_contra = result["contradiction_runs"]["static"]; drawer_contra = result["contradiction_runs"]["drawer"]
    return {**result["source_metadata"], "experiment_set": kind,
            "experimental_identity_preference": result["experimental_identity_preference"],
            "continuity_advantage_static_minus_drawer": result["continuity_advantage_static_minus_drawer"],
            "static_continuity_median": result["static"]["continuity_score_median"],
            "drawer_continuity_median": result["drawer"]["continuity_score_median"],
            "static_switch_count": result["static"]["switch_count"], "drawer_switch_count": result["drawer"]["switch_count"],
            "static_switch_fraction": result["static"]["switch_fraction"], "drawer_switch_fraction": result["drawer"]["switch_fraction"],
            "static_disocclusion_score": result["temporal_cues"]["static_disocclusion_score"],
            "drawer_attachment_score": result["temporal_cues"]["drawer_attachment_score"],
            "strong_static_disocclusion": result["temporal_cues"]["strong_static_disocclusion"],
            "strong_drawer_attachment": result["temporal_cues"]["strong_drawer_attachment"],
            "static_contradiction_run": static_contra["longest_consecutive_contradiction_run"],
            "drawer_contradiction_run": drawer_contra["longest_consecutive_contradiction_run"],
            "coherent_static_contradiction": static_contra["coherent_contradiction_run"],
            "coherent_drawer_contradiction": drawer_contra["coherent_contradiction_run"],
            "cues_used": json.dumps(result["cues_used"], sort_keys=True), "region_directory": str(directory)}


def control_report(control_rows: dict[str, list[dict]], cfg: dict) -> dict:
    def accuracy(rows: list[dict], expected: str) -> float:
        return sum(row["experimental_identity_preference"] == expected for row in rows) / max(len(rows), 1)
    front_accuracy = accuracy(control_rows["drawer_front"], "drawer_preferred")
    static_accuracy = accuracy(control_rows["static"], "static_preferred")
    comoving_fraction = accuracy(control_rows["comoving"], "drawer_preferred")
    passed = (bool(control_rows["drawer_front"]) and bool(control_rows["static"]) and bool(control_rows["comoving"])
              and front_accuracy >= float(cfg["minimum_control_accuracy"])
              and static_accuracy >= float(cfg["minimum_control_accuracy"])
              and comoving_fraction >= float(cfg["require_comoving_drawer_fraction"]))
    return {"passed": bool(passed), "thresholds_frozen_before_ambiguity_run": True,
            "drawer_front_count": len(control_rows["drawer_front"]), "drawer_front_preference_accuracy": front_accuracy,
            "static_control_count": len(control_rows["static"]), "static_control_preference_accuracy": static_accuracy,
            "layer2_comoving_count": len(control_rows["comoving"]), "layer2_drawer_preference_fraction": comoving_fraction,
            "actual_controls": control_rows,
            "failure_policy": "stop before 116 ambiguity regions if controls do not pass"}


def select_representatives(rows: list[dict], count: int) -> dict[str, list[dict]]:
    chosen = {}
    static = [row for row in rows if row["experimental_identity_preference"] == "static_preferred"]
    drawer = [row for row in rows if row["experimental_identity_preference"] == "drawer_preferred"]
    unresolved = [row for row in rows if row["experimental_identity_preference"] == "unresolved"]
    chosen["strongest_static_preference"] = sorted(static, key=lambda row: (-row["continuity_advantage_static_minus_drawer"], -row["static_disocclusion_score"]))[:count]
    chosen["strongest_drawer_preference"] = sorted(drawer, key=lambda row: (row["continuity_advantage_static_minus_drawer"], -row["drawer_attachment_score"]))[:count]
    chosen["most_unresolved"] = sorted(unresolved, key=lambda row: (abs(row["continuity_advantage_static_minus_drawer"]), row["static_switch_count"] + row["drawer_switch_count"]))[:count]
    chosen["highest_surface_switching"] = sorted(rows, key=lambda row: -(row["static_switch_count"] + row["drawer_switch_count"]))[:count]
    chosen["strongest_static_disocclusion"] = sorted(rows, key=lambda row: -row["static_disocclusion_score"])[:count]
    chosen["strongest_contradiction_run"] = sorted(rows, key=lambda row: -max(row["static_contradiction_run"], row["drawer_contradiction_run"]))[:count]
    return chosen


def run(config_path: Path) -> None:
    audit_cfg = yaml.safe_load(config_path.read_text()); base_path = Path(audit_cfg["inputs"]["assignment_v4_config"])
    base_cfg = yaml.safe_load(base_path.read_text()); v4 = Path(audit_cfg["inputs"]["assignment_v4_output"])
    ambiguity = Path(audit_cfg["inputs"]["ambiguity_audit_output"]); output = Path(audit_cfg["output_dir"])
    review = Path(audit_cfg["review_bundle_dir"])
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"non-overwrite output exists: {output}")
    if review.exists() and any(review.iterdir()): raise FileExistsError(f"non-overwrite review bundle exists: {review}")
    required = [v4 / name for name in ("region_evidence.json", "config_resolved.yaml", "scale_consistency_gate_report.json",
                                       "sam2_seed_manifest.json", "frozen_input_audit.json")]
    required += [ambiguity / name for name in ("ambiguity_regions.json", "ambiguity_audit_summary.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(json.dumps({"code": "missing_verified_audit_inputs", "paths": missing}, indent=2))
    source_gate = json.loads((v4 / "scale_consistency_gate_report.json").read_text())
    if not source_gate.get("passed"): raise RuntimeError("source verified-depth v4 scale gate did not pass")
    repeated_gate = validate_registered_depth_scale(base_cfg)
    if not repeated_gate["passed"]: raise RuntimeError("surface identity audit scale gate did not pass")
    output.mkdir(parents=True); (output / "controls").mkdir(); (output / "ambiguity_regions").mkdir(); (output / "visualization").mkdir()
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(audit_cfg, sort_keys=False))
    (output / "scale_consistency_gate_report.json").write_text(json.dumps(repeated_gate, indent=2) + "\n")
    (output / "execution_manifest.json").write_text(json.dumps({"argv": sys.argv, "cwd": str(Path.cwd()),
        "entry_script": "tools/audit_assignment_v4_surface_identity_visibility.py", "parent_commit": "8f62bc3d09c9634588b63b800d2e1d9a5947c007",
        "source_hashes": {path.name: sha256(path) for path in required}, "sam2_rerun": False,
        "assignment_v4_full_rerun": False, "loftr_lk_tapip3d_ran": False, "reconstruction_ran": False}, indent=2) + "\n")
    context = load_and_validate(base_cfg); frames = context["frames"]["interaction"]; attach_interaction_proposals(frames, base_cfg)
    frames_by_source = {int(frame["source"]): frame for frame in frames}
    evidence = json.loads((v4 / "region_evidence.json").read_text()); evidence_by_key = {(int(item["original_frame_id"]), item["proposal_id"]): item for item in evidence}
    occluders, occluder_report = build_trusted_occluders(context, base_cfg, audit_cfg, v4, frames)
    (output / "trusted_occluder_report.json").write_text(json.dumps(occluder_report, indent=2) + "\n")
    controls = select_controls(evidence, frames_by_source, v4, context, audit_cfg["controls"])
    descriptor_cache = {}; control_summary = defaultdict(list)
    for kind, items in controls.items():
        for ordinal, item in enumerate(items):
            frame = frames_by_source[int(item["original_frame_id"])]
            proposal = next(p for p in frame["proposals"] if p["proposal_id"] == item["proposal_id"])
            result = analyze_region(item, frame, proposal, frames_by_source, context, base_cfg, audit_cfg, occluders, descriptor_cache)
            directory = output / "controls" / kind / f"{ordinal:03d}_{item['proposal_id']}"
            save_region(result, directory, frame, proposal, frames_by_source, context, base_cfg, occluders)
            control_summary[kind].append(summary_row(result, kind, directory))
            print(f"[control {kind}] {item['proposal_id']} -> {result['experimental_identity_preference']}", flush=True)
    controls_report = control_report(dict(control_summary), audit_cfg["controls"])
    (output / "control_report.json").write_text(json.dumps(controls_report, indent=2) + "\n")
    if not controls_report["passed"]:
        failed = {"status": "failed_controls", "interpretation": "failed", "ready_for_dual_tsdf": False,
                  "controls": controls_report, "input_both_supported_regions": 116,
                  "assignment_v4_modified": False, "sam2_ran": False, "tsdf_ran": False, "nksr_ran": False, "mesh_ran": False}
        (output / "surface_identity_summary.json").write_text(json.dumps(failed, indent=2) + "\n")
        raise RuntimeError("surface identity controls failed; ambiguity experiment stopped")
    ambiguity_rows = json.loads((ambiguity / "ambiguity_regions.json").read_text())
    selected = [row for row in ambiguity_rows if row["audit_category"] == "static_drawer_both_supported"]
    expected = int(audit_cfg["audit"]["expected_both_supported_regions"])
    if len(selected) != expected: raise RuntimeError(f"both-supported cardinality mismatch: {len(selected)} != {expected}")
    summaries = []; overview_by_preference = defaultdict(list)
    for ordinal, selection in enumerate(selected):
        key = (int(selection["original_frame_id"]), selection["proposal_id"]); item = evidence_by_key[key]
        if item["label"] != "unknown": raise RuntimeError(f"formal v4 label changed for {item['proposal_id']}")
        frame = frames_by_source[key[0]]; proposal = next(p for p in frame["proposals"] if p["proposal_id"] == key[1])
        result = analyze_region(item, frame, proposal, frames_by_source, context, base_cfg, audit_cfg, occluders, descriptor_cache)
        directory = output / "ambiguity_regions" / f"{ordinal:03d}_{item['proposal_id']}"
        save_region(result, directory, frame, proposal, frames_by_source, context, base_cfg, occluders)
        row = summary_row(result, "ambiguity", directory); summaries.append(row)
        overview_by_preference[result["experimental_identity_preference"]].append(directory / "overview.jpg")
        print(f"[identity {ordinal + 1:03d}/{expected:03d}] {item['proposal_id']} -> {result['experimental_identity_preference']}", flush=True)
    save_csv(output / "per_region_identity_summary.csv", summaries)
    (output / "per_region_identity_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    visualization = audit_cfg["visualization"]
    for preference in PREFERENCES:
        contact_sheet(overview_by_preference[preference], output / "visualization" / f"{preference}_contact_sheet.jpg",
                      int(visualization["contact_sheet_columns"]), int(visualization["contact_sheet_thumbnail_width"]))
    preference_counts = Counter(row["experimental_identity_preference"] for row in summaries)
    advantage_threshold = float(audit_cfg["continuity"]["continuity_advantage_threshold"])
    continuity_counts = {"static_advantage_count": sum(row["continuity_advantage_static_minus_drawer"] >= advantage_threshold for row in summaries),
                         "drawer_advantage_count": sum(row["continuity_advantage_static_minus_drawer"] <= -advantage_threshold for row in summaries),
                         "indistinguishable_count": sum(abs(row["continuity_advantage_static_minus_drawer"]) < advantage_threshold for row in summaries)}
    unresolved_fraction = preference_counts["unresolved"] / max(len(summaries), 1)
    interpretation = ("promising" if unresolved_fraction <= float(audit_cfg["acceptance"]["maximum_unresolved_fraction_for_promising"])
                      else "inconclusive")
    summary = {"input_both_supported_regions": len(summaries),
               "experimental_preference": {"static": preference_counts["static_preferred"], "drawer": preference_counts["drawer_preferred"],
                                           "unresolved": preference_counts["unresolved"], "conflicting": preference_counts["conflicting"]},
               "controls": {key: controls_report[key] for key in ("drawer_front_count", "drawer_front_preference_accuracy", "static_control_count", "static_control_preference_accuracy", "layer2_comoving_count", "layer2_drawer_preference_fraction")},
               "continuity": continuity_counts,
               "occlusion": {"explained_by_drawer_count": sum(any(row2["explained_drawer_occlusion_fraction"] > 0 for model in ("static", "drawer") for row2 in json.loads((Path(row["region_directory"]) / "region_identity_audit.json").read_text())["per_target"][model]) for row in summaries),
                             "explained_by_static_count": sum(any(row2["explained_static_occlusion_fraction"] > 0 for model in ("static", "drawer") for row2 in json.loads((Path(row["region_directory"]) / "region_identity_audit.json").read_text())["per_target"][model]) for row in summaries),
                             "unknown_occluder_count": sum(any(row2["unknown_occlusion_fraction"] > 0 for model in ("static", "drawer") for row2 in json.loads((Path(row["region_directory"]) / "region_identity_audit.json").read_text())["per_target"][model]) for row in summaries)},
               "disocclusion": {"strong_static_disocclusion_count": sum(row["strong_static_disocclusion"] for row in summaries),
                                "strong_drawer_attachment_count": sum(row["strong_drawer_attachment"] for row in summaries)},
               "contradiction": {"coherent_static_contradiction_count": sum(row["coherent_static_contradiction"] for row in summaries),
                                 "coherent_drawer_contradiction_count": sum(row["coherent_drawer_contradiction"] for row in summaries)},
               "interpretation": interpretation, "unresolved_fraction": unresolved_fraction,
               "ready_for_dual_tsdf": False,
               "assignment_v4_modified": False, "camera_pose_axis_q_moving_map_modified": False,
               "sam2_ran": False, "loftr_lk_tapip3d_ran": False, "tsdf_ran": False, "nksr_ran": False, "mesh_ran": False}
    (output / "surface_identity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    representatives = select_representatives(summaries, int(visualization["representative_per_category"]))
    (output / "representative_cases.json").write_text(json.dumps(representatives, indent=2) + "\n")
    review.mkdir(parents=True); (review / "representatives").mkdir()
    for name in ("config_resolved.yaml", "surface_identity_summary.json", "per_region_identity_summary.csv", "per_region_identity_summary.json", "control_report.json", "trusted_occluder_report.json", "representative_cases.json", "scale_consistency_gate_report.json", "execution_manifest.json"):
        shutil.copy2(output / name, review / name)
    for preference in PREFERENCES:
        source = output / "visualization" / f"{preference}_contact_sheet.jpg"
        if source.is_file(): shutil.copy2(source, review / source.name)
    copied = set()
    for category, rows in representatives.items():
        for row in rows:
            source_dir = Path(row["region_directory"]); key = source_dir.name
            if key in copied: continue
            copied.add(key); target = review / "representatives" / key; target.mkdir()
            for name in ("overview.jpg", "q_time_strip.png", "continuity_world_static.png", "continuity_drawer_canonical.png", "surface_switch.png", "occlusion_explanation.jpg"):
                shutil.copy2(source_dir / name, target / name)
    for kind, rows in control_summary.items():
        for row in rows:
            source_dir = Path(row["region_directory"]); target = review / "representatives" / f"control_{kind}_{source_dir.name}"; target.mkdir()
            for name in ("overview.jpg", "q_time_strip.png", "surface_switch.png", "occlusion_explanation.jpg"):
                shutil.copy2(source_dir / name, target / name)
    print(json.dumps({"output": str(output), "review_bundle": str(review), "controls_passed": True,
                      "preference_counts": summary["experimental_preference"], "ready_for_dual_tsdf": False}, indent=2))
