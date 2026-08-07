"""Measured trusted-drawer silhouette motion and local causal reveal events."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from tools.itaco_region_assignment_v4.projective_models import (
    predicted_surface_front,
    project_world,
    unproject_pixels,
)
from tools.itaco_region_assignment_v4.region_evidence import deterministic_sample


def load_trusted_front_masks(
    assignment_v4_output: Path, frames: list[dict], interaction_start: int, minimum_agreement: int = 3
) -> tuple[dict[int, np.ndarray], dict]:
    manifest = json.loads((assignment_v4_output / "sam2_seed_manifest.json").read_text())
    seeds = [row for row in manifest["seeds"] if row["seed_kind"] == "front"]
    if len(seeds) != 4:
        raise RuntimeError(f"trusted drawer front requires exactly four audited front seeds, got {len(seeds)}")
    masks = {}
    per_frame = []
    for frame in frames:
        local = int(frame["source"]) - int(interaction_start)
        propagated = []
        paths = []
        for seed in seeds:
            path = assignment_v4_output / "sam2_propagation_per_seed" / seed["seed_id"] / f"{local:06d}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            propagated.append(np.load(path).astype(bool))
            paths.append(str(path))
        count = np.stack(propagated).sum(axis=0)
        masks[int(frame["source"])] = count >= int(minimum_agreement)
        per_frame.append({
            "original_frame_id": int(frame["source"]),
            "trusted_front_pixels": int(masks[int(frame["source"])].sum()),
            "agreement_required": int(minimum_agreement),
            "propagation_paths": paths,
        })
    return masks, {
        "front_seed_count": 4,
        "front_seed_frame_ids": [int(row["original_frame_id"]) for row in seeds],
        "front_seed_proposal_ids": [row["proposal_id"] for row in seeds],
        "minimum_agreement": int(minimum_agreement),
        "payload_seeds_used": False,
        "sam2_rerun": False,
        "per_frame": per_frame,
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(np.asarray(mask, np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return np.asarray(mask, bool) & ~eroded


def _boundary_displacement(a: np.ndarray, b: np.ndarray) -> float | None:
    ba, bb = _mask_boundary(a), _mask_boundary(b)
    if not np.any(ba) or not np.any(bb):
        return None
    da = distance_transform_edt(~ba)
    db = distance_transform_edt(~bb)
    return float(0.5 * (np.median(da[bb]) + np.median(db[ba])))


def transform_trusted_drawer_silhouette(
    source_frame: dict,
    target_frame: dict,
    source_mask: np.ndarray,
    axis_world: np.ndarray,
    intrinsic: np.ndarray,
    cfg: dict,
) -> dict:
    valid = np.asarray(source_mask, bool) & np.asarray(source_frame["valid"], bool)
    uv = deterministic_sample(valid, int(cfg["maximum_silhouette_samples"]))
    shape = target_frame["depth"].shape
    predicted_mask = np.zeros(shape, bool)
    residuals = np.empty(0, float)
    if len(uv):
        world = unproject_pixels(
            uv, source_frame["depth"][uv[:, 1], uv[:, 0]], source_frame["pose"], intrinsic)
        predicted_world = world + (float(target_frame["q"]) - float(source_frame["q"])) * np.asarray(axis_world, float)
        projected, z, inside = project_world(predicted_world, target_frame["pose"], intrinsic, shape)
        front = predicted_surface_front(projected, z, inside, shape)
        if len(front):
            predicted_mask[projected[front, 1], projected[front, 0]] = True
            radius = int(cfg["silhouette_dilation_pixels"])
            if radius:
                size = 2 * radius + 1
                predicted_mask = cv2.dilate(predicted_mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
            ids = front[np.asarray(target_frame["depth_valid"], bool)[projected[front, 1], projected[front, 0]]]
            if len(ids):
                observed = target_frame["depth"][projected[ids, 1], projected[ids, 0]]
                residuals = np.abs(observed - z[ids])
    return {
        "predicted_mask": predicted_mask,
        "source_sample_count": int(len(uv)),
        "predicted_pixel_count": int(predicted_mask.sum()),
        "geometry_depth_residual_median": float(np.median(residuals)) if len(residuals) else None,
        "geometry_depth_residual_p90": float(np.percentile(residuals, 90)) if len(residuals) else None,
    }


def compare_predicted_actual_silhouette(predicted: np.ndarray, actual: np.ndarray) -> dict:
    predicted = np.asarray(predicted, bool); actual = np.asarray(actual, bool)
    intersection = int((predicted & actual).sum())
    union = int((predicted | actual).sum())
    return {
        "silhouette_iou": float(intersection / max(union, 1)),
        "boundary_displacement_px": _boundary_displacement(predicted, actual),
        "predicted_pixels": int(predicted.sum()),
        "actual_pixels": int(actual.sum()),
    }


def _project_static_samples(patch: dict, frame: dict, intrinsic: np.ndarray) -> dict:
    world = np.asarray(patch["source_world"], float)
    uv, z, inside = project_world(world, frame["pose"], intrinsic, frame["depth"].shape)
    front = predicted_surface_front(uv, z, inside, frame["depth"].shape)
    valid = np.zeros(len(world), bool)
    observed = np.full(len(world), np.nan, float)
    residual = np.full(len(world), np.nan, float)
    ids = front[np.asarray(frame["valid"], bool)[uv[front, 1], uv[front, 0]]] if len(front) else front
    if len(ids):
        valid[ids] = True
        observed[ids] = frame["depth"][uv[ids, 1], uv[ids, 0]]
        observed_world = unproject_pixels(uv[ids], observed[ids], frame["pose"], intrinsic)
        residual[ids] = np.linalg.norm(observed_world - world[ids], axis=1)
    return {"uv": uv, "predicted_depth_m": z, "observed_depth_m": observed,
            "world_residual_m": residual, "valid": valid}


def measure_causal_reveal(
    patch: dict,
    before_frame: dict,
    after_frame: dict,
    future_frames: list[dict],
    trusted_masks: dict[int, np.ndarray],
    axis_world: np.ndarray,
    intrinsic: np.ndarray,
    transition: dict,
    cfg: dict,
) -> dict:
    """Measure every causal field from registered RGB-D and trusted masks."""
    before_mask = trusted_masks[int(before_frame["source"])]
    after_mask = trusted_masks[int(after_frame["source"])]
    silhouette = transform_trusted_drawer_silhouette(
        before_frame, after_frame, before_mask, axis_world, intrinsic, cfg["silhouette"])
    comparison = compare_predicted_actual_silhouette(silhouette["predicted_mask"], after_mask)
    boundary_displacement = comparison["boundary_displacement_px"]
    motion_matches = (
        comparison["silhouette_iou"] >= float(cfg["minimum_silhouette_iou"])
        and boundary_displacement is not None
        and boundary_displacement <= float(cfg["maximum_silhouette_boundary_displacement_px"])
    )
    before = _project_static_samples(patch, before_frame, intrinsic)
    after = _project_static_samples(patch, after_frame, intrinsic)
    n = len(patch["source_world"])
    before_inside = before["valid"].copy()
    ids = np.flatnonzero(before_inside)
    before_trusted = np.zeros(n, bool)
    boundary_distance = distance_transform_edt(before_mask)
    local_gap = np.full(n, np.nan, float)
    near_boundary = np.zeros(n, bool)
    if len(ids):
        uv = before["uv"][ids]
        before_trusted[ids] = before_mask[uv[:, 1], uv[:, 0]]
        local_gap[ids] = before["predicted_depth_m"][ids] - before["observed_depth_m"][ids]
        near_boundary[ids] = boundary_distance[uv[:, 1], uv[:, 0]] <= float(cfg["maximum_boundary_distance_px"])
    local_occluded = (before_inside & before_trusted & near_boundary &
                      (local_gap > 0) & (local_gap <= float(cfg["maximum_local_depth_gap_m"])))
    after_ids = np.flatnonzero(after["valid"])
    actual_left = np.zeros(n, bool); predicted_left = np.zeros(n, bool)
    new_support = np.zeros(n, bool)
    if len(after_ids):
        uv = after["uv"][after_ids]
        actual_left[after_ids] = ~after_mask[uv[:, 1], uv[:, 0]]
        predicted_left[after_ids] = ~silhouette["predicted_mask"][uv[:, 1], uv[:, 0]]
        new_support[after_ids] = after["world_residual_m"][after_ids] <= float(cfg["maximum_world_residual_m"])
    reveal_band = local_occluded & actual_left & predicted_left
    newly_supported = reveal_band & new_support
    persistence_rows = []
    for frame in [after_frame] + [row for row in future_frames if int(row["source"]) > int(after_frame["source"])]:
        observation = _project_static_samples(patch, frame, intrinsic)
        supported = reveal_band & observation["valid"] & (
            observation["world_residual_m"] <= float(cfg["maximum_world_residual_m"]))
        residuals = observation["world_residual_m"][supported]
        persistence_rows.append({
            "frame_id": int(frame["source"]), "q_m": float(frame["q"]),
            "supported_sample_count": int(supported.sum()),
            "supported_sample_fraction": float(supported.sum() / max(reveal_band.sum(), 1)),
            "median_world_residual_m": float(np.median(residuals)) if len(residuals) else None,
            "p90_world_residual_m": float(np.percentile(residuals, 90)) if len(residuals) else None,
        })
    persistent = [row for row in persistence_rows
                  if row["supported_sample_fraction"] >= float(cfg["minimum_persistent_sample_fraction"])
                  and row["median_world_residual_m"] is not None]
    all_residuals = [row["median_world_residual_m"] for row in persistent]
    persistent_q_span = (float(max(row["q_m"] for row in persistent) - min(row["q_m"] for row in persistent))
                         if len(persistent) > 1 else 0.0)
    local_count = int(local_occluded.sum())
    revealed_count = int(newly_supported.sum())
    fields = {
        "active_transition": bool(transition["valid"]),
        "verified_moving_occluder": bool(before_mask.sum() >= int(cfg["minimum_trusted_occluder_pixels"])),
        "occluder_motion_matches_articulation": bool(motion_matches),
        "local_depth_gap": bool(local_count >= int(cfg["minimum_reveal_samples"])),
        "near_silhouette_boundary": bool(np.any(local_occluded)),
        "temporally_adjacent_reveal": bool(
            int(transition["frame_gap"]) <= int(cfg["maximum_reveal_frame_gap"]) and
            float(transition["elapsed_s"]) <= float(cfg["maximum_reveal_elapsed_s"])),
        "silhouette_leaves_location": bool(
            reveal_band.sum() / max(local_count, 1) >= float(cfg["minimum_reveal_fraction"])),
        "new_surface_appears": bool(
            revealed_count >= int(cfg["minimum_reveal_samples"]) and
            revealed_count / max(reveal_band.sum(), 1) >= float(cfg["minimum_reveal_fraction"])),
        "world_static_persistence": bool(
            len(persistent) >= int(cfg["minimum_persistent_frames"]) and all_residuals and
            float(np.median(all_residuals)) <= float(cfg["maximum_world_residual_m"])),
    }
    accepted = all(fields.values())
    local_gaps = local_gap[local_occluded]
    return {
        "source_frame_id": int(before_frame["source"]),
        "target_frame_id": int(after_frame["source"]),
        "frame_gap": int(transition["frame_gap"]),
        "elapsed_s": float(transition["elapsed_s"]),
        "delta_q_m": float(transition["delta_q"]),
        "chronological_direction": transition["chronological_direction"],
        "positive_causal_static_disocclusion": bool(accepted),
        "reason": ("measured_local_drawer_departure_reveals_persistent_world_surface"
                   if accepted else "measured_causal_chain_incomplete"),
        "checks": fields,
        "failed_checks": [name for name, passed in fields.items() if not passed],
        "field_provenance": {
            "active_transition": "directed_local_transition",
            "verified_moving_occluder": "four_front_seed_propagations_3of4_agreement",
            "occluder_motion_matches_articulation": "predicted_vs_actual_trusted_silhouette",
            "local_depth_gap": "registered_depth_same_predicted_ray",
            "near_silhouette_boundary": "actual_before_trusted_mask_boundary_distance",
            "temporally_adjacent_reveal": "raw_frame_id_and_timestamp_gap",
            "silhouette_leaves_location": "predicted_and_actual_after_masks_at_candidate_ray",
            "new_surface_appears": "after_registered_depth_world_support",
            "world_static_persistence": "measured_after_and_future_world_residuals",
        },
        "silhouette": {key: value for key, value in {**silhouette, **comparison}.items()
                       if key != "predicted_mask"},
        "local_occluded_sample_count": local_count,
        "reveal_band_sample_count": int(reveal_band.sum()),
        "newly_supported_sample_count": revealed_count,
        "local_depth_gap_median": float(np.median(local_gaps)) if len(local_gaps) else None,
        "local_depth_gap_p90": float(np.percentile(local_gaps, 90)) if len(local_gaps) else None,
        "persistent_frame_count": len(persistent),
        "persistent_q_span_m": persistent_q_span,
        "persistent_sample_fraction_median": (float(np.median([
            row["supported_sample_fraction"] for row in persistent])) if persistent else 0.0),
        "persistent_world_residual_median": float(np.median(all_residuals)) if all_residuals else None,
        "persistent_world_residual_p90": float(np.percentile(all_residuals, 90)) if all_residuals else None,
        "persistence_rows": persistence_rows,
        "arrays": {
            "before_trusted_mask": before_mask,
            "predicted_after_mask": silhouette["predicted_mask"],
            "actual_after_mask": after_mask,
            "reveal_band_source_indices": np.flatnonzero(reveal_band),
            "newly_supported_source_indices": np.flatnonzero(newly_supported),
            "after_uv": after["uv"],
            "after_world_residual_m": after["world_residual_m"],
        },
    }


def serializable_causal(event: dict) -> dict:
    return {key: value for key, value in event.items() if key != "arrays"}
