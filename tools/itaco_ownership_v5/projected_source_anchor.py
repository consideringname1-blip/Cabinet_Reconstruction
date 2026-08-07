"""Source-indexed projective finite-surface identity evidence.

Formal identity is evaluated only at pixels predicted from the immutable source
patch. Target AutoSeg proposals are absent from this module by construction.
"""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from tools.itaco_region_assignment_v4.projective_models import (
    predicted_surface_front,
    project_world,
    transform_model,
    unproject_pixels,
)
from tools.itaco_region_assignment_v4.region_evidence import deterministic_sample


class AnchorState(str, Enum):
    VERIFIED_SOURCE_ANCHOR = "VERIFIED_SOURCE_ANCHOR"
    OCCLUDED_BY_TRUSTED_DRAWER = "OCCLUDED_BY_TRUSTED_DRAWER"
    OCCLUDED_BY_OTHER = "OCCLUDED_BY_OTHER"
    FREE_SPACE_CONTRADICTION = "FREE_SPACE_CONTRADICTION"
    IDENTITY_LOST = "IDENTITY_LOST"
    UNOBSERVABLE = "UNOBSERVABLE"


def extract_source_patch(frame: dict, mask: np.ndarray, intrinsic: np.ndarray, cfg: dict) -> dict:
    valid_mask = np.asarray(mask, bool) & np.asarray(frame["valid"], bool)
    uv = deterministic_sample(valid_mask, int(cfg["maximum_source_samples"]))
    if len(uv):
        depth = frame["depth"][uv[:, 1], uv[:, 0]]
        world = unproject_pixels(uv, depth, frame["pose"], intrinsic)
    else:
        depth = np.empty(0, float)
        world = np.empty((0, 3), float)
    return {
        "source_frame_id": int(frame["source"]),
        "source_q_m": float(frame["q"]),
        "source_sample_indices": np.arange(len(uv), dtype=np.int64),
        "source_uv": uv,
        "source_depth_m": depth,
        "source_world": world,
        "valid_region_pixel_count": int(valid_mask.sum()),
        "sample_count": int(len(uv)),
    }


def _spatial_coverage(source_uv: np.ndarray, selected: np.ndarray, grid_size: int) -> float:
    selected = np.asarray(selected, bool)
    if not np.any(selected) or not len(source_uv):
        return 0.0
    uv = np.asarray(source_uv, float)
    extent = np.maximum(np.ptp(uv, axis=0), 1.0)
    cell = np.floor((uv - uv.min(axis=0)) / extent * grid_size).astype(int)
    cell = np.clip(cell, 0, grid_size - 1)
    all_cells = np.unique(cell[:, 1] * grid_size + cell[:, 0])
    hit_cells = np.unique(cell[selected, 1] * grid_size + cell[selected, 0])
    return float(len(hit_cells) / max(len(all_cells), 1))


def _coherent_fraction(uv: np.ndarray, selected: np.ndarray, shape: tuple[int, int], dilation: int) -> float:
    points = np.asarray(uv, np.int64)[np.asarray(selected, bool)]
    if not len(points):
        return 0.0
    inside = ((points[:, 0] >= 0) & (points[:, 0] < shape[1]) &
              (points[:, 1] >= 0) & (points[:, 1] < shape[0]))
    points = points[inside]
    if not len(points):
        return 0.0
    mask = np.zeros(shape, np.uint8)
    mask[points[:, 1], points[:, 0]] = 1
    if dilation:
        size = 2 * int(dilation) + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    count, labels = cv2.connectedComponents(mask)
    if count <= 1:
        return 0.0
    values, counts = np.unique(labels[points[:, 1], points[:, 0]], return_counts=True)
    counts = counts[values != 0]
    return float(counts.max() / len(points)) if len(counts) else 0.0


def evaluate_source_indexed_patch(
    patch: dict,
    target_frame: dict,
    axis_world: np.ndarray,
    intrinsic: np.ndarray,
    model: str,
    cfg: dict,
    trusted_drawer_mask: np.ndarray | None = None,
) -> dict:
    """Evaluate a static or drawer hypothesis using paired source sample IDs."""
    source_world = np.asarray(patch["source_world"], float)
    source_q = float(patch["source_q_m"])
    target_q = float(target_frame["q"])
    predicted_world = transform_model(source_world, source_q, target_q, axis_world, model)
    uv, predicted_depth, inside = project_world(
        predicted_world, target_frame["pose"], intrinsic, target_frame["depth"].shape)
    front = predicted_surface_front(uv, predicted_depth, inside, target_frame["depth"].shape)
    count = len(source_world)
    observable = np.zeros(count, bool)
    observed_depth = np.full(count, np.nan, float)
    observed_world = np.full((count, 3), np.nan, float)
    world_residual = np.full(count, np.nan, float)
    signed_depth_residual = np.full(count, np.nan, float)
    if len(front):
        u, v = uv[front, 0], uv[front, 1]
        valid = np.asarray(target_frame["valid"], bool)[v, u]
        ids = front[valid]
        observable[ids] = True
        if len(ids):
            u, v = uv[ids, 0], uv[ids, 1]
            observed_depth[ids] = target_frame["depth"][v, u]
            observed_world[ids] = unproject_pixels(
                uv[ids], observed_depth[ids], target_frame["pose"], intrinsic)
            signed_depth_residual[ids] = observed_depth[ids] - predicted_depth[ids]
            if model == "static":
                world_residual[ids] = np.linalg.norm(observed_world[ids] - source_world[ids], axis=1)
            elif model == "drawer":
                source_canonical = source_world[ids] - source_q * np.asarray(axis_world, float)
                observed_canonical = observed_world[ids] - target_q * np.asarray(axis_world, float)
                world_residual[ids] = np.linalg.norm(observed_canonical - source_canonical, axis=1)
            else:
                raise ValueError(f"unknown model: {model}")
    support = (observable &
               (np.abs(signed_depth_residual) <= float(cfg["depth_support_threshold_m"])) &
               (world_residual <= float(cfg["residual_inlier_threshold_m"])))
    occluded = observable & (signed_depth_residual < -float(cfg["occlusion_margin_m"]))
    contradiction = observable & (signed_depth_residual > float(cfg["free_space_margin_m"]))
    other = observable & ~support & ~occluded & ~contradiction
    trusted_occlusion = np.zeros(count, bool)
    if trusted_drawer_mask is not None and np.any(occluded):
        ids = np.flatnonzero(occluded)
        trusted_occlusion[ids] = np.asarray(trusted_drawer_mask, bool)[uv[ids, 1], uv[ids, 0]]
    valid_count = int(observable.sum())
    support_count = int(support.sum())
    inlier_fraction = float(support_count / max(valid_count, 1))
    source_index_coverage = float(valid_count / max(count, 1))
    spatial_coverage = _spatial_coverage(patch["source_uv"], observable, int(cfg["coverage_grid_size"]))
    residual_values = world_residual[observable]
    residual_values = residual_values[np.isfinite(residual_values)]
    median = float(np.median(residual_values)) if len(residual_values) else None
    p90 = float(np.percentile(residual_values, 90)) if len(residual_values) else None
    contradiction_fraction = float(contradiction.sum() / max(valid_count, 1))
    contradiction_coherence = _coherent_fraction(
        uv, contradiction, target_frame["depth"].shape, int(cfg["spatial_dilation_pixels"]))
    occlusion_fraction = float(occluded.sum() / max(valid_count, 1))
    verified = (
        valid_count >= int(cfg["minimum_valid_projected_samples"])
        and source_index_coverage >= float(cfg["minimum_source_index_coverage"])
        and spatial_coverage >= float(cfg["minimum_spatial_coverage"])
        and inlier_fraction >= float(cfg["minimum_inlier_fraction"])
        and median is not None and median <= float(cfg["maximum_median_residual_m"])
        and p90 is not None and p90 <= float(cfg["maximum_p90_residual_m"])
    )
    if valid_count < int(cfg["minimum_valid_projected_samples"]):
        state, reason = AnchorState.UNOBSERVABLE, "insufficient_valid_projected_source_samples"
    elif verified:
        state, reason = AnchorState.VERIFIED_SOURCE_ANCHOR, "paired_source_samples_verify_immutable_anchor"
    elif (contradiction_fraction >= float(cfg["minimum_contradiction_fraction"]) and
          contradiction_coherence >= float(cfg["minimum_contradiction_spatial_coherence"])):
        state, reason = AnchorState.FREE_SPACE_CONTRADICTION, "coherent_predicted_surface_in_measured_free_space"
    elif occlusion_fraction >= float(cfg["minimum_occlusion_fraction"]):
        trusted_fraction = float(trusted_occlusion.sum() / max(occluded.sum(), 1))
        if trusted_fraction >= float(cfg["minimum_trusted_drawer_occlusion_fraction"]):
            state, reason = (AnchorState.OCCLUDED_BY_TRUSTED_DRAWER,
                             "predicted_source_patch_occluded_by_measured_trusted_drawer")
        else:
            state, reason = AnchorState.OCCLUDED_BY_OTHER, "predicted_source_patch_occluded_by_other_surface"
    else:
        state, reason = AnchorState.IDENTITY_LOST, "paired_source_samples_do_not_verify_hypothesis"
    return {
        "model": model,
        "source_frame_id": int(patch["source_frame_id"]),
        "target_frame_id": int(target_frame["source"]),
        "source_q_m": source_q,
        "target_q_m": target_q,
        "delta_q_m": float(target_q - source_q),
        "identity_state": state.value,
        "identity_reason": reason,
        "source_sample_count": int(count),
        "valid_projected_samples": valid_count,
        "supported_samples": support_count,
        "occluded_samples": int(occluded.sum()),
        "free_space_contradiction_samples": int(contradiction.sum()),
        "unexplained_samples": int(other.sum()),
        "source_sample_index_coverage": source_index_coverage,
        "spatial_coverage": spatial_coverage,
        "inlier_fraction": inlier_fraction,
        "median_residual_m": median,
        "p90_residual_m": p90,
        "contradiction_fraction": contradiction_fraction,
        "contradiction_spatial_coherence": contradiction_coherence,
        "occlusion_fraction": occlusion_fraction,
        "trusted_drawer_occlusion_fraction": float(trusted_occlusion.sum() / max(occluded.sum(), 1)),
        "target_proposals_used_for_identity": False,
        "source_anchor_immutable": True,
        "arrays": {
            "source_sample_indices": np.asarray(patch["source_sample_indices"], np.int64),
            "source_uv": np.asarray(patch["source_uv"], np.int64),
            "predicted_uv": uv,
            "predicted_world": predicted_world,
            "predicted_depth_m": predicted_depth,
            "observed_depth_m": observed_depth,
            "observed_world": observed_world,
            "world_residual_m": world_residual,
            "signed_depth_residual_m": signed_depth_residual,
            "observable": observable,
            "supported": support,
            "occluded": occluded,
            "trusted_occlusion": trusted_occlusion,
            "contradiction": contradiction,
        },
    }


def posthoc_proposal_provenance(evaluation: dict, proposals: list[dict]) -> list[dict]:
    """Report target segmentation overlap after identity has been decided."""
    arrays = evaluation["arrays"]
    ids = np.flatnonzero(arrays["observable"])
    if not len(ids):
        return []
    uv = arrays["predicted_uv"][ids]
    rows = []
    for proposal in proposals:
        mask = np.asarray(proposal["mask"], bool)
        hits = mask[uv[:, 1], uv[:, 0]]
        count = int(hits.sum())
        if count:
            rows.append({
                "proposal_id": proposal["proposal_id"],
                "source_layer": int(proposal["source_layer"]),
                "hit_count": count,
                "observable_overlap_fraction": float(count / len(ids)),
                "role": "posthoc_provenance_only_not_identity",
            })
    return sorted(rows, key=lambda row: (-row["hit_count"], row["proposal_id"]))


def serializable_evaluation(evaluation: dict) -> dict:
    return {key: value for key, value in evaluation.items() if key != "arrays"}
