"""Frozen registered-depth residual noise calibrated from motion plateaus."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import cv2
import numpy as np

from .local_transitions import LocalMotionState
from .projected_source_anchor import evaluate_source_indexed_patch, extract_source_patch


CATEGORIES = ("surface_interior", "depth_or_geometry_edge")


def depth_geometry_edge_mask(frame: dict, cfg: dict) -> np.ndarray:
    """Observed depth/geometry edge mask; no object labels or controls are read."""
    depth = np.asarray(frame["depth"], np.float32)
    valid = np.asarray(frame["depth_valid"], bool)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    near_registered_discontinuity = (
        np.asarray(frame["depth_edge_distance"], float)
        <= float(cfg["depth_discontinuity_distance_pixels"])
    )
    local_geometry_edge = valid & (gradient >= float(cfg["minimum_local_depth_gradient_m_per_pixel"]))
    return near_registered_discontinuity | local_geometry_edge


@dataclass(frozen=True)
class FrozenPlateauNoiseModel:
    residuals: dict[str, np.ndarray]
    summary: dict

    def category_for_uv(self, frame: dict, uv: np.ndarray, cfg: dict) -> np.ndarray:
        edge = depth_geometry_edge_mask(frame, cfg)
        points = np.asarray(uv, np.int64)
        return np.where(edge[points[:, 1], points[:, 0]], CATEGORIES[1], CATEGORIES[0])

    def surprisal(self, residual: np.ndarray, categories: np.ndarray) -> np.ndarray:
        """One-sided empirical noise surprisal with a frozen tail extrapolation."""
        values = np.asarray(residual, float)
        labels = np.asarray(categories, object)
        result = np.full(values.shape, np.nan, float)
        for name in CATEGORIES:
            selected = (labels == name) & np.isfinite(values)
            calibration = np.asarray(self.residuals[name], float)
            if not np.any(selected):
                continue
            x = values[selected]
            ranks = np.searchsorted(calibration, x, side="left")
            survival = (len(calibration) - ranks + 0.5) / (len(calibration) + 1.0)
            score = -np.log(np.maximum(survival, np.finfo(float).tiny))
            beyond = x > calibration[-1]
            if np.any(beyond):
                tail = float(self.summary["categories"][name]["tail_scale_m"])
                score[beyond] += (x[beyond] - calibration[-1]) / tail
            result[selected] = score
        return result


def _summary(values: np.ndarray) -> dict:
    values = np.sort(np.asarray(values, float))
    if not len(values):
        raise RuntimeError("empty plateau residual category")
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    tail = max(float(p99 - p90), float(0.25 * p90), 1e-6)
    return {
        "sample_count": int(len(values)), "median_m": float(p50),
        "p90_m": float(p90), "p95_m": float(p95), "p99_m": float(p99),
        "tail_scale_m": tail,
    }


def fit_frozen_plateau_noise_model(
    runtime: dict, cfg: dict, anchor_cfg: dict, source_patch_cfg: dict
) -> tuple[FrozenPlateauNoiseModel, list[dict]]:
    """Fit before controls from all valid geometry in same-state plateau pairs."""
    by_state: dict[LocalMotionState, list] = {
        LocalMotionState.CLOSED_PLATEAU: [], LocalMotionState.OPEN_PLATEAU: []}
    for row in runtime["states"]:
        if row.state in by_state:
            by_state[row.state].append(row)
    collected = {name: [] for name in CATEGORIES}
    pair_rows = []
    maximum_gap = int(cfg["maximum_calibration_frame_gap"])
    edge_cfg = cfg["edge_definition"]
    patch_cfg = {"maximum_source_samples": int(cfg["maximum_source_samples_per_pair"])}
    for state, rows in by_state.items():
        for source in rows:
            source_frame = runtime["frames_by_id"][int(source.frame_id)]
            patch = extract_source_patch(
                source_frame, source_frame["valid"], runtime["context"]["intrinsic"], patch_cfg)
            categories = FrozenPlateauNoiseModel({}, {}).category_for_uv(
                source_frame, patch["source_uv"], edge_cfg)
            for target in rows:
                if source.frame_id == target.frame_id or abs(target.frame_id - source.frame_id) > maximum_gap:
                    continue
                target_frame = runtime["frames_by_id"][int(target.frame_id)]
                evaluation = evaluate_source_indexed_patch(
                    patch, target_frame, runtime["context"]["axis"],
                    runtime["context"]["intrinsic"], "static", anchor_cfg)
                observable = np.asarray(evaluation["arrays"]["observable"], bool)
                residual = np.asarray(evaluation["arrays"]["world_residual_m"], float)
                valid = observable & np.isfinite(residual)
                counts = {}
                for name in CATEGORIES:
                    selected = valid & (categories == name)
                    collected[name].append(residual[selected])
                    counts[name] = int(selected.sum())
                pair_rows.append({
                    "plateau_state": state.value, "source_frame_id": int(source.frame_id),
                    "target_frame_id": int(target.frame_id),
                    "frame_gap": abs(int(target.frame_id) - int(source.frame_id)),
                    "source_sample_count": int(len(patch["source_uv"])),
                    "observable_count": int(valid.sum()), "category_counts": counts,
                    "median_residual_m": (float(np.median(residual[valid])) if np.any(valid) else None),
                    "p90_residual_m": (float(np.percentile(residual[valid], 90)) if np.any(valid) else None),
                })
    residuals = {
        name: np.sort(np.concatenate(collected[name]) if collected[name] else np.empty(0))
        for name in CATEGORIES
    }
    minimum = int(cfg["minimum_samples_per_category"])
    failures = [name for name in CATEGORIES if len(residuals[name]) < minimum]
    if failures:
        raise RuntimeError(f"insufficient plateau noise samples: {failures}")
    category_summary = {name: _summary(residuals[name]) for name in CATEGORIES}
    all_values = np.sort(np.concatenate(list(residuals.values())))
    payload = {
        "schema_version": 1,
        "source": "registered_depth_same_plateau_source_indexed_projection",
        "calibration_uses_control_annotations": False,
        "calibration_uses_box_results": False,
        "plateau_states": [item.value for item in by_state],
        "pair_count": len(pair_rows),
        "categories": category_summary,
        "combined": _summary(all_values),
        "edge_definition": edge_cfg,
        "frozen_before_control_evaluation": True,
    }
    provisional = FrozenPlateauNoiseModel(residuals, payload)
    calibration_surprisal = np.concatenate([
        provisional.surprisal(values, np.full(len(values), name, object))
        for name, values in residuals.items()
    ])
    payload["observation_mean_surprisal_p95"] = float(
        np.percentile(calibration_surprisal, 95))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["frozen_model_sha256"] = hashlib.sha256(canonical).hexdigest()
    return FrozenPlateauNoiseModel(residuals, payload), pair_rows
