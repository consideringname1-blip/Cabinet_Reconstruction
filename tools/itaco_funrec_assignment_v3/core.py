"""Testable tracking-motion, region-voting and propagation primitives."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


LABEL_INVALID = np.uint8(0)
LABEL_STATIC = np.uint8(1)
LABEL_DRAWER = np.uint8(2)
LABEL_UNKNOWN = np.uint8(3)


def robust_spread(points: np.ndarray) -> tuple[float, float, float, float]:
    """Return median, p90, maximum radial residual and robust spread."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return (float("inf"),) * 4
    center = np.median(points, axis=0)
    residual = np.linalg.norm(points - center, axis=1)
    median = float(np.median(residual))
    p90 = float(np.percentile(residual, 90))
    maximum = float(np.max(residual))
    return median, p90, maximum, p90


def classify_track(observations: list[dict], axis: np.ndarray, travel_m: float, cfg: dict) -> dict:
    frames = np.asarray([o["original_frame_id"] for o in observations], dtype=np.int64)
    world = np.asarray([o["point_world"] for o in observations], dtype=np.float64)
    q = np.asarray([o["q_t"] for o in observations], dtype=np.float64)
    confidence = np.asarray([o["tracking_confidence"] for o in observations], dtype=np.float64)
    unique_frames = int(len(np.unique(frames)))
    q_span = float(np.ptp(q)) if len(q) else 0.0
    static_stats = robust_spread(world)
    drawer_stats = robust_spread(world - q[:, None] * np.asarray(axis, dtype=np.float64))
    if unique_frames < int(cfg["min_track_observations"]):
        label, reason = "unknown", "unknown_insufficient_support"
    elif q_span < float(cfg["min_track_q_span_fraction"]) * float(travel_m):
        label, reason = "unknown", "unknown_low_excitation"
    elif min(static_stats[3], drawer_stats[3]) > float(cfg.get("max_absolute_residual_m", float("inf"))):
        label, reason = "unknown", "unknown_absolute_residual"
    elif drawer_stats[3] + float(cfg["residual_margin_m"]) < static_stats[3]:
        label, reason = "moving", "moving_track"
    elif static_stats[3] + float(cfg["residual_margin_m"]) < drawer_stats[3]:
        label, reason = "static", "static_track"
    else:
        label, reason = "unknown", "unknown_ambiguous_track"
    return {
        "track_id": int(observations[0]["track_id"]) if observations else -1,
        "label": label,
        "reason": reason,
        "unique_frame_count": unique_frames,
        "q_span_m": q_span,
        "q_span_fraction": float(q_span / max(float(travel_m), 1e-12)),
        "static_median_residual_m": static_stats[0],
        "static_p90_residual_m": static_stats[1],
        "static_max_residual_m": static_stats[2],
        "E_static_m": static_stats[3],
        "drawer_median_residual_m": drawer_stats[0],
        "drawer_p90_residual_m": drawer_stats[1],
        "drawer_max_residual_m": drawer_stats[2],
        "E_drawer_m": drawer_stats[3],
        "observation_confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
    }


def vote_region(track_ids: Iterable[int], track_labels: dict[int, str], cfg: dict) -> dict:
    unique = sorted(set(int(x) for x in track_ids))
    counts = {name: sum(track_labels.get(i, "unknown") == name for i in unique) for name in ("moving", "static", "unknown")}
    labeled = counts["moving"] + counts["static"]
    motion_ratio = counts["moving"] / max(labeled, 1)
    unknown_fraction = counts["unknown"] / max(len(unique), 1)
    if labeled < int(cfg["min_labeled_tracks"]):
        label, reason = "unknown", "region_unknown_insufficient_tracks"
    elif unknown_fraction > float(cfg["max_unknown_track_fraction"]):
        label, reason = "unknown", "region_unknown_track_fraction"
    elif counts["moving"] and counts["static"] and float(cfg["static_ratio_threshold"]) < motion_ratio < float(cfg["moving_ratio_threshold"]):
        label, reason = "unknown", "unknown_region_mixed_motion"
    elif motion_ratio >= float(cfg["moving_ratio_threshold"]):
        label, reason = "drawer", "drawer_region"
    elif motion_ratio <= float(cfg["static_ratio_threshold"]):
        label, reason = "static", "static_region"
    else:
        label, reason = "unknown", "unknown_region_mixed_motion"
    return {"label": label, "reason": reason, "n_moving": counts["moving"], "n_static": counts["static"],
            "n_unknown": counts["unknown"], "n_unique_tracks": len(unique), "motion_ratio": float(motion_ratio),
            "unknown_track_fraction": float(unknown_fraction)}


def merge_propagations(votes: np.ndarray, valid: np.ndarray, minimum_votes: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-seed {-1,0,1} support; disagreements are conflict/unknown."""
    votes = np.asarray(votes, dtype=np.int8)
    positive = np.sum(votes > 0, axis=0)
    negative = np.sum(votes < 0, axis=0)
    conflict = (positive > 0) & (negative > 0)
    drawer = valid & (positive >= int(minimum_votes)) & ~conflict
    return drawer, conflict & valid


def four_state(valid: np.ndarray, drawer_evidence: np.ndarray, static_evidence: np.ndarray,
               conflict: np.ndarray | None = None) -> np.ndarray:
    valid = np.asarray(valid, bool)
    drawer = np.asarray(drawer_evidence, bool) & valid
    static = np.asarray(static_evidence, bool) & valid
    conflict = np.zeros_like(valid) if conflict is None else np.asarray(conflict, bool) & valid
    ambiguous = (drawer & static) | conflict
    labels = np.full(valid.shape, LABEL_INVALID, np.uint8)
    labels[valid] = LABEL_UNKNOWN
    labels[static & ~drawer & ~ambiguous] = LABEL_STATIC
    labels[drawer & ~static & ~ambiguous] = LABEL_DRAWER
    labels[ambiguous] = LABEL_UNKNOWN
    return labels


def periodic_seed_frames(frame_count: int, interval: int) -> list[int]:
    if frame_count <= 0 or interval <= 0:
        return []
    result = list(range(0, frame_count, interval))
    if result[-1] != frame_count - 1:
        result.append(frame_count - 1)
    return result


def core_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, np.float32).reshape(-1, 3)
    colors = np.zeros((len(points), 3), np.uint8) if colors is None else np.clip(np.asarray(colors), 0, 255).astype(np.uint8)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points.T if len(points) else ([], [], [])
    if len(points):
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii")); data.tofile(stream)
