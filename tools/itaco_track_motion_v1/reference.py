"""Automatic reference-frame selection without a frame-zero default."""

from __future__ import annotations

import cv2
import numpy as np

from .errors import Failure, Phase1Error
from .geometry import camera_speed


def _load_depth(record: dict, scale: float) -> np.ndarray:
    raw = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise Phase1Error(Failure("reference_selection", "depth_decode_failed", "Depth image could not be decoded", record["original_frame_id"]))
    return raw.astype(np.float32) * scale


def rgb_footprint(rgb: np.ndarray, config: dict) -> np.ndarray:
    intensity = rgb.max(axis=2)
    mask = (intensity >= int(config["rgb_min_intensity"])).astype(np.uint8)
    close_radius = int(config["rgb_close_radius_px"])
    if close_radius > 0:
        kernel = np.ones((2 * close_radius + 1, 2 * close_radius + 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if bool(config["rgb_keep_largest_component"]):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)
    return mask.astype(bool)


def depth_components(depth: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth) & (depth >= float(config["depth_min_m"])) & (depth <= float(config["depth_max_m"]))
    distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    return valid, distance


def _tracking_quality(gray_images: list[np.ndarray], base_masks: list[np.ndarray], config: dict) -> np.ndarray:
    count = len(gray_images)
    qualities = np.zeros(count, dtype=np.float64)
    samples = np.zeros(count, dtype=np.float64)
    feature_cfg = config["tracking_probe"]
    for index in range(count - 1):
        points = cv2.goodFeaturesToTrack(
            gray_images[index], maxCorners=int(feature_cfg["max_corners"]),
            qualityLevel=float(feature_cfg["quality_level"]),
            minDistance=float(feature_cfg["min_distance_px"]),
            mask=(base_masks[index].astype(np.uint8) * 255),
        )
        if points is None or len(points) == 0:
            continue
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(gray_images[index], gray_images[index + 1], points, None)
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(gray_images[index + 1], gray_images[index], forward, None)
        fb = np.linalg.norm(points[:, 0] - backward[:, 0], axis=1)
        good = status_f[:, 0].astype(bool) & status_b[:, 0].astype(bool) & (fb <= float(feature_cfg["fb_max_error_px"]))
        ratio = float(good.mean())
        qualities[index] += ratio
        qualities[index + 1] += ratio
        samples[index] += 1
        samples[index + 1] += 1
    return np.divide(qualities, np.maximum(samples, 1.0))


def _normalize(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype=np.float64)
    if not finite.any():
        return result
    lo, hi = np.quantile(values[finite], [0.05, 0.95])
    if hi - lo < 1e-12:
        result[finite] = 0.5
    else:
        result[finite] = np.clip((values[finite] - lo) / (hi - lo), 0.0, 1.0)
    return result if higher_is_better else 1.0 - result


def select_reference_frame(records: list[dict], poses: np.ndarray, config: dict, state: np.ndarray) -> tuple[int, list[dict], dict]:
    validity_cfg = config["validity"]
    reference_cfg = config["reference_selection"]
    gray_images, base_masks = [], []
    hand_fraction, depth_fraction, static_fraction, blur = [], [], [], []
    for record in records:
        rgb = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        hand = np.load(record["hand_mask_path"]).squeeze().astype(bool)
        depth = _load_depth(record, float(validity_cfg["depth_scale_to_m"]))
        footprint = rgb_footprint(rgb, validity_cfg)
        depth_valid, boundary_distance = depth_components(depth, validity_cfg)
        static_valid = footprint & depth_valid & (~hand) & (boundary_distance >= float(validity_cfg["depth_boundary_erosion_px"]))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)
        base_masks.append(static_valid)
        hand_fraction.append(float(hand.mean()))
        depth_fraction.append(float((footprint & depth_valid).sum() / max(footprint.sum(), 1)))
        static_fraction.append(float(static_valid.sum() / max(footprint.sum(), 1)))
        values = cv2.Laplacian(gray, cv2.CV_64F)[footprint]
        blur.append(float(values.var()) if len(values) else 0.0)
    tracking = _tracking_quality(gray_images, base_masks, reference_cfg)
    timestamps = np.asarray([record["timestamp"] for record in records], dtype=np.float64)
    linear_speed, angular_speed = camera_speed(poses, timestamps, float(reference_cfg["timestamp_scale_seconds"]))
    q = np.asarray(state, dtype=np.float64)
    q_range = float(q.max() - q.min())
    baseline = float(np.median(np.sort(q)[:max(1, int(np.ceil(len(q) * float(reference_cfg["pre_action_baseline_fraction"]))))]))
    before_action = np.abs(q - baseline) <= max(float(reference_cfg["pre_action_max_state_delta"]), q_range * float(reference_cfg["pre_action_max_range_fraction"]))

    metrics = {
        "hand": np.asarray(hand_fraction), "depth": np.asarray(depth_fraction),
        "static": np.asarray(static_fraction), "linear_speed": linear_speed,
        "angular_speed": angular_speed, "blur": np.asarray(blur), "tracking": tracking,
    }
    normalized = {
        "hand": _normalize(metrics["hand"], False), "depth": _normalize(metrics["depth"], True),
        "static": _normalize(metrics["static"], True), "linear_speed": _normalize(metrics["linear_speed"], False),
        "angular_speed": _normalize(metrics["angular_speed"], False), "blur": _normalize(metrics["blur"], True),
        "tracking": _normalize(metrics["tracking"], True),
    }
    weights = reference_cfg["weights"]
    scores = sum(float(weights[key]) * normalized[key] for key in normalized)
    scores += before_action.astype(np.float64) * float(weights["before_action"])
    candidates = []
    for index, record in enumerate(records):
        reasons = []
        if metrics["hand"][index] > float(reference_cfg["max_hand_fraction"]): reasons.append("hand_coverage_too_high")
        if metrics["depth"][index] < float(reference_cfg["min_valid_depth_fraction"]): reasons.append("valid_depth_too_low")
        if metrics["static"][index] < float(reference_cfg["min_static_valid_fraction"]): reasons.append("static_valid_area_too_low")
        if metrics["linear_speed"][index] > float(reference_cfg["max_camera_speed_mps"]): reasons.append("camera_speed_too_high")
        if metrics["angular_speed"][index] > float(reference_cfg["max_camera_angular_speed_radps"]): reasons.append("camera_angular_speed_too_high")
        if metrics["blur"][index] < float(reference_cfg["min_laplacian_variance"]): reasons.append("image_too_blurry")
        if metrics["tracking"][index] < float(reference_cfg["min_tracking_quality"]): reasons.append("tracking_status_too_weak")
        if bool(reference_cfg["require_pre_action"]) and not bool(before_action[index]): reasons.append("not_before_interaction")
        candidates.append({
            "processing_index": index, "original_frame_id": record["original_frame_id"], "timestamp": record["timestamp"],
            "score": float(scores[index]), "accepted": not reasons, "rejection_reasons": reasons,
            "metrics": {key: float(value[index]) for key, value in metrics.items()},
            "normalized_metrics": {key: float(value[index]) for key, value in normalized.items()},
            "before_interaction": bool(before_action[index]),
        })
    accepted = [item for item in candidates if item["accepted"]]
    if not accepted:
        raise Phase1Error(Failure("reference_selection", "no_acceptable_reference", "No frame passed reference-frame gates", details={"candidates": candidates}))
    selected = max(accepted, key=lambda item: item["score"])
    selection = {
        "reference_frame_id": selected["original_frame_id"], "processing_index": selected["processing_index"],
        "timestamp": selected["timestamp"], "score": selected["score"], "reason": "highest_score_among_accepted_candidates",
        "frame_zero_default_used": False, "selected_is_processing_frame_zero": selected["processing_index"] == 0,
        "pose_convention_before": "T_world_camera", "rebase_formula": "T_reference_camera(t) = inverse(T_world_camera(r)) @ T_world_camera(t)",
    }
    return int(selected["processing_index"]), candidates, selection
