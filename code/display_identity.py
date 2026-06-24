from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from artifact_layout import model_worker_file
from depth_camera_config import depth_sensor_limits_for_task
from task_db import (
    create_display_object,
    get_capture_instance,
    get_capture_instance_by_task_id,
    list_identity_candidate_captures,
    record_capture_binding_log,
    upsert_capture_instance,
)
from task_json import (
    load_task_json,
    normalize_path_for_storage,
    resolve_project_path,
    resolve_task_json_path,
    save_task_json,
)


BIND_DISTANCE_THRESHOLD = 0.36
GRAY_DISTANCE_THRESHOLD = 0.41
STRONG_DISTANCE_THRESHOLD = 0.24
CLOSE_CANDIDATE_MARGIN = 0.035
MIN_MASK_PIXELS = 250
MIN_VALID_DEPTH_PIXELS = 50
DEFAULT_CANDIDATE_LIMIT = 500


class IdentityFeatureError(RuntimeError):
    pass


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        loaded = json.loads(str(value))
    except Exception:
        return default
    return loaded if loaded is not None else default


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _cap01(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _log_distance(a: float, b: float, scale: float) -> float:
    a = max(float(a or 0.0), 1.0e-9)
    b = max(float(b or 0.0), 1.0e-9)
    return _cap01(abs(math.log(a / b)) / float(scale))


def _bhattacharyya(a: Any, b: Any) -> float:
    left = _as_float_array(a)
    right = _as_float_array(b)
    if left.size != right.size or left.size == 0:
        return 1.0
    return float(cv2.compareHist(left.astype(np.float32), right.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def _task_id(task: dict[str, Any]) -> str:
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is missing")
    return task_id


def _capture_instance_id(task: dict[str, Any]) -> str:
    existing = task.get("DisplayIdentity") if isinstance(task.get("DisplayIdentity"), dict) else {}
    raw = task.get("capture_instance_id") or existing.get("capture_instance_id")
    if raw:
        return str(raw)
    return f"capture_{_task_id(task)}"


def _resolve_existing_capture(capture_instance_id: str, task_id: str) -> dict[str, Any] | None:
    existing = get_capture_instance(capture_instance_id)
    if existing is not None:
        return existing
    return get_capture_instance_by_task_id(task_id)


def _resolve_existing_binding(existing_capture: dict[str, Any] | None) -> str | None:
    if not existing_capture:
        return None
    if str(existing_capture.get("binding_status") or "") != "bound":
        return None
    display_object_id = str(existing_capture.get("display_object_id") or "").strip()
    return display_object_id or None


def _resolve_sam3_path(task: dict[str, Any], key: str, label: str) -> Path:
    sam3 = task.get("sam3Name") if isinstance(task.get("sam3Name"), dict) else {}
    name = str(sam3.get(key) or "").strip()
    if not name:
        raise IdentityFeatureError(f"sam3Name.{key} is missing")
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise IdentityFeatureError("task_timestamp is required for SAM3 artifacts")
    artifact_key = {"mask": "sam3.mask", "color": "sam3.color", "depth": "sam3.depth"}.get(key)
    if not artifact_key:
        raise IdentityFeatureError(f"unsupported SAM3 artifact key: {key}")
    path = model_worker_file(task_timestamp, artifact_key).resolve()
    if not path.is_file():
        raise IdentityFeatureError(f"{label} not found: {path}")
    return path


def _resolve_color_path(task: dict[str, Any]) -> Path:
    sam3 = task.get("sam3Name") if isinstance(task.get("sam3Name"), dict) else {}
    color_name = str(sam3.get("color") or "").strip()
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise IdentityFeatureError("task_timestamp is required for color artifacts")
    if color_name:
        path = model_worker_file(task_timestamp, "sam3.color").resolve()
        if path.is_file():
            return path
    pv_name = str((task.get("PVCamera") or {}).get("name") or "").strip()
    if not pv_name:
        raise IdentityFeatureError("sam3Name.color and PVCamera.name are missing")
    path = model_worker_file(task_timestamp, "input.color").resolve()
    if not path.is_file():
        raise IdentityFeatureError(f"color image not found: {path}")
    return path


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IdentityFeatureError(f"failed to read mask: {path}")
    if image.ndim == 2:
        return image > 0
    if image.shape[2] == 4:
        return image[:, :, 3] > 0
    return np.any(image > 0, axis=2)


def _read_color_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IdentityFeatureError(f"failed to read color: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3]


def _read_depth_mm(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise IdentityFeatureError(f"failed to read depth: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.dtype != np.uint16:
        raise IdentityFeatureError(f"depth must be uint16 mm, got {depth.dtype}: {path}")
    return depth


def _camera_matrix(task: dict[str, Any]) -> np.ndarray:
    raw = (task.get("PVCamera") or {}).get("k")
    if raw is None:
        frames = task.get("PVCameraFrames") if isinstance(task.get("PVCameraFrames"), list) else []
        if frames:
            raw = (frames[0] or {}).get("k")
    if raw is None:
        raise IdentityFeatureError("PVCamera.k is missing")
    matrix = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or float(matrix[0, 0]) == 0.0 or float(matrix[1, 1]) == 0.0:
        raise IdentityFeatureError("PVCamera.k is invalid")
    return matrix


def _histogram_features(color_bgr: np.ndarray, mask_u8: np.ndarray) -> dict[str, Any]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hist_hs = cv2.calcHist([hsv], [0, 1], mask_u8, [24, 16], [0, 180, 0, 256]).astype(np.float32)
    hist_hs /= max(float(hist_hs.sum()), 1.0)

    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    hist_ab = cv2.calcHist([lab], [1, 2], mask_u8, [20, 20], [0, 256, 0, 256]).astype(np.float32)
    hist_ab /= max(float(hist_ab.sum()), 1.0)

    pixels = color_bgr[mask_u8 > 0].astype(np.float32)
    return {
        "hist_hs": [float(v) for v in hist_hs.reshape(-1)],
        "hist_ab": [float(v) for v in hist_ab.reshape(-1)],
        "mean_bgr": [float(v) for v in pixels.mean(axis=0)],
        "std_bgr": [float(v) for v in pixels.std(axis=0)],
    }


def _shape_features(mask: np.ndarray, mask_u8: np.ndarray, color_bgr: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise IdentityFeatureError("mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = max(1, x1 - x0 + 1)
    bbox_h = max(1, y1 - y0 + 1)
    area = int(mask.sum())

    moments = cv2.moments(mask_u8, binaryImage=True)
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1.0e-30)

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[y0 : y1 + 1, x0 : x1 + 1]
    crop_mask = mask_u8[y0 : y1 + 1, x0 : x1 + 1]
    edges = cv2.Canny(crop_gray, 50, 150)
    edge_density = float((edges[crop_mask > 0] > 0).mean()) if area else 0.0

    h, w = mask.shape
    return {
        "mask_pixels": area,
        "mask_area_ratio": float(area / max(h * w, 1)),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_width_px": int(bbox_w),
        "bbox_height_px": int(bbox_h),
        "bbox_aspect": float(bbox_w / float(bbox_h)),
        "mask_extent": float(area / float(max(bbox_w * bbox_h, 1))),
        "hu_moments_log": [float(v) for v in hu[:4]],
        "edge_density": edge_density,
    }


def _visible_physical_features(
    task: dict[str, Any],
    mask: np.ndarray,
    depth_mm: np.ndarray,
    shape: dict[str, Any],
) -> dict[str, Any]:
    k = _camera_matrix(task)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    limits = depth_sensor_limits_for_task(task)
    valid = mask & (depth_mm >= int(limits.min_depth_mm)) & (depth_mm <= int(limits.max_reliable_depth_mm))
    valid_count = int(np.count_nonzero(valid))
    if valid_count < max(MIN_VALID_DEPTH_PIXELS, int(int(shape["mask_pixels"]) * 0.03)):
        raise IdentityFeatureError(f"insufficient masked depth: {valid_count}/{shape['mask_pixels']}")

    depth_values = depth_mm[valid].astype(np.float64)
    depth_median_mm = float(np.median(depth_values))
    depth_iqr_mm = float(np.percentile(depth_values, 75) - np.percentile(depth_values, 25))
    depth_median_m = depth_median_mm / 1000.0

    bbox_width_m = float(int(shape["bbox_width_px"]) * depth_median_m / fx)
    bbox_height_m = float(int(shape["bbox_height_px"]) * depth_median_m / fy)

    ys, xs = np.nonzero(valid)
    z_m = depth_mm[valid].astype(np.float64) / 1000.0
    x_m = (xs.astype(np.float64) - cx) * z_m / fx
    y_m = -((ys.astype(np.float64) - cy) * z_m / fy)
    if z_m.size >= 64:
        x_low, x_high = np.percentile(x_m, [5.0, 95.0])
        y_low, y_high = np.percentile(y_m, [5.0, 95.0])
        point_width_m = float(max(x_high - x_low, 1.0e-6))
        point_height_m = float(max(y_high - y_low, 1.0e-6))
    else:
        point_width_m = bbox_width_m
        point_height_m = bbox_height_m

    visible_width_m = max(point_width_m, bbox_width_m * 0.25, 1.0e-6)
    visible_height_m = max(point_height_m, bbox_height_m * 0.25, 1.0e-6)

    return {
        "depth_sensor": str(limits.sensor),
        "valid_depth_pixels": valid_count,
        "valid_depth_ratio": float(valid_count / max(int(shape["mask_pixels"]), 1)),
        "depth_median_mm": depth_median_mm,
        "depth_iqr_mm": depth_iqr_mm,
        "bbox_visible_width_m": bbox_width_m,
        "bbox_visible_height_m": bbox_height_m,
        "point_visible_width_m": point_width_m,
        "point_visible_height_m": point_height_m,
        "visible_width_m": visible_width_m,
        "visible_height_m": visible_height_m,
        "visible_area_m2": float(visible_width_m * visible_height_m),
        "visible_size_source": "mask_depth_point_percentile" if z_m.size >= 64 else "mask_bbox_depth_median",
    }


def extract_capture_identity_feature(json_path_arg: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    mask_path = _resolve_sam3_path(task, "mask", "SAM3 mask")
    color_path = _resolve_color_path(task)
    depth_path = _resolve_sam3_path(task, "depth", "SAM3 depth")

    mask = _read_mask(mask_path)
    color_bgr = _read_color_bgr(color_path)
    depth_mm = _read_depth_mm(depth_path)
    if mask.shape[:2] != color_bgr.shape[:2]:
        raise IdentityFeatureError(f"mask/color shape mismatch: {mask.shape} vs {color_bgr.shape}")
    if mask.shape[:2] != depth_mm.shape[:2]:
        raise IdentityFeatureError(f"mask/depth shape mismatch: {mask.shape} vs {depth_mm.shape}")
    if int(mask.sum()) < MIN_MASK_PIXELS:
        raise IdentityFeatureError(f"mask too small: {int(mask.sum())}")

    mask_u8 = (mask.astype(np.uint8) * 255)
    shape = _shape_features(mask, mask_u8, color_bgr)
    color = _histogram_features(color_bgr, mask_u8)
    physical = _visible_physical_features(task, mask, depth_mm, shape)

    feature = {
        "version": 1,
        **shape,
        **color,
        **physical,
    }
    evidence = {
        "json_path": normalize_path_for_storage(json_path),
        "mask_path": normalize_path_for_storage(mask_path),
        "color_path": normalize_path_for_storage(color_path),
        "depth_path": normalize_path_for_storage(depth_path),
        "task_name": str(task.get("task_name") or ""),
        "server_received_utc": task.get("server_received_utc"),
        "sam3Name": task.get("sam3Name") or {},
    }
    return feature, evidence


def compute_identity_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    color_hs = _bhattacharyya(left.get("hist_hs"), right.get("hist_hs"))
    color_ab = _bhattacharyya(left.get("hist_ab"), right.get("hist_ab"))
    mean = _cap01(float(np.linalg.norm(_as_float_array(left.get("mean_bgr")) - _as_float_array(right.get("mean_bgr"))) / 180.0))
    std = _cap01(float(np.linalg.norm(_as_float_array(left.get("std_bgr")) - _as_float_array(right.get("std_bgr"))) / 110.0))
    masked_color = 0.45 * color_hs + 0.35 * color_ab + 0.13 * mean + 0.07 * std

    left_hu = _as_float_array(left.get("hu_moments_log"))[:4]
    right_hu = _as_float_array(right.get("hu_moments_log"))[:4]
    if left_hu.size != 4 or right_hu.size != 4:
        hu_distance = 1.0
    else:
        hu_distance = _cap01(float(np.linalg.norm(left_hu - right_hu) / 12.0))

    mask_shape = (
        0.30 * _log_distance(float(left.get("mask_pixels") or 0), float(right.get("mask_pixels") or 0), 1.6)
        + 0.24 * _log_distance(float(left.get("bbox_aspect") or 0), float(right.get("bbox_aspect") or 0), 1.0)
        + 0.20 * abs(float(left.get("mask_extent") or 0.0) - float(right.get("mask_extent") or 0.0))
        + 0.12 * hu_distance
        + 0.14 * abs(float(left.get("edge_density") or 0.0) - float(right.get("edge_density") or 0.0))
    )

    visible_physical_size = (
        0.42 * _log_distance(float(left.get("visible_width_m") or 0), float(right.get("visible_width_m") or 0), 1.2)
        + 0.42 * _log_distance(float(left.get("visible_height_m") or 0), float(right.get("visible_height_m") or 0), 1.2)
        + 0.16 * _log_distance(float(left.get("visible_area_m2") or 0), float(right.get("visible_area_m2") or 0), 1.5)
    )

    depth_distribution = (
        0.60 * _cap01(abs(float(left.get("depth_median_mm") or 0.0) - float(right.get("depth_median_mm") or 0.0)) / 1800.0)
        + 0.40 * _cap01(abs(float(left.get("depth_iqr_mm") or 0.0) - float(right.get("depth_iqr_mm") or 0.0)) / 800.0)
    )

    identity_distance = 0.58 * masked_color + 0.24 * mask_shape + 0.18 * visible_physical_size
    return {
        "identity_distance": float(identity_distance),
        "masked_color_distance": float(masked_color),
        "mask_shape_distance": float(mask_shape),
        "visible_physical_size_distance": float(visible_physical_size),
        "depth_distribution_distance": float(depth_distribution),
        "debug_components": {
            "color_hs": float(color_hs),
            "color_ab": float(color_ab),
            "color_mean": float(mean),
            "color_std": float(std),
            "hu_moments": float(hu_distance),
        },
    }


def _candidate_scores_for_feature(
    feature: dict[str, Any],
    *,
    exclude_capture_instance_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = list_identity_candidate_captures(
        limit=limit,
        exclude_capture_instance_id=exclude_capture_instance_id,
    )
    best_by_display: dict[str, dict[str, Any]] = {}
    for row in rows:
        display_object_id = str(row.get("display_object_id") or "").strip()
        if not display_object_id:
            continue
        candidate_feature = _json_loads(row.get("feature_json"), {})
        if not isinstance(candidate_feature, dict) or not candidate_feature:
            continue
        try:
            score = compute_identity_distance(feature, candidate_feature)
        except Exception:
            continue
        candidate = {
            "display_object_id": display_object_id,
            "capture_instance_id": row.get("capture_instance_id"),
            "task_id": row.get("task_id"),
            **score,
        }
        old = best_by_display.get(display_object_id)
        if old is None or float(candidate["identity_distance"]) < float(old["identity_distance"]):
            best_by_display[display_object_id] = candidate
    return sorted(best_by_display.values(), key=lambda item: float(item.get("identity_distance") or 999.0))


def _feature_summary(feature: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mask_pixels",
        "bbox_xyxy",
        "bbox_width_px",
        "bbox_height_px",
        "visible_width_m",
        "visible_height_m",
        "visible_area_m2",
        "depth_median_mm",
        "depth_iqr_mm",
        "valid_depth_pixels",
        "valid_depth_ratio",
        "visible_size_source",
    )
    return {key: feature.get(key) for key in keys if key in feature}


def _store_result_in_task_json(json_path: Path, task: dict[str, Any], result: dict[str, Any]) -> None:
    task["capture_instance_id"] = result.get("capture_instance_id")
    task["display_object_id"] = result.get("display_object_id")
    task["DisplayIdentity"] = result
    save_task_json(json_path, task)


def bind_capture_identity(
    json_path_arg: str | Path,
    *,
    force_rebind: bool = False,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = _task_id(task)
    capture_instance_id = _capture_instance_id(task)
    existing_capture = _resolve_existing_capture(capture_instance_id, task_id)
    if existing_capture and existing_capture.get("capture_instance_id"):
        capture_instance_id = str(existing_capture["capture_instance_id"])
    existing_display_object_id = _resolve_existing_binding(existing_capture)
    timestamp = str(task.get("server_received_utc") or (task.get("device") or {}).get("time") or _utc_now_text())

    try:
        feature, evidence = extract_capture_identity_feature(json_path)
    except Exception as exc:
        reason = f"feature_unavailable:{exc}"
        if existing_display_object_id and not force_rebind:
            result = {
                "capture_instance_id": capture_instance_id,
                "display_object_id": existing_display_object_id,
                "is_new_display_object": False,
                "binding_status": "bound",
                "decision": "reuse_existing_binding",
                "binding_reason": reason,
                "identity_distance": None,
                "candidate_scores": [],
                "candidate_scores_close": False,
                "thresholds": _thresholds_payload(),
                "feature_summary": {},
                "evidence": {"json_path": normalize_path_for_storage(json_path)},
            }
            _store_result_in_task_json(json_path, task, result)
            return result

        evidence = {"json_path": normalize_path_for_storage(json_path), "error": str(exc)}
        result = {
            "capture_instance_id": capture_instance_id,
            "display_object_id": None,
            "is_new_display_object": False,
            "binding_status": "unbound",
            "decision": "keep_unbound",
            "binding_reason": reason,
            "identity_distance": None,
            "candidate_scores": [],
            "candidate_scores_close": False,
            "thresholds": _thresholds_payload(),
            "feature_summary": {},
            "evidence": evidence,
        }
        upsert_capture_instance(
            capture_instance_id=capture_instance_id,
            task_id=task_id,
            display_object_id=None,
            source="hololens",
            timestamp=timestamp,
            binding_status="unbound",
            binding_reason=reason,
            identity_distance=None,
            candidate_scores=[],
            feature={},
            evidence=evidence,
        )
        record_capture_binding_log(
            capture_instance_id=capture_instance_id,
            task_id=task_id,
            display_object_id=None,
            decision="keep_unbound",
            binding_status="unbound",
            reason=reason,
            candidate_scores=[],
            detail=result,
        )
        _store_result_in_task_json(json_path, task, result)
        return result

    candidates = _candidate_scores_for_feature(
        feature,
        exclude_capture_instance_id=capture_instance_id,
        limit=candidate_limit,
    )
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    best_distance = float(best["identity_distance"]) if best else None
    candidate_scores_close = bool(
        best is not None
        and second is not None
        and float(second["identity_distance"]) - float(best["identity_distance"]) <= CLOSE_CANDIDATE_MARGIN
    )

    if existing_display_object_id and not force_rebind:
        display_object_id = existing_display_object_id
        decision = "reuse_existing_binding"
        reason = "existing_capture_binding_reused"
        is_new_display_object = False
        binding_status = "bound"
    elif best is None:
        display_object_id = str(uuid.uuid4())
        create_display_object(display_object_id=display_object_id, canonical_capture_instance_id=capture_instance_id)
        decision = "create_new"
        reason = "no_existing_candidates_create_new"
        is_new_display_object = True
        binding_status = "bound"
    elif best_distance is not None and best_distance <= BIND_DISTANCE_THRESHOLD:
        display_object_id = str(best["display_object_id"])
        decision = "bind_existing"
        reason = "close_candidates_bound_to_best" if candidate_scores_close else "high_similarity_bind"
        is_new_display_object = False
        binding_status = "bound"
    else:
        display_object_id = str(uuid.uuid4())
        create_display_object(display_object_id=display_object_id, canonical_capture_instance_id=capture_instance_id)
        decision = "create_new"
        if best_distance is not None and best_distance <= GRAY_DISTANCE_THRESHOLD:
            reason = "gray_zone_create_new"
        else:
            reason = "distance_above_gray_create_new"
        is_new_display_object = True
        binding_status = "bound"

    upsert_capture_instance(
        capture_instance_id=capture_instance_id,
        task_id=task_id,
        display_object_id=display_object_id,
        source="hololens",
        timestamp=timestamp,
        binding_status=binding_status,
        binding_reason=reason,
        identity_distance=best_distance,
        candidate_scores=candidates,
        feature=feature,
        evidence=evidence,
    )

    result = {
        "capture_instance_id": capture_instance_id,
        "display_object_id": display_object_id,
        "is_new_display_object": bool(is_new_display_object),
        "binding_status": binding_status,
        "decision": decision,
        "binding_reason": reason,
        "identity_distance": best_distance,
        "candidate_scores": candidates[:10],
        "candidate_scores_close": candidate_scores_close,
        "close_candidates": candidates[:3] if candidate_scores_close else [],
        "thresholds": _thresholds_payload(),
        "feature_summary": _feature_summary(feature),
        "evidence": evidence,
    }
    record_capture_binding_log(
        capture_instance_id=capture_instance_id,
        task_id=task_id,
        display_object_id=display_object_id,
        decision=decision,
        binding_status=binding_status,
        reason=reason,
        candidate_scores=candidates,
        detail=result,
    )
    _store_result_in_task_json(json_path, task, result)
    return result


def _thresholds_payload() -> dict[str, Any]:
    return {
        "strong_distance_threshold": STRONG_DISTANCE_THRESHOLD,
        "bind_distance_threshold": BIND_DISTANCE_THRESHOLD,
        "gray_distance_threshold": GRAY_DISTANCE_THRESHOLD,
        "close_candidate_margin": CLOSE_CANDIDATE_MARGIN,
        "gray_zone_default": "create_new",
    }


def load_display_identity_for_task_json(task_json: dict[str, Any]) -> dict[str, Any] | None:
    value = task_json.get("DisplayIdentity")
    return value if isinstance(value, dict) else None
