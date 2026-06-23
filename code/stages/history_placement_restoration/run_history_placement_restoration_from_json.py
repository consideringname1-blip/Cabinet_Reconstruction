from __future__ import annotations

import base64
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import HISTORY_PLACEMENT_OUTPUT_ROOT, SHIGURE_HISTORY_CACHE_ROOT
from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS, quat_xyzw_to_rotation_matrix
from stages.history_placement_restoration import settings
from stages.shigure_history.cache import CachedRgbdSample, CachedSampleMetadata, RosStamp, ShigureRgbdCache, load_json, sample_key
from stages.shigure_history.marker_history import latest_marker_pose_path
from task_json import load_task_json, resolve_task_json_path, save_task_json


STATUS_ORIGINAL = "ORIGINAL"
STATUS_MOVED = "MOVED"
STATUS_MISSING = "MISSING"
STATUS_OCCLUDED_REUSE_LAST = "OCCLUDED_REUSE_LAST"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ObjectCenterProjection:
    pixel_xy: tuple[float, float]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_xy": [float(self.pixel_xy[0]), float(self.pixel_xy[1])],
            "depth_m": float(self.depth_m),
            "camera_xyz_m": [float(v) for v in self.camera_xyz_m],
            "source": self.source,
        }


@dataclass(frozen=True)
class YoloEvent:
    sample: CachedSampleMetadata
    payload: dict[str, Any]

    @property
    def seconds(self) -> float:
        return self.sample.stamp.seconds


@dataclass(frozen=True)
class YoloObjectObservation:
    event: YoloEvent
    object_id: str
    center_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    mask_pixels: int
    median_depth_m: float | None
    aruco_position: tuple[float, float, float] | None
    signature: dict[str, Any]

    @property
    def stamp(self) -> RosStamp:
        return self.event.sample.stamp

    @property
    def seconds(self) -> float:
        return self.event.seconds

    @property
    def bbox_diag(self) -> float:
        x0, y0, x1, y1 = self.bbox_xyxy
        return float(math.hypot(max(0.0, x1 - x0), max(0.0, y1 - y0)))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "stamp": self.stamp.to_dict(),
            "object_id": self.object_id,
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "mask_pixels": int(self.mask_pixels),
            "median_depth_m": self.median_depth_m,
            "yolo_hash": self.event.sample.yolo_hash,
            "signature": self.signature,
        }
        if self.aruco_position is not None:
            payload["aruco_position"] = [float(v) for v in self.aruco_position]
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_status(json_path: Path, task: dict[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    payload = dict(task.get("HistoryPlacementRestoration") or {})
    payload.update(fields)
    payload["status"] = status
    payload["updated_at"] = _utc_now()
    task["HistoryPlacementRestoration"] = _jsonable(payload)
    save_task_json(json_path, task)
    return payload


def _parse_iso_timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        if "." not in text:
            return None
        head, tail = text.split(".", 1)
        timezone_pos = min([p for p in (tail.find("+"), tail.find("-")) if p >= 0], default=-1)
        fraction = tail if timezone_pos < 0 else tail[:timezone_pos]
        suffix = "" if timezone_pos < 0 else tail[timezone_pos:]
        try:
            parsed = datetime.fromisoformat(f"{head}.{fraction[:6]}{suffix}")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _task_capture_time_seconds(task: dict[str, Any]) -> tuple[float | None, str | None]:
    for frame in task.get("PVCameraFrames") or []:
        if isinstance(frame, dict):
            seconds = _parse_iso_timestamp_seconds(frame.get("time"))
            if seconds is not None:
                return seconds, "PVCameraFrames.time"
    for key in ("PVCamera", "device"):
        payload = task.get(key)
        if isinstance(payload, dict):
            seconds = _parse_iso_timestamp_seconds(payload.get("time"))
            if seconds is not None:
                return seconds, f"{key}.time"
    seconds = _parse_iso_timestamp_seconds(task.get("server_received_utc"))
    return (seconds, "server_received_utc") if seconds is not None else (None, None)


def _stamp_from_seconds(seconds: float) -> RosStamp:
    sec = math.floor(float(seconds))
    nanosec = int(round((float(seconds) - sec) * 1_000_000_000.0))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return RosStamp(sec=int(sec), nanosec=int(nanosec))


def _depth_raw_to_m(depth_raw: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_raw).astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and float(np.nanmedian(finite)) > 20.0:
        depth = depth / 1000.0
    return depth.astype(np.float32)


def _sample_depth_m(sample: CachedRgbdSample) -> np.ndarray:
    return _depth_raw_to_m(sample.depth)


def _camera_info_message(camera_info: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(camera_info, dict):
        return {}
    message = camera_info.get("message")
    if isinstance(message, dict):
        merged = dict(message)
        for key, value in camera_info.items():
            if key not in merged and key != "message":
                merged[key] = value
        return merged
    return camera_info


def _parse_camera_k(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)]
    else:
        try:
            values = list(np.asarray(raw, dtype=np.float64).reshape(-1))
        except Exception:
            return None
    if len(values) != 9:
        return None
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] == 0 or matrix[1, 1] == 0:
        return None
    return matrix


def _camera_matrix_from_info(camera_info: dict[str, Any] | None) -> np.ndarray | None:
    info = _camera_info_message(camera_info)
    raw = info.get("k") or info.get("K") or info.get("camera_matrix")
    return _parse_camera_k(raw)


def _shape_from_camera_info(camera_info: dict[str, Any] | None) -> tuple[int, int] | None:
    info = _camera_info_message(camera_info)
    try:
        height = int(info.get("height") or 0)
        width = int(info.get("width") or 0)
    except Exception:
        return None
    return (height, width) if height > 0 and width > 0 else None


def _parse_float_array(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    return array


def _find_marker_pose_path() -> Path | None:
    return latest_marker_pose_path()


def _load_marker_pose_cv() -> tuple[np.ndarray, np.ndarray, Path] | None:
    path = _find_marker_pose_path()
    if path is None or not path.is_file():
        return None
    payload = load_json(path)
    pose = payload.get("opencv_camera_pose") if isinstance(payload.get("opencv_camera_pose"), dict) else payload
    if not isinstance(pose, dict):
        return None
    try:
        if pose.get("rotation_matrix") is not None:
            rotation = _parse_float_array(pose.get("rotation_matrix"), 9, "marker rotation_matrix").reshape(3, 3)
        elif pose.get("rotation_quaternion_xyzw") is not None:
            rotation = quat_xyzw_to_rotation_matrix(_parse_float_array(pose.get("rotation_quaternion_xyzw"), 4, "marker quaternion"))
        else:
            return None
        translation = _parse_float_array(
            pose.get("tvec_m") if pose.get("tvec_m") is not None else pose.get("position"),
            3,
            "marker translation",
        )
    except Exception:
        return None
    return rotation.astype(np.float64), translation.astype(np.float64), path


def _object_center_aruco(task: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    bounds = task.get("ModelBounds") if isinstance(task.get("ModelBounds"), dict) else None
    if bounds and bounds.get("aabb_min_aruco") is not None and bounds.get("aabb_max_aruco") is not None:
        try:
            a = _parse_float_array(bounds.get("aabb_min_aruco"), 3, "ModelBounds.aabb_min_aruco")
            b = _parse_float_array(bounds.get("aabb_max_aruco"), 3, "ModelBounds.aabb_max_aruco")
            return (a + b) * 0.5, "ModelBounds.aabb_center_aruco"
        except Exception:
            pass
    obj = task.get("object_aruco") if isinstance(task.get("object_aruco"), dict) else None
    if obj and obj.get("position") is not None:
        try:
            return _parse_float_array(obj.get("position"), 3, "object_aruco.position"), "object_aruco.position"
        except Exception:
            pass
    return None, "missing_object_center_aruco"


def _object_height_m(task: dict[str, Any]) -> float:
    bounds = task.get("ModelBounds") if isinstance(task.get("ModelBounds"), dict) else None
    if bounds and bounds.get("aabb_min_aruco") is not None and bounds.get("aabb_max_aruco") is not None:
        try:
            a = _parse_float_array(bounds.get("aabb_min_aruco"), 3, "ModelBounds.aabb_min_aruco")
            b = _parse_float_array(bounds.get("aabb_max_aruco"), 3, "ModelBounds.aabb_max_aruco")
            return float(abs(b[1] - a[1]))
        except Exception:
            pass
    return 0.10


def _original_pose_aruco(task: dict[str, Any]) -> dict[str, Any] | None:
    pose = task.get("object_aruco") if isinstance(task.get("object_aruco"), dict) else None
    if not pose:
        center, _source = _object_center_aruco(task)
        if center is None:
            return None
        return {
            "position": [float(v) for v in center],
            "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    position = pose.get("position")
    rotation = pose.get("rotation_quaternion_xyzw")
    if not (isinstance(position, list) and len(position) == 3):
        return None
    if not (isinstance(rotation, list) and len(rotation) == 4):
        rotation = [0.0, 0.0, 0.0, 1.0]
    return {
        "position": [float(v) for v in position],
        "rotation_quaternion_xyzw": [float(v) for v in rotation],
    }


def _project_object_center_to_shigure(
    task: dict[str, Any],
    camera_info: dict[str, Any] | None,
    image_shape: tuple[int, int],
) -> tuple[ObjectCenterProjection | None, dict[str, Any]]:
    camera_matrix = _camera_matrix_from_info(camera_info)
    if camera_matrix is None:
        return None, {"source": "model_center_projection", "reason": "camera_matrix_missing"}
    center_aruco, center_source = _object_center_aruco(task)
    if center_aruco is None:
        return None, {"source": "model_center_projection", "reason": center_source}
    marker_pose = _load_marker_pose_cv()
    if marker_pose is None:
        return None, {"source": "model_center_projection", "reason": "marker_pose_missing"}
    marker_rotation, marker_translation, marker_path = marker_pose
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    center_marker_cv = basis @ center_aruco.reshape(3)
    center_camera = marker_rotation @ center_marker_cv + marker_translation.reshape(3)
    z = float(center_camera[2])
    if not np.isfinite(z) or z <= 0.0:
        return None, {
            "source": "model_center_projection",
            "reason": "projected_center_behind_camera",
            "camera_xyz_m": center_camera.tolist(),
        }
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x = fx * float(center_camera[0]) / z + cx
    y = fy * float(center_camera[1]) / z + cy
    h, w = image_shape
    info = {
        "source": "model_center_projection",
        "center_source": center_source,
        "marker_pose_path": str(marker_path),
        "object_center_aruco": center_aruco.tolist(),
        "camera_matrix": camera_matrix.tolist(),
        "pixel_xy": [x, y],
        "depth_m": z,
        "camera_xyz_m": center_camera.tolist(),
        "image_shape": [h, w],
    }
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, {**info, "reason": "projected_center_outside_image"}
    return (
        ObjectCenterProjection(
            pixel_xy=(x, y),
            depth_m=z,
            camera_xyz_m=tuple(float(v) for v in center_camera),
            source=center_source,
        ),
        info,
    )


def _pixel_depth_to_aruco(
    pixel_xy: tuple[float, float],
    depth_m: float | None,
    camera_info: dict[str, Any] | None,
) -> tuple[float, float, float] | None:
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0:
        return None
    camera_matrix = _camera_matrix_from_info(camera_info)
    marker_pose = _load_marker_pose_cv()
    if camera_matrix is None or marker_pose is None:
        return None
    marker_rotation, marker_translation, _marker_path = marker_pose
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    z = float(depth_m)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    camera_cv = np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
    marker_cv = marker_rotation.T @ (camera_cv.reshape(3) - marker_translation.reshape(3))
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    aruco = basis @ marker_cv.reshape(3)
    if not np.isfinite(aruco).all():
        return None
    return tuple(float(v) for v in aruco)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _decode_yolo_mask(obj: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    raw = obj.get("mask_b64")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        data = base64.b64decode(raw)
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if image is None:
        return None
    return _resize_mask(image > 0, shape)


def _bbox_from_object(obj: dict[str, Any], shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    h, w = shape
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    x0 = max(0.0, min(float(w), x0))
    x1 = max(0.0, min(float(w), x1))
    y0 = max(0.0, min(float(h), y0))
    y1 = max(0.0, min(float(h), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _center_from_object(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    try:
        return float(obj.get("x")), float(obj.get("y"))
    except Exception:
        x0, y0, x1, y1 = bbox
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def _mask_median_depth(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
    if depth_m.shape != mask.shape or not np.any(mask):
        return None
    values = depth_m[mask]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.nanmedian(values))


def _mask_signature(
    sample: CachedRgbdSample | None,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    median_depth_m: float | None,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    signature: dict[str, Any] = {
        "area_ratio": float(np.count_nonzero(mask)) / float(mask.size),
        "aspect_ratio": float(width / height),
        "median_depth_m": median_depth_m,
    }
    if sample is not None and sample.rgb_bgr.shape[:2] == mask.shape and np.any(mask):
        rgb = sample.rgb_bgr[:, :, ::-1].astype(np.float32)
        mean_rgb = np.mean(rgb[mask], axis=0)
        signature["mean_rgb"] = [float(v) for v in mean_rgb]
    return signature


def _signature_score(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    color_score = 0.0
    if isinstance(candidate.get("mean_rgb"), list) and isinstance(reference.get("mean_rgb"), list):
        a = np.asarray(candidate["mean_rgb"], dtype=np.float32)
        b = np.asarray(reference["mean_rgb"], dtype=np.float32)
        color_score = float(np.linalg.norm(a - b) / (255.0 * math.sqrt(3.0)))

    area_candidate = max(float(candidate.get("area_ratio") or 0.0), 1e-8)
    area_reference = max(float(reference.get("area_ratio") or 0.0), 1e-8)
    area_score = min(3.0, abs(math.log(area_candidate / area_reference)))

    aspect_candidate = max(float(candidate.get("aspect_ratio") or 0.0), 1e-6)
    aspect_reference = max(float(reference.get("aspect_ratio") or 0.0), 1e-6)
    aspect_score = min(3.0, abs(math.log(aspect_candidate / aspect_reference)))

    depth_score = 0.0
    if candidate.get("median_depth_m") is not None and reference.get("median_depth_m") is not None:
        depth_score = min(
            3.0,
            abs(float(candidate["median_depth_m"]) - float(reference["median_depth_m"]))
            / max(1e-6, settings.YOLO_MATCH_DEPTH_TOLERANCE_M),
        )

    score = (
        settings.SIGNATURE_COLOR_WEIGHT * color_score
        + settings.SIGNATURE_AREA_WEIGHT * area_score
        + settings.SIGNATURE_ASPECT_WEIGHT * aspect_score
        + settings.SIGNATURE_DEPTH_WEIGHT * depth_score
    )
    return {
        "score": float(score),
        "color_score": float(color_score),
        "area_score": float(area_score),
        "aspect_score": float(aspect_score),
        "depth_score": float(depth_score),
    }


def _unique_yolo_events(metadata: list[CachedSampleMetadata]) -> list[YoloEvent]:
    events: list[YoloEvent] = []
    last_hash: str | None = None
    for sample in metadata:
        if not sample.yolo_hash or sample.yolo_hash == last_hash:
            continue
        last_hash = sample.yolo_hash
        payload = sample.load_yolo()
        if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
            continue
        events.append(YoloEvent(sample=sample, payload=payload))
    return events


def _observation_from_object(
    cache: ShigureRgbdCache,
    event: YoloEvent,
    obj: dict[str, Any],
    shape: tuple[int, int],
) -> YoloObjectObservation | None:
    bbox = _bbox_from_object(obj, shape)
    if bbox is None:
        return None
    mask = _decode_yolo_mask(obj, shape)
    if mask is None or int(np.count_nonzero(mask)) < settings.YOLO_MIN_MASK_PIXELS:
        return None
    center = _center_from_object(obj, bbox)
    sample = cache.get_sample(event.sample.stamp, mode="nearest")
    depth = _sample_depth_m(sample) if sample is not None else None
    median_depth = _mask_median_depth(depth, mask) if depth is not None else None
    camera_info = sample.camera_info if sample is not None else event.sample.camera_info
    aruco_position = _pixel_depth_to_aruco(center, median_depth, camera_info)
    signature = _mask_signature(sample, mask, bbox, median_depth)
    return YoloObjectObservation(
        event=event,
        object_id=str(obj.get("object_id")),
        center_xy=center,
        bbox_xyxy=bbox,
        mask=mask,
        mask_pixels=int(np.count_nonzero(mask)),
        median_depth_m=median_depth,
        aruco_position=aruco_position,
        signature=signature,
    )


def _observations_for_events(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    shape: tuple[int, int],
) -> list[YoloObjectObservation]:
    observations: list[YoloObjectObservation] = []
    for event in events:
        for obj in event.payload.get("objects") or []:
            if isinstance(obj, dict):
                obs = _observation_from_object(cache, event, obj, shape)
                if obs is not None:
                    observations.append(obs)
    return observations


def _collect_target_observations(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    object_id: str,
    shape: tuple[int, int],
) -> list[YoloObjectObservation]:
    observations: list[YoloObjectObservation] = []
    for event in events:
        for obj in event.payload.get("objects") or []:
            if isinstance(obj, dict) and str(obj.get("object_id")) == str(object_id):
                obs = _observation_from_object(cache, event, obj, shape)
                if obs is not None:
                    observations.append(obs)
                break
    return observations


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _score_yolo_match(obs: YoloObjectObservation, projection: ObjectCenterProjection) -> dict[str, Any]:
    px, py = projection.pixel_xy
    center_distance = _center_distance((px, py), obs.center_xy)
    depth_diff = None if obs.median_depth_m is None else abs(float(obs.median_depth_m) - float(projection.depth_m))
    h, w = obs.mask.shape
    ix = int(round(px))
    iy = int(round(py))
    point_inside = 0 <= ix < w and 0 <= iy < h and bool(obs.mask[iy, ix])
    x0, y0, x1, y1 = obs.bbox_xyxy
    point_in_bbox = x0 <= px <= x1 and y0 <= py <= y1
    depth_score = 1.0 if depth_diff is None else min(3.0, depth_diff / max(1e-6, settings.YOLO_MATCH_DEPTH_TOLERANCE_M))
    center_score = min(3.0, center_distance / max(1.0, settings.YOLO_MATCH_MAX_CENTER_PX))
    score = center_score + depth_score
    if not point_inside:
        score += 0.75
    if not point_in_bbox:
        score += 0.75
    return {
        "score": float(score),
        "center_distance_px": center_distance,
        "depth_diff_m": depth_diff,
        "point_inside_mask": point_inside,
        "point_inside_bbox": point_in_bbox,
        "observation": obs,
    }


def _find_yolo_target(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    projection: ObjectCenterProjection,
    shape: tuple[int, int],
) -> tuple[YoloObjectObservation | None, dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for obs in _observations_for_events(cache, events, shape):
        scored.append(_score_yolo_match(obs, projection))
    if not scored:
        return None, {"reason": "no_yolo_object_with_mask"}
    scored.sort(key=lambda item: float(item["score"]))
    best = scored[0]
    best_obs: YoloObjectObservation = best["observation"]
    accept = (
        float(best["score"]) <= settings.YOLO_MATCH_MAX_SCORE
        and float(best["center_distance_px"]) <= settings.YOLO_MATCH_MAX_CENTER_PX
        and (best["depth_diff_m"] is None or float(best["depth_diff_m"]) <= settings.YOLO_MATCH_DEPTH_TOLERANCE_M * 2.0)
    )
    return (best_obs if accept else None), {
        "reason": "matched" if accept else "best_match_rejected",
        "best": {k: v for k, v in best.items() if k != "observation"},
        "best_observation": best_obs.to_dict(),
        "candidate_count": len(scored),
        "top_candidates": [
            {**{k: v for k, v in item.items() if k != "observation"}, "observation": item["observation"].to_dict()}
            for item in scored[:5]
        ],
    }


def _stable_observation_window(
    observations: list[YoloObjectObservation],
    *,
    minimum_count: int,
) -> tuple[list[YoloObjectObservation], dict[str, Any]]:
    if len(observations) < minimum_count:
        return [], {"reason": "not_enough_unique_yolo", "available": len(observations), "required": minimum_count}
    for start_index in range(0, len(observations) - minimum_count + 1):
        window = observations[start_index : start_index + minimum_count]
        anchor = window[0]
        max_center = max(settings.ORIGINAL_MAX_CENTER_PX, anchor.bbox_diag * settings.ORIGINAL_MAX_CENTER_BBOX_RATIO)
        center_distances = [_center_distance(anchor.center_xy, obs.center_xy) for obs in window[1:]]
        depth_diffs = [
            abs(float(obs.median_depth_m) - float(anchor.median_depth_m))
            for obs in window[1:]
            if obs.median_depth_m is not None and anchor.median_depth_m is not None
        ]
        stable = (
            all(distance <= max_center for distance in center_distances)
            and all(delta <= settings.ORIGINAL_MAX_DEPTH_DELTA_M for delta in depth_diffs)
        )
        stats = {
            "reason": "stable" if stable else "unstable_yolo_window",
            "start_index": start_index,
            "required": minimum_count,
            "max_center_px": max_center,
            "center_distances_px": center_distances,
            "depth_diffs_m": depth_diffs,
            "window": [obs.to_dict() for obs in window],
        }
        if stable:
            return window, stats
    return [], {"reason": "no_stable_yolo_window", "available": len(observations), "required": minimum_count}


def _save_sample_backup(task: dict[str, Any], sample: CachedRgbdSample, output_dir: Path, *, kind: str) -> Path:
    task_name = str(task.get("task_name") or task.get("task_id") or "task").strip()
    backup_dir = output_dir / f"{task_name}_{kind}_{sample_key(sample.stamp)}"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(backup_dir / "rgb.png"), sample.rgb_bgr)
    cv2.imwrite(str(backup_dir / "depth.png"), np.asarray(sample.depth, dtype=np.uint16))
    if sample.camera_info is not None:
        _write_json(backup_dir / "camera_info.json", sample.camera_info)
    elif sample.camera_info_path and sample.camera_info_path.is_file():
        shutil.copy2(sample.camera_info_path, backup_dir / "camera_info.json")
    if sample.yolo is not None:
        _write_json(backup_dir / "active_objects.json", sample.yolo)
        _write_json(backup_dir / "yolo.json", sample.yolo)
    elif sample.yolo_path and sample.yolo_path.is_file():
        shutil.copy2(sample.yolo_path, backup_dir / "active_objects.json")
        shutil.copy2(sample.yolo_path, backup_dir / "yolo.json")
    marker_pose = _find_marker_pose_path()
    if marker_pose is not None:
        shutil.copy2(marker_pose, backup_dir / "marker_6d_pose.json")
    _write_json(
        backup_dir / "meta.json",
        {
            "backup_kind": kind,
            "stamp": sample.stamp.to_dict(),
            "source_chunk_id": sample.chunk_id,
            "source_frame_index": sample.frame_index,
            "source_yolo_hash": sample.yolo_hash,
            "marker_pose_copied": marker_pose is not None,
            "written_at": _utc_now(),
        },
    )
    return backup_dir


def _save_baseline_arrays(output_dir: Path, obs: YoloObjectObservation, sample: CachedRgbdSample) -> dict[str, str]:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    depth = _sample_depth_m(sample)
    valid = obs.mask & np.isfinite(depth) & (depth > 0.0)
    reference_depth = np.where(valid, depth, 0.0).astype(np.float32)
    mask_path = debug_dir / "baseline_mask.png"
    depth_path = debug_dir / "baseline_reference_depth_m.npy"
    cv2.imwrite(str(mask_path), obs.mask.astype(np.uint8) * 255)
    np.save(depth_path, reference_depth)
    return {
        "baseline_mask_path": str(mask_path),
        "baseline_reference_depth_m_path": str(depth_path),
    }


def _load_baseline_arrays(baseline: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    mask_path = Path(str(baseline.get("baseline_mask_path") or ""))
    depth_path = Path(str(baseline.get("baseline_reference_depth_m_path") or ""))
    mask = None
    reference_depth = None
    if mask_path.is_file():
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            mask = raw > 0
    if depth_path.is_file():
        try:
            reference_depth = np.load(depth_path).astype(np.float32)
        except Exception:
            reference_depth = None
    return mask, reference_depth


def _establish_baseline(
    json_path: Path,
    task: dict[str, Any],
    cache: ShigureRgbdCache,
    output_dir: Path,
    *,
    capture_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    start = _stamp_from_seconds(capture_seconds - max(0.0, settings.BASELINE_PRE_CAPTURE_LOOKBACK_SECONDS))
    end = _stamp_from_seconds(capture_seconds + max(0.0, settings.BASELINE_POST_CAPTURE_SECONDS))
    metadata = list(cache.iter_sample_metadata(start=start, end=end))
    after_metadata = [sample for sample in metadata if sample.stamp.seconds >= capture_seconds]
    first_after = after_metadata[0] if after_metadata else None
    if first_after is None:
        return None, {"reason": "no_shigure_frames_for_baseline", "metadata_frame_count": len(metadata)}
    first_sample = cache.get_sample(first_after.stamp, mode="nearest")
    if first_sample is None:
        return None, {"reason": "baseline_first_sample_unavailable"}
    shape = _shape_from_camera_info(first_after.camera_info) or first_sample.depth.shape[:2]
    projection, projection_info = _project_object_center_to_shigure(task, first_after.camera_info or first_sample.camera_info, shape)
    if projection is None:
        return None, {"reason": "baseline_projection_failed", "projection": projection_info}

    unique_events = _unique_yolo_events(metadata)
    post_events = [event for event in unique_events if event.seconds >= capture_seconds]
    match_events = unique_events[-settings.RECENT_UNIQUE_COUNT :] + post_events
    matched_obs, match_info = _find_yolo_target(cache, match_events, projection, shape)
    if matched_obs is None:
        return None, {
            "reason": "baseline_yolo_match_failed",
            "projection": projection_info,
            "match": match_info,
            "unique_yolo_count": len(unique_events),
        }

    target_observations = _collect_target_observations(cache, post_events, matched_obs.object_id, shape)
    stable, stable_info = _stable_observation_window(
        target_observations,
        minimum_count=max(1, settings.STABLE_UNIQUE_COUNT),
    )
    if not stable:
        return None, {
            "reason": "baseline_yolo_not_stable",
            "projection": projection_info,
            "match": match_info,
            "stable": stable_info,
            "target_object_id": matched_obs.object_id,
        }
    baseline_obs = stable[0]
    baseline_sample = cache.get_sample(baseline_obs.stamp, mode="nearest")
    if baseline_sample is None:
        return None, {"reason": "baseline_sample_unavailable", "target_object_id": matched_obs.object_id}

    backup_dir = _save_sample_backup(task, baseline_sample, output_dir, kind="baseline")
    array_files = _save_baseline_arrays(output_dir, baseline_obs, baseline_sample)
    baseline = {
        "status": "ready",
        "created_at": _utc_now(),
        "capture_seconds": float(capture_seconds),
        "target_object_id": baseline_obs.object_id,
        "reference_observation": baseline_obs.to_dict(),
        "reference_signature": baseline_obs.signature,
        "projection": projection_info,
        "match": match_info,
        "stable": stable_info,
        "baseline_backup_dir": str(backup_dir),
        **array_files,
    }
    return baseline, {"reason": "baseline_ready", "baseline": baseline}


def _best_signature_match(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    shape: tuple[int, int],
    reference_signature: dict[str, Any],
) -> tuple[YoloObjectObservation | None, dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for obs in _observations_for_events(cache, events, shape):
        signature = _signature_score(obs.signature, reference_signature)
        scored.append({"signature": signature, "observation": obs})
    if not scored:
        return None, {"reason": "no_signature_candidates"}
    scored.sort(key=lambda item: float(item["signature"]["score"]))
    best = scored[0]
    best_obs: YoloObjectObservation = best["observation"]
    accept = float(best["signature"]["score"]) <= settings.SIGNATURE_MAX_SCORE
    return (best_obs if accept else None), {
        "reason": "matched" if accept else "best_signature_rejected",
        "best_signature": best["signature"],
        "best_observation": best_obs.to_dict(),
        "candidate_count": len(scored),
        "top_candidates": [
            {"signature": item["signature"], "observation": item["observation"].to_dict()}
            for item in scored[:5]
        ],
    }


def _is_original_position(obs: YoloObjectObservation, baseline: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    reference = baseline.get("reference_observation") if isinstance(baseline.get("reference_observation"), dict) else {}
    reference_center = reference.get("center_xy") if isinstance(reference.get("center_xy"), list) else None
    reference_depth = reference.get("median_depth_m")
    center_distance = None
    depth_delta = None
    aruco_delta = None
    checks = []
    if reference_center and len(reference_center) >= 2:
        center_distance = _center_distance(obs.center_xy, (float(reference_center[0]), float(reference_center[1])))
        max_center = max(settings.ORIGINAL_MAX_CENTER_PX, obs.bbox_diag * settings.ORIGINAL_MAX_CENTER_BBOX_RATIO)
        checks.append(center_distance <= max_center)
    if reference_depth is not None and obs.median_depth_m is not None:
        depth_delta = abs(float(obs.median_depth_m) - float(reference_depth))
        checks.append(depth_delta <= settings.ORIGINAL_MAX_DEPTH_DELTA_M)
    reference_aruco = reference.get("aruco_position") if isinstance(reference.get("aruco_position"), list) else None
    if reference_aruco and obs.aruco_position is not None:
        aruco_delta = float(np.linalg.norm(np.asarray(obs.aruco_position) - np.asarray(reference_aruco, dtype=np.float64)))
        checks.append(aruco_delta <= settings.ORIGINAL_MAX_ARUCO_DELTA_M)
    if not checks:
        return False, {"reason": "no_position_checks_available"}
    return all(checks), {
        "center_distance_px": center_distance,
        "depth_delta_m": depth_delta,
        "aruco_delta_m": aruco_delta,
        "checks": checks,
    }


def _classify_depth_state(
    sample: CachedRgbdSample,
    baseline_mask: np.ndarray | None,
    reference_depth: np.ndarray | None,
) -> tuple[str, dict[str, Any]]:
    if baseline_mask is None or reference_depth is None:
        return STATUS_UNKNOWN, {"reason": "baseline_depth_unavailable"}
    current_depth = _sample_depth_m(sample)
    if current_depth.shape != baseline_mask.shape or reference_depth.shape != baseline_mask.shape:
        return STATUS_UNKNOWN, {
            "reason": "baseline_depth_shape_mismatch",
            "current_depth_shape": list(current_depth.shape),
            "baseline_mask_shape": list(baseline_mask.shape),
            "reference_depth_shape": list(reference_depth.shape),
        }
    valid = baseline_mask & np.isfinite(current_depth) & (current_depth > 0.0) & np.isfinite(reference_depth) & (reference_depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels <= 0:
        return STATUS_UNKNOWN, {"reason": "no_valid_depth_pixels"}
    delta = current_depth - reference_depth
    closer = valid & (delta <= settings.OCCLUSION_DELTA_M)
    deeper = valid & (delta >= settings.MISSING_DELTA_M)
    closer_ratio = float(np.count_nonzero(closer)) / max(1, valid_pixels)
    deeper_ratio = float(np.count_nonzero(deeper)) / max(1, valid_pixels)
    info = {
        "reason": "depth_state",
        "valid_pixels": valid_pixels,
        "closer_ratio": closer_ratio,
        "deeper_ratio": deeper_ratio,
        "occlusion_delta_m": settings.OCCLUSION_DELTA_M,
        "missing_delta_m": settings.MISSING_DELTA_M,
    }
    if closer_ratio >= settings.OCCLUSION_CLOSER_RATIO and deeper_ratio >= settings.MISSING_DEEPER_RATIO:
        return STATUS_UNKNOWN, {**info, "reason": "depth_conflict_closer_and_deeper"}
    if closer_ratio >= settings.OCCLUSION_CLOSER_RATIO:
        return STATUS_OCCLUDED_REUSE_LAST, info
    if deeper_ratio >= settings.MISSING_DEEPER_RATIO:
        return STATUS_MISSING, info
    return STATUS_UNKNOWN, {**info, "reason": "target_id_missing_without_depth_change"}


def _pose_from_observation(task: dict[str, Any], obs: YoloObjectObservation | None) -> dict[str, Any] | None:
    if obs is None or obs.aruco_position is None:
        return None
    original = _original_pose_aruco(task) or {}
    rotation = original.get("rotation_quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0]
    return {
        "position": [float(v) for v in obs.aruco_position],
        "rotation_quaternion_xyzw": [float(v) for v in rotation],
        "source": "shigure_yolo_mask_center_depth",
    }


def _polyhedron_pose(position: list[float] | None, task: dict[str, Any]) -> dict[str, Any] | None:
    if not position or len(position) < 3:
        return None
    height = _object_height_m(task)
    y_offset = max(0.0, height * 0.5) + settings.POLYHEDRON_ABOVE_MARGIN_M + settings.POLYHEDRON_EDGE_LENGTH_M * 0.5
    pos = [float(position[0]), float(position[1]) + y_offset, float(position[2])]
    return {
        "position": pos,
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "source": "object_top_hint",
    }


def _build_display_payload(
    task: dict[str, Any],
    *,
    status: str,
    current_pose_aruco: dict[str, Any] | None,
    id_changed: bool = False,
) -> dict[str, Any]:
    original_pose = _original_pose_aruco(task)
    original_position = original_pose.get("position") if original_pose else None
    current_position = current_pose_aruco.get("position") if current_pose_aruco else None

    display: dict[str, Any] = {
        "status": status,
        "original_pose_aruco": original_pose,
        "current_pose_aruco": current_pose_aruco,
        "id_changed": bool(id_changed),
        "polyhedron": {
            "enabled": False,
            "shape": None,
            "edge_length_m": settings.POLYHEDRON_EDGE_LENGTH_M,
            "pose_aruco": None,
        },
        "animation": {
            "enabled": False,
            "from_pose_aruco": None,
            "to_pose_aruco": original_pose,
            "duration_seconds": settings.ANIMATION_DURATION_SECONDS,
        },
    }
    if status == STATUS_MOVED and current_pose_aruco is not None:
        display["mode"] = "current_polyhedron_to_original"
        display["polyhedron"] = {
            "enabled": True,
            "shape": "cube",
            "edge_length_m": settings.POLYHEDRON_EDGE_LENGTH_M,
            "pose_aruco": _polyhedron_pose(current_position, task),
            "attach_to": "current_object",
        }
        display["animation"]["enabled"] = original_pose is not None
        display["animation"]["from_pose_aruco"] = current_pose_aruco
    elif status == STATUS_UNKNOWN:
        display["mode"] = "unknown_original_octahedron"
        display["polyhedron"] = {
            "enabled": True,
            "shape": "octahedron",
            "edge_length_m": settings.POLYHEDRON_EDGE_LENGTH_M,
            "pose_aruco": _polyhedron_pose(original_position, task),
            "attach_to": "original_model",
        }
    elif status == STATUS_MISSING:
        display["mode"] = "restore_original_only"
    elif status == STATUS_OCCLUDED_REUSE_LAST:
        display["mode"] = "occluded_reuse_last"
        if current_pose_aruco is not None:
            display["polyhedron"] = {
                "enabled": True,
                "shape": "cube",
                "edge_length_m": settings.POLYHEDRON_EDGE_LENGTH_M,
                "pose_aruco": _polyhedron_pose(current_position, task),
                "attach_to": "last_known_object",
            }
    else:
        display["mode"] = "original_only"
    return display


def _save_visualization(
    output_dir: Path,
    sample: CachedRgbdSample,
    *,
    status: str,
    observation: YoloObjectObservation | None,
    projection: dict[str, Any] | None,
) -> str | None:
    image = np.asarray(sample.rgb_bgr).copy()
    if observation is not None:
        color = {
            STATUS_ORIGINAL: (40, 220, 40),
            STATUS_MOVED: (255, 180, 40),
            STATUS_OCCLUDED_REUSE_LAST: (255, 220, 80),
            STATUS_UNKNOWN: (255, 80, 255),
        }.get(status, (80, 80, 255))
        overlay = image.copy()
        overlay[observation.mask] = color
        image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0)
        x0, y0, x1, y1 = [int(round(v)) for v in observation.bbox_xyxy]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cx, cy = [int(round(v)) for v in observation.center_xy]
        cv2.circle(image, (cx, cy), 5, color, -1)
    if projection and isinstance(projection.get("pixel_xy"), list):
        px, py = projection["pixel_xy"][:2]
        cv2.drawMarker(
            image,
            (int(round(px)), int(round(py))),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
    cv2.putText(image, status, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, status, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    path = output_dir / "state_visualization.png"
    cv2.imwrite(str(path), image)
    return str(path)


def _classify_current(
    task: dict[str, Any],
    cache: ShigureRgbdCache,
    baseline: dict[str, Any],
    current_sample: CachedRgbdSample,
    recent_events: list[YoloEvent],
    shape: tuple[int, int],
) -> dict[str, Any]:
    target_id = str(baseline.get("target_object_id") or "")
    reference_signature = baseline.get("reference_signature") if isinstance(baseline.get("reference_signature"), dict) else {}
    baseline_mask, reference_depth = _load_baseline_arrays(baseline)

    target_observations = _collect_target_observations(cache, recent_events, target_id, shape) if target_id else []
    selected_obs: YoloObjectObservation | None = target_observations[-1] if target_observations else None
    id_changed = False
    validation: dict[str, Any] = {}

    if selected_obs is not None and reference_signature:
        signature = _signature_score(selected_obs.signature, reference_signature)
        validation["target_id_signature"] = signature
        if float(signature["score"]) > settings.SIGNATURE_MAX_SCORE:
            fallback, fallback_info = _best_signature_match(cache, recent_events, shape, reference_signature)
            validation["fallback_signature_search"] = fallback_info
            if fallback is not None and fallback.object_id != selected_obs.object_id:
                selected_obs = fallback
                id_changed = True
            else:
                return {
                    "status": STATUS_UNKNOWN,
                    "reason": "target_id_signature_conflict",
                    "selected_observation": selected_obs.to_dict(),
                    "validation": validation,
                    "current_pose_aruco": None,
                    "id_changed": False,
                }

    if selected_obs is None and reference_signature:
        fallback, fallback_info = _best_signature_match(cache, recent_events, shape, reference_signature)
        validation["fallback_signature_search"] = fallback_info
        if fallback is not None:
            selected_obs = fallback
            id_changed = bool(target_id and fallback.object_id != target_id)

    if selected_obs is not None:
        is_original, position_check = _is_original_position(selected_obs, baseline)
        status = STATUS_ORIGINAL if is_original else STATUS_MOVED
        current_pose = _pose_from_observation(task, selected_obs)
        return {
            "status": status,
            "reason": "same_yolo_id" if not id_changed else "signature_matched_id_changed",
            "selected_observation": selected_obs.to_dict(),
            "position_check": position_check,
            "validation": validation,
            "current_pose_aruco": current_pose,
            "id_changed": id_changed,
        }

    if len(recent_events) < max(1, settings.MISSING_UNIQUE_COUNT):
        return {
            "status": STATUS_UNKNOWN,
            "reason": "not_enough_recent_yolo_for_missing_or_occlusion",
            "recent_unique_yolo_count": len(recent_events),
            "required_unique_yolo_count": max(1, settings.MISSING_UNIQUE_COUNT),
            "selected_observation": None,
            "validation": validation,
            "current_pose_aruco": None,
            "id_changed": False,
        }

    depth_status, depth_info = _classify_depth_state(current_sample, baseline_mask, reference_depth)
    return {
        "status": depth_status,
        "reason": depth_info.get("reason"),
        "depth_check": depth_info,
        "selected_observation": None,
        "validation": validation,
        "current_pose_aruco": None,
        "id_changed": False,
    }


def _resolve_target_seconds(cache: ShigureRgbdCache, task: dict[str, Any], target_time: str | None) -> tuple[float | None, str]:
    parsed = _parse_iso_timestamp_seconds(target_time)
    if parsed is not None:
        return parsed, "request.target_time"
    request_payload = task.get("HistoryPlacementRequest") if isinstance(task.get("HistoryPlacementRequest"), dict) else {}
    parsed = _parse_iso_timestamp_seconds(request_payload.get("target_time"))
    if parsed is not None:
        return parsed, "HistoryPlacementRequest.target_time"
    newest = cache.newest_sample()
    if newest is not None:
        return newest.stamp.seconds, "shigure_history.newest_sample"
    capture_seconds, source = _task_capture_time_seconds(task)
    return capture_seconds, source or "task_capture_time"


def run_history_placement_restoration(
    json_path_arg: str | Path,
    *,
    target_time: str | None = None,
    request_source: str = "stage",
) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = str(task.get("task_id") or task.get("task_name") or json_path.stem)
    output_dir = HISTORY_PLACEMENT_OUTPUT_ROOT / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if not settings.ENABLE:
        payload = _write_status(json_path, task, STATUS_SKIPPED, reason="disabled", output_dir=str(output_dir))
        return {"status": STATUS_SKIPPED, "payload": payload}

    cache = ShigureRgbdCache(SHIGURE_HISTORY_CACHE_ROOT)
    capture_seconds, capture_source = _task_capture_time_seconds(task)
    if capture_seconds is None:
        payload = _write_status(json_path, task, STATUS_UNKNOWN, reason="capture_time_missing", output_dir=str(output_dir))
        return {"status": STATUS_UNKNOWN, "payload": payload}

    existing = task.get("HistoryPlacementRestoration") if isinstance(task.get("HistoryPlacementRestoration"), dict) else {}
    baseline = existing.get("baseline") if isinstance(existing.get("baseline"), dict) and existing["baseline"].get("status") == "ready" else None
    if baseline is None:
        baseline, baseline_info = _establish_baseline(json_path, task, cache, output_dir, capture_seconds=capture_seconds)
        if baseline is None:
            payload = _write_status(
                json_path,
                task,
                STATUS_UNKNOWN,
                reason="baseline_failed",
                baseline_attempt=baseline_info,
                capture_time_source=capture_source,
                output_dir=str(output_dir),
            )
            return {"status": STATUS_UNKNOWN, "payload": payload}
        task = load_task_json(json_path)
    else:
        baseline_info = {"reason": "reuse_existing_baseline"}

    target_seconds, target_source = _resolve_target_seconds(cache, task, target_time)
    if target_seconds is None:
        payload = _write_status(
            json_path,
            task,
            STATUS_UNKNOWN,
            reason="target_time_missing",
            baseline=baseline,
            output_dir=str(output_dir),
        )
        return {"status": STATUS_UNKNOWN, "payload": payload}

    target_stamp = _stamp_from_seconds(target_seconds)
    current_sample = cache.get_sample(target_stamp, mode="before") or cache.get_sample(target_stamp, mode="nearest")
    if current_sample is None:
        payload = _write_status(
            json_path,
            task,
            STATUS_UNKNOWN,
            reason="no_shigure_history",
            baseline=baseline,
            output_dir=str(output_dir),
        )
        return {"status": STATUS_UNKNOWN, "payload": payload}

    shape = current_sample.depth.shape[:2]
    current_start = _stamp_from_seconds(target_seconds - max(0.0, settings.CURRENT_LOOKBACK_SECONDS))
    current_end = _stamp_from_seconds(target_seconds + max(0.0, settings.CURRENT_FORWARD_SECONDS))
    metadata = list(cache.iter_sample_metadata(start=current_start, end=current_end))
    events = _unique_yolo_events(metadata)
    recent_events = events[-max(1, settings.RECENT_UNIQUE_COUNT) :]
    current_backup_dir = _save_sample_backup(task, current_sample, output_dir, kind="current")
    classification = _classify_current(task, cache, baseline, current_sample, recent_events, shape)

    selected_observation = classification.get("selected_observation")
    observation_obj = None
    if selected_observation is not None:
        # Re-resolve the observation object so the visualization can use its mask without serializing it.
        selected_id = str(selected_observation.get("object_id") or "")
        selected_stamp = RosStamp.from_dict(selected_observation.get("stamp") or {})
        for obs in _collect_target_observations(cache, recent_events, selected_id, shape):
            if obs.stamp == selected_stamp:
                observation_obj = obs
                break

    status = str(classification["status"])
    current_pose = classification.get("current_pose_aruco")
    display = _build_display_payload(
        task,
        status=status,
        current_pose_aruco=current_pose if isinstance(current_pose, dict) else None,
        id_changed=bool(classification.get("id_changed")),
    )
    visualization_path = _save_visualization(
        output_dir,
        current_sample,
        status=status,
        observation=observation_obj,
        projection=baseline.get("projection") if isinstance(baseline.get("projection"), dict) else None,
    )

    payload = _write_status(
        json_path,
        task,
        status,
        reason=classification.get("reason"),
        request_source=request_source,
        capture_time_source=capture_source,
        target_time_source=target_source,
        target_timestamp=current_sample.stamp.to_dict(),
        target_object_id=baseline.get("target_object_id"),
        baseline=baseline,
        baseline_info=baseline_info,
        current={
            "sample": current_sample.to_dict(),
            "backup_shigurei_dir": str(current_backup_dir),
            "metadata_frame_count": len(metadata),
            "unique_yolo_count": len(events),
            "recent_unique_yolo_count": len(recent_events),
        },
        classification=classification,
        display=display,
        state_visualization_path=visualization_path,
        output_dir=str(output_dir),
    )
    _write_json(output_dir / "summary.json", payload)
    return {"status": status, "payload": payload}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python run_history_placement_restoration_from_json.py <task_meta.json>", file=sys.stderr)
        return 2
    try:
        result = run_history_placement_restoration(argv[1])
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        print("[OK] history_placement_restoration")
        return 0
    except Exception as exc:
        try:
            json_path = resolve_task_json_path(argv[1])
            task = load_task_json(json_path)
            _write_status(json_path, task, STATUS_UNKNOWN, reason="exception", error_message=str(exc))
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
