from __future__ import annotations

import base64
import json
import math
import re
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from artifact_layout import SHIGURE_HISTORY_CACHE_ROOT, model_worker_dir
from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS, quat_xyzw_to_rotation_matrix
from stages.history_placement_restoration import settings
from stages.shigure_history.cache import CachedRgbdSample, CachedSampleMetadata, RosStamp, ShigureRgbdCache, load_json, sample_key
from stages.shigure_history.marker_history import latest_marker_pose_path
from task_db import record_task_timing_event
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


class StageTimingCollector:
    def __init__(self, *, task_id: str, stage_name: str) -> None:
        self.task_id = str(task_id)
        self.stage_name = str(stage_name)
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def span(self, event_name: str, detail: dict[str, Any] | None = None):
        payload: dict[str, Any] = dict(detail or {})
        start_wall = time.time()
        start_perf = time.perf_counter()
        status = "completed"
        error_message = None
        try:
            yield payload
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            raise
        finally:
            completed_wall = time.time()
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            event = {
                "event_name": str(event_name),
                "status": status,
                "duration_ms": int(round(duration_ms)),
                "detail": _jsonable(payload),
            }
            if error_message:
                event["error_message"] = error_message
            self.events.append(event)
            try:
                record_task_timing_event(
                    task_id=self.task_id,
                    stage_name=self.stage_name,
                    event_name=str(event_name),
                    status=status,
                    duration_ms=duration_ms,
                    started_at_unix=start_wall,
                    completed_at_unix=completed_wall,
                    detail=_jsonable(payload),
                    error_message=error_message,
                )
            except Exception as db_exc:
                print(f"[history-placement] failed to record timing {event_name}: {db_exc}", file=sys.stderr)


def _timing_span(timings: StageTimingCollector | None, event_name: str, detail: dict[str, Any] | None = None):
    if timings is None:
        return nullcontext(detail if detail is not None else {})
    return timings.span(event_name, detail)


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


def _yolo_events_from_metadata(metadata: list[CachedSampleMetadata]) -> list[YoloEvent]:
    events: list[YoloEvent] = []
    for sample in metadata:
        if not sample.yolo_hash:
            continue
        payload = sample.load_yolo()
        if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
            continue
        events.append(YoloEvent(sample=sample, payload=payload))
    return events


def _nearest_yolo_event(metadata: list[CachedSampleMetadata], target_seconds: float) -> tuple[YoloEvent | None, dict[str, Any]]:
    events = _yolo_events_from_metadata(metadata)
    if not events:
        return None, {"reason": "no_yolo_payload_in_window", "metadata_frame_count": len(metadata)}
    event = min(events, key=lambda item: abs(float(item.seconds) - float(target_seconds)))
    delta = abs(float(event.seconds) - float(target_seconds))
    return event, {
        "reason": "nearest_yolo_payload",
        "yolo_payload_time": event.sample.stamp.to_dict(),
        "yolo_delta_to_target_seconds": float(delta),
        "is_stale": bool(delta > settings.YOLO_MAX_DELTA_TO_TARGET_SECONDS),
        "max_delta_to_target_seconds": settings.YOLO_MAX_DELTA_TO_TARGET_SECONDS,
        "candidate_event_count": len(events),
    }


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


def _bbox_center_xy(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)


def _union_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    return (
        float(min(b[0] for b in bboxes)),
        float(min(b[1] for b in bboxes)),
        float(max(b[2] for b in bboxes)),
        float(max(b[3] for b in bboxes)),
    )


def _tracking_region_ids() -> set[str]:
    return {str(value) for value in settings.TRACKING_REGION_YOLO_IDS if str(value).strip()}


def _build_tracking_search_region(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    shape: tuple[int, int],
    *,
    reference_region: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured_ids = _tracking_region_ids()
    if not configured_ids:
        return {
            "source": "unrestricted",
            "configured_yolo_ids": [],
            "is_unrestricted": True,
            "valid": True,
            "reason": "no_tracking_region_yolo_ids_configured",
        }

    observations: list[YoloObjectObservation] = []
    for obs in _observations_for_events(cache, events, shape):
        if obs.object_id in configured_ids:
            observations.append(obs)

    if not observations:
        return {
            "source": "configured_yolo_ids",
            "configured_yolo_ids": sorted(configured_ids),
            "is_unrestricted": False,
            "valid": False,
            "reason": "configured_region_yolo_ids_missing",
        }

    bbox = _union_bbox([obs.bbox_xyxy for obs in observations])
    support_depth_values = [float(obs.median_depth_m) for obs in observations if obs.median_depth_m is not None]
    support_depth = float(np.nanmedian(np.asarray(support_depth_values, dtype=np.float32))) if support_depth_values else None
    centers = [_bbox_center_xy(obs.bbox_xyxy) for obs in observations]
    center_xy = [
        float(np.nanmean([center[0] for center in centers])),
        float(np.nanmean([center[1] for center in centers])),
    ]
    region: dict[str, Any] = {
        "source": "configured_yolo_ids",
        "configured_yolo_ids": sorted(configured_ids),
        "matched_yolo_ids": sorted({obs.object_id for obs in observations}),
        "excluded_candidate_yolo_ids": sorted(configured_ids),
        "bbox_xyxy": [float(v) for v in bbox] if bbox is not None else None,
        "support_depth_summary": {"median_depth_m": support_depth, "object_count": len(observations)},
        "support_center_xy": center_xy,
        "is_unrestricted": False,
        "valid": True,
        "reason": "configured_region_ready",
    }

    if isinstance(reference_region, dict) and reference_region.get("source") == "configured_yolo_ids":
        checks: dict[str, Any] = {}
        reference_depth = None
        reference_summary = reference_region.get("support_depth_summary")
        if isinstance(reference_summary, dict) and reference_summary.get("median_depth_m") is not None:
            reference_depth = float(reference_summary["median_depth_m"])
        if support_depth is not None and reference_depth is not None:
            depth_delta = abs(float(support_depth) - reference_depth)
            checks["support_depth_delta_m"] = depth_delta
            checks["support_depth_static"] = depth_delta <= settings.TRACKING_REGION_SUPPORT_STATIC_DEPTH_DELTA_M
        reference_center = reference_region.get("support_center_xy")
        if isinstance(reference_center, list) and len(reference_center) >= 2:
            center_delta = _center_distance((float(reference_center[0]), float(reference_center[1])), (center_xy[0], center_xy[1]))
            checks["support_center_delta_px"] = center_delta
            checks["support_center_static"] = center_delta <= settings.TRACKING_REGION_SUPPORT_STATIC_CENTER_DELTA_PX
        region["support_plane_static_check"] = checks
        if any(value is False for value in checks.values() if isinstance(value, bool)):
            region["valid"] = False
            region["reason"] = "support_plane_moved"
    return region


def _tracking_region_allows_observation(obs: YoloObjectObservation, region: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if not isinstance(region, dict) or region.get("is_unrestricted"):
        return True, {"reason": "unrestricted"}
    excluded = {str(value) for value in region.get("excluded_candidate_yolo_ids") or []}
    if obs.object_id in excluded:
        return False, {"reason": "candidate_is_tracking_region_anchor"}
    bbox = region.get("bbox_xyxy")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False, {"reason": "tracking_region_bbox_missing"}
    cx, cy = obs.center_xy
    inside = float(bbox[0]) <= cx <= float(bbox[2]) and float(bbox[1]) <= cy <= float(bbox[3])
    if not inside:
        return False, {"reason": "outside_tracking_region_bbox"}
    support_summary = region.get("support_depth_summary") if isinstance(region.get("support_depth_summary"), dict) else {}
    support_depth = support_summary.get("median_depth_m")
    if support_depth is not None and obs.median_depth_m is not None:
        behind_delta = float(obs.median_depth_m) - float(support_depth)
        if behind_delta > settings.TRACKING_REGION_SUPPORT_MAX_BEHIND_DEPTH_M:
            return False, {"reason": "behind_support_plane", "behind_delta_m": behind_delta}
        return True, {"reason": "inside_tracking_region", "behind_delta_m": behind_delta}
    return True, {"reason": "inside_tracking_region_depth_unavailable"}


def _search_current_candidate(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    shape: tuple[int, int],
    baseline: dict[str, Any],
    tracking_region: dict[str, Any],
) -> tuple[YoloObjectObservation | None, dict[str, Any]]:
    reference_signature = baseline.get("reference_signature") if isinstance(baseline.get("reference_signature"), dict) else {}
    reference = baseline.get("reference_observation") if isinstance(baseline.get("reference_observation"), dict) else {}
    reference_center = reference.get("center_xy") if isinstance(reference.get("center_xy"), list) else None
    if not reference_signature:
        return None, {"reason": "reference_signature_missing"}

    scored: list[dict[str, Any]] = []
    for obs in _observations_for_events(cache, events, shape):
        allowed, region_check = _tracking_region_allows_observation(obs, tracking_region)
        if not allowed:
            scored.append({"observation": obs, "region_check": region_check, "rejected": True})
            continue
        signature = _signature_score(obs.signature, reference_signature)
        spatial_distance = None
        if isinstance(reference_center, list) and len(reference_center) >= 2:
            spatial_distance = _center_distance(obs.center_xy, (float(reference_center[0]), float(reference_center[1])))
        hard_thresholds_passed = float(signature["score"]) <= settings.SIGNATURE_MAX_SCORE
        scored.append(
            {
                "observation": obs,
                "signature": signature,
                "spatial_distance_px": spatial_distance,
                "region_check": region_check,
                "hard_thresholds_passed": hard_thresholds_passed,
                "rejected": False,
            }
        )

    if not scored:
        return None, {"reason": "no_candidates"}

    viable = [item for item in scored if not item.get("rejected") and item.get("hard_thresholds_passed")]
    viable.sort(
        key=lambda item: (
            float("inf") if item.get("spatial_distance_px") is None else float(item["spatial_distance_px"]),
            float(item["signature"]["score"]),
        )
    )
    selected = viable[0] if viable else None

    def serialize(item: dict[str, Any]) -> dict[str, Any]:
        payload = {k: v for k, v in item.items() if k != "observation"}
        payload["observation"] = item["observation"].to_dict()
        return payload

    info = {
        "reason": "matched" if selected is not None else "no_candidate_passed_hard_thresholds",
        "candidate_count": len(scored),
        "viable_candidate_count": len(viable),
        "tracking_region": tracking_region,
        "top_candidates": [serialize(item) for item in sorted(
            scored,
            key=lambda item: (
                1 if item.get("rejected") else 0,
                float("inf") if item.get("spatial_distance_px") is None else float(item.get("spatial_distance_px")),
                float(item.get("signature", {}).get("score", 999.0)),
            ),
        )[:8]],
    }
    if selected is not None:
        info["selected"] = serialize(selected)
    return (selected["observation"] if selected is not None else None), info


def _establish_baseline(
    json_path: Path,
    task: dict[str, Any],
    cache: ShigureRgbdCache,
    output_dir: Path,
    *,
    capture_seconds: float,
    timings: StageTimingCollector | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    start = _stamp_from_seconds(capture_seconds)
    end = _stamp_from_seconds(capture_seconds + max(0.0, settings.BASELINE_POST_CAPTURE_SECONDS))
    with _timing_span(
        timings,
        "baseline_metadata_scan",
        {"start_stamp": start.to_dict(), "end_stamp": end.to_dict()},
    ) as timing:
        metadata = list(cache.iter_sample_metadata(start=start, end=end))
        timing["metadata_frame_count"] = len(metadata)
    if not metadata:
        return None, {"reason": "no_shigure_frames_for_baseline", "metadata_frame_count": 0}

    with _timing_span(timings, "baseline_first_sample_load", {"stamp": metadata[0].stamp.to_dict()}) as timing:
        first_sample = cache.get_sample(metadata[0].stamp, mode="nearest")
        timing["sample_available"] = first_sample is not None
    if first_sample is None:
        return None, {"reason": "baseline_first_sample_unavailable"}

    with _timing_span(timings, "baseline_yolo_payload_load", {"metadata_frame_count": len(metadata)}) as timing:
        yolo_events = _yolo_events_from_metadata(metadata)
        timing["yolo_event_count"] = len(yolo_events)
    if not yolo_events:
        return None, {"reason": "no_yolo_payload_for_baseline", "metadata_frame_count": len(metadata)}

    attempts: list[dict[str, Any]] = []
    selected_obs: YoloObjectObservation | None = None
    selected_projection_info: dict[str, Any] | None = None
    selected_match_info: dict[str, Any] | None = None
    selected_sample: CachedRgbdSample | None = None

    with _timing_span(timings, "baseline_yolo_target_search", {"yolo_event_count": len(yolo_events)}) as timing:
        for event in sorted(yolo_events, key=lambda item: abs(float(item.seconds) - float(capture_seconds))):
            paired_sample = cache.get_sample(event.sample.stamp, mode="nearest")
            if paired_sample is None:
                attempts.append({"event_stamp": event.sample.stamp.to_dict(), "reason": "paired_sample_unavailable"})
                continue
            shape = paired_sample.depth.shape[:2]
            projection, projection_info = _project_object_center_to_shigure(task, paired_sample.camera_info or event.sample.camera_info, shape)
            if projection is None:
                attempts.append({"event_stamp": event.sample.stamp.to_dict(), "reason": "projection_failed", "projection": projection_info})
                continue
            matched_obs, match_info = _find_yolo_target(cache, [event], projection, shape)
            attempts.append(
                {
                    "event_stamp": event.sample.stamp.to_dict(),
                    "seconds_from_capture": float(event.seconds - capture_seconds),
                    "match": match_info,
                }
            )
            if matched_obs is None:
                continue
            selected_obs = matched_obs
            selected_projection_info = projection_info
            selected_match_info = match_info
            selected_sample = paired_sample
            break
        timing["attempt_count"] = len(attempts)
        timing["selected"] = selected_obs is not None
        if selected_obs is not None:
            timing["selected_object_id"] = selected_obs.object_id
            timing["selected_stamp"] = selected_obs.stamp.to_dict()

    if selected_obs is None or selected_sample is None:
        return None, {
            "reason": "baseline_yolo_match_failed",
            "metadata_frame_count": len(metadata),
            "yolo_event_count": len(yolo_events),
            "attempts": attempts[:10],
        }

    with _timing_span(timings, "baseline_backup_write", {"stamp": selected_sample.stamp.to_dict()}) as timing:
        backup_dir = _save_sample_backup(task, selected_sample, output_dir, kind="baseline")
        timing["backup_dir"] = str(backup_dir)
    with _timing_span(timings, "baseline_array_write", {"object_id": selected_obs.object_id}) as timing:
        array_files = _save_baseline_arrays(output_dir, selected_obs, selected_sample)
        timing.update(array_files)
    with _timing_span(timings, "baseline_tracking_region_build", {"object_id": selected_obs.object_id}) as timing:
        tracking_region_reference = _build_tracking_search_region(
            cache,
            [selected_obs.event],
            selected_sample.depth.shape[:2],
            reference_region=None,
        )
        timing["source"] = tracking_region_reference.get("source")
        timing["is_unrestricted"] = bool(tracking_region_reference.get("is_unrestricted"))
        timing["valid"] = bool(tracking_region_reference.get("valid", True))
        timing["reason"] = tracking_region_reference.get("reason")
    baseline = {
        "status": "ready",
        "created_at": _utc_now(),
        "capture_seconds": float(capture_seconds),
        "reference_observation": selected_obs.to_dict(),
        "reference_signature": selected_obs.signature,
        "projection": selected_projection_info or {},
        "match": selected_match_info or {},
        "baseline_visibility": {
            "reason": "nearest_unoccluded_candidate",
            "seconds_from_capture": float(selected_obs.seconds - capture_seconds),
            "partial_initialization": False,
            "valid_mask_pixels": int(selected_obs.mask_pixels),
        },
        "tracking_search_region_reference": tracking_region_reference,
        "baseline_backup_dir": str(backup_dir),
        **array_files,
    }
    return baseline, {"reason": "baseline_ready", "baseline": baseline, "attempts": attempts[:10]}

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
    return STATUS_UNKNOWN, {**info, "reason": "target_missing_without_depth_change"}


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


def _polyhedron_payload(task: dict[str, Any], shape: str, position: list[float] | None, attach_to: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "shape": shape,
        "edge_length_m": settings.POLYHEDRON_EDGE_LENGTH_M,
        "pose_aruco": _polyhedron_pose(position, task),
        "attach_to": attach_to,
    }


def _build_display_payload(
    task: dict[str, Any],
    *,
    status: str,
    current_pose_aruco: dict[str, Any] | None,
) -> dict[str, Any]:
    original_pose = _original_pose_aruco(task)
    original_position = original_pose.get("position") if original_pose else None
    current_position = current_pose_aruco.get("position") if current_pose_aruco else None

    display: dict[str, Any] = {
        "status": status,
        "original_pose_aruco": original_pose,
        "current_pose_aruco": current_pose_aruco,
        "show_model": False,
        "model_pose": original_pose,
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
    if status == STATUS_ORIGINAL:
        display["mode"] = "original_tetrahedron"
        display["polyhedron"] = _polyhedron_payload(
            task,
            "tetrahedron",
            current_position or original_position,
            "current_object" if current_position is not None else "original_model",
        )
    elif status == STATUS_MOVED and current_pose_aruco is not None:
        display["mode"] = "moved_cube_to_original"
        display["polyhedron"] = _polyhedron_payload(task, "cube", current_position, "current_object")
        display["animation"]["enabled"] = original_pose is not None
        display["animation"]["from_pose_aruco"] = current_pose_aruco
    elif status == STATUS_MISSING:
        display["mode"] = "missing_original_octahedron"
        display["show_model"] = True
        display["polyhedron"] = _polyhedron_payload(task, "octahedron", original_position, "original_model")
    elif status == STATUS_OCCLUDED_REUSE_LAST:
        display["mode"] = "occluded_original_dodecahedron"
        display["show_model"] = True
        display["polyhedron"] = _polyhedron_payload(task, "dodecahedron", original_position, "original_model")
    elif status == STATUS_UNKNOWN:
        display["mode"] = "unknown_original_icosahedron"
        display["show_model"] = True
        display["polyhedron"] = _polyhedron_payload(task, "icosahedron", original_position, "original_model")
    else:
        display["mode"] = "skipped" if status == STATUS_SKIPPED else "original_only"
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
    yolo_event: YoloEvent | None,
    yolo_timing: dict[str, Any],
    shape: tuple[int, int],
    timings: StageTimingCollector | None = None,
) -> dict[str, Any]:
    with _timing_span(timings, "baseline_arrays_load") as timing:
        baseline_mask, reference_depth = _load_baseline_arrays(baseline)
        timing["mask_available"] = baseline_mask is not None
        timing["reference_depth_available"] = reference_depth is not None
    validation: dict[str, Any] = {"yolo_timing": yolo_timing}
    with _timing_span(
        timings,
        "current_tracking_region_build",
        {"has_yolo_event": yolo_event is not None},
    ) as timing:
        tracking_region = _build_tracking_search_region(
            cache,
            [yolo_event] if yolo_event is not None else [],
            shape,
            reference_region=baseline.get("tracking_search_region_reference") if isinstance(baseline.get("tracking_search_region_reference"), dict) else None,
        )
        timing["source"] = tracking_region.get("source")
        timing["is_unrestricted"] = bool(tracking_region.get("is_unrestricted"))
        timing["valid"] = bool(tracking_region.get("valid", True))
        timing["reason"] = tracking_region.get("reason")
        timing["matched_yolo_ids"] = tracking_region.get("matched_yolo_ids")
    validation["tracking_search_region"] = tracking_region

    if not tracking_region.get("valid", True) and settings.TRACKING_REGION_INVALID_POLICY != "unrestricted":
        return {
            "status": STATUS_UNKNOWN,
            "reason": tracking_region.get("reason") or "tracking_region_invalid",
            "selected_observation": None,
            "validation": validation,
            "tracking_search_region": tracking_region,
            "current_pose_aruco": None,
        }
    if not tracking_region.get("valid", True):
        tracking_region = {
            "source": "unrestricted",
            "configured_yolo_ids": tracking_region.get("configured_yolo_ids") or [],
            "is_unrestricted": True,
            "valid": True,
            "reason": f"degraded_from_{tracking_region.get('reason') or 'invalid_tracking_region'}",
        }
        validation["tracking_search_region_degraded"] = tracking_region

    selected_obs: YoloObjectObservation | None = None
    candidate_info: dict[str, Any] = {"reason": "no_yolo_event"}
    yolo_stale = bool(yolo_timing.get("is_stale")) if isinstance(yolo_timing, dict) else False
    if yolo_event is not None and not yolo_stale:
        with _timing_span(
            timings,
            "current_candidate_search",
            {
                "tracking_region_source": tracking_region.get("source"),
                "tracking_region_unrestricted": bool(tracking_region.get("is_unrestricted")),
            },
        ) as timing:
            selected_obs, candidate_info = _search_current_candidate(cache, [yolo_event], shape, baseline, tracking_region)
            timing["reason"] = candidate_info.get("reason")
            timing["candidate_count"] = candidate_info.get("candidate_count")
            timing["viable_candidate_count"] = candidate_info.get("viable_candidate_count")
            timing["selected"] = selected_obs is not None
            if selected_obs is not None:
                timing["selected_object_id"] = selected_obs.object_id
    elif yolo_event is not None:
        candidate_info = {"reason": "yolo_payload_stale", "yolo_timing": yolo_timing}
    validation["candidate_search"] = candidate_info

    if selected_obs is not None:
        with _timing_span(timings, "current_original_position_check", {"object_id": selected_obs.object_id}) as timing:
            is_original, position_check = _is_original_position(selected_obs, baseline)
            timing["is_original"] = bool(is_original)
            timing.update({k: v for k, v in position_check.items() if k != "checks"})
        status = STATUS_ORIGINAL if is_original else STATUS_MOVED
        current_pose = _pose_from_observation(task, selected_obs)
        return {
            "status": status,
            "reason": "candidate_matched_original_position" if is_original else "candidate_matched_moved",
            "selected_observation": selected_obs.to_dict(),
            "position_check": position_check,
            "validation": validation,
            "tracking_search_region": tracking_region,
            "current_pose_aruco": current_pose,
        }

    with _timing_span(timings, "current_depth_state_classification") as timing:
        depth_status, depth_info = _classify_depth_state(current_sample, baseline_mask, reference_depth)
        timing["status"] = depth_status
        timing["reason"] = depth_info.get("reason")
        for key in ("valid_pixels", "closer_ratio", "deeper_ratio"):
            if key in depth_info:
                timing[key] = depth_info[key]
    return {
        "status": depth_status,
        "reason": depth_info.get("reason"),
        "depth_check": depth_info,
        "selected_observation": None,
        "validation": validation,
        "tracking_search_region": tracking_region,
        "current_pose_aruco": None,
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
    artifact_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = str(task.get("task_id") or task.get("task_name") or json_path.stem)
    timings = StageTimingCollector(task_id=task_id, stage_name="history_placement_restoration")
    if artifact_output_dir is not None:
        output_dir = Path(artifact_output_dir)
    else:
        task_timestamp = str(task.get("task_timestamp") or "").strip()
        if not task_timestamp:
            raise ValueError("task_timestamp is required for history placement artifacts")
        output_dir = model_worker_dir(task_timestamp) / "09_history_placement_working"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not settings.ENABLE:
        payload = _write_status(
            json_path,
            task,
            STATUS_SKIPPED,
            reason="disabled",
            output_dir=str(output_dir),
            timings=timings.events,
        )
        return {"status": STATUS_SKIPPED, "payload": payload}

    with timings.span("cache_initialize", {"cache_root": str(SHIGURE_HISTORY_CACHE_ROOT)}):
        cache = ShigureRgbdCache(SHIGURE_HISTORY_CACHE_ROOT)
    with timings.span("capture_time_resolve") as timing:
        capture_seconds, capture_source = _task_capture_time_seconds(task)
        timing["capture_time_source"] = capture_source
        timing["capture_seconds_available"] = capture_seconds is not None
    if capture_seconds is None:
        payload = _write_status(
            json_path,
            task,
            STATUS_UNKNOWN,
            reason="capture_time_missing",
            output_dir=str(output_dir),
            timings=timings.events,
        )
        return {"status": STATUS_UNKNOWN, "payload": payload}

    with timings.span("baseline_establish_total") as timing:
        baseline, baseline_info = _establish_baseline(
            json_path,
            task,
            cache,
            output_dir,
            capture_seconds=capture_seconds,
            timings=timings,
        )
        timing["reason"] = baseline_info.get("reason")
        timing["baseline_ready"] = baseline is not None
    if baseline is None:
        payload = _write_status(
            json_path,
            task,
            STATUS_UNKNOWN,
            reason="baseline_failed",
            baseline_attempt=baseline_info,
            capture_time_source=capture_source,
            output_dir=str(output_dir),
            timings=timings.events,
        )
        return {"status": STATUS_UNKNOWN, "payload": payload}
    with timings.span("reload_task_json_after_baseline"):
        task = load_task_json(json_path)

    with timings.span("target_time_resolve") as timing:
        target_seconds, target_source = _resolve_target_seconds(cache, task, target_time)
        timing["target_time_source"] = target_source
        timing["target_seconds_available"] = target_seconds is not None
    if target_seconds is None:
        payload = _write_status(
            json_path,
            task,
            STATUS_UNKNOWN,
            reason="target_time_missing",
            baseline=baseline,
            output_dir=str(output_dir),
            timings=timings.events,
        )
        return {"status": STATUS_UNKNOWN, "payload": payload}

    target_stamp = _stamp_from_seconds(target_seconds)
    with timings.span("current_sample_load", {"target_stamp": target_stamp.to_dict()}) as timing:
        current_sample = cache.get_sample(target_stamp, mode="before") or cache.get_sample(target_stamp, mode="nearest")
        timing["sample_available"] = current_sample is not None
        if current_sample is not None:
            timing["sample_stamp"] = current_sample.stamp.to_dict()
    if current_sample is None:
        payload = _write_status(
            json_path,
            task,
            STATUS_UNKNOWN,
            reason="no_shigure_history",
            baseline=baseline,
            output_dir=str(output_dir),
            timings=timings.events,
        )
        return {"status": STATUS_UNKNOWN, "payload": payload}

    yolo_search_seconds = max(0.0, settings.YOLO_NEAREST_SEARCH_SECONDS, settings.YOLO_MAX_DELTA_TO_TARGET_SECONDS)
    yolo_start = _stamp_from_seconds(target_seconds - yolo_search_seconds)
    yolo_end = _stamp_from_seconds(target_seconds + yolo_search_seconds)
    with timings.span(
        "current_yolo_metadata_scan",
        {"start_stamp": yolo_start.to_dict(), "end_stamp": yolo_end.to_dict()},
    ) as timing:
        metadata = list(cache.iter_sample_metadata(start=yolo_start, end=yolo_end))
        timing["metadata_frame_count"] = len(metadata)
    with timings.span("current_nearest_yolo_event_select", {"metadata_frame_count": len(metadata)}) as timing:
        yolo_event, yolo_timing = _nearest_yolo_event(metadata, target_seconds)
        timing.update({k: v for k, v in yolo_timing.items() if k in {"reason", "candidate_event_count", "yolo_delta_to_target_seconds", "is_stale"}})
        timing["has_yolo_event"] = yolo_event is not None
    with timings.span("current_yolo_paired_sample_load", {"has_yolo_event": yolo_event is not None}) as timing:
        yolo_paired_sample = cache.get_sample(yolo_event.sample.stamp, mode="nearest") if yolo_event is not None else None
        timing["sample_available"] = yolo_paired_sample is not None
        if yolo_paired_sample is not None:
            timing["sample_stamp"] = yolo_paired_sample.stamp.to_dict()
    if yolo_event is not None and yolo_paired_sample is not None:
        yolo_timing = {
            **yolo_timing,
            "target_rgbd_time": current_sample.stamp.to_dict(),
            "yolo_paired_rgbd_time": yolo_paired_sample.stamp.to_dict(),
            "yolo_pairing_delta_seconds": abs(float(yolo_paired_sample.stamp.seconds) - float(yolo_event.seconds)),
        }
    shape = yolo_paired_sample.depth.shape[:2] if yolo_paired_sample is not None else current_sample.depth.shape[:2]
    with timings.span("current_backup_write", {"stamp": current_sample.stamp.to_dict()}) as timing:
        current_backup_dir = _save_sample_backup(task, current_sample, output_dir, kind="current")
        timing["backup_dir"] = str(current_backup_dir)
    paired_backup_dir = None
    if yolo_paired_sample is not None and yolo_paired_sample.stamp != current_sample.stamp:
        with timings.span("current_yolo_paired_backup_write", {"stamp": yolo_paired_sample.stamp.to_dict()}) as timing:
            paired_backup_dir = _save_sample_backup(task, yolo_paired_sample, output_dir, kind="current_yolo_paired")
            timing["backup_dir"] = str(paired_backup_dir)
    with timings.span("current_classify_total") as timing:
        classification = _classify_current(task, cache, baseline, current_sample, yolo_event, yolo_timing, shape, timings=timings)
        timing["status"] = classification.get("status")
        timing["reason"] = classification.get("reason")

    selected_observation = classification.get("selected_observation")
    observation_obj = None
    if selected_observation is not None and yolo_event is not None:
        with timings.span("selected_observation_hydrate") as timing:
            selected_id = str(selected_observation.get("object_id") or "")
            selected_stamp = RosStamp.from_dict(selected_observation.get("stamp") or {})
            timing["selected_object_id"] = selected_id
            for obs in _observations_for_events(cache, [yolo_event], shape):
                if obs.object_id == selected_id and obs.stamp == selected_stamp:
                    observation_obj = obs
                    break
            timing["hydrated"] = observation_obj is not None

    status = str(classification["status"])
    current_pose = classification.get("current_pose_aruco")
    display = _build_display_payload(
        task,
        status=status,
        current_pose_aruco=current_pose if isinstance(current_pose, dict) else None,
    )
    visualization_sample = yolo_paired_sample if observation_obj is not None and yolo_paired_sample is not None else current_sample
    with timings.span("visualization_write", {"status": status}) as timing:
        visualization_path = _save_visualization(
            output_dir,
            visualization_sample,
            status=status,
            observation=observation_obj,
            projection=baseline.get("projection") if isinstance(baseline.get("projection"), dict) else None,
        )
        timing["visualization_path"] = visualization_path

    payload = _write_status(
        json_path,
        task,
        status,
        reason=classification.get("reason"),
        request_source=request_source,
        capture_time_source=capture_source,
        target_time_source=target_source,
        target_timestamp=current_sample.stamp.to_dict(),
        baseline=baseline,
        baseline_info=baseline_info,
        current={
            "sample": current_sample.to_dict(),
            "backup_shigurei_dir": str(current_backup_dir),
            "yolo_paired_sample": yolo_paired_sample.to_dict() if yolo_paired_sample is not None else None,
            "yolo_paired_backup_shigurei_dir": str(paired_backup_dir) if paired_backup_dir is not None else None,
            "metadata_frame_count": len(metadata),
            "yolo_timing": yolo_timing,
        },
        classification=classification,
        display=display,
        state_visualization_path=visualization_path,
        output_dir=str(output_dir),
        timings=timings.events,
    )
    with timings.span("summary_json_write", {"summary_path": str(output_dir / "summary.json")}):
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
