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

from artifact_layout import SHIGURE_HISTORY_CACHE_ROOT, model_debug_dir, model_result_dir, model_worker_dir
from coordinate_systems import quat_xyzw_to_rotation_matrix
from spatial_transforms import (
    aruco_points_to_shigure_camera,
    pixel_depth_to_aruco as spatial_pixel_depth_to_aruco,
)
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
        compact_signature = dict(self.signature)
        for key in ("hist_hs", "hist_ab", "gray_hist", "hu_moments_log"):
            values = compact_signature.pop(key, None)
            if isinstance(values, list):
                compact_signature[f"{key}_bins"] = len(values)
        payload = {
            "stamp": self.stamp.to_dict(),
            "object_id": self.object_id,
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "mask_pixels": int(self.mask_pixels),
            "median_depth_m": self.median_depth_m,
            "yolo_hash": self.event.sample.yolo_hash,
            "signature": compact_signature,
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


_LARGE_SIGNATURE_ARRAY_KEYS = {"hist_hs", "hist_ab", "gray_hist", "hu_moments_log"}


def _compact_feature_arrays(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in _LARGE_SIGNATURE_ARRAY_KEYS and isinstance(item, list):
                compact[f"{key}_bins"] = len(item)
            else:
                compact[str(key)] = _compact_feature_arrays(item)
        return compact
    if isinstance(value, (list, tuple)):
        return [_compact_feature_arrays(v) for v in value]
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


@contextmanager
def _shared_context_lock(shared_context: dict[str, Any] | None, lock_key: str = "_current_lock"):
    lock = None
    if isinstance(shared_context, dict):
        lock = shared_context.get(lock_key)
        if lock is None and lock_key != "_lock":
            lock = shared_context.get("_lock")
    if lock is None:
        yield
        return
    with lock:
        yield


def _write_status(json_path: Path, task: dict[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    payload = _compact_feature_arrays(dict(fields))
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


def _aruco_local_world_up(task: dict[str, Any]) -> np.ndarray:
    reference = task.get("aruco_reference") if isinstance(task.get("aruco_reference"), dict) else None
    if reference and reference.get("rotation_quaternion_xyzw") is not None:
        try:
            rotation = quat_xyzw_to_rotation_matrix(
                _parse_float_array(reference.get("rotation_quaternion_xyzw"), 4, "aruco_reference.rotation_quaternion_xyzw")
            )
            local_up = rotation.T @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            norm = float(np.linalg.norm(local_up))
            if np.isfinite(norm) and norm > 1.0e-9:
                return (local_up / norm).astype(np.float64)
        except Exception:
            pass
    return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _object_height_m(task: dict[str, Any], local_up: np.ndarray | None = None) -> float:
    bounds = task.get("ModelBounds") if isinstance(task.get("ModelBounds"), dict) else None
    if bounds:
        up = np.asarray(local_up if local_up is not None else _aruco_local_world_up(task), dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(up))
        if np.isfinite(norm) and norm > 1.0e-9:
            up = up / norm
            corners = bounds.get("corners_aruco")
            if isinstance(corners, list) and corners:
                try:
                    points = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
                    projected = points @ up
                    height = float(np.nanmax(projected) - np.nanmin(projected))
                    if np.isfinite(height) and height > 0.0:
                        return height
                except Exception:
                    pass
        if bounds.get("aabb_min_aruco") is not None and bounds.get("aabb_max_aruco") is not None:
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
    center_camera = aruco_points_to_shigure_camera(
        center_aruco.reshape(1, 3),
        marker_rotation,
        marker_translation,
    ).reshape(3)
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
    try:
        aruco = spatial_pixel_depth_to_aruco(
            pixel_xy,
            float(depth_m),
            camera_matrix,
            marker_rotation,
            marker_translation,
        )
    except Exception:
        return None
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


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _center_from_mask(mask: np.ndarray, fallback_bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return _bbox_center_xy(fallback_bbox)
    return float(np.mean(xs)), float(np.mean(ys))


def _mask_median_depth(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
    if depth_m.shape != mask.shape or not np.any(mask):
        return None
    values = depth_m[mask]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.nanmedian(values))



def _model_bbox_size_signature(task: dict[str, Any]) -> dict[str, Any]:
    bounds = task.get("ModelBounds") if isinstance(task.get("ModelBounds"), dict) else None
    if not bounds:
        return {"model_bbox_status": "missing"}
    try:
        a = _parse_float_array(bounds.get("aabb_min_aruco"), 3, "ModelBounds.aabb_min_aruco")
        b = _parse_float_array(bounds.get("aabb_max_aruco"), 3, "ModelBounds.aabb_max_aruco")
    except Exception as exc:
        return {"model_bbox_status": "parse_failed", "model_bbox_error": str(exc)}
    dims = np.abs(b - a).astype(np.float64)
    if not np.isfinite(dims).all() or float(np.max(dims)) <= 0.0:
        return {"model_bbox_status": "invalid"}
    sorted_dims = sorted((float(v) for v in dims), reverse=True)
    return {
        "model_bbox_status": "ready",
        "model_bbox_dims_m": [float(v) for v in dims],
        "model_bbox_sorted_dims_m": sorted_dims,
        "model_bbox_top2_area_m2": float(sorted_dims[0] * sorted_dims[1]),
        "model_bbox_volume_m3": float(max(dims[0] * dims[1] * dims[2], 1.0e-9)),
    }


def _pointcloud_size_features(sample: CachedRgbdSample | None, mask: np.ndarray) -> dict[str, Any]:
    if sample is None or sample.depth.shape[:2] != mask.shape:
        return {"point_bbox_status": "sample_unavailable"}
    camera_matrix = _camera_matrix_from_info(sample.camera_info)
    if camera_matrix is None:
        return {"point_bbox_status": "camera_matrix_missing"}
    depth = _sample_depth_m(sample)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < max(32, int(np.count_nonzero(mask) * 0.02)):
        return {"point_bbox_status": "insufficient_depth", "valid_depth_pixels": valid_count}
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    ys, xs = np.nonzero(valid)
    z = depth[valid].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points = (x, y, z)
    lows = [float(np.percentile(values, 5.0)) for values in points]
    highs = [float(np.percentile(values, 95.0)) for values in points]
    dims = [max(float(high - low), 1.0e-6) for low, high in zip(lows, highs)]
    sorted_dims = sorted(dims, reverse=True)
    depth_iqr = float(np.percentile(z, 75.0) - np.percentile(z, 25.0))
    return {
        "point_bbox_status": "ready",
        "valid_depth_pixels": valid_count,
        "valid_depth_ratio": float(valid_count / max(int(np.count_nonzero(mask)), 1)),
        "point_bbox_dims_m": [float(v) for v in dims],
        "point_bbox_sorted_dims_m": [float(v) for v in sorted_dims],
        "point_bbox_top2_area_m2": float(sorted_dims[0] * sorted_dims[1]),
        "point_bbox_depth_extent_m": float(dims[2]),
        "depth_iqr_m": depth_iqr,
    }


def _visual_mask_features(sample: CachedRgbdSample | None, mask: np.ndarray, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    if sample is None or sample.rgb_bgr.shape[:2] != mask.shape or not np.any(mask):
        return {"visual_status": "sample_unavailable"}
    mask_u8 = (mask.astype(np.uint8) * 255)
    color_bgr = sample.rgb_bgr
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hist_hs = cv2.calcHist([hsv], [0, 1], mask_u8, [24, 16], [0, 180, 0, 256]).astype(np.float32)
    hist_hs /= max(float(hist_hs.sum()), 1.0)
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    hist_ab = cv2.calcHist([lab], [1, 2], mask_u8, [20, 20], [0, 256, 0, 256]).astype(np.float32)
    hist_ab /= max(float(hist_ab.sum()), 1.0)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], mask_u8, [32], [0, 256]).astype(np.float32)
    gray_hist /= max(float(gray_hist.sum()), 1.0)

    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = mask.shape
    x0, x1 = max(0, x0), min(w, max(x0 + 1, x1))
    y0, y1 = max(0, y0), min(h, max(y0 + 1, y1))
    crop_gray = gray[y0:y1, x0:x1]
    crop_mask = mask_u8[y0:y1, x0:x1]
    edges = cv2.Canny(crop_gray, 50, 150)
    edge_density = float((edges[crop_mask > 0] > 0).mean()) if np.any(crop_mask > 0) else 0.0
    moments = cv2.moments(mask_u8, binaryImage=True)
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1.0e-30)
    mask_pixels = int(np.count_nonzero(mask))
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    return {
        "visual_status": "ready",
        "hist_hs": [float(v) for v in hist_hs.reshape(-1)],
        "hist_ab": [float(v) for v in hist_ab.reshape(-1)],
        "gray_hist": [float(v) for v in gray_hist.reshape(-1)],
        "hu_moments_log": [float(v) for v in hu[:4]],
        "edge_density": edge_density,
        "mask_extent": float(mask_pixels / float(bbox_area)),
    }


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
    signature.update(_visual_mask_features(sample, mask, bbox))
    signature.update(_pointcloud_size_features(sample, mask))
    return signature



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
    sample: CachedRgbdSample | None = None,
) -> YoloObjectObservation | None:
    bbox = _bbox_from_object(obj, shape)
    if bbox is None:
        return None
    mask = _decode_yolo_mask(obj, shape)
    if mask is None or int(np.count_nonzero(mask)) < settings.YOLO_MIN_MASK_PIXELS:
        return None
    center = _center_from_object(obj, bbox)
    if sample is None:
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
    sample_cache: dict[str, CachedRgbdSample | None] = {}
    for event in events:
        sample = sample_cache.get(event.sample.key)
        if event.sample.key not in sample_cache:
            sample = cache.get_sample(event.sample.stamp, mode="nearest")
            sample_cache[event.sample.key] = sample
        for obj in event.payload.get("objects") or []:
            if isinstance(obj, dict):
                obs = _observation_from_object(cache, event, obj, shape, sample=sample)
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


def _save_baseline_arrays(task_timestamp: str, obs: YoloObjectObservation, sample: CachedRgbdSample) -> dict[str, str]:
    if not task_timestamp:
        raise ValueError("task_timestamp is required for history placement debug artifacts")
    debug_dir = model_debug_dir(task_timestamp)
    debug_dir.mkdir(parents=True, exist_ok=True)
    depth = _sample_depth_m(sample)
    valid = obs.mask & np.isfinite(depth) & (depth > 0.0)
    reference_depth = np.where(valid, depth, 0.0).astype(np.float32)
    mask_path = debug_dir / "09_history_baseline_mask.png"
    depth_path = debug_dir / "09_history_baseline_reference_depth_m.npy"
    cv2.imwrite(str(mask_path), obs.mask.astype(np.uint8) * 255)
    np.save(depth_path, reference_depth)
    return {
        "baseline_mask_path": str(mask_path),
        "baseline_reference_depth_m_path": str(depth_path),
    }



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


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(float(ax0), float(bx0))
    iy0 = max(float(ay0), float(by0))
    ix1 = min(float(ax1), float(bx1))
    iy1 = min(float(ay1), float(by1))
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    intersection = iw * ih
    area_a = max(0.0, float(ax1) - float(ax0)) * max(0.0, float(ay1) - float(ay0))
    area_b = max(0.0, float(bx1) - float(bx0)) * max(0.0, float(by1) - float(by0))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _is_duplicate_tracking_anchor(obs: YoloObjectObservation, anchors: list[YoloObjectObservation]) -> bool:
    for anchor in anchors:
        center_delta = _center_distance(obs.center_xy, anchor.center_xy)
        if center_delta > settings.TRACKING_REGION_DUPLICATE_MAX_CENTER_DELTA_PX:
            continue
        if _bbox_iou(obs.bbox_xyxy, anchor.bbox_xyxy) < settings.TRACKING_REGION_DUPLICATE_MIN_IOU:
            continue
        if obs.median_depth_m is not None and anchor.median_depth_m is not None:
            depth_delta = abs(float(obs.median_depth_m) - float(anchor.median_depth_m))
            if depth_delta > settings.TRACKING_REGION_DUPLICATE_MAX_DEPTH_DELTA_M:
                continue
        return True
    return False


def _tracking_region_ids() -> set[str]:
    return {str(value) for value in settings.TRACKING_REGION_YOLO_IDS if str(value).strip()}


def _build_tracking_search_region(
    cache: ShigureRgbdCache,
    events: list[YoloEvent],
    shape: tuple[int, int],
    *,
    reference_region: dict[str, Any] | None = None,
    observations: list[YoloObjectObservation] | None = None,
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

    all_observations = observations if observations is not None else _observations_for_events(cache, events, shape)
    observations = [obs for obs in all_observations if obs.object_id in configured_ids]

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
    duplicate_anchor_ids = {
        obs.object_id
        for obs in all_observations
        if obs.object_id not in configured_ids and _is_duplicate_tracking_anchor(obs, observations)
    }
    excluded_candidate_ids = set(configured_ids) | duplicate_anchor_ids

    region: dict[str, Any] = {
        "source": "configured_yolo_ids",
        "configured_yolo_ids": sorted(configured_ids),
        "matched_yolo_ids": sorted({obs.object_id for obs in observations}),
        "overlap_excluded_yolo_ids": sorted(duplicate_anchor_ids),
        "excluded_candidate_yolo_ids": sorted(excluded_candidate_ids),
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
        reference_bbox = reference_region.get("bbox_xyxy")
        bbox_static = False
        if isinstance(reference_bbox, list) and len(reference_bbox) == 4 and bbox is not None:
            try:
                ref_bbox_tuple = tuple(float(v) for v in reference_bbox)
                bbox_iou = _bbox_iou(tuple(float(v) for v in bbox), ref_bbox_tuple)
                ix0 = max(float(bbox[0]), ref_bbox_tuple[0])
                iy0 = max(float(bbox[1]), ref_bbox_tuple[1])
                ix1 = min(float(bbox[2]), ref_bbox_tuple[2])
                iy1 = min(float(bbox[3]), ref_bbox_tuple[3])
                intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                ref_area = max(0.0, ref_bbox_tuple[2] - ref_bbox_tuple[0]) * max(0.0, ref_bbox_tuple[3] - ref_bbox_tuple[1])
                reference_coverage = intersection / ref_area if ref_area > 0.0 else 0.0
                bbox_static = bbox_iou >= 0.45 or reference_coverage >= 0.80
                checks["support_bbox_iou"] = bbox_iou
                checks["support_bbox_reference_coverage"] = reference_coverage
                checks["support_bbox_static"] = bbox_static
            except Exception:
                bbox_static = False
        reference_center = reference_region.get("support_center_xy")
        if isinstance(reference_center, list) and len(reference_center) >= 2:
            center_delta = _center_distance((float(reference_center[0]), float(reference_center[1])), (center_xy[0], center_xy[1]))
            checks["support_center_delta_px"] = center_delta
            checks["support_center_static"] = bbox_static or center_delta <= settings.TRACKING_REGION_SUPPORT_STATIC_CENTER_DELTA_PX
        region["support_plane_static_check"] = checks
        if any(value is False for value in checks.values() if isinstance(value, bool)):
            region["valid"] = False
            region["reason"] = "support_plane_moved"
    return region



def _resolve_artifact_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = CODE_ROOT.parent / path
    return path


def _load_backup_sample(backup_dir: Path) -> CachedRgbdSample | None:
    if not backup_dir.is_dir():
        return None
    meta_path = backup_dir / "meta.json"
    meta = load_json(meta_path) if meta_path.is_file() else {}
    stamp_payload = meta.get("stamp") if isinstance(meta.get("stamp"), dict) else None
    if stamp_payload is None:
        return None
    rgb_path = backup_dir / "rgb.png"
    depth_path = backup_dir / "depth.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path.is_file() else None
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
    if rgb is None or depth is None:
        return None
    camera_info_path = backup_dir / "camera_info.json"
    camera_info = load_json(camera_info_path) if camera_info_path.is_file() else None
    yolo_path = backup_dir / "yolo.json"
    if not yolo_path.is_file():
        yolo_path = backup_dir / "active_objects.json"
    yolo = load_json(yolo_path) if yolo_path.is_file() else None
    return CachedRgbdSample(
        stamp=RosStamp.from_dict(stamp_payload),
        rgb_bgr=rgb,
        depth=depth,
        camera_info_path=camera_info_path if camera_info_path.is_file() else None,
        camera_info=camera_info,
        yolo_path=yolo_path if yolo_path.is_file() else None,
        yolo=yolo,
        yolo_hash=str(meta.get("source_yolo_hash") or "") or None,
        chunk_id=str(meta.get("source_chunk_id") or backup_dir.name),
        frame_index=int(meta.get("source_frame_index") or 0),
        rgb_path=rgb_path,
        depth_path=depth_path,
    )


def _taken_detection_payload(task: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    payload = task.get("TakenObjectDetection")
    if isinstance(payload, dict) and payload:
        return payload, "task.TakenObjectDetection"
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if task_timestamp:
        result_path = model_result_dir(task_timestamp) / "07_taken_detection_result.json"
        if result_path.is_file():
            try:
                return load_json(result_path), str(result_path)
            except Exception:
                return None, f"failed_to_load:{result_path}"
    return None, "missing"


def _with_trusted_mask(
    obs: YoloObjectObservation,
    sample: CachedRgbdSample,
    trusted_mask: np.ndarray | None,
) -> YoloObjectObservation:
    if trusted_mask is None or trusted_mask.shape != obs.mask.shape or not np.any(trusted_mask):
        return obs
    mask = trusted_mask.astype(bool)
    bbox = _bbox_from_mask(mask) or obs.bbox_xyxy
    center = _center_from_mask(mask, bbox)
    depth = _sample_depth_m(sample)
    median_depth = _mask_median_depth(depth, mask)
    camera_info = sample.camera_info or obs.event.sample.camera_info
    aruco_position = _pixel_depth_to_aruco(center, median_depth, camera_info)
    signature = _mask_signature(sample, mask, bbox, median_depth)
    return YoloObjectObservation(
        event=obs.event,
        object_id=obs.object_id,
        center_xy=center,
        bbox_xyxy=bbox,
        mask=mask,
        mask_pixels=int(np.count_nonzero(mask)),
        median_depth_m=median_depth,
        aruco_position=aruco_position,
        signature=signature,
    )


def _restore_baseline_from_taken_detection(
    task: dict[str, Any],
    cache: ShigureRgbdCache,
    *,
    capture_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    payload, payload_source = _taken_detection_payload(task)
    if not isinstance(payload, dict):
        return None, {"reason": "taken_detection_payload_missing", "payload_source": payload_source}
    init = payload.get("init") if isinstance(payload.get("init"), dict) else {}
    explicit_history_baseline = payload.get("history_baseline") if isinstance(payload.get("history_baseline"), dict) else None
    if explicit_history_baseline is None and isinstance(init.get("history_baseline"), dict):
        explicit_history_baseline = init.get("history_baseline")
    yolo_tracking = payload.get("yolo_tracking") if isinstance(payload.get("yolo_tracking"), dict) else {}
    backup_dir = _resolve_artifact_path(
        payload.get("init_backup_shigurei_dir")
        or init.get("init_backup_shigurei_dir")
        or init.get("backup_shigurei_dir")
        or yolo_tracking.get("init_backup_shigurei_dir")
    )
    if backup_dir is None:
        return None, {"reason": "taken_init_backup_missing", "payload_source": payload_source}
    sample = _load_backup_sample(backup_dir)
    if sample is None or not isinstance(sample.yolo, dict):
        return None, {"reason": "taken_init_backup_unreadable", "payload_source": payload_source, "backup_dir": str(backup_dir)}

    shape = sample.depth.shape[:2]
    sample_metadata = CachedSampleMetadata(
        stamp=sample.stamp,
        camera_info_path=sample.camera_info_path,
        camera_info=sample.camera_info,
        yolo_path=sample.yolo_path,
        yolo_hash=sample.yolo_hash,
        chunk_id=sample.chunk_id,
        frame_index=sample.frame_index,
    )
    event = YoloEvent(sample=sample_metadata, payload=sample.yolo)
    init_obs = init.get("selected_observation") if isinstance(init.get("selected_observation"), dict) else {}
    target_object_id = str(
        init_obs.get("object_id")
        or init.get("target_object_id")
        or yolo_tracking.get("tracked_object_id")
        or ""
    ).strip()
    if not target_object_id:
        return None, {"reason": "taken_init_object_id_missing", "payload_source": payload_source, "backup_dir": str(backup_dir)}

    all_observations: list[YoloObjectObservation] = []
    selected_obs: YoloObjectObservation | None = None
    needed_object_ids = set(_tracking_region_ids())
    needed_object_ids.add(target_object_id)
    skipped_object_count = 0
    for obj in sample.yolo.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("object_id"))
        if object_id not in needed_object_ids:
            skipped_object_count += 1
            continue
        obs = _observation_from_object(cache, event, obj, shape, sample=sample)
        if obs is None:
            continue
        all_observations.append(obs)
        if obs.object_id == target_object_id:
            selected_obs = obs
    if selected_obs is None:
        return None, {
            "reason": "taken_init_object_not_found_in_yolo_backup",
            "payload_source": payload_source,
            "backup_dir": str(backup_dir),
            "target_object_id": target_object_id,
            "observation_count": len(all_observations),
            "skipped_object_count": skipped_object_count,
        }

    debug_files = payload.get("debug_files") if isinstance(payload.get("debug_files"), dict) else {}
    init_debug_files = init.get("debug_files") if isinstance(init.get("debug_files"), dict) else {}
    yolo_debug_files = yolo_tracking.get("debug_files") if isinstance(yolo_tracking.get("debug_files"), dict) else {}
    trusted_mask_path = _resolve_artifact_path(
        debug_files.get("init_trusted_shigure_mask_path")
        or init_debug_files.get("init_trusted_shigure_mask_path")
        or yolo_debug_files.get("init_trusted_shigure_mask_path")
    )
    trusted_mask = None
    if trusted_mask_path is not None and trusted_mask_path.is_file():
        raw_mask = cv2.imread(str(trusted_mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is not None:
            trusted_mask = raw_mask > 0
    selected_obs = _with_trusted_mask(selected_obs, sample, trusted_mask)

    reference_depth_path = _resolve_artifact_path(
        debug_files.get("init_reference_depth_m_path")
        or init_debug_files.get("init_reference_depth_m_path")
        or yolo_debug_files.get("init_reference_depth_m_path")
    )
    baseline_mask_path = trusted_mask_path if trusted_mask_path is not None and trusted_mask_path.is_file() else None
    array_files: dict[str, str]
    if baseline_mask_path is not None and reference_depth_path is not None and reference_depth_path.is_file():
        array_files = {
            "baseline_mask_path": str(baseline_mask_path),
            "baseline_reference_depth_m_path": str(reference_depth_path),
        }
    else:
        array_files = _save_baseline_arrays(str(task.get("task_timestamp") or "").strip(), selected_obs, sample)

    tracking_region_reference = _build_tracking_search_region(
        cache,
        [event],
        shape,
        reference_region=None,
        observations=all_observations,
    )
    reference_signature = dict(selected_obs.signature)
    reference_signature.update(_model_bbox_size_signature(task))
    projection = payload.get("projection") if isinstance(payload.get("projection"), dict) else init.get("projection") if isinstance(init.get("projection"), dict) else {}
    match = init.get("candidate_match") if isinstance(init.get("candidate_match"), dict) else {}
    history_baseline = explicit_history_baseline or {
        "source": "taken_detection_init_recovered",
        "coordinate_space": "fixed_shigure_image",
        "old_rgb_path": str(backup_dir / "rgb.png"),
        "old_depth_path": str(backup_dir / "depth.png"),
        "old_mask_path": str(array_files.get("baseline_mask_path") or ""),
        "old_reference_depth_m_path": str(array_files.get("baseline_reference_depth_m_path") or ""),
        "camera_info_path": str(backup_dir / "camera_info.json"),
    }
    baseline = {
        "status": "ready",
        "created_at": _utc_now(),
        "source": "taken_detection_init",
        "history_baseline": history_baseline,
        "source_payload": payload_source,
        "capture_seconds": float(capture_seconds),
        "reference_observation": selected_obs.to_dict(),
        "reference_signature": reference_signature,
        "projection": projection,
        "match": match,
        "baseline_visibility": {
            "reason": "taken_detection_yolo_init",
            "seconds_from_capture": float(selected_obs.seconds - capture_seconds),
            "partial_initialization": False,
            "valid_mask_pixels": int(selected_obs.mask_pixels),
        },
        "tracking_search_region_reference": tracking_region_reference,
        "baseline_backup_dir": str(backup_dir),
        **array_files,
    }
    return baseline, {
        "reason": "baseline_reused_taken_detection_init",
        "payload_source": payload_source,
        "backup_dir": str(backup_dir),
        "target_object_id": target_object_id,
        "observation_count": len(all_observations),
        "skipped_object_count": skipped_object_count,
        "trusted_mask_path": str(trusted_mask_path) if trusted_mask_path is not None else None,
        "reference_depth_path": str(reference_depth_path) if reference_depth_path is not None else None,
        "baseline": baseline,
    }


def _establish_baseline(
    json_path: Path,
    task: dict[str, Any],
    cache: ShigureRgbdCache,
    output_dir: Path,
    *,
    capture_seconds: float,
    timings: StageTimingCollector | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    with _timing_span(timings, "baseline_taken_detection_restore") as timing:
        restored_baseline, restored_info = _restore_baseline_from_taken_detection(
            task,
            cache,
            capture_seconds=capture_seconds,
        )
        timing["reason"] = restored_info.get("reason")
        timing["restored"] = restored_baseline is not None
        timing["payload_source"] = restored_info.get("payload_source")
        timing["backup_dir"] = restored_info.get("backup_dir")
    if restored_baseline is not None:
        return restored_baseline, restored_info

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
        array_files = _save_baseline_arrays(str(task.get("task_timestamp") or "").strip(), selected_obs, selected_sample)
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
    reference_signature = dict(selected_obs.signature)
    reference_signature.update(_model_bbox_size_signature(task))
    history_baseline = {
        "source": "history_baseline_established_from_shigure_object_mask",
        "coordinate_space": "fixed_shigure_image",
        "old_rgb_path": str(backup_dir / "rgb.png"),
        "old_depth_path": str(backup_dir / "depth.png"),
        "old_mask_path": str(array_files.get("baseline_mask_path") or ""),
        "old_reference_depth_m_path": str(array_files.get("baseline_reference_depth_m_path") or ""),
        "camera_info_path": str(backup_dir / "camera_info.json"),
    }
    baseline = {
        "status": "ready",
        "created_at": _utc_now(),
        "history_baseline": history_baseline,
        "capture_seconds": float(capture_seconds),
        "reference_observation": selected_obs.to_dict(),
        "reference_signature": reference_signature,
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


def _polyhedron_pose(position: list[float] | None, task: dict[str, Any]) -> dict[str, Any] | None:
    if not position or len(position) < 3:
        return None
    local_up = _aruco_local_world_up(task)
    height = _object_height_m(task, local_up)
    offset = max(0.0, height * 0.5) + settings.POLYHEDRON_ABOVE_MARGIN_M + settings.POLYHEDRON_EDGE_LENGTH_M * 0.5
    base = np.asarray([float(position[0]), float(position[1]), float(position[2])], dtype=np.float64)
    pos = base + local_up * offset
    return {
        "position": [float(v) for v in pos],
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "source": "aruco_local_up_hint",
        "local_up_aruco": [float(v) for v in local_up],
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
        display["mode"] = "still_octahedron_no_model"
        display["show_model"] = False
        display["polyhedron"] = _polyhedron_payload(task, "octahedron", original_position, "original_model")
    elif status == STATUS_MOVED and current_pose_aruco is not None:
        display["mode"] = "moved_cube_no_model"
        display["show_model"] = False
        display["polyhedron"] = _polyhedron_payload(task, "cube", current_position, "current_object")
        display["animation"]["enabled"] = original_pose is not None
        display["animation"]["from_pose_aruco"] = current_pose_aruco
    elif status == STATUS_MOVED:
        display["mode"] = "moved_cube_no_model"
        display["show_model"] = False
        display["polyhedron"] = _polyhedron_payload(task, "cube", original_position, "original_model")
    elif status == STATUS_MISSING:
        display["mode"] = "missing_cube_no_model"
        display["show_model"] = False
        display["polyhedron"] = _polyhedron_payload(task, "cube", original_position, "original_model")
    elif status == STATUS_OCCLUDED_REUSE_LAST:
        display["mode"] = "occluded_tetrahedron_no_model"
        display["show_model"] = False
        display["polyhedron"] = _polyhedron_payload(task, "tetrahedron", original_position, "original_model")
    elif status == STATUS_UNKNOWN:
        display["mode"] = "unknown_dodecahedron_no_model"
        display["show_model"] = False
        display["polyhedron"] = _polyhedron_payload(task, "dodecahedron", original_position, "original_model")
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
    path = output_dir / ("09_history_state_visualization.png" if output_dir.name == "worker" else "state_visualization.png")
    cv2.imwrite(str(path), image)
    return str(path)


def _load_direct_baseline_assets(baseline: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    history = baseline.get("history_baseline") if isinstance(baseline.get("history_baseline"), dict) else {}
    backup_dir = _resolve_artifact_path(baseline.get("baseline_backup_dir"))

    old_rgb_path = _resolve_artifact_path(history.get("old_rgb_path"))
    if old_rgb_path is None and backup_dir is not None:
        old_rgb_path = backup_dir / "rgb.png"
    old_depth_path = _resolve_artifact_path(history.get("old_depth_path"))
    if old_depth_path is None and backup_dir is not None:
        old_depth_path = backup_dir / "depth.png"
    old_mask_path = _resolve_artifact_path(history.get("old_mask_path"))
    if old_mask_path is None:
        old_mask_path = _resolve_artifact_path(baseline.get("baseline_mask_path"))
    old_reference_depth_path = _resolve_artifact_path(history.get("old_reference_depth_m_path"))
    if old_reference_depth_path is None:
        old_reference_depth_path = _resolve_artifact_path(baseline.get("baseline_reference_depth_m_path"))

    old_rgb = cv2.imread(str(old_rgb_path), cv2.IMREAD_COLOR) if old_rgb_path is not None and old_rgb_path.is_file() else None
    old_depth_raw = cv2.imread(str(old_depth_path), cv2.IMREAD_UNCHANGED) if old_depth_path is not None and old_depth_path.is_file() else None
    old_depth = _depth_raw_to_m(old_depth_raw) if old_depth_raw is not None else None
    old_mask = None
    if old_mask_path is not None and old_mask_path.is_file():
        raw = cv2.imread(str(old_mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            old_mask = raw > 0
    if old_reference_depth_path is not None and old_reference_depth_path.is_file():
        try:
            reference_depth = np.load(str(old_reference_depth_path)).astype(np.float32)
            if old_depth is None or old_depth.shape != reference_depth.shape:
                old_depth = reference_depth
        except Exception:
            pass
    return old_rgb, old_depth, old_mask, {
        "old_rgb_path": str(old_rgb_path) if old_rgb_path is not None else None,
        "old_depth_path": str(old_depth_path) if old_depth_path is not None else None,
        "old_mask_path": str(old_mask_path) if old_mask_path is not None else None,
        "old_reference_depth_m_path": str(old_reference_depth_path) if old_reference_depth_path is not None else None,
        "baseline_backup_dir": str(backup_dir) if backup_dir is not None else None,
    }


def _write_direct_compare_debug(task: dict[str, Any], current_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        return {}
    debug_dir = model_debug_dir(task_timestamp)
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}
    for name, mask in masks.items():
        if mask is None:
            continue
        path = debug_dir / f"09_history_direct_{name}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        paths[f"direct_{name}_path"] = str(path)
    changed = masks.get("changed_mask")
    nearer = masks.get("nearer_mask")
    farther = masks.get("farther_mask")
    if changed is not None and current_rgb.shape[:2] == changed.shape:
        overlay = np.asarray(current_rgb).copy()
        if farther is not None:
            overlay[farther] = cv2.addWeighted(
                np.full_like(overlay[farther], (255, 170, 40)), 0.55, overlay[farther], 0.45, 0.0
            )
        if nearer is not None:
            overlay[nearer] = cv2.addWeighted(
                np.full_like(overlay[nearer], (40, 220, 255)), 0.55, overlay[nearer], 0.45, 0.0
            )
        path = debug_dir / "09_history_direct_compare_overlay.png"
        cv2.imwrite(str(path), overlay)
        paths["direct_compare_overlay_path"] = str(path)
    return paths


def _classify_current_direct(
    task: dict[str, Any],
    baseline: dict[str, Any],
    current_sample: CachedRgbdSample,
) -> dict[str, Any]:
    old_rgb, old_depth, old_mask, asset_info = _load_direct_baseline_assets(baseline)
    current_rgb = np.asarray(current_sample.rgb_bgr)
    current_depth = _sample_depth_m(current_sample)
    if old_rgb is None or old_depth is None or old_mask is None:
        return {
            "status": STATUS_UNKNOWN,
            "reason": "direct_baseline_assets_missing",
            "selected_observation": None,
            "current_pose_aruco": None,
            "validation": {"direct_compare": asset_info},
        }
    shape = current_depth.shape
    if old_mask.shape != shape:
        old_mask = cv2.resize(old_mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if old_rgb.shape[:2] != shape or old_depth.shape != shape:
        return {
            "status": STATUS_UNKNOWN,
            "reason": "direct_baseline_shape_mismatch",
            "selected_observation": None,
            "current_pose_aruco": None,
            "validation": {
                "direct_compare": {
                    **asset_info,
                    "old_rgb_shape": list(old_rgb.shape[:2]) if old_rgb is not None else None,
                    "old_depth_shape": list(old_depth.shape) if old_depth is not None else None,
                    "old_mask_shape": list(old_mask.shape) if old_mask is not None else None,
                    "current_shape": list(shape),
                }
            },
        }

    valid = old_mask & np.isfinite(old_depth) & (old_depth > 0.0) & np.isfinite(current_depth) & (current_depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels < int(settings.DIRECT_COMPARE_MIN_VALID_PIXELS):
        return {
            "status": STATUS_UNKNOWN,
            "reason": "direct_compare_not_enough_valid_pixels",
            "selected_observation": None,
            "current_pose_aruco": None,
            "validation": {"direct_compare": {**asset_info, "valid_pixels": valid_pixels}},
        }

    depth_delta = current_depth - old_depth
    threshold = float(settings.DIRECT_COMPARE_DEPTH_DELTA_M)
    changed = valid & (np.abs(depth_delta) > threshold)
    nearer = valid & (depth_delta < -threshold)
    farther = valid & (depth_delta > threshold)
    old_lab = cv2.cvtColor(old_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_lab = cv2.cvtColor(current_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_delta = np.linalg.norm(current_lab - old_lab, axis=2)
    rgb_changed = valid & (lab_delta > float(settings.DIRECT_COMPARE_RGB_LAB_DELTA_THRESHOLD))

    depth_changed_ratio = float(np.count_nonzero(changed) / max(1, valid_pixels))
    depth_nearer_ratio = float(np.count_nonzero(nearer) / max(1, valid_pixels))
    depth_farther_ratio = float(np.count_nonzero(farther) / max(1, valid_pixels))
    rgb_changed_ratio = float(np.count_nonzero(rgb_changed) / max(1, valid_pixels))
    lab_mean = float(np.nanmean(lab_delta[valid])) if valid_pixels > 0 else None

    metrics = {
        **asset_info,
        "source": "fixed_shigure_old_mask_direct_compare",
        "coordinate_space": "fixed_shigure_image",
        "valid_pixels": valid_pixels,
        "mask_pixels": int(np.count_nonzero(old_mask)),
        "depth_delta_threshold_m": threshold,
        "depth_changed_ratio": depth_changed_ratio,
        "depth_nearer_ratio": depth_nearer_ratio,
        "depth_farther_ratio": depth_farther_ratio,
        "rgb_lab_delta_threshold": float(settings.DIRECT_COMPARE_RGB_LAB_DELTA_THRESHOLD),
        "rgb_lab_mean_delta": lab_mean,
        "rgb_changed_ratio": rgb_changed_ratio,
        "occluded_nearer_ratio_threshold": float(settings.DIRECT_COMPARE_OCCLUDED_NEARER_RATIO),
        "missing_farther_ratio_threshold": float(settings.DIRECT_COMPARE_MISSING_FARTHER_RATIO),
        "still_max_depth_changed_ratio": float(settings.DIRECT_COMPARE_STILL_MAX_DEPTH_CHANGED_RATIO),
        "still_max_rgb_changed_ratio": float(settings.DIRECT_COMPARE_STILL_MAX_RGB_CHANGED_RATIO),
    }
    debug_paths = _write_direct_compare_debug(
        task,
        current_rgb,
        {
            "old_mask": old_mask,
            "changed_mask": changed,
            "nearer_mask": nearer,
            "farther_mask": farther,
            "rgb_changed_mask": rgb_changed,
        },
    )
    metrics.update(debug_paths)

    if depth_nearer_ratio >= float(settings.DIRECT_COMPARE_OCCLUDED_NEARER_RATIO):
        status = STATUS_OCCLUDED_REUSE_LAST
        reason = "direct_compare_occluded_nearer_depth"
    elif depth_farther_ratio >= float(settings.DIRECT_COMPARE_MISSING_FARTHER_RATIO):
        status = STATUS_MISSING
        reason = "direct_compare_not_in_original_place_farther_depth"
    elif (
        depth_changed_ratio <= float(settings.DIRECT_COMPARE_STILL_MAX_DEPTH_CHANGED_RATIO)
        and rgb_changed_ratio <= float(settings.DIRECT_COMPARE_STILL_MAX_RGB_CHANGED_RATIO)
    ):
        status = STATUS_ORIGINAL
        reason = "direct_compare_still_in_original_place"
    else:
        status = STATUS_UNKNOWN
        reason = "direct_compare_ambiguous"

    return {
        "status": status,
        "reason": reason,
        "selected_observation": None,
        "current_pose_aruco": None,
        "validation": {"direct_compare": metrics},
        "direct_compare": metrics,
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


def prepare_shared_current_context(
    json_path_arg: str | Path,
    *,
    target_time: str | None = None,
    shared_current_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preload request-wide current Shigure/Yolo context for parallel item workers."""
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    shared_context = shared_current_context if isinstance(shared_current_context, dict) else {}
    events: list[dict[str, Any]] = []

    @contextmanager
    def span(event_name: str, detail: dict[str, Any] | None = None):
        started = time.perf_counter()
        payload = dict(detail or {})
        event: dict[str, Any] = {"event_name": event_name, "detail": payload}
        try:
            yield payload
            event["status"] = "ok"
        except Exception as exc:
            event["status"] = "error"
            event["error"] = str(exc)
            raise
        finally:
            event["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            events.append(_jsonable(event))

    started_total = time.perf_counter()
    target_context_key = str(target_time or "")
    with _shared_context_lock(shared_context):
        with _shared_context_lock(shared_context, "_cache_lock"):
            with span("cache_initialize", {"cache_root": str(SHIGURE_HISTORY_CACHE_ROOT)}) as timing:
                cached_cache = shared_context.get("cache") if shared_context is not None else None
                if isinstance(cached_cache, ShigureRgbdCache):
                    cache = cached_cache
                    timing["shared_context_hit"] = True
                else:
                    cache = ShigureRgbdCache(SHIGURE_HISTORY_CACHE_ROOT)
                    timing["shared_context_hit"] = False
                    shared_context["cache"] = cache

        with span("target_time_resolve") as timing:
            shared_target = shared_context.get("target")
            if isinstance(shared_target, dict) and shared_target.get("request_target_time") == target_context_key:
                target_seconds = shared_target.get("target_seconds")
                target_source = str(shared_target.get("target_source") or "shared_current_context")
                timing["shared_context_hit"] = True
            else:
                target_seconds, target_source = _resolve_target_seconds(cache, task, target_time)
                timing["shared_context_hit"] = False
                shared_context["target"] = {
                    "request_target_time": target_context_key,
                    "target_seconds": float(target_seconds) if target_seconds is not None else None,
                    "target_source": target_source,
                }
            timing["target_time_source"] = target_source
            timing["target_seconds_available"] = target_seconds is not None
        if target_seconds is None:
            return _jsonable({
                "success": False,
                "reason": "target_time_missing",
                "elapsed_seconds": round(time.perf_counter() - started_total, 3),
                "timings": events,
            })

        target_stamp = _stamp_from_seconds(target_seconds)
        with span("current_sample_load", {"target_stamp": target_stamp.to_dict()}) as timing:
            shared_sample = shared_context.get("current_sample")
            if (
                isinstance(shared_sample, dict)
                and abs(float(shared_sample.get("target_seconds", float("nan"))) - float(target_seconds)) < 1e-6
                and isinstance(shared_sample.get("sample"), CachedRgbdSample)
            ):
                current_sample = shared_sample["sample"]
                timing["shared_context_hit"] = True
            else:
                current_sample = cache.get_sample(target_stamp, mode="before") or cache.get_sample(target_stamp, mode="nearest")
                timing["shared_context_hit"] = False
                if current_sample is not None:
                    shared_context["current_sample"] = {"target_seconds": float(target_seconds), "sample": current_sample}
            timing["sample_available"] = current_sample is not None
            if current_sample is not None:
                timing["sample_stamp"] = current_sample.stamp.to_dict()
        if current_sample is None:
            return _jsonable({
                "success": False,
                "reason": "no_shigure_history",
                "target_seconds": float(target_seconds),
                "target_time_source": target_source,
                "elapsed_seconds": round(time.perf_counter() - started_total, 3),
                "timings": events,
            })

    return _jsonable({
        "success": True,
        "target_seconds": float(target_seconds),
        "target_time_source": target_source,
        "current_sample": current_sample.to_dict(),
        "metadata_frame_count": None,
        "has_yolo_event": False,
        "yolo_timing": {"status": "skipped_fixed_shigure_direct_compare"},
        "observation_count": 0,
        "elapsed_seconds": round(time.perf_counter() - started_total, 3),
        "timings": events,
    })


def run_history_placement_restoration(
    json_path_arg: str | Path,
    *,
    target_time: str | None = None,
    request_source: str = "stage",
    artifact_output_dir: str | Path | None = None,
    shared_current_context: dict[str, Any] | None = None,
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
        output_dir = model_worker_dir(task_timestamp)
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

    shared_context = shared_current_context if isinstance(shared_current_context, dict) else None
    with _shared_context_lock(shared_context, "_cache_lock"):
        with timings.span(
            "cache_initialize",
            {"cache_root": str(SHIGURE_HISTORY_CACHE_ROOT), "shared_context_enabled": shared_context is not None},
        ) as timing:
            cached_cache = shared_context.get("cache") if shared_context is not None else None
            if isinstance(cached_cache, ShigureRgbdCache):
                cache = cached_cache
                timing["shared_context_hit"] = True
            else:
                cache = ShigureRgbdCache(SHIGURE_HISTORY_CACHE_ROOT)
                timing["shared_context_hit"] = False
                if shared_context is not None:
                    shared_context["cache"] = cache
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

    with _shared_context_lock(shared_context):
        target_context_key = str(target_time or "")
        with timings.span("target_time_resolve") as timing:
            shared_target = shared_context.get("target") if shared_context is not None else None
            if isinstance(shared_target, dict) and shared_target.get("request_target_time") == target_context_key:
                target_seconds = shared_target.get("target_seconds")
                target_source = str(shared_target.get("target_source") or "shared_current_context")
                timing["shared_context_hit"] = True
            else:
                target_seconds, target_source = _resolve_target_seconds(cache, task, target_time)
                timing["shared_context_hit"] = False
                if shared_context is not None:
                    shared_context["target"] = {
                        "request_target_time": target_context_key,
                        "target_seconds": float(target_seconds) if target_seconds is not None else None,
                        "target_source": target_source,
                    }
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
            shared_sample = shared_context.get("current_sample") if shared_context is not None else None
            if (
                isinstance(shared_sample, dict)
                and abs(float(shared_sample.get("target_seconds", float("nan"))) - float(target_seconds)) < 1e-6
                and isinstance(shared_sample.get("sample"), CachedRgbdSample)
            ):
                current_sample = shared_sample["sample"]
                timing["shared_context_hit"] = True
            else:
                current_sample = cache.get_sample(target_stamp, mode="before") or cache.get_sample(target_stamp, mode="nearest")
                timing["shared_context_hit"] = False
                if shared_context is not None and current_sample is not None:
                    shared_context["current_sample"] = {"target_seconds": float(target_seconds), "sample": current_sample}
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

        with timings.span("current_backup_write", {"stamp": current_sample.stamp.to_dict()}) as timing:
            current_backup_dir = _save_sample_backup(task, current_sample, output_dir, kind="current")
            timing["backup_dir"] = str(current_backup_dir)

        with timings.span("current_direct_compare_total") as timing:
            classification = _classify_current_direct(task, baseline, current_sample)
            timing["status"] = classification.get("status")
            timing["reason"] = classification.get("reason")

        status = str(classification["status"])
        display = _build_display_payload(task, status=status, current_pose_aruco=None)
        with timings.span("visualization_write", {"status": status}) as timing:
            visualization_path = _save_visualization(
                output_dir,
                current_sample,
                status=status,
                observation=None,
                projection=None,
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
                "metadata_frame_count": None,
                "yolo_timing": {"status": "skipped_fixed_shigure_direct_compare"},
            },
            classification=classification,
            direct_compare=classification.get("direct_compare"),
            display=display,
            state_visualization_path=visualization_path,
            output_dir=str(output_dir),
            timings=timings.events,
        )
        summary_path = output_dir / ("09_history_summary.json" if output_dir.name == "worker" else "summary.json")
        with timings.span("summary_json_write", {"summary_path": str(summary_path)}):
            _write_json(summary_path, payload)
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
