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


def _cap01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _as_float_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float32)


def _log_distance(a: float, b: float, scale: float) -> float:
    a = max(float(a or 0.0), 1.0e-9)
    b = max(float(b or 0.0), 1.0e-9)
    return _cap01(abs(math.log(a / b)) / max(float(scale), 1.0e-9))


def _bhattacharyya(a: Any, b: Any) -> float:
    left = _as_float_array(a)
    right = _as_float_array(b)
    if left.size != right.size or left.size == 0:
        return 1.0
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    left /= max(float(left.sum()), 1.0e-9)
    right /= max(float(right.sum()), 1.0e-9)
    coeff = float(np.sum(np.sqrt(left * right)))
    return _cap01(math.sqrt(max(0.0, 1.0 - coeff)))


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


def _sorted_dim_distance(candidate_dims: Any, reference_dims: Any) -> float:
    c = sorted((float(v) for v in _as_float_array(candidate_dims) if float(v) > 0.0), reverse=True)
    r = sorted((float(v) for v in _as_float_array(reference_dims) if float(v) > 0.0), reverse=True)
    if len(c) < 2 or len(r) < 2:
        return 1.0
    top1 = _log_distance(c[0], r[0], 1.2)
    top2 = _log_distance(c[1], r[1], 1.2)
    area = _log_distance(c[0] * c[1], r[0] * r[1], 1.5)
    return float(0.42 * top1 + 0.42 * top2 + 0.16 * area)


def _signature_score(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    visual_score = (
        0.44 * _bhattacharyya(candidate.get("hist_hs"), reference.get("hist_hs"))
        + 0.36 * _bhattacharyya(candidate.get("hist_ab"), reference.get("hist_ab"))
        + 0.20 * _bhattacharyya(candidate.get("gray_hist"), reference.get("gray_hist"))
    )

    candidate_hu = _as_float_array(candidate.get("hu_moments_log"))[:4]
    reference_hu = _as_float_array(reference.get("hu_moments_log"))[:4]
    hu_score = 1.0 if candidate_hu.size != 4 or reference_hu.size != 4 else _cap01(float(np.linalg.norm(candidate_hu - reference_hu) / 12.0))
    extent_score = abs(float(candidate.get("mask_extent") or 0.0) - float(reference.get("mask_extent") or 0.0))
    edge_score = abs(float(candidate.get("edge_density") or 0.0) - float(reference.get("edge_density") or 0.0))
    aspect_score = _log_distance(float(candidate.get("aspect_ratio") or 0.0), float(reference.get("aspect_ratio") or 0.0), 1.0)
    shape_score = float(0.42 * hu_score + 0.24 * edge_score + 0.20 * extent_score + 0.14 * aspect_score)

    model_reference_dims = reference.get("model_bbox_sorted_dims_m") or reference.get("point_bbox_sorted_dims_m")
    candidate_dims = sorted((float(v) for v in _as_float_array(candidate.get("point_bbox_sorted_dims_m")) if float(v) > 0.0), reverse=True)
    reference_dims = sorted((float(v) for v in _as_float_array(model_reference_dims) if float(v) > 0.0), reverse=True)
    model_size_score = _sorted_dim_distance(candidate_dims, reference_dims)
    model_largest_dim_ratio = None
    model_top2_area_ratio = None
    if len(candidate_dims) >= 2 and len(reference_dims) >= 2:
        model_largest_dim_ratio = float(candidate_dims[0] / max(reference_dims[0], 1.0e-9))
        model_top2_area_ratio = float((candidate_dims[0] * candidate_dims[1]) / max(reference_dims[0] * reference_dims[1], 1.0e-9))

    depth_score = 0.0
    if candidate.get("median_depth_m") is not None and reference.get("median_depth_m") is not None:
        depth_score = min(
            3.0,
            abs(float(candidate["median_depth_m"]) - float(reference["median_depth_m"]))
            / max(1e-6, settings.YOLO_MATCH_DEPTH_TOLERANCE_M),
        )

    change_overlap_ratio = candidate.get("change_overlap_ratio")
    change_score = 0.0 if change_overlap_ratio is None else 1.0 - max(0.0, min(1.0, float(change_overlap_ratio)))

    score = (
        settings.SIGNATURE_VISUAL_WEIGHT * visual_score
        + settings.SIGNATURE_MODEL_SIZE_WEIGHT * model_size_score
        + settings.SIGNATURE_SHAPE_WEIGHT * shape_score
        + settings.SIGNATURE_DEPTH_WEIGHT * depth_score
        + settings.SIGNATURE_CHANGE_WEIGHT * change_score
    )
    return {
        "score": float(score),
        "visual_score": float(visual_score),
        "model_size_score": float(model_size_score),
        "model_largest_dim_ratio": model_largest_dim_ratio,
        "model_top2_area_ratio": model_top2_area_ratio,
        "shape_score": float(shape_score),
        "depth_score": float(depth_score),
        "change_score": float(change_score),
        "change_overlap_ratio": change_overlap_ratio,
        "color_score": float(visual_score),
        "area_score": float(model_size_score),
        "aspect_score": float(shape_score),
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


def _load_baseline_backup_images(baseline: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    backup_dir = Path(str(baseline.get("baseline_backup_dir") or ""))
    if not backup_dir.is_dir():
        return None, None, {"reason": "baseline_backup_dir_missing", "baseline_backup_dir": str(backup_dir)}
    rgb_path = backup_dir / "rgb.png"
    depth_path = backup_dir / "depth.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path.is_file() else None
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
    depth = _depth_raw_to_m(depth_raw) if depth_raw is not None else None
    return rgb, depth, {"baseline_rgb_path": str(rgb_path), "baseline_depth_path": str(depth_path)}


def _tracking_region_image_mask(shape: tuple[int, int], region: dict[str, Any]) -> np.ndarray:
    h, w = shape
    mask = np.ones((h, w), dtype=bool)
    if not isinstance(region, dict) or region.get("is_unrestricted"):
        return mask
    bbox = region.get("bbox_xyxy")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return mask
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    region_mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        region_mask[y0:y1, x0:x1] = True
    return region_mask


def _write_difference_debug(task: dict[str, Any], current_rgb: np.ndarray, change_mask: np.ndarray, region: dict[str, Any]) -> dict[str, Any]:
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        return {}
    debug_dir = model_debug_dir(task_timestamp)
    debug_dir.mkdir(parents=True, exist_ok=True)
    mask_path = debug_dir / "09_history_difference_mask.png"
    overlay_path = debug_dir / "09_history_difference_overlay.png"
    cv2.imwrite(str(mask_path), change_mask.astype(np.uint8) * 255)
    overlay = np.asarray(current_rgb).copy()
    color = np.zeros_like(overlay)
    color[:, :] = (40, 40, 255)
    overlay[change_mask] = cv2.addWeighted(color[change_mask], 0.55, overlay[change_mask], 0.45, 0.0)
    bbox = region.get("bbox_xyxy") if isinstance(region, dict) else None
    if isinstance(bbox, list) and len(bbox) == 4:
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 220, 255), 2)
    cv2.imwrite(str(overlay_path), overlay)
    return {"difference_mask_path": str(mask_path), "difference_overlay_path": str(overlay_path)}


def _build_difference_mask(
    task: dict[str, Any],
    baseline: dict[str, Any],
    current_sample: CachedRgbdSample,
    tracking_region: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    baseline_rgb, baseline_depth, info = _load_baseline_backup_images(baseline)
    current_rgb = np.asarray(current_sample.rgb_bgr)
    if baseline_rgb is None or baseline_depth is None:
        return None, {**info, "reason": "baseline_backup_images_unavailable"}
    if baseline_rgb.shape[:2] != current_rgb.shape[:2] or baseline_depth.shape != current_rgb.shape[:2]:
        return None, {
            **info,
            "reason": "baseline_current_shape_mismatch",
            "baseline_rgb_shape": list(baseline_rgb.shape[:2]) if baseline_rgb is not None else None,
            "baseline_depth_shape": list(baseline_depth.shape) if baseline_depth is not None else None,
            "current_shape": list(current_rgb.shape[:2]),
        }
    current_depth = _sample_depth_m(current_sample)
    rgb_delta = np.mean(cv2.absdiff(baseline_rgb, current_rgb).astype(np.float32), axis=2)
    rgb_changed = rgb_delta >= float(settings.CHANGE_RGB_DIFF_THRESHOLD)
    depth_valid = np.isfinite(baseline_depth) & (baseline_depth > 0.0) & np.isfinite(current_depth) & (current_depth > 0.0)
    depth_changed = depth_valid & (np.abs(current_depth - baseline_depth) >= float(settings.CHANGE_DEPTH_DIFF_THRESHOLD_M))
    region_mask = _tracking_region_image_mask(current_rgb.shape[:2], tracking_region)
    change = (rgb_changed | depth_changed) & region_mask
    kernel_size = max(1, int(settings.CHANGE_MORPH_KERNEL_PX))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        change_u8 = change.astype(np.uint8) * 255
        change_u8 = cv2.morphologyEx(change_u8, cv2.MORPH_OPEN, kernel)
        change_u8 = cv2.morphologyEx(change_u8, cv2.MORPH_CLOSE, kernel)
        change_u8 = cv2.dilate(change_u8, kernel, iterations=1)
        change = change_u8 > 0
    debug_paths = _write_difference_debug(task, current_rgb, change, tracking_region)
    return change, {
        **info,
        **debug_paths,
        "reason": "difference_mask_ready",
        "rgb_diff_threshold": settings.CHANGE_RGB_DIFF_THRESHOLD,
        "depth_diff_threshold_m": settings.CHANGE_DEPTH_DIFF_THRESHOLD_M,
        "changed_pixels": int(np.count_nonzero(change)),
        "region_pixels": int(np.count_nonzero(region_mask)),
        "changed_ratio_in_region": float(np.count_nonzero(change) / max(1, np.count_nonzero(region_mask))),
    }


def _change_overlap_features(mask: np.ndarray, change_mask: np.ndarray | None) -> dict[str, Any]:
    if change_mask is None or change_mask.shape != mask.shape or not np.any(mask):
        return {"change_overlap_ratio": None, "change_overlap_pixels": None}
    overlap = int(np.count_nonzero(mask & change_mask))
    pixels = int(np.count_nonzero(mask))
    return {
        "change_overlap_ratio": float(overlap / max(1, pixels)),
        "change_overlap_pixels": overlap,
    }


def _masked_crop(image: np.ndarray, mask: np.ndarray, bbox: tuple[float, float, float, float], *, pad: int = 8) -> np.ndarray | None:
    if image.shape[:2] != mask.shape or not np.any(mask):
        return None
    h, w = mask.shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 0
    return crop


def _save_candidate_match_debug(
    task: dict[str, Any],
    baseline: dict[str, Any],
    current_sample: CachedRgbdSample,
    selected_obs: YoloObjectObservation,
) -> dict[str, Any]:
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        return {}
    baseline_rgb, _baseline_depth, _info = _load_baseline_backup_images(baseline)
    baseline_mask, _reference_depth = _load_baseline_arrays(baseline)
    reference = baseline.get("reference_observation") if isinstance(baseline.get("reference_observation"), dict) else {}
    bbox_values = reference.get("bbox_xyxy") if isinstance(reference.get("bbox_xyxy"), list) else None
    if baseline_rgb is None or baseline_mask is None or not (isinstance(bbox_values, list) and len(bbox_values) == 4):
        return {}
    debug_dir = model_debug_dir(task_timestamp)
    debug_dir.mkdir(parents=True, exist_ok=True)
    baseline_crop = _masked_crop(baseline_rgb, baseline_mask, tuple(float(v) for v in bbox_values))
    current_crop = _masked_crop(np.asarray(current_sample.rgb_bgr), selected_obs.mask, selected_obs.bbox_xyxy)
    paths: dict[str, Any] = {}
    if baseline_crop is not None:
        path = debug_dir / "09_history_baseline_masked_crop.png"
        cv2.imwrite(str(path), baseline_crop)
        paths["baseline_masked_crop_path"] = str(path)
    if current_crop is not None:
        path = debug_dir / "09_history_current_candidate_masked_crop.png"
        cv2.imwrite(str(path), current_crop)
        paths["current_candidate_masked_crop_path"] = str(path)
    if baseline_crop is not None and current_crop is not None:
        target_h = max(baseline_crop.shape[0], current_crop.shape[0], 1)
        def resize_to_height(img: np.ndarray) -> np.ndarray:
            scale = target_h / max(img.shape[0], 1)
            return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), target_h), interpolation=cv2.INTER_AREA)
        panel = np.concatenate([resize_to_height(baseline_crop), resize_to_height(current_crop)], axis=1)
        path = debug_dir / "09_history_masked_crop_compare.png"
        cv2.imwrite(str(path), panel)
        paths["masked_crop_compare_path"] = str(path)
    return paths


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


def _signature_hard_thresholds(signature: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        ("score", settings.SIGNATURE_MAX_SCORE),
        ("visual_score", settings.SIGNATURE_MAX_VISUAL_SCORE),
        ("model_size_score", settings.SIGNATURE_MAX_MODEL_SIZE_SCORE),
        ("model_largest_dim_ratio", settings.SIGNATURE_MAX_MODEL_LARGEST_DIM_RATIO),
        ("model_top2_area_ratio", settings.SIGNATURE_MAX_MODEL_TOP2_AREA_RATIO),
        ("shape_score", settings.SIGNATURE_MAX_SHAPE_SCORE),
        ("depth_score", settings.SIGNATURE_MAX_DEPTH_SCORE),
        ("change_score", settings.SIGNATURE_MAX_CHANGE_SCORE),
    )
    for key, limit in checks:
        value = signature.get(key)
        if value is not None and float(value) > float(limit):
            failures.append(f"{key}_too_high")
    return not failures, failures


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
    *,
    observations: list[YoloObjectObservation] | None = None,
    change_mask: np.ndarray | None = None,
) -> tuple[YoloObjectObservation | None, dict[str, Any]]:
    reference_signature = baseline.get("reference_signature") if isinstance(baseline.get("reference_signature"), dict) else {}
    reference = baseline.get("reference_observation") if isinstance(baseline.get("reference_observation"), dict) else {}
    reference_center = reference.get("center_xy") if isinstance(reference.get("center_xy"), list) else None
    if not reference_signature:
        return None, {"reason": "reference_signature_missing"}

    scored: list[dict[str, Any]] = []
    source_observations = observations if observations is not None else _observations_for_events(cache, events, shape)
    for obs in source_observations:
        allowed, region_check = _tracking_region_allows_observation(obs, tracking_region)
        if not allowed:
            scored.append({"observation": obs, "region_check": region_check, "rejected": True})
            continue
        candidate_signature = dict(obs.signature)
        candidate_signature.update(_change_overlap_features(obs.mask, change_mask))
        spatial_distance = None
        near_reference_position = False
        if isinstance(reference_center, list) and len(reference_center) >= 2:
            spatial_distance = _center_distance(obs.center_xy, (float(reference_center[0]), float(reference_center[1])))
            original_center_limit = max(settings.ORIGINAL_MAX_CENTER_PX, obs.bbox_diag * settings.ORIGINAL_MAX_CENTER_BBOX_RATIO)
            near_reference_position = spatial_distance <= original_center_limit
        signature = _signature_score(candidate_signature, reference_signature)
        hard_thresholds_passed, hard_threshold_failures = _signature_hard_thresholds(signature)
        if near_reference_position and "change_score_too_high" in hard_threshold_failures:
            hard_threshold_failures = [
                failure for failure in hard_threshold_failures if failure != "change_score_too_high"
            ]
            hard_thresholds_passed = not hard_threshold_failures
        change_overlap = candidate_signature.get("change_overlap_ratio")
        if (
            change_mask is not None
            and not near_reference_position
            and change_overlap is not None
            and float(change_overlap) < settings.CANDIDATE_MIN_CHANGE_OVERLAP_RATIO
        ):
            hard_thresholds_passed = False
            hard_threshold_failures = list(hard_threshold_failures) + ["change_overlap_too_low"]
        scored.append(
            {
                "observation": obs,
                "signature": signature,
                "candidate_signature_summary": {
                    key: candidate_signature.get(key)
                    for key in (
                        "point_bbox_status",
                        "valid_depth_pixels",
                        "valid_depth_ratio",
                        "point_bbox_sorted_dims_m",
                        "point_bbox_top2_area_m2",
                        "visual_status",
                        "edge_density",
                        "mask_extent",
                        "change_overlap_ratio",
                        "change_overlap_pixels",
                    )
                    if key in candidate_signature
                },
                "spatial_distance_px": spatial_distance,
                "near_reference_position": near_reference_position,
                "region_check": region_check,
                "hard_thresholds_passed": hard_thresholds_passed,
                "hard_threshold_failures": hard_threshold_failures,
                "rejected": False,
            }
        )

    if not scored:
        return None, {"reason": "no_candidates"}

    viable = [item for item in scored if not item.get("rejected") and item.get("hard_thresholds_passed")]
    viable.sort(
        key=lambda item: (
            float(item["signature"]["score"]),
            float("inf") if item.get("spatial_distance_px") is None else float(item["spatial_distance_px"]),
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
        "change_mask_available": change_mask is not None,
        "candidate_min_change_overlap_ratio": settings.CANDIDATE_MIN_CHANGE_OVERLAP_RATIO,
        "top_candidates": [serialize(item) for item in sorted(
            scored,
            key=lambda item: (
                1 if item.get("rejected") else 0,
                0 if item.get("hard_thresholds_passed") else 1,
                float(item.get("signature", {}).get("score", 999.0)),
                float("inf") if item.get("spatial_distance_px") is None else float(item.get("spatial_distance_px")),
            ),
        )[:8]],
    }
    if selected is not None:
        info["selected"] = serialize(selected)
    return (selected["observation"] if selected is not None else None), info

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
    baseline = {
        "status": "ready",
        "created_at": _utc_now(),
        "source": "taken_detection_init",
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
    baseline = {
        "status": "ready",
        "created_at": _utc_now(),
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
    path = output_dir / ("09_history_state_visualization.png" if output_dir.name == "worker" else "state_visualization.png")
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
    current_observations: list[YoloObjectObservation] | None = None,
    current_visual_sample: CachedRgbdSample | None = None,
) -> dict[str, Any]:
    if current_visual_sample is None:
        current_visual_sample = current_sample
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
            observations=current_observations,
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

    with _timing_span(timings, "current_difference_mask_build") as timing:
        change_mask, difference_info = _build_difference_mask(task, baseline, current_visual_sample, tracking_region)
        timing["reason"] = difference_info.get("reason")
        timing["change_mask_available"] = change_mask is not None
        timing["changed_pixels"] = difference_info.get("changed_pixels")
        timing["changed_ratio_in_region"] = difference_info.get("changed_ratio_in_region")
    validation["difference_map"] = difference_info

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
            selected_obs, candidate_info = _search_current_candidate(
                cache,
                [yolo_event],
                shape,
                baseline,
                tracking_region,
                observations=current_observations,
                change_mask=change_mask,
            )
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
        debug_paths = _save_candidate_match_debug(task, baseline, current_visual_sample, selected_obs)
        if debug_paths:
            validation["candidate_match_debug"] = debug_paths
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
    final_status = depth_status
    final_reason = depth_info.get("reason")
    if (
        depth_status == STATUS_UNKNOWN
        and depth_info.get("reason") == "target_missing_without_depth_change"
        and candidate_info.get("reason") in {"no_candidates", "no_candidate_passed_hard_thresholds", "no_yolo_event"}
    ):
        final_status = STATUS_MISSING
        final_reason = "no_current_candidate_after_signature_filter"
    return {
        "status": final_status,
        "reason": final_reason,
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

        yolo_search_seconds = max(0.0, settings.YOLO_NEAREST_SEARCH_SECONDS, settings.YOLO_MAX_DELTA_TO_TARGET_SECONDS)
        yolo_start = _stamp_from_seconds(target_seconds - yolo_search_seconds)
        yolo_end = _stamp_from_seconds(target_seconds + yolo_search_seconds)
        yolo_window_key = f"{float(target_seconds):.9f}:{float(yolo_search_seconds):.9f}"
        with span("current_yolo_metadata_scan", {"start_stamp": yolo_start.to_dict(), "end_stamp": yolo_end.to_dict()}) as timing:
            shared_metadata = shared_context.get("current_yolo_metadata")
            if isinstance(shared_metadata, dict) and shared_metadata.get("window_key") == yolo_window_key:
                metadata = shared_metadata.get("metadata") or []
                timing["shared_context_hit"] = True
            else:
                metadata = list(cache.iter_sample_metadata(start=yolo_start, end=yolo_end))
                timing["shared_context_hit"] = False
                shared_context["current_yolo_metadata"] = {"window_key": yolo_window_key, "metadata": metadata}
            timing["metadata_frame_count"] = len(metadata)

        with span("current_nearest_yolo_event_select", {"metadata_frame_count": len(metadata)}) as timing:
            shared_event = shared_context.get("current_yolo_event")
            if isinstance(shared_event, dict) and shared_event.get("window_key") == yolo_window_key:
                yolo_event = shared_event.get("event")
                yolo_timing = dict(shared_event.get("timing") or {})
                timing["shared_context_hit"] = True
            else:
                yolo_event, yolo_timing = _nearest_yolo_event(metadata, target_seconds)
                timing["shared_context_hit"] = False
                shared_context["current_yolo_event"] = {"window_key": yolo_window_key, "event": yolo_event, "timing": dict(yolo_timing)}
            timing.update({k: v for k, v in yolo_timing.items() if k in {"reason", "candidate_event_count", "yolo_delta_to_target_seconds", "is_stale"}})
            timing["has_yolo_event"] = yolo_event is not None

        yolo_event_key = yolo_event.sample.key if yolo_event is not None else ""
        with span("current_yolo_paired_sample_load", {"has_yolo_event": yolo_event is not None}) as timing:
            shared_paired = shared_context.get("current_yolo_paired_sample")
            if (
                yolo_event is not None
                and isinstance(shared_paired, dict)
                and shared_paired.get("event_key") == yolo_event_key
                and isinstance(shared_paired.get("sample"), CachedRgbdSample)
            ):
                yolo_paired_sample = shared_paired["sample"]
                timing["shared_context_hit"] = True
            else:
                yolo_paired_sample = cache.get_sample(yolo_event.sample.stamp, mode="nearest") if yolo_event is not None else None
                timing["shared_context_hit"] = False
                if yolo_event is not None and yolo_paired_sample is not None:
                    shared_context["current_yolo_paired_sample"] = {"event_key": yolo_event_key, "sample": yolo_paired_sample}
            timing["sample_available"] = yolo_paired_sample is not None
            if yolo_paired_sample is not None:
                timing["sample_stamp"] = yolo_paired_sample.stamp.to_dict()

        shape = yolo_paired_sample.depth.shape[:2] if yolo_paired_sample is not None else current_sample.depth.shape[:2]
        current_observations: list[YoloObjectObservation] | None = None
        with span("current_observations_prepare", {"has_yolo_event": yolo_event is not None}) as timing:
            if yolo_event is None:
                timing["shared_context_hit"] = False
                timing["observation_count"] = 0
            else:
                observation_key = f"{yolo_event.sample.key}:{int(shape[0])}x{int(shape[1])}"
                timing["observation_cache_key"] = observation_key
                observations_cache = shared_context.get("current_observations_by_key")
                if not isinstance(observations_cache, dict):
                    observations_cache = {}
                    shared_context["current_observations_by_key"] = observations_cache
                cached_observations = observations_cache.get(observation_key)
                if isinstance(cached_observations, list):
                    current_observations = cached_observations
                    timing["shared_context_hit"] = True
                else:
                    current_observations = _observations_for_events(cache, [yolo_event], shape)
                    timing["shared_context_hit"] = False
                    observations_cache[observation_key] = current_observations
                timing["observation_count"] = len(current_observations)

    return _jsonable({
        "success": True,
        "target_seconds": float(target_seconds),
        "target_time_source": target_source,
        "current_sample": current_sample.to_dict(),
        "yolo_paired_sample": yolo_paired_sample.to_dict() if yolo_paired_sample is not None else None,
        "metadata_frame_count": len(metadata),
        "has_yolo_event": yolo_event is not None,
        "yolo_timing": yolo_timing,
        "observation_count": len(current_observations or []),
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

        yolo_search_seconds = max(0.0, settings.YOLO_NEAREST_SEARCH_SECONDS, settings.YOLO_MAX_DELTA_TO_TARGET_SECONDS)
        yolo_start = _stamp_from_seconds(target_seconds - yolo_search_seconds)
        yolo_end = _stamp_from_seconds(target_seconds + yolo_search_seconds)
        yolo_window_key = f"{float(target_seconds):.9f}:{float(yolo_search_seconds):.9f}"
        with timings.span(
            "current_yolo_metadata_scan",
            {"start_stamp": yolo_start.to_dict(), "end_stamp": yolo_end.to_dict()},
        ) as timing:
            shared_metadata = shared_context.get("current_yolo_metadata") if shared_context is not None else None
            if isinstance(shared_metadata, dict) and shared_metadata.get("window_key") == yolo_window_key:
                metadata = shared_metadata.get("metadata") or []
                timing["shared_context_hit"] = True
            else:
                metadata = list(cache.iter_sample_metadata(start=yolo_start, end=yolo_end))
                timing["shared_context_hit"] = False
                if shared_context is not None:
                    shared_context["current_yolo_metadata"] = {"window_key": yolo_window_key, "metadata": metadata}
            timing["metadata_frame_count"] = len(metadata)
        with timings.span("current_nearest_yolo_event_select", {"metadata_frame_count": len(metadata)}) as timing:
            shared_event = shared_context.get("current_yolo_event") if shared_context is not None else None
            if isinstance(shared_event, dict) and shared_event.get("window_key") == yolo_window_key:
                yolo_event = shared_event.get("event")
                yolo_timing = dict(shared_event.get("timing") or {})
                timing["shared_context_hit"] = True
            else:
                yolo_event, yolo_timing = _nearest_yolo_event(metadata, target_seconds)
                timing["shared_context_hit"] = False
                if shared_context is not None:
                    shared_context["current_yolo_event"] = {"window_key": yolo_window_key, "event": yolo_event, "timing": dict(yolo_timing)}
            timing.update({k: v for k, v in yolo_timing.items() if k in {"reason", "candidate_event_count", "yolo_delta_to_target_seconds", "is_stale"}})
            timing["has_yolo_event"] = yolo_event is not None
        yolo_event_key = yolo_event.sample.key if yolo_event is not None else ""
        with timings.span("current_yolo_paired_sample_load", {"has_yolo_event": yolo_event is not None}) as timing:
            shared_paired = shared_context.get("current_yolo_paired_sample") if shared_context is not None else None
            if (
                yolo_event is not None
                and isinstance(shared_paired, dict)
                and shared_paired.get("event_key") == yolo_event_key
                and isinstance(shared_paired.get("sample"), CachedRgbdSample)
            ):
                yolo_paired_sample = shared_paired["sample"]
                timing["shared_context_hit"] = True
            else:
                yolo_paired_sample = cache.get_sample(yolo_event.sample.stamp, mode="nearest") if yolo_event is not None else None
                timing["shared_context_hit"] = False
                if shared_context is not None and yolo_event is not None and yolo_paired_sample is not None:
                    shared_context["current_yolo_paired_sample"] = {"event_key": yolo_event_key, "sample": yolo_paired_sample}
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
        current_observations: list[YoloObjectObservation] | None = None
        with timings.span("current_observations_prepare", {"has_yolo_event": yolo_event is not None}) as timing:
            if yolo_event is None:
                timing["shared_context_hit"] = False
                timing["observation_count"] = 0
            else:
                observation_key = f"{yolo_event.sample.key}:{int(shape[0])}x{int(shape[1])}"
                timing["observation_cache_key"] = observation_key
                observations_cache = None
                if shared_context is not None:
                    existing_cache = shared_context.get("current_observations_by_key")
                    if not isinstance(existing_cache, dict):
                        existing_cache = {}
                        shared_context["current_observations_by_key"] = existing_cache
                    observations_cache = existing_cache
                cached_observations = observations_cache.get(observation_key) if observations_cache is not None else None
                if isinstance(cached_observations, list):
                    current_observations = cached_observations
                    timing["shared_context_hit"] = True
                else:
                    current_observations = _observations_for_events(cache, [yolo_event], shape)
                    timing["shared_context_hit"] = False
                    if observations_cache is not None:
                        observations_cache[observation_key] = current_observations
                timing["observation_count"] = len(current_observations)
    with timings.span("current_backup_write", {"stamp": current_sample.stamp.to_dict()}) as timing:
        current_backup_dir = _save_sample_backup(task, current_sample, output_dir, kind="current")
        timing["backup_dir"] = str(current_backup_dir)
    paired_backup_dir = None
    if yolo_paired_sample is not None and yolo_paired_sample.stamp != current_sample.stamp:
        with timings.span("current_yolo_paired_backup_write", {"stamp": yolo_paired_sample.stamp.to_dict()}) as timing:
            paired_backup_dir = _save_sample_backup(task, yolo_paired_sample, output_dir, kind="current_yolo_paired")
            timing["backup_dir"] = str(paired_backup_dir)
    with timings.span("current_classify_total") as timing:
        classification = _classify_current(
            task,
            cache,
            baseline,
            current_sample,
            yolo_event,
            yolo_timing,
            shape,
            timings=timings,
            current_observations=current_observations,
            current_visual_sample=yolo_paired_sample if yolo_paired_sample is not None else current_sample,
        )
        timing["status"] = classification.get("status")
        timing["reason"] = classification.get("reason")

    selected_observation = classification.get("selected_observation")
    observation_obj = None
    if selected_observation is not None and yolo_event is not None:
        with timings.span("selected_observation_hydrate") as timing:
            selected_id = str(selected_observation.get("object_id") or "")
            selected_stamp = RosStamp.from_dict(selected_observation.get("stamp") or {})
            timing["selected_object_id"] = selected_id
            source_observations = current_observations if current_observations is not None else _observations_for_events(cache, [yolo_event], shape)
            timing["used_current_observations"] = current_observations is not None
            for obs in source_observations:
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
