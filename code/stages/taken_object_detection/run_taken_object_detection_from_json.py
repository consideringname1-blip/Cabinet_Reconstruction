from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from artifact_layout import SHIGURE_HISTORY_CACHE_ROOT, model_debug_dir, model_result_file, model_worker_dir, model_worker_file
from coordinate_systems import quat_xyzw_to_rotation_matrix
from spatial_transforms import aruco_points_to_shigure_camera
from stages.shigure_history.cache import CachedRgbdSample, CachedSampleMetadata, RosStamp, ShigureRgbdCache, load_json, sample_key
from stages.shigure_history.marker_history import latest_marker_pose_path
from task_json import load_task_json, resolve_task_json_path, save_task_json

from stages.taken_object_detection import settings


@dataclass(frozen=True)
class FrameDecision:
    stamp: RosStamp
    status: str
    valid_pixels: int
    occluded_pixels: int
    unoccluded_pixels: int
    taken_pixels: int
    occluded_ratio: float
    taken_ratio: float
    candidate: bool
    full_occlusion: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            'stamp': self.stamp.to_dict(),
            'status': self.status,
            'valid_pixels': self.valid_pixels,
            'occluded_pixels': self.occluded_pixels,
            'unoccluded_pixels': self.unoccluded_pixels,
            'taken_pixels': self.taken_pixels,
            'occluded_ratio': self.occluded_ratio,
            'taken_ratio': self.taken_ratio,
            'candidate': self.candidate,
            'full_occlusion': self.full_occlusion,
        }


@dataclass(frozen=True)
class ObjectCenterProjection:
    pixel_xy: tuple[float, float]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'pixel_xy': [float(self.pixel_xy[0]), float(self.pixel_xy[1])],
            'depth_m': float(self.depth_m),
            'camera_xyz_m': [float(v) for v in self.camera_xyz_m],
            'source': self.source,
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
        return {
            'stamp': self.stamp.to_dict(),
            'object_id': self.object_id,
            'center_xy': [float(self.center_xy[0]), float(self.center_xy[1])],
            'bbox_xyxy': [float(v) for v in self.bbox_xyxy],
            'mask_pixels': int(self.mask_pixels),
            'median_depth_m': self.median_depth_m,
            'yolo_hash': self.event.sample.yolo_hash,
        }


@dataclass(frozen=True)
class YoloInitResult:
    trusted_mask: np.ndarray
    reference_depth: np.ndarray
    init_frame: CachedRgbdSample
    object_id: str
    projection: dict[str, Any]
    init_stats: dict[str, Any]
    yolo_observations: list[YoloObjectObservation]
    yolo_events: list[YoloEvent]
    projected_mask: np.ndarray | None = None



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write('\n')


def _publish_taken_result_artifacts(task: Mapping[str, Any], backup_dir: Path) -> dict[str, Any]:
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for taken object result artifacts')
    mappings = {
        'rgb.png': ('result_rgb', model_result_file(task_timestamp, 'taken.result_rgb')),
        'depth.png': ('result_depth', model_result_file(task_timestamp, 'taken.result_depth')),
        'camera_info.json': ('camera_info', model_result_file(task_timestamp, 'taken.camera_info')),
        'active_objects.json': ('active_objects', model_result_file(task_timestamp, 'taken.active_objects')),
        'marker_6d_pose.json': ('marker_pose', model_result_file(task_timestamp, 'taken.marker_pose')),
    }
    published: dict[str, Any] = {'artifact_root': 'model_result'}
    for source_name, (payload_key, target_path) in mappings.items():
        source_path = backup_dir / source_name
        if source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            published[payload_key] = target_path.name
    return published


def _write_status(json_path: Path, task: dict[str, Any], status: str, **fields: Any) -> None:
    payload = dict(fields)
    payload['status'] = status
    payload['updated_at'] = _utc_now()
    task['TakenObjectDetection'] = _jsonable(payload)
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for taken object status artifacts')
    _write_json(model_result_file(task_timestamp, 'taken.result'), payload)
    save_task_json(json_path, task)


def _parse_iso_timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        if '.' not in text:
            return None
        head, tail = text.split('.', 1)
        timezone_pos = min([p for p in (tail.find('+'), tail.find('-')) if p >= 0], default=-1)
        fraction = tail if timezone_pos < 0 else tail[:timezone_pos]
        suffix = '' if timezone_pos < 0 else tail[timezone_pos:]
        try:
            parsed = datetime.fromisoformat(f'{head}.{fraction[:6]}{suffix}')
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _task_capture_time_seconds(task: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for frame in task.get('PVCameraFrames') or []:
        if isinstance(frame, Mapping):
            seconds = _parse_iso_timestamp_seconds(frame.get('time'))
            if seconds is not None:
                return seconds, 'PVCameraFrames.time'
    for key in ('PVCamera', 'device'):
        payload = task.get(key)
        if isinstance(payload, Mapping):
            seconds = _parse_iso_timestamp_seconds(payload.get('time'))
            if seconds is not None:
                return seconds, f'{key}.time'
    seconds = _parse_iso_timestamp_seconds(task.get('server_received_utc'))
    return (seconds, 'server_received_utc') if seconds is not None else (None, None)


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


def _sample_rgb(sample: CachedRgbdSample) -> np.ndarray:
    return np.asarray(sample.rgb_bgr[:, :, ::-1], dtype=np.float32)


def _resolve_existing_path(name: str | None) -> Path | None:
    if not name:
        return None
    path = Path(str(name)).expanduser()
    return path if path.is_file() else None


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _mask_from_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L')) > 0
    return _resize_mask(mask, shape)


def _selection_box_mask(task: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    if not settings.ALLOW_SELECTION_BOX_MASK_FALLBACK:
        return None
    box = task.get('SelectionBox') if isinstance(task.get('SelectionBox'), Mapping) else None
    if not box:
        return None
    top_left = box.get('top_left')
    bottom_right = box.get('bottom_right')
    if not (isinstance(top_left, list) and isinstance(bottom_right, list) and len(top_left) == 2 and len(bottom_right) == 2):
        return None
    h, w = shape
    x0 = int(max(0, min(w - 1, math.floor(float(top_left[0]) * w))))
    y0 = int(max(0, min(h - 1, math.floor(float(top_left[1]) * h))))
    x1 = int(max(0, min(w, math.ceil(float(bottom_right[0]) * w))))
    y1 = int(max(0, min(h, math.ceil(float(bottom_right[1]) * h))))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _parse_float_array(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f'{name} expected {size} values, got {array.size}')
    return array


def _camera_info_message(camera_info: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(camera_info, Mapping):
        return {}
    message = camera_info.get('message')
    if isinstance(message, Mapping):
        merged = dict(message)
        for key, value in camera_info.items():
            if key != 'message' and key not in merged:
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


def _camera_matrix_from_info(camera_info: Mapping[str, Any] | None) -> np.ndarray | None:
    info = _camera_info_message(camera_info)
    raw = info.get('k') or info.get('K') or info.get('camera_matrix')
    return _parse_camera_k(raw)


def _load_marker_pose_cv() -> tuple[np.ndarray, np.ndarray, Path] | None:
    path = _find_marker_pose_path()
    if path is None or not path.is_file():
        return None
    payload = load_json(path)
    pose = payload.get('opencv_camera_pose') if isinstance(payload.get('opencv_camera_pose'), Mapping) else payload
    if not isinstance(pose, Mapping):
        return None
    try:
        if pose.get('rotation_matrix') is not None:
            rotation = _parse_float_array(pose.get('rotation_matrix'), 9, 'marker rotation_matrix').reshape(3, 3)
        elif pose.get('rotation_quaternion_xyzw') is not None:
            rotation = quat_xyzw_to_rotation_matrix(_parse_float_array(pose.get('rotation_quaternion_xyzw'), 4, 'marker quaternion'))
        else:
            return None
        translation = _parse_float_array(pose.get('tvec_m') if pose.get('tvec_m') is not None else pose.get('position'), 3, 'marker translation')
    except Exception:
        return None
    return rotation.astype(np.float64), translation.astype(np.float64), path


def _object_center_aruco(task: Mapping[str, Any]) -> tuple[np.ndarray | None, str]:
    bounds = task.get('ModelBounds') if isinstance(task.get('ModelBounds'), Mapping) else None
    if bounds and bounds.get('aabb_min_aruco') is not None and bounds.get('aabb_max_aruco') is not None:
        try:
            a = _parse_float_array(bounds.get('aabb_min_aruco'), 3, 'ModelBounds.aabb_min_aruco')
            b = _parse_float_array(bounds.get('aabb_max_aruco'), 3, 'ModelBounds.aabb_max_aruco')
            return (a + b) * 0.5, 'ModelBounds.aabb_center_aruco'
        except Exception:
            pass
    obj = task.get('object_aruco') if isinstance(task.get('object_aruco'), Mapping) else None
    if obj and obj.get('position') is not None:
        try:
            return _parse_float_array(obj.get('position'), 3, 'object_aruco.position'), 'object_aruco.position'
        except Exception:
            pass
    return None, 'missing_object_center_aruco'


def _project_object_center_to_shigure(task: Mapping[str, Any], camera_info: Mapping[str, Any] | None, image_shape: tuple[int, int]) -> tuple[ObjectCenterProjection | None, dict[str, Any]]:
    camera_matrix = _camera_matrix_from_info(camera_info)
    if camera_matrix is None:
        return None, {'source': 'model_center_projection', 'reason': 'camera_matrix_missing'}
    center_aruco, center_source = _object_center_aruco(task)
    if center_aruco is None:
        return None, {'source': 'model_center_projection', 'reason': center_source}
    marker_pose = _load_marker_pose_cv()
    if marker_pose is None:
        return None, {'source': 'model_center_projection', 'reason': 'marker_pose_missing'}
    marker_rotation, marker_translation, marker_path = marker_pose
    center_camera = aruco_points_to_shigure_camera(
        center_aruco.reshape(1, 3),
        marker_rotation,
        marker_translation,
    ).reshape(3)
    z = float(center_camera[2])
    if not np.isfinite(z) or z <= 0.0:
        return None, {
            'source': 'model_center_projection',
            'reason': 'projected_center_behind_camera',
            'camera_xyz_m': center_camera.tolist(),
        }
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x = fx * float(center_camera[0]) / z + cx
    y = fy * float(center_camera[1]) / z + cy
    h, w = image_shape
    info = {
        'source': 'model_center_projection',
        'center_source': center_source,
        'marker_pose_path': str(marker_path),
        'object_center_aruco': center_aruco.tolist(),
        'camera_matrix': camera_matrix.tolist(),
        'pixel_xy': [x, y],
        'depth_m': z,
        'camera_xyz_m': center_camera.tolist(),
        'image_shape': [h, w],
    }
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, {**info, 'reason': 'projected_center_outside_image'}
    return ObjectCenterProjection(pixel_xy=(x, y), depth_m=z, camera_xyz_m=tuple(float(v) for v in center_camera), source=center_source), info


def _unique_yolo_events(metadata: list[CachedSampleMetadata]) -> list[YoloEvent]:
    events: list[YoloEvent] = []
    last_hash: str | None = None
    for sample in metadata:
        if not sample.yolo_hash or sample.yolo_hash == last_hash:
            continue
        last_hash = sample.yolo_hash
        payload = sample.load_yolo()
        if not isinstance(payload, Mapping):
            continue
        if not isinstance(payload.get('objects'), list):
            continue
        events.append(YoloEvent(sample=sample, payload=dict(payload)))
    return events


def _decode_yolo_mask(obj: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    raw = obj.get('mask_b64')
    if not isinstance(raw, str) or not raw:
        return None
    try:
        data = base64.b64decode(raw)
        with Image.open(BytesIO(data)) as image:
            mask = np.asarray(image.convert('L')) > 0
    except Exception:
        return None
    return _resize_mask(mask, shape)


def _bbox_from_object(obj: Mapping[str, Any], shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    bbox = obj.get('bbox')
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


def _center_from_object(obj: Mapping[str, Any], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    try:
        return float(obj.get('x')), float(obj.get('y'))
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


def _observation_from_object(cache: ShigureRgbdCache, event: YoloEvent, obj: Mapping[str, Any], shape: tuple[int, int]) -> YoloObjectObservation | None:
    bbox = _bbox_from_object(obj, shape)
    if bbox is None:
        return None
    mask = _decode_yolo_mask(obj, shape)
    if mask is None or not np.any(mask):
        return None
    center = _center_from_object(obj, bbox)
    object_id = str(obj.get('object_id'))
    sample = cache.get_sample(event.sample.stamp, mode='nearest')
    median_depth = _mask_median_depth(_sample_depth_m(sample), mask) if sample is not None else None
    return YoloObjectObservation(
        event=event,
        object_id=object_id,
        center_xy=center,
        bbox_xyxy=bbox,
        mask=mask,
        mask_pixels=int(np.count_nonzero(mask)),
        median_depth_m=median_depth,
    )


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / max(1, union)


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _score_yolo_match(obs: YoloObjectObservation, projection: ObjectCenterProjection) -> dict[str, Any]:
    px, py = projection.pixel_xy
    cx, cy = obs.center_xy
    center_distance = _center_distance((px, py), (cx, cy))
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
        'score': float(score),
        'center_distance_px': center_distance,
        'depth_diff_m': depth_diff,
        'point_inside_mask': point_inside,
        'point_inside_bbox': point_in_bbox,
        'observation': obs,
    }


def _find_yolo_target(cache: ShigureRgbdCache, events: list[YoloEvent], projection: ObjectCenterProjection, shape: tuple[int, int]) -> tuple[str | None, dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for event in events:
        for obj in event.payload.get('objects') or []:
            if not isinstance(obj, Mapping):
                continue
            obs = _observation_from_object(cache, event, obj, shape)
            if obs is None:
                continue
            item = _score_yolo_match(obs, projection)
            scored.append(item)
    if not scored:
        return None, {'reason': 'no_yolo_object_with_mask'}
    scored.sort(key=lambda item: float(item['score']))
    best = scored[0]
    best_obs: YoloObjectObservation = best['observation']
    accept = (
        float(best['score']) <= settings.YOLO_MATCH_MAX_SCORE
        and float(best['center_distance_px']) <= settings.YOLO_MATCH_MAX_CENTER_PX
        and (best['depth_diff_m'] is None or float(best['depth_diff_m']) <= settings.YOLO_MATCH_DEPTH_TOLERANCE_M * 2.0)
    )
    return (best_obs.object_id if accept else None), {
        'reason': 'matched' if accept else 'best_match_rejected',
        'best': {k: v for k, v in best.items() if k != 'observation'},
        'best_observation': best_obs.to_dict(),
        'candidate_count': len(scored),
        'top_candidates': [
            {**{k: v for k, v in item.items() if k != 'observation'}, 'observation': item['observation'].to_dict()}
            for item in scored[:5]
        ],
    }


def _collect_target_observations(cache: ShigureRgbdCache, events: list[YoloEvent], object_id: str, shape: tuple[int, int]) -> list[YoloObjectObservation]:
    observations: list[YoloObjectObservation] = []
    for event in events:
        for obj in event.payload.get('objects') or []:
            if not isinstance(obj, Mapping) or str(obj.get('object_id')) != object_id:
                continue
            obs = _observation_from_object(cache, event, obj, shape)
            if obs is not None:
                observations.append(obs)
            break
    return observations


def _stable_observation_prefix(observations: list[YoloObjectObservation], *, minimum_count: int) -> tuple[list[YoloObjectObservation], dict[str, Any]]:
    if len(observations) < minimum_count:
        return [], {'reason': 'not_enough_unique_yolo', 'available': len(observations), 'required': minimum_count}
    for start_index in range(0, len(observations) - minimum_count + 1):
        window = observations[start_index:start_index + minimum_count]
        anchor = window[0]
        max_center = max(settings.YOLO_STABLE_MAX_CENTER_PX, anchor.bbox_diag * settings.YOLO_STABLE_MAX_CENTER_BBOX_RATIO)
        center_distances = [_center_distance(anchor.center_xy, obs.center_xy) for obs in window[1:]]
        mask_ious = [_mask_iou(anchor.mask, obs.mask) for obs in window[1:]]
        depth_diffs = [abs(float(obs.median_depth_m) - float(anchor.median_depth_m)) for obs in window[1:] if obs.median_depth_m is not None and anchor.median_depth_m is not None]
        stable = (
            all(distance <= max_center for distance in center_distances)
            and all(iou >= settings.YOLO_STABLE_MIN_MASK_IOU for iou in mask_ious)
            and all(delta <= settings.YOLO_STABLE_MAX_DEPTH_DELTA_M for delta in depth_diffs)
        )
        stats = {
            'reason': 'stable' if stable else 'unstable_yolo_window',
            'start_index': start_index,
            'required': minimum_count,
            'max_center_px': max_center,
            'center_distances_px': center_distances,
            'mask_ious': mask_ious,
            'depth_diffs_m': depth_diffs,
            'window': [obs.to_dict() for obs in window],
        }
        if stable:
            return window, stats
    return [], {'reason': 'no_stable_yolo_window', 'available': len(observations), 'required': minimum_count}


def _model_box_corners_aruco(task: Mapping[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    bounds = task.get('ModelBounds') if isinstance(task.get('ModelBounds'), Mapping) else None
    if not bounds:
        return None, {'reason': 'model_bounds_missing'}
    corners = bounds.get('corners_aruco')
    if isinstance(corners, list) and len(corners) >= 8:
        try:
            points = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
            if len(points) >= 8:
                return points, {'source': 'ModelBounds.corners_aruco', 'point_count': int(len(points))}
        except Exception:
            pass
    if bounds.get('aabb_min_aruco') is not None and bounds.get('aabb_max_aruco') is not None:
        try:
            a = _parse_float_array(bounds.get('aabb_min_aruco'), 3, 'ModelBounds.aabb_min_aruco')
            b = _parse_float_array(bounds.get('aabb_max_aruco'), 3, 'ModelBounds.aabb_max_aruco')
            points = np.asarray(
                [[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])],
                dtype=np.float64,
            )
            return points, {'source': 'ModelBounds.aabb_min_max', 'point_count': int(len(points))}
        except Exception as exc:
            return None, {'reason': 'model_bounds_parse_failed', 'error_message': str(exc)}
    return None, {'reason': 'model_bounds_corners_missing'}


def _project_model_box_to_shigure(task: Mapping[str, Any], camera_info: Mapping[str, Any] | None, image_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    camera_matrix = _camera_matrix_from_info(camera_info)
    if camera_matrix is None:
        return None, {'source': 'model_box_projection', 'reason': 'camera_matrix_missing'}
    marker_pose = _load_marker_pose_cv()
    if marker_pose is None:
        return None, {'source': 'model_box_projection', 'reason': 'marker_pose_missing'}
    corners_aruco, corners_info = _model_box_corners_aruco(task)
    if corners_aruco is None:
        return None, {'source': 'model_box_projection', **corners_info}
    center_aruco, center_source = _object_center_aruco(task)
    if center_aruco is None:
        return None, {'source': 'model_box_projection', 'reason': center_source, 'corners': corners_info}

    marker_rotation, marker_translation, marker_path = marker_pose
    points_camera = aruco_points_to_shigure_camera(
        corners_aruco.reshape(-1, 3),
        marker_rotation,
        marker_translation,
    )
    visible = points_camera[:, 2] > 1e-6
    if not np.any(visible):
        return None, {
            'source': 'model_box_projection',
            'reason': 'projected_box_behind_camera',
            'camera_xyz_m': points_camera.astype(float).tolist(),
        }

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    visible_points = points_camera[visible]
    pixels_x = fx * visible_points[:, 0] / visible_points[:, 2] + cx
    pixels_y = fy * visible_points[:, 1] / visible_points[:, 2] + cy
    h, w = image_shape
    if pixels_x.size == 0 or pixels_y.size == 0:
        return None, {'source': 'model_box_projection', 'reason': 'no_projected_box_pixels'}
    raw_x0, raw_x1 = float(np.nanmin(pixels_x)), float(np.nanmax(pixels_x))
    raw_y0, raw_y1 = float(np.nanmin(pixels_y)), float(np.nanmax(pixels_y))
    box_w = max(1.0, raw_x1 - raw_x0)
    box_h = max(1.0, raw_y1 - raw_y0)
    pad = max(float(settings.MODEL_BOX_PADDING_PX), max(box_w, box_h) * float(settings.MODEL_BOX_PADDING_RATIO))
    x0 = int(max(0, math.floor(raw_x0 - pad)))
    y0 = int(max(0, math.floor(raw_y0 - pad)))
    x1 = int(min(w, math.ceil(raw_x1 + pad)))
    y1 = int(min(h, math.ceil(raw_y1 + pad)))
    if x1 <= x0 or y1 <= y0:
        return None, {
            'source': 'model_box_projection',
            'reason': 'projected_box_outside_image',
            'raw_bbox_xyxy': [raw_x0, raw_y0, raw_x1, raw_y1],
            'image_shape': [h, w],
        }

    center_camera = aruco_points_to_shigure_camera(
        center_aruco.reshape(1, 3),
        marker_rotation,
        marker_translation,
    ).reshape(3)
    center_z = float(center_camera[2])
    center_pixel = None
    if np.isfinite(center_z) and center_z > 1e-6:
        center_pixel = [
            float(fx * float(center_camera[0]) / center_z + cx),
            float(fy * float(center_camera[1]) / center_z + cy),
        ]
    projected_mask = np.zeros((h, w), dtype=bool)
    projected_mask[y0:y1, x0:x1] = True
    z_values = visible_points[:, 2]
    info = {
        'source': 'model_box_projection',
        'reason': 'projected_box_ready',
        'corners': corners_info,
        'center_source': center_source,
        'marker_pose_path': str(marker_path),
        'object_center_aruco': center_aruco.astype(float).tolist(),
        'center_pixel_xy': center_pixel,
        'center_depth_m': center_z if np.isfinite(center_z) else None,
        'box_depth_min_m': float(np.nanmin(z_values)),
        'box_depth_max_m': float(np.nanmax(z_values)),
        'raw_bbox_xyxy': [raw_x0, raw_y0, raw_x1, raw_y1],
        'bbox_xyxy': [x0, y0, x1, y1],
        'padding_px': float(pad),
        'camera_matrix': camera_matrix.astype(float).tolist(),
        'image_shape': [h, w],
        'projected_pixels': int(np.count_nonzero(projected_mask)),
    }
    return projected_mask, info


def _build_reference_from_model_box(sample: CachedRgbdSample, projected_mask: np.ndarray, projection: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    depth = _sample_depth_m(sample)
    if depth.shape != projected_mask.shape:
        return None, None, {'reason': 'depth_mask_shape_mismatch', 'depth_shape': list(depth.shape), 'mask_shape': list(projected_mask.shape)}
    box_depth_min = projection.get('box_depth_min_m')
    box_depth_max = projection.get('box_depth_max_m')
    center_depth = projection.get('center_depth_m')
    margin = float(settings.MODEL_BOX_DEPTH_MARGIN_M)
    if center_depth is not None and np.isfinite(float(center_depth)):
        depth_min = float(center_depth) - margin
        depth_max = float(center_depth) + margin
        depth_source = 'model_center_depth'
    elif box_depth_min is not None and box_depth_max is not None:
        depth_min = float(box_depth_min) - margin
        depth_max = float(box_depth_max) + margin
        depth_source = 'model_box_depth_range'
    else:
        return None, None, {'reason': 'projected_depth_missing'}
    valid_box = projected_mask & np.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid_box))
    if valid_pixels <= 0:
        return None, None, {'reason': 'no_valid_depth_pixels_in_projected_box'}
    occlusion_reference = float(box_depth_min) if box_depth_min is not None else depth_min
    occluded = valid_box & (depth < occlusion_reference - abs(settings.OCCLUSION_DELTA_M))
    occluded_ratio = int(np.count_nonzero(occluded)) / max(1, valid_pixels)
    if occluded_ratio > settings.MODEL_BOX_INIT_MAX_OCCLUSION_RATIO:
        return None, None, {
            'reason': 'projected_box_occluded',
            'valid_pixels': valid_pixels,
            'occluded_pixels': int(np.count_nonzero(occluded)),
            'occluded_ratio': occluded_ratio,
            'max_occlusion_ratio': settings.MODEL_BOX_INIT_MAX_OCCLUSION_RATIO,
        }
    trusted = valid_box & (depth >= depth_min) & (depth <= depth_max)
    trusted_pixels = int(np.count_nonzero(trusted))
    image_ratio = trusted_pixels / float(trusted.size)
    if trusted_pixels < settings.MODEL_BOX_MIN_TRUSTED_PIXELS or image_ratio < settings.TRUSTED_MASK_MIN_IMAGE_RATIO:
        return None, None, {
            'reason': 'trusted_mask_too_small',
            'valid_pixels': valid_pixels,
            'trusted_pixels': trusted_pixels,
            'trusted_image_ratio': image_ratio,
            'required_pixels': settings.MODEL_BOX_MIN_TRUSTED_PIXELS,
            'required_image_ratio': settings.TRUSTED_MASK_MIN_IMAGE_RATIO,
            'occluded_ratio': occluded_ratio,
        }
    reference_depth = np.where(trusted, depth, 0.0).astype(np.float32)
    values = depth[trusted]
    return trusted, reference_depth, {
        'reason': 'initialized',
        'mode': 'model_box_depth',
        'valid_pixels': valid_pixels,
        'trusted_pixels': trusted_pixels,
        'trusted_image_ratio': image_ratio,
        'occluded_pixels': int(np.count_nonzero(occluded)),
        'occluded_ratio': occluded_ratio,
        'depth_source': depth_source,
        'trusted_depth_min_m': depth_min,
        'trusted_depth_max_m': depth_max,
        'box_depth_min_m': float(box_depth_min) if box_depth_min is not None else None,
        'box_depth_max_m': float(box_depth_max) if box_depth_max is not None else None,
        'center_depth_m': float(center_depth) if center_depth is not None else None,
        'depth_margin_m': margin,
        'median_depth_m': float(np.nanmedian(values)) if values.size else None,
        'min_depth_m': float(np.nanmin(values)) if values.size else None,
        'max_depth_m': float(np.nanmax(values)) if values.size else None,
        'init_stamp': sample.stamp.to_dict(),
    }


def _build_reference_from_yolo_mask(sample: CachedRgbdSample, mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    depth = _sample_depth_m(sample)
    if depth.shape != mask.shape:
        return None, None, {'reason': 'depth_mask_shape_mismatch', 'depth_shape': list(depth.shape), 'mask_shape': list(mask.shape)}
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid))
    image_ratio = valid_pixels / float(mask.size)
    if image_ratio < settings.TRUSTED_MASK_MIN_IMAGE_RATIO:
        return None, None, {'reason': 'trusted_mask_too_small', 'trusted_pixels': valid_pixels, 'trusted_image_ratio': image_ratio}
    reference_depth = np.where(valid, depth, 0.0).astype(np.float32)
    values = depth[valid]
    return valid, reference_depth, {
        'reason': 'initialized',
        'trusted_pixels': valid_pixels,
        'trusted_image_ratio': image_ratio,
        'mask_pixels': int(np.count_nonzero(mask)),
        'median_depth_m': float(np.nanmedian(values)) if values.size else None,
        'min_depth_m': float(np.nanmin(values)) if values.size else None,
        'max_depth_m': float(np.nanmax(values)) if values.size else None,
        'init_stamp': sample.stamp.to_dict(),
    }



def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(float(ax0), float(bx0))
    iy0 = max(float(ay0), float(by0))
    ix1 = min(float(ax1), float(bx1))
    iy1 = min(float(ay1), float(by1))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1.0, union)


def _mask_rgb_stats(sample: CachedRgbdSample, mask: np.ndarray) -> dict[str, Any]:
    try:
        rgb = _sample_rgb(sample)
    except Exception:
        return {'reason': 'rgb_unavailable'}
    if rgb.shape[:2] != mask.shape or not np.any(mask):
        return {'reason': 'mask_rgb_shape_mismatch_or_empty'}
    values = rgb[mask]
    if values.size == 0:
        return {'reason': 'empty_mask'}
    return {
        'mean_rgb': [float(v) for v in np.mean(values, axis=0)],
        'std_rgb': [float(v) for v in np.std(values, axis=0)],
    }


def _check_model_box_visibility(sample: CachedRgbdSample, projected_mask: np.ndarray, projection: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    depth = _sample_depth_m(sample)
    if depth.shape != projected_mask.shape:
        return False, {'reason': 'depth_mask_shape_mismatch', 'depth_shape': list(depth.shape), 'mask_shape': list(projected_mask.shape)}
    valid_box = projected_mask & np.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid_box))
    if valid_pixels <= 0:
        return False, {'reason': 'no_valid_depth_pixels_in_projected_box', 'valid_pixels': 0}
    box_depth_min = projection.get('box_depth_min_m')
    center_depth = projection.get('center_depth_m')
    if box_depth_min is not None and np.isfinite(float(box_depth_min)):
        occlusion_reference = float(box_depth_min)
        occlusion_reference_source = 'model_box_front_depth'
    elif center_depth is not None and np.isfinite(float(center_depth)):
        occlusion_reference = float(center_depth)
        occlusion_reference_source = 'model_center_depth'
    else:
        return False, {'reason': 'projected_depth_missing', 'valid_pixels': valid_pixels}
    occluded = valid_box & (depth < occlusion_reference - abs(settings.OCCLUSION_DELTA_M))
    occluded_pixels = int(np.count_nonzero(occluded))
    occluded_ratio = occluded_pixels / max(1, valid_pixels)
    center_depth_value = float(center_depth) if center_depth is not None and np.isfinite(float(center_depth)) else None
    depth_band_pixels = 0
    depth_band_ratio = 0.0
    if center_depth_value is not None:
        margin = float(settings.MODEL_BOX_DEPTH_MARGIN_M)
        in_band = valid_box & (depth >= center_depth_value - margin) & (depth <= center_depth_value + margin)
        depth_band_pixels = int(np.count_nonzero(in_band))
        depth_band_ratio = depth_band_pixels / max(1, valid_pixels)
    ok = occluded_ratio <= settings.MODEL_BOX_INIT_MAX_OCCLUSION_RATIO
    return ok, {
        'reason': 'visible' if ok else 'projected_box_occluded',
        'valid_pixels': valid_pixels,
        'occluded_pixels': occluded_pixels,
        'occluded_ratio': occluded_ratio,
        'max_occlusion_ratio': settings.MODEL_BOX_INIT_MAX_OCCLUSION_RATIO,
        'occlusion_reference_m': occlusion_reference,
        'occlusion_reference_source': occlusion_reference_source,
        'center_depth_m': center_depth_value,
        'depth_band_pixels': depth_band_pixels,
        'depth_band_ratio': depth_band_ratio,
    }


def _score_model_box_candidate(obs: YoloObjectObservation, sample: CachedRgbdSample, projected_mask: np.ndarray, projection: Mapping[str, Any]) -> dict[str, Any]:
    bbox_values = projection.get('bbox_xyxy')
    if isinstance(bbox_values, list) and len(bbox_values) == 4:
        projected_bbox = tuple(float(v) for v in bbox_values)
    else:
        h, w = projected_mask.shape
        ys, xs = np.where(projected_mask)
        if xs.size == 0 or ys.size == 0:
            projected_bbox = (0.0, 0.0, float(w), float(h))
        else:
            projected_bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    projected_pixels = int(np.count_nonzero(projected_mask))
    projected_area = float(max(1, projected_pixels))
    center = projection.get('center_pixel_xy')
    if not (isinstance(center, list) and len(center) == 2 and np.all(np.isfinite(np.asarray(center, dtype=float)))):
        x0, y0, x1, y1 = projected_bbox
        center_xy = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    else:
        center_xy = (float(center[0]), float(center[1]))
    center_distance = _center_distance(obs.center_xy, center_xy)
    center_depth = projection.get('center_depth_m')
    depth_diff = None
    if obs.median_depth_m is not None and center_depth is not None and np.isfinite(float(center_depth)):
        depth_diff = abs(float(obs.median_depth_m) - float(center_depth))
    overlap_pixels = int(np.count_nonzero(obs.mask & projected_mask))
    mask_overlap_ratio = overlap_pixels / max(1, int(obs.mask_pixels))
    box_coverage_ratio = overlap_pixels / max(1, projected_pixels)
    bbox_iou = _bbox_iou(obs.bbox_xyxy, projected_bbox)
    area_ratio = float(obs.mask_pixels) / projected_area
    h, w = obs.mask.shape
    ix = int(round(center_xy[0]))
    iy = int(round(center_xy[1]))
    point_inside_mask = 0 <= ix < w and 0 <= iy < h and bool(obs.mask[iy, ix])
    x0, y0, x1, y1 = obs.bbox_xyxy
    point_inside_bbox = x0 <= center_xy[0] <= x1 and y0 <= center_xy[1] <= y1
    max_center = float(settings.MODEL_BOX_CANDIDATE_MAX_CENTER_PX)
    max_depth = float(settings.MODEL_BOX_CANDIDATE_MAX_DEPTH_DIFF_M)
    center_score = min(3.0, center_distance / max(1.0, max_center))
    depth_score = 0.5 if depth_diff is None else min(3.0, depth_diff / max(1.0e-6, max_depth))
    area_score = min(3.0, abs(math.log(max(area_ratio, 1.0e-6)))) * 0.35
    overlap_score = (1.0 - min(1.0, mask_overlap_ratio)) * 0.7
    score = center_score + depth_score + area_score + overlap_score
    if not point_inside_mask:
        score += 0.75
    if not point_inside_bbox:
        score += 0.5
    reject_reasons: list[str] = []
    if center_distance > max_center and not point_inside_bbox:
        reject_reasons.append('center_too_far')
    if depth_diff is not None and depth_diff > max_depth:
        reject_reasons.append('depth_too_different')
    if box_coverage_ratio < settings.MODEL_BOX_OBJECTMASK_MIN_BOX_COVERAGE:
        reject_reasons.append('object_mask_does_not_cover_projected_box')
    if mask_overlap_ratio < settings.MODEL_BOX_CANDIDATE_MIN_MASK_OVERLAP_RATIO and bbox_iou < settings.MODEL_BOX_CANDIDATE_MIN_BBOX_IOU and not point_inside_mask:
        reject_reasons.append('mask_not_near_projected_box')
    if area_ratio < settings.MODEL_BOX_CANDIDATE_MIN_AREA_RATIO:
        reject_reasons.append('mask_too_small_for_box')
    if area_ratio > settings.MODEL_BOX_CANDIDATE_MAX_AREA_RATIO:
        reject_reasons.append('mask_too_large_for_box')
    if score > settings.MODEL_BOX_CANDIDATE_MAX_SCORE:
        reject_reasons.append('score_too_high')
    return {
        'score': float(score),
        'accepted': not reject_reasons,
        'reject_reasons': reject_reasons,
        'center_distance_px': center_distance,
        'depth_diff_m': depth_diff,
        'bbox_iou': bbox_iou,
        'mask_overlap_pixels': overlap_pixels,
        'mask_overlap_ratio': mask_overlap_ratio,
        'projected_box_coverage_ratio': box_coverage_ratio,
        'min_projected_box_coverage_ratio': settings.MODEL_BOX_OBJECTMASK_MIN_BOX_COVERAGE,
        'area_ratio_to_projected_box': area_ratio,
        'point_inside_mask': point_inside_mask,
        'point_inside_bbox': point_inside_bbox,
        'ray_selection': {'status': 'not_available_in_current_server_inputs', 'fallback': 'smallest_accepted_projected_box_mask'},
        'rgb_signature': _mask_rgb_stats(sample, obs.mask),
        'observation': obs,
    }


def _candidate_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in item.items() if k != 'observation'}
    obs = item.get('observation')
    if isinstance(obs, YoloObjectObservation):
        payload['observation'] = obs.to_dict()
    return payload


def _select_model_box_yolo_candidate(cache: ShigureRgbdCache, event: YoloEvent, sample: CachedRgbdSample, shape: tuple[int, int], projected_mask: np.ndarray, projection: Mapping[str, Any]) -> tuple[YoloObjectObservation | None, dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for obj in event.payload.get('objects') or []:
        if not isinstance(obj, Mapping):
            continue
        obs = _observation_from_object(cache, event, obj, shape)
        if obs is None:
            continue
        scored.append(_score_model_box_candidate(obs, sample, projected_mask, projection))
    if not scored:
        return None, {'reason': 'no_yolo_object_with_mask'}
    scored.sort(key=lambda item: (not bool(item.get('accepted')), float(item.get('score', math.inf))))
    accepted = [item for item in scored if item.get('accepted')]
    if accepted:
        accepted.sort(
            key=lambda item: (
                int(item['observation'].mask_pixels),
                -float(item.get('projected_box_coverage_ratio') or 0.0),
                float(item.get('center_distance_px') or math.inf),
            )
        )
    best = accepted[0] if accepted else scored[0]
    reason = 'matched_projected_box_shigure_candidate_mask' if accepted else 'best_projected_box_candidate_rejected'
    return (best['observation'] if accepted else None), {
        'reason': reason,
        'candidate_scope': 'shigure_object_masks_covering_projected_hololens_depth_box',
        'selection_policy': 'smallest_accepted_mask_after_projected_box_coverage',
        'ray_selection': {'status': 'not_available_in_current_server_inputs', 'fallback': 'smallest_accepted_projected_box_mask'},
        'candidate_count': len(scored),
        'accepted_count': len(accepted),
        'best': _candidate_payload(best),
        'top_candidates': [_candidate_payload(item) for item in scored[:8]],
    }


def _init_model_box_primary(cache: ShigureRgbdCache, task: Mapping[str, Any], metadata: list[CachedSampleMetadata], first_after: CachedSampleMetadata, capture_seconds: float) -> tuple[YoloInitResult | None, dict[str, Any]]:
    first_sample = cache.get_sample(first_after.stamp, mode='nearest')
    if first_sample is None:
        return None, {'mode': 'model_box_yolo_mask_init', 'reason': 'first_frame_unavailable'}
    info = _camera_info_message(first_after.camera_info or first_sample.camera_info)
    h = int(info.get('height') or 0) if info else 0
    w = int(info.get('width') or 0) if info else 0
    shape = (h, w) if h > 0 and w > 0 else first_sample.depth.shape[:2]
    projected_mask, projection = _project_model_box_to_shigure(task, first_after.camera_info or first_sample.camera_info, shape)
    init_info: dict[str, Any] = {
        'mode': 'model_box_yolo_mask_init',
        'projection': projection,
        'metadata_frame_count': len(metadata),
        'reference_source': 'selected_shigure_yolo_mask',
        'candidate_scope': 'shigure_object_masks_covering_projected_hololens_depth_box',
        'ray_selection': {'status': 'not_available_in_current_server_inputs', 'fallback': 'smallest_accepted_projected_box_mask'},
    }
    if projected_mask is None:
        return None, {**init_info, 'reason': projection.get('reason')}

    unique_events = _unique_yolo_events(metadata)
    deadline = capture_seconds + settings.INIT_TIMEOUT_SECONDS
    post_events = [event for event in unique_events if capture_seconds <= event.seconds <= deadline]
    init_info['unique_yolo_count'] = len(unique_events)
    init_info['post_capture_unique_count'] = len(post_events)
    checked: list[dict[str, Any]] = []
    for event in post_events:
        sample = cache.get_sample(event.sample.stamp, mode='nearest')
        if sample is None:
            checked.append({'stamp': event.sample.stamp.to_dict(), 'reason': 'sample_unavailable'})
            continue
        visible, visibility_stats = _check_model_box_visibility(sample, projected_mask, projection)
        record: dict[str, Any] = {'stamp': sample.stamp.to_dict(), 'visibility': visibility_stats}
        if not visible:
            checked.append(record)
            continue
        selected_obs, candidate_info = _select_model_box_yolo_candidate(cache, event, sample, shape, projected_mask, projection)
        record['candidate_match'] = candidate_info
        checked.append(record)
        if selected_obs is None:
            continue
        trusted, reference_depth, depth_stats = _build_reference_from_yolo_mask(sample, selected_obs.mask)
        record['depth_init'] = depth_stats
        if trusted is None or reference_depth is None:
            continue
        post_observations = _collect_target_observations(cache, post_events, selected_obs.object_id, shape)
        init_stats = {
            **init_info,
            'reason': 'initialized',
            'target_object_id': selected_obs.object_id,
            'selected_observation': selected_obs.to_dict(),
            'selected_signature': _mask_rgb_stats(sample, selected_obs.mask),
            'reference_stamp': sample.stamp.to_dict(),
            'init_start_stamp': sample.stamp.to_dict(),
            'init_end_stamp': sample.stamp.to_dict(),
            'visibility_init': visibility_stats,
            'candidate_match': candidate_info,
            'depth_init': depth_stats,
            'checked_init_frames': checked[:25],
        }
        return YoloInitResult(
            trusted_mask=trusted,
            reference_depth=reference_depth,
            init_frame=sample,
            object_id=selected_obs.object_id,
            projection=projection,
            init_stats=init_stats,
            yolo_observations=post_observations,
            yolo_events=post_events,
            projected_mask=projected_mask,
        ), init_stats
    reason = 'no_post_capture_yolo_events' if not post_events else 'no_visible_projected_box_candidate_mask'
    if checked:
        last = checked[-1]
        reason = str((last.get('candidate_match') or last.get('visibility') or {}).get('reason') or reason)
    return None, {**init_info, 'reason': reason, 'checked_init_frames': checked[:25]}

def _init_yolo_primary(cache: ShigureRgbdCache, task: Mapping[str, Any], metadata: list[CachedSampleMetadata], first_after: CachedSampleMetadata, capture_seconds: float) -> tuple[YoloInitResult | None, dict[str, Any]]:
    shape = None
    if first_after.camera_info:
        info = _camera_info_message(first_after.camera_info)
        h = int(info.get('height') or 0)
        w = int(info.get('width') or 0)
        if h > 0 and w > 0:
            shape = (h, w)
    first_sample = cache.get_sample(first_after.stamp, mode='nearest')
    if first_sample is None:
        return None, {'reason': 'first_frame_unavailable'}
    if shape is None:
        shape = first_sample.depth.shape[:2]
    projection, projection_info = _project_object_center_to_shigure(task, first_after.camera_info or first_sample.camera_info, shape)
    if projection is None:
        return None, {'reason': 'projection_failed', 'projection': projection_info}
    unique_events = _unique_yolo_events(metadata)
    pre_events = [event for event in unique_events if event.seconds < capture_seconds]
    pre_events = pre_events[-max(0, settings.YOLO_PRE_CAPTURE_UNIQUE_COUNT):]
    post_deadline = capture_seconds + settings.YOLO_INIT_HARD_TIMEOUT_SECONDS
    post_events = [event for event in unique_events if capture_seconds <= event.seconds <= post_deadline]
    match_events = pre_events + post_events
    target_id, match_info = _find_yolo_target(cache, match_events, projection, shape)
    init_info: dict[str, Any] = {
        'mode': 'yolo_primary',
        'projection': projection_info,
        'unique_yolo_count': len(unique_events),
        'pre_capture_unique_count': len(pre_events),
        'post_capture_unique_count': len(post_events),
        'match': match_info,
    }
    if target_id is None:
        return None, {**init_info, 'reason': 'yolo_target_match_failed'}
    post_observations = _collect_target_observations(cache, post_events, target_id, shape)
    stable, stable_stats = _stable_observation_prefix(post_observations, minimum_count=max(1, settings.YOLO_INIT_STABLE_UNIQUE_COUNT))
    init_info['target_object_id'] = target_id
    init_info['stable'] = stable_stats
    init_info['post_observations'] = [obs.to_dict() for obs in post_observations]
    if not stable:
        return None, {**init_info, 'reason': stable_stats.get('reason', 'yolo_not_stable')}
    init_obs = stable[0]
    init_sample = cache.get_sample(init_obs.stamp, mode='nearest')
    if init_sample is None:
        return None, {**init_info, 'reason': 'init_sample_unavailable'}
    trusted, reference_depth, depth_stats = _build_reference_from_yolo_mask(init_sample, init_obs.mask)
    init_info['depth_init'] = depth_stats
    if trusted is None or reference_depth is None:
        return None, {**init_info, 'reason': depth_stats.get('reason')}
    init_stats = {
        **init_info,
        'reason': 'initialized',
        'init_frame_count': len(stable),
        'init_start_stamp': stable[0].stamp.to_dict(),
        'init_end_stamp': stable[-1].stamp.to_dict(),
        'reference_stamp': init_sample.stamp.to_dict(),
    }
    return YoloInitResult(
        trusted_mask=trusted,
        reference_depth=reference_depth,
        init_frame=init_sample,
        object_id=target_id,
        projection=projection_info,
        init_stats=init_stats,
        yolo_observations=post_observations,
        yolo_events=post_events,
        projected_mask=init_obs.mask,
    ), init_stats


def _is_partial_occlusion(decision: FrameDecision) -> bool:
    if decision.valid_pixels <= 0:
        return False
    unchanged_pixels = max(0, decision.unoccluded_pixels - decision.taken_pixels)
    unchanged_ratio = unchanged_pixels / max(1, decision.valid_pixels)
    deeper_ratio = decision.taken_pixels / max(1, decision.valid_pixels)
    return (
        decision.occluded_ratio >= settings.PARTIAL_OCCLUSION_CLOSER_RATIO
        and unchanged_ratio >= settings.PARTIAL_OCCLUSION_UNCHANGED_RATIO
        and deeper_ratio <= settings.PARTIAL_OCCLUSION_DEEPER_MAX_RATIO
    )


def _depth_scan_for_taken(frames: list[CachedRgbdSample], trusted_mask: np.ndarray, reference_depth: np.ndarray, *, after_seconds: float) -> tuple[CachedRgbdSample | None, CachedRgbdSample | None, list[FrameDecision], dict[str, Any]]:
    taken_count = 0
    candidate_start: CachedRgbdSample | None = None
    confirmed: CachedRgbdSample | None = None
    decisions: list[FrameDecision] = []
    partial_occlusions = 0
    for frame in frames:
        if frame.stamp.seconds < after_seconds:
            continue
        decision, _occluded, _taken = _classify_frame(frame, trusted_mask, reference_depth)
        decisions.append(decision)
        if decision.full_occlusion or _is_partial_occlusion(decision):
            partial_occlusions += 1
            taken_count = 0
            candidate_start = None
            continue
        if decision.candidate:
            if taken_count == 0:
                candidate_start = frame
            taken_count += 1
            if taken_count >= max(1, settings.TAKEN_CONSECUTIVE_FRAMES):
                confirmed = frame
                break
        else:
            taken_count = 0
            candidate_start = None
    return candidate_start, confirmed, decisions, {'partial_occlusion_frames': partial_occlusions}


def _event_has_object(event: YoloEvent, object_id: str) -> bool:
    for obj in event.payload.get('objects') or []:
        if isinstance(obj, Mapping) and str(obj.get('object_id')) == object_id:
            return True
    return False


def _run_yolo_primary_tracking(cache: ShigureRgbdCache, init: YoloInitResult, end: RosStamp) -> tuple[CachedRgbdSample | None, CachedRgbdSample | None, list[FrameDecision], dict[str, Any], list[CachedRgbdSample]]:
    if settings.YOLO_TRACKING_TRIGGER_MODE in {'direct', 'direct_depth_scan', 'state_machine'}:
        scan_start_seconds = init.init_frame.stamp.seconds
        frames = list(cache.iter_depth_samples(start=init.init_frame.stamp, end=end))
        candidate_start, confirmed, decisions, depth_info = _depth_scan_for_taken(
            frames,
            init.trusted_mask,
            init.reference_depth,
            after_seconds=scan_start_seconds,
        )
        info = {
            'trigger_reason': 'direct_depth_scan_after_shigure_mask_init',
            'trigger_mode': settings.YOLO_TRACKING_TRIGGER_MODE,
            'trigger_scan_start_seconds': scan_start_seconds,
            'tracked_object_id': init.object_id,
            'yolo_observation_count': len(init.yolo_observations),
            'depth_scan': depth_info,
        }
        return candidate_start, confirmed, decisions, info, frames

    observations_by_key = {obs.event.sample.key: obs for obs in init.yolo_observations}
    move_votes = 0
    last_yolo_seconds = init.init_frame.stamp.seconds
    trigger_seconds: float | None = None
    trigger_reason = 'no_yolo_trigger'
    anchor = init.yolo_observations[0] if init.yolo_observations else None
    if anchor is not None:
        normal_threshold = max(settings.YOLO_MOVE_CENTER_PX, anchor.bbox_diag * settings.YOLO_MOVE_CENTER_BBOX_RATIO)
        strong_threshold = max(settings.YOLO_MOVE_STRONG_CENTER_PX, anchor.bbox_diag * settings.YOLO_MOVE_STRONG_BBOX_RATIO)
        for event in init.yolo_events:
            if event.seconds <= init.init_frame.stamp.seconds:
                continue
            obs = observations_by_key.get(event.sample.key)
            if obs is None and not _event_has_object(event, init.object_id):
                trigger_seconds = last_yolo_seconds
                trigger_reason = 'target_id_missing'
                break
            if obs is None:
                last_yolo_seconds = event.seconds
                continue
            distance = _center_distance(anchor.center_xy, obs.center_xy)
            if distance >= strong_threshold:
                trigger_seconds = last_yolo_seconds
                trigger_reason = 'strong_yolo_center_move'
                break
            if distance >= normal_threshold:
                move_votes += 1
                if move_votes >= max(1, settings.YOLO_MOVE_STABLE_UNIQUE_COUNT):
                    trigger_seconds = last_yolo_seconds
                    trigger_reason = 'debounced_yolo_center_move'
                    break
            else:
                move_votes = 0
            last_yolo_seconds = event.seconds
    if trigger_seconds is None:
        info = {
            'trigger_reason': trigger_reason,
            'trigger_scan_start_seconds': None,
            'tracked_object_id': init.object_id,
            'yolo_observation_count': len(init.yolo_observations),
            'depth_scan': {'status': 'skipped_no_yolo_trigger'},
        }
        return None, None, [], info, []
    scan_start = _stamp_from_seconds(trigger_seconds)
    frames = list(cache.iter_depth_samples(start=scan_start, end=end))
    candidate_start, confirmed, decisions, depth_info = _depth_scan_for_taken(
        frames,
        init.trusted_mask,
        init.reference_depth,
        after_seconds=trigger_seconds,
    )
    info = {
        'trigger_reason': trigger_reason,
        'trigger_scan_start_seconds': trigger_seconds,
        'tracked_object_id': init.object_id,
        'yolo_observation_count': len(init.yolo_observations),
        'depth_scan': depth_info,
    }
    return candidate_start, confirmed, decisions, info, frames



def _resolve_projected_mask(task: Mapping[str, Any], first_depth: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    shape = first_depth.shape[:2]
    candidates: list[tuple[str, Path]] = []
    env_path = os.environ.get('TAKEN_OBJECT_PROJECTED_MASK') or os.environ.get('TAKEN_OBJECT_PROJECTED_MASK_PATH')
    if env_path:
        candidates.append(('env_projected_mask', Path(env_path)))
    projection = task.get('TakenObjectProjection') if isinstance(task.get('TakenObjectProjection'), Mapping) else {}
    if projection.get('mask_path'):
        candidates.append(('task_projected_mask', Path(str(projection.get('mask_path')))))
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for taken object projected mask')
    candidates.append(('sam3_mask', model_worker_file(task_timestamp, 'sam3.mask')))

    for source, raw_path in candidates:
        path = raw_path if raw_path.is_file() else _resolve_existing_path(str(raw_path))
        if not path or not path.is_file():
            continue
        mask = _mask_from_image(path, shape)
        if np.count_nonzero(mask) > 0:
            return mask, {'source': source, 'path': str(path), 'shape': list(shape)}

    fallback = _selection_box_mask(task, shape)
    if fallback is not None and np.count_nonzero(fallback) > 0:
        return fallback, {'source': 'selection_box_scaled_fallback', 'shape': list(shape)}
    return None, {'source': 'missing', 'shape': list(shape)}


def _init_trusted_mask(frames: list[CachedRgbdSample], projected_mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, CachedRgbdSample | None, dict[str, Any]]:
    if not frames:
        return None, None, None, {'reason': 'no_init_frames'}
    depths = []
    used_frames: list[CachedRgbdSample] = []
    for sample in frames:
        depth = _sample_depth_m(sample)
        if depth.shape != projected_mask.shape:
            continue
        depths.append(depth)
        used_frames.append(sample)
    if len(depths) < 2:
        return None, None, None, {'reason': 'not_enough_init_depth_frames', 'frame_count': len(depths)}
    stack = np.stack(depths, axis=0)
    valid = np.isfinite(stack) & (stack > 0.0) & projected_mask.reshape(1, *projected_mask.shape)
    valid_count = np.count_nonzero(valid, axis=0)
    has_depth = valid_count > 0
    masked = np.where(valid, stack, np.nan)
    depth_min = np.nanmin(masked, axis=0)
    depth_max = np.nanmax(masked, axis=0)
    depth_mean = np.nanmean(masked, axis=0)
    stable = projected_mask & has_depth & ((depth_max - depth_min) <= settings.INIT_STABLE_DEPTH_DELTA_M)
    stable_ratio = float(np.count_nonzero(stable)) / max(1, int(np.count_nonzero(projected_mask)))
    if stable_ratio < settings.INIT_STABLE_PIXEL_RATIO:
        return None, None, used_frames[-1], {
            'reason': 'unstable_init_depth',
            'stable_ratio': stable_ratio,
            'projected_pixels': int(np.count_nonzero(projected_mask)),
            'stable_pixels': int(np.count_nonzero(stable)),
        }
    stable_depth = depth_mean[stable]
    front_min = float(np.nanmin(stable_depth)) if stable_depth.size else 0.0
    front_max = float(np.nanmax(stable_depth)) if stable_depth.size else 0.0
    trusted = stable & (depth_mean >= front_min - settings.DEPTH_MARGIN_M) & (depth_mean <= front_max + settings.DEPTH_MARGIN_M)
    image_ratio = float(np.count_nonzero(trusted)) / float(trusted.size)
    if image_ratio < settings.TRUSTED_MASK_MIN_IMAGE_RATIO:
        return None, None, used_frames[-1], {
            'reason': 'trusted_mask_too_small',
            'trusted_image_ratio': image_ratio,
            'trusted_pixels': int(np.count_nonzero(trusted)),
        }
    reference_depth = np.where(trusted, depth_mean, 0.0).astype(np.float32)
    return trusted, reference_depth, used_frames[-1], {
        'reason': 'initialized',
        'init_frame_count': len(used_frames),
        'init_start_stamp': used_frames[0].stamp.to_dict(),
        'init_end_stamp': used_frames[-1].stamp.to_dict(),
        'projected_pixels': int(np.count_nonzero(projected_mask)),
        'stable_pixels': int(np.count_nonzero(stable)),
        'stable_ratio': stable_ratio,
        'trusted_pixels': int(np.count_nonzero(trusted)),
        'trusted_image_ratio': image_ratio,
        'mesh_front_surface_min_depth_m': front_min,
        'mesh_front_surface_max_depth_m': front_max,
    }


def _classify_frame(sample: CachedRgbdSample, trusted_mask: np.ndarray, reference_depth: np.ndarray) -> tuple[FrameDecision, np.ndarray, np.ndarray]:
    depth = _sample_depth_m(sample)
    if depth.shape != trusted_mask.shape:
        raise ValueError(f'depth shape {depth.shape} does not match trusted mask {trusted_mask.shape}')
    valid = trusted_mask & np.isfinite(depth) & (depth > 0.0) & (reference_depth > 0.0)
    delta = depth - reference_depth
    occluded = valid & (delta <= settings.OCCLUSION_DELTA_M)
    unoccluded = valid & ~occluded
    taken = unoccluded & (delta >= settings.TAKEN_DELTA_M)
    valid_pixels = int(np.count_nonzero(valid))
    occluded_pixels = int(np.count_nonzero(occluded))
    unoccluded_pixels = int(np.count_nonzero(unoccluded))
    taken_pixels = int(np.count_nonzero(taken))
    occluded_ratio = occluded_pixels / max(1, valid_pixels)
    taken_ratio = taken_pixels / max(1, unoccluded_pixels)
    full_occlusion = valid_pixels > 0 and occluded_ratio >= settings.FULL_OCCLUSION_RATIO
    candidate = unoccluded_pixels > 0 and taken_ratio >= settings.TAKEN_RATIO
    status = 'full_occlusion' if full_occlusion else ('taken_candidate' if candidate else 'present')
    return (
        FrameDecision(
            stamp=sample.stamp,
            status=status,
            valid_pixels=valid_pixels,
            occluded_pixels=occluded_pixels,
            unoccluded_pixels=unoccluded_pixels,
            taken_pixels=taken_pixels,
            occluded_ratio=occluded_ratio,
            taken_ratio=taken_ratio,
            candidate=candidate,
            full_occlusion=full_occlusion,
        ),
        occluded,
        taken,
    )


def _rgb_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if a.shape[:2] != b.shape[:2] or mask.shape != a.shape[:2] or not np.any(mask):
        return None
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _backtrack_rgb_frame(frames: list[CachedRgbdSample], depth_frame: CachedRgbdSample, init_frame: CachedRgbdSample, trusted_mask: np.ndarray) -> tuple[CachedRgbdSample, dict[str, Any]]:
    start_seconds = depth_frame.stamp.seconds - max(0.0, settings.RGB_BACKTRACK_SECONDS)
    candidates = [f for f in frames if start_seconds <= f.stamp.seconds <= depth_frame.stamp.seconds]
    if not candidates:
        return depth_frame, {'status': 'fallback_no_rgb_candidates'}
    try:
        init_rgb = _sample_rgb(init_frame)
    except Exception as exc:
        return depth_frame, {'status': 'fallback_no_init_rgb', 'reason': str(exc)}
    rgb_cache: dict[str, np.ndarray] = {}
    records = []
    selected: CachedRgbdSample | None = None
    for index in range(len(candidates) - 1, -1, -1):
        frame = candidates[index]
        try:
            key = sample_key(frame.stamp)
            rgb = rgb_cache.setdefault(key, _sample_rgb(frame))
            prev_rgb = None
            if index > 0:
                prev_key = sample_key(candidates[index - 1].stamp)
                prev_rgb = rgb_cache.setdefault(prev_key, _sample_rgb(candidates[index - 1]))
            init_diff = _rgb_diff(rgb, init_rgb, trusted_mask)
            adjacent_diff = _rgb_diff(rgb, prev_rgb, trusted_mask) if prev_rgb is not None else 0.0
        except Exception:
            continue
        record = {
            'stamp': frame.stamp.to_dict(),
            'init_diff': init_diff,
            'adjacent_diff': adjacent_diff,
        }
        records.append(record)
        if (
            init_diff is not None
            and adjacent_diff is not None
            and init_diff <= settings.RGB_INIT_DIFF_THRESHOLD
            and adjacent_diff <= settings.RGB_ADJACENT_DIFF_THRESHOLD
        ):
            selected = frame
            break
    if selected is None:
        return depth_frame, {'status': 'fallback_no_quiet_frame', 'checked': records[:25]}
    return selected, {'status': 'found', 'selected_stamp': selected.stamp.to_dict(), 'checked': records[:25]}


def _nearest_sample_by_stamp(frames: list[CachedRgbdSample], stamp: RosStamp) -> CachedRgbdSample | None:
    if not frames:
        return None
    target = stamp.seconds
    return min(frames, key=lambda frame: abs(frame.stamp.seconds - target))


def _rgb_window_for_taken_result(
    cache: ShigureRgbdCache,
    candidate_start: CachedRgbdSample,
    confirmed: CachedRgbdSample,
    init_frame: CachedRgbdSample,
) -> tuple[list[CachedRgbdSample], CachedRgbdSample, CachedRgbdSample, dict[str, Any]]:
    start_seconds = max(init_frame.stamp.seconds, candidate_start.stamp.seconds - max(0.0, settings.RGB_BACKTRACK_SECONDS))
    start = _stamp_from_seconds(start_seconds)
    frames = list(cache.iter_samples(start=start, end=confirmed.stamp))
    candidate_full = _nearest_sample_by_stamp(frames, candidate_start.stamp)
    confirmed_full = _nearest_sample_by_stamp(frames, confirmed.stamp)
    if candidate_full is None:
        fallback = cache.get_sample(candidate_start.stamp, mode='nearest')
        candidate_full = fallback if fallback is not None else candidate_start
        if fallback is not None:
            frames.append(fallback)
    if confirmed_full is None:
        fallback = cache.get_sample(confirmed.stamp, mode='nearest')
        confirmed_full = fallback if fallback is not None else confirmed
        if fallback is not None:
            frames.append(fallback)
    frames.sort(key=lambda frame: frame.stamp.seconds)
    info = {
        'status': 'loaded_rgb_backtrack_window' if frames else 'fallback_no_rgb_window',
        'start_seconds': start_seconds,
        'end_stamp': confirmed.stamp.to_dict(),
        'frame_count': len(frames),
        'candidate_rgb_delta_s': abs(candidate_full.stamp.seconds - candidate_start.stamp.seconds),
        'confirmed_rgb_delta_s': abs(confirmed_full.stamp.seconds - confirmed.stamp.seconds),
    }
    return frames, candidate_full, confirmed_full, info


def _find_marker_pose_path() -> Path | None:
    # Shigurei ArMarker is treated as a stable camera-side calibration.
    # Consumers read the recorder-maintained history instead of scanning test
    # snapshots or per-frame cache metadata.
    return latest_marker_pose_path()


def _backup_sample(
    task: Mapping[str, Any],
    sample: CachedRgbdSample,
    output_root: Path,
    *,
    kind: str | None = None,
) -> Path:
    task_name = str(task.get('task_name') or task.get('task_id') or 'task').strip()
    key = sample_key(sample.stamp)
    suffix = f'_{kind}' if kind else ''
    backup_dir = output_root / f'{task_name}{suffix}_{key}'
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample.rgb_bgr[:, :, ::-1]).save(backup_dir / 'rgb.png')
    Image.fromarray(np.asarray(sample.depth, dtype=np.uint16)).save(backup_dir / 'depth.png')
    if sample.camera_info is not None:
        _write_json(backup_dir / 'camera_info.json', sample.camera_info)
    elif sample.camera_info_path and sample.camera_info_path.is_file():
        shutil.copy2(sample.camera_info_path, backup_dir / 'camera_info.json')
    if sample.yolo is not None:
        _write_json(backup_dir / 'active_objects.json', sample.yolo)
        _write_json(backup_dir / 'yolo.json', sample.yolo)
    elif sample.yolo_path and sample.yolo_path.is_file():
        shutil.copy2(sample.yolo_path, backup_dir / 'active_objects.json')
        shutil.copy2(sample.yolo_path, backup_dir / 'yolo.json')
    marker_pose = _find_marker_pose_path()
    if marker_pose is not None:
        shutil.copy2(marker_pose, backup_dir / 'marker_6d_pose.json')
    _write_json(
        backup_dir / 'meta.json',
        {
            'backup_kind': kind or 'result',
            'stamp': sample.stamp.to_dict(),
            'source_chunk_id': sample.chunk_id,
            'source_frame_index': sample.frame_index,
            'source_camera_info_path': str(sample.camera_info_path) if sample.camera_info_path else None,
            'source_yolo_hash': sample.yolo_hash,
            'marker_pose_copied': marker_pose is not None,
            'written_at': _utc_now(),
        },
    )
    return backup_dir


def _taken_debug_dir(task_timestamp: str) -> Path:
    if not task_timestamp:
        raise ValueError('task_timestamp is required for taken debug artifacts')
    debug_dir = model_debug_dir(task_timestamp)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _save_debug_masks(task_timestamp: str, trusted: np.ndarray, reference_depth: np.ndarray, projected: np.ndarray) -> dict[str, str]:
    debug_dir = _taken_debug_dir(task_timestamp)
    projected_path = debug_dir / '07_taken_projected_mask.png'
    trusted_path = debug_dir / '07_taken_trusted_mask.png'
    reference_path = debug_dir / '07_taken_reference_depth_m.npy'
    Image.fromarray(projected.astype(np.uint8) * 255).save(projected_path)
    Image.fromarray(trusted.astype(np.uint8) * 255).save(trusted_path)
    np.save(reference_path, reference_depth.astype(np.float32))
    return {
        'projected_mask_path': str(projected_path),
        'trusted_mask_path': str(trusted_path),
        'reference_depth_m_path': str(reference_path),
    }


def _draw_debug_bbox(draw: ImageDraw.ImageDraw, bbox: Any, label: str, color: tuple[int, int, int], *, width: int = 3) -> None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    if label:
        text_xy = (max(0.0, x0 + 4.0), max(0.0, y0 + 4.0))
        draw.text(text_xy, label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))


def _save_init_debug_files(
    task_timestamp: str,
    sample: CachedRgbdSample,
    trusted: np.ndarray,
    reference_depth: np.ndarray,
    projected: np.ndarray,
    projection: Mapping[str, Any],
    object_id: str,
    selected_observation: Mapping[str, Any] | None,
) -> dict[str, str]:
    debug_dir = _taken_debug_dir(task_timestamp)
    projected_path = debug_dir / '07_taken_init_projected_box_mask.png'
    trusted_path = debug_dir / '07_taken_init_trusted_shigure_mask.png'
    reference_path = debug_dir / '07_taken_init_reference_depth_m.npy'
    overlay_path = debug_dir / '07_taken_init_mask_rgb_overlay.png'
    Image.fromarray(projected.astype(np.uint8) * 255).save(projected_path)
    Image.fromarray(trusted.astype(np.uint8) * 255).save(trusted_path)
    np.save(reference_path, reference_depth.astype(np.float32))

    rgb = np.asarray(sample.rgb_bgr[:, :, ::-1], dtype=np.uint8)
    overlay = rgb.astype(np.float32).copy()
    if projected.shape == rgb.shape[:2] and np.any(projected):
        projected_color = np.asarray([255.0, 210.0, 0.0], dtype=np.float32)
        overlay[projected] = overlay[projected] * 0.65 + projected_color * 0.35
    if trusted.shape == rgb.shape[:2] and np.any(trusted):
        trusted_color = np.asarray([0.0, 190.0, 255.0], dtype=np.float32)
        overlay[trusted] = overlay[trusted] * 0.45 + trusted_color * 0.55
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    _draw_debug_bbox(draw, projection.get('bbox_xyxy'), 'projected model box', (255, 210, 0), width=3)
    if isinstance(selected_observation, Mapping):
        _draw_debug_bbox(draw, selected_observation.get('bbox_xyxy'), f'shigure mask id {object_id}', (0, 190, 255), width=3)
    image.save(overlay_path)

    return {
        'init_projected_box_mask_path': str(projected_path),
        'init_trusted_shigure_mask_path': str(trusted_path),
        'init_reference_depth_m_path': str(reference_path),
        'init_mask_rgb_overlay_path': str(overlay_path),
        'model_debug_init_mask_rgb_overlay_path': str(overlay_path),
    }


def _save_decisions_debug(task_timestamp: str, payload: Mapping[str, Any]) -> str:
    decisions_path = _taken_debug_dir(task_timestamp) / '07_taken_decisions.json'
    _write_json(decisions_path, payload)
    return str(decisions_path)


def _write_history_baseline_artifacts(
    backup_dir: Path,
    sample: CachedRgbdSample,
    trusted_mask: np.ndarray,
    reference_depth: np.ndarray,
) -> dict[str, Any]:
    mask_path = backup_dir / 'old_mask.png'
    reference_path = backup_dir / 'old_reference_depth_m.npy'
    Image.fromarray(trusted_mask.astype(np.uint8) * 255).save(mask_path)
    np.save(reference_path, np.asarray(reference_depth, dtype=np.float32))
    payload = {
        'source': 'taken_object_detection_shigure_object_mask',
        'coordinate_space': 'fixed_shigure_image',
        'old_rgb_path': str(backup_dir / 'rgb.png'),
        'old_depth_path': str(backup_dir / 'depth.png'),
        'old_mask_path': str(mask_path),
        'old_reference_depth_m_path': str(reference_path),
        'camera_info_path': str(backup_dir / 'camera_info.json'),
        'stamp': sample.stamp.to_dict(),
        'valid_mask_pixels': int(np.count_nonzero(trusted_mask)),
    }
    _write_json(backup_dir / 'history_baseline.json', payload)
    return payload


def _tracking_window_payload(capture_seconds: float, capture_source: str | None, first_after: CachedSampleMetadata | None, input_frame_count: int) -> dict[str, Any]:
    payload = {
        'capture_time_source': capture_source,
        'capture_time_seconds': capture_seconds,
        'tracking_start_seconds': capture_seconds,
        'tracking_end_seconds': capture_seconds + settings.TRACKING_DURATION_SECONDS,
        'input_frame_count': input_frame_count,
        'offline_stop_when_history_exhausted': settings.OFFLINE_STOP_WHEN_HISTORY_EXHAUSTED,
    }
    if first_after is not None:
        first_delay = first_after.stamp.seconds - capture_seconds
        payload['first_frame_stamp'] = first_after.stamp.to_dict()
        payload['first_frame_delay_seconds'] = first_delay
    return payload


def _write_not_taken(json_path: Path, task: dict[str, Any], *, tracking_window: dict[str, Any], projection: dict[str, Any], init: dict[str, Any], decisions: list[FrameDecision], debug_files: dict[str, str], output_dir: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        'result_timestamp': None,
        'backup_shigurei_dir': None,
        'tracking_window': tracking_window,
        'projection': projection,
        'init': init,
        'history_baseline': init.get('history_baseline') if isinstance(init, Mapping) else None,
        'checked_frame_count': len(decisions),
        'debug_files': debug_files,
        'output_dir': str(output_dir),
    }
    if extra:
        payload.update(dict(extra))
    _write_status(json_path, task, 'NOT_TAKEN', **payload)
    return {'status': 'NOT_TAKEN', 'checked_frame_count': len(decisions)}


def _write_taken(json_path: Path, task: dict[str, Any], *, frames: list[CachedRgbdSample], candidate_start: CachedRgbdSample, confirmed: CachedRgbdSample, init_frame: CachedRgbdSample, trusted: np.ndarray, tracking_window: dict[str, Any], projection: dict[str, Any], init_stats: dict[str, Any], decisions: list[FrameDecision], debug_files: dict[str, str], output_dir: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result_frame, rgb_backtrack = _backtrack_rgb_frame(frames, candidate_start, init_frame, trusted)
    backup_dir = _backup_sample(task, result_frame, output_dir)
    result_timestamp = result_frame.stamp.to_dict()
    payload = {
        'result_timestamp': result_timestamp,
        'backup_shigurei_dir': str(backup_dir),
        'tracking_window': tracking_window,
        'projection': projection,
        'init': init_stats,
        'history_baseline': init_stats.get('history_baseline') if isinstance(init_stats, Mapping) else None,
        'depth_taken_timestamp': candidate_start.stamp.to_dict(),
        'depth_confirm_timestamp': confirmed.stamp.to_dict(),
        'full_occlusion_start_timestamp': None,
        'used_full_occlusion_start': False,
        'rgb_backtrack': rgb_backtrack,
        'checked_frame_count': len(decisions),
        'debug_files': debug_files,
        'output_dir': str(output_dir),
    }
    if extra:
        payload.update(dict(extra))
    payload.update(_publish_taken_result_artifacts(task, backup_dir))
    _write_status(json_path, task, 'TAKEN', **payload)
    return {'status': 'TAKEN', 'result_timestamp': result_timestamp, 'backup_shigurei_dir': str(backup_dir)}


def _run_legacy_projected_mask_detection(json_path: Path, task: dict[str, Any], cache: ShigureRgbdCache, output_dir: Path, *, capture_seconds: float, capture_source: str | None, start: RosStamp, end: RosStamp, fallback_reason: Mapping[str, Any] | None = None) -> dict[str, Any]:
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    frames = list(cache.iter_samples(start=start, end=end))
    tracking_window = _tracking_window_payload(capture_seconds, capture_source, None, len(frames))
    tracking_window['mode'] = 'legacy_projected_mask'
    if fallback_reason:
        tracking_window['fallback_reason'] = dict(fallback_reason)
    if not frames:
        _write_status(json_path, task, 'INIT_FAILED', reason='no_shigure_frames_in_window', tracking_window=tracking_window, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'no_shigure_frames_in_window'}
    first_delay = frames[0].stamp.seconds - capture_seconds
    tracking_window['first_frame_stamp'] = frames[0].stamp.to_dict()
    tracking_window['first_frame_delay_seconds'] = first_delay
    if first_delay > settings.INIT_MAX_START_DELAY_SECONDS:
        _write_status(json_path, task, 'INIT_FAILED', reason='first_frame_too_late', tracking_window=tracking_window, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'first_frame_too_late'}

    first_depth = _sample_depth_m(frames[0])
    projected_mask, projection = _resolve_projected_mask(task, first_depth)
    projection['mode'] = 'legacy_projected_mask'
    if projected_mask is None:
        _write_status(json_path, task, 'INIT_FAILED', reason='projected_mask_missing', tracking_window=tracking_window, projection=projection, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'projected_mask_missing'}

    init_deadline = frames[0].stamp.seconds + min(settings.INIT_TIMEOUT_SECONDS, settings.INIT_STABLE_WINDOW_SECONDS)
    init_frames = [frame for frame in frames if frame.stamp.seconds <= init_deadline]
    trusted, reference_depth, init_end_frame, init_stats = _init_trusted_mask(init_frames, projected_mask)
    init_stats['mode'] = 'legacy_projected_mask'
    if trusted is None or reference_depth is None or init_end_frame is None:
        _write_status(json_path, task, 'INIT_FAILED', reason=init_stats.get('reason'), tracking_window=tracking_window, projection=projection, init=init_stats, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': init_stats.get('reason')}

    _write_status(json_path, task, 'RUNNING', tracking_window=tracking_window, projection=projection, init=init_stats, output_dir=str(output_dir))

    decisions: list[FrameDecision] = []
    taken_count = 0
    candidate_start: CachedRgbdSample | None = None
    confirmed: CachedRgbdSample | None = None
    full_occlusion_start: CachedRgbdSample | None = None
    active_full_occlusion = False
    for frame in frames:
        if frame.stamp.seconds <= init_end_frame.stamp.seconds:
            continue
        decision, _occluded, _taken = _classify_frame(frame, trusted, reference_depth)
        decisions.append(decision)
        if decision.full_occlusion:
            taken_count = 0
            candidate_start = None
            if not active_full_occlusion:
                full_occlusion_start = frame
                active_full_occlusion = True
            continue
        if active_full_occlusion and not decision.full_occlusion:
            active_full_occlusion = False
        if decision.candidate:
            if taken_count == 0:
                candidate_start = frame
            taken_count += 1
            if taken_count >= max(1, settings.TAKEN_CONSECUTIVE_FRAMES):
                confirmed = frame
                break
        else:
            taken_count = 0
            candidate_start = None

    debug_files: dict[str, str] = {}
    if settings.FULL_OUTPUT:
        debug_files = _save_debug_masks(task_timestamp, trusted, reference_depth, projected_mask)
        debug_files['decisions_path'] = _save_decisions_debug(task_timestamp, {'frames': [d.to_dict() for d in decisions]})

    if confirmed is None or candidate_start is None:
        return _write_not_taken(json_path, task, tracking_window=tracking_window, projection=projection, init=init_stats, decisions=decisions, debug_files=debug_files, output_dir=output_dir)

    used_full_occlusion = full_occlusion_start is not None and full_occlusion_start.stamp.seconds <= candidate_start.stamp.seconds
    if used_full_occlusion:
        result_frame = full_occlusion_start
        backup_dir = _backup_sample(task, result_frame, output_dir)
        result_timestamp = result_frame.stamp.to_dict()
        payload = {
            'result_timestamp': result_timestamp,
            'backup_shigurei_dir': str(backup_dir),
            'tracking_window': tracking_window,
            'projection': projection,
            'init': init_stats,
            'depth_taken_timestamp': candidate_start.stamp.to_dict(),
            'depth_confirm_timestamp': confirmed.stamp.to_dict(),
            'full_occlusion_start_timestamp': full_occlusion_start.stamp.to_dict(),
            'used_full_occlusion_start': True,
            'rgb_backtrack': {'status': 'skipped_full_occlusion'},
            'checked_frame_count': len(decisions),
            'debug_files': debug_files,
            'output_dir': str(output_dir),
        }
        payload.update(_publish_taken_result_artifacts(task, backup_dir))
        _write_status(json_path, task, 'TAKEN', **payload)
        return {'status': 'TAKEN', 'result_timestamp': result_timestamp, 'backup_shigurei_dir': str(backup_dir)}

    return _write_taken(
        json_path,
        task,
        frames=frames,
        candidate_start=candidate_start,
        confirmed=confirmed,
        init_frame=init_end_frame,
        trusted=trusted,
        tracking_window=tracking_window,
        projection=projection,
        init_stats=init_stats,
        decisions=decisions,
        debug_files=debug_files,
        output_dir=output_dir,
    )


def run_taken_object_detection(json_path_arg: str | Path) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = str(task.get('task_id') or task.get('task_name') or json_path.stem)
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for taken object artifacts')
    output_dir = model_worker_dir(task_timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = ShigureRgbdCache(SHIGURE_HISTORY_CACHE_ROOT)
    capture_seconds, capture_source = _task_capture_time_seconds(task)
    if capture_seconds is None:
        _write_status(json_path, task, 'INIT_FAILED', reason='capture_time_missing', output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'capture_time_missing'}

    start = _stamp_from_seconds(capture_seconds)
    end = _stamp_from_seconds(capture_seconds + settings.TRACKING_DURATION_SECONDS)
    metadata_start = _stamp_from_seconds(capture_seconds - max(0.0, settings.YOLO_PRE_CAPTURE_LOOKBACK_SECONDS))
    metadata = list(cache.iter_sample_metadata(start=metadata_start, end=end))
    after_metadata = [sample for sample in metadata if sample.stamp.seconds >= capture_seconds]
    first_after = after_metadata[0] if after_metadata else None
    tracking_window = _tracking_window_payload(capture_seconds, capture_source, first_after, len(after_metadata))
    tracking_window['mode'] = settings.TRACKING_MODE
    tracking_window['metadata_frame_count'] = len(metadata)
    tracking_window['metadata_start_seconds'] = metadata_start.seconds

    if first_after is None:
        _write_status(json_path, task, 'INIT_FAILED', reason='no_shigure_frames_in_window', tracking_window=tracking_window, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'no_shigure_frames_in_window'}
    first_delay = first_after.stamp.seconds - capture_seconds
    if first_delay > settings.INIT_MAX_START_DELAY_SECONDS:
        _write_status(json_path, task, 'INIT_FAILED', reason='first_frame_too_late', tracking_window=tracking_window, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'first_frame_too_late'}

    if settings.TRACKING_MODE in {'legacy', 'legacy_projected_mask', 'projected_mask'}:
        if not settings.ENABLE_LEGACY_PROJECTED_MASK:
            _write_status(json_path, task, 'INIT_FAILED', reason='legacy_projected_mask_disabled', tracking_window=tracking_window, output_dir=str(output_dir))
            return {'status': 'INIT_FAILED', 'reason': 'legacy_projected_mask_disabled'}
        return _run_legacy_projected_mask_detection(json_path, task, cache, output_dir, capture_seconds=capture_seconds, capture_source=capture_source, start=start, end=end)

    if settings.TRACKING_MODE in {'model_box', 'model_box_depth', 'projected_model_box'}:
        yolo_init, yolo_info = _init_model_box_primary(cache, task, metadata, first_after, capture_seconds)
    else:
        yolo_init, yolo_info = _init_yolo_primary(cache, task, metadata, first_after, capture_seconds)
    if yolo_init is None:
        fallback_reason = {'mode': 'yolo_primary', 'reason': yolo_info.get('reason'), 'details': yolo_info}
        if settings.ENABLE_LEGACY_FALLBACK:
            return _run_legacy_projected_mask_detection(json_path, task, cache, output_dir, capture_seconds=capture_seconds, capture_source=capture_source, start=start, end=end, fallback_reason=fallback_reason)
        _write_status(json_path, task, 'INIT_FAILED', reason=yolo_info.get('reason'), tracking_window=tracking_window, projection=yolo_info.get('projection'), init=yolo_info, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': yolo_info.get('reason')}

    tracking_window['mode'] = yolo_init.init_stats.get('mode') or settings.TRACKING_MODE
    tracking_window['yolo_init_soft_timeout_seconds'] = settings.YOLO_INIT_SOFT_TIMEOUT_SECONDS
    tracking_window['yolo_init_hard_timeout_seconds'] = settings.YOLO_INIT_HARD_TIMEOUT_SECONDS
    tracking_window['tracking_trigger_mode'] = settings.YOLO_TRACKING_TRIGGER_MODE
    init_backup_dir = _backup_sample(task, yolo_init.init_frame, output_dir, kind='yolo_init')
    debug_projected_mask = yolo_init.projected_mask if yolo_init.projected_mask is not None else yolo_init.trusted_mask
    debug_files: dict[str, str] = _save_init_debug_files(
        task_timestamp,
        yolo_init.init_frame,
        yolo_init.trusted_mask,
        yolo_init.reference_depth,
        debug_projected_mask,
        yolo_init.projection,
        yolo_init.object_id,
        yolo_init.init_stats.get('selected_observation') if isinstance(yolo_init.init_stats, Mapping) else None,
    )
    history_baseline = _write_history_baseline_artifacts(
        init_backup_dir,
        yolo_init.init_frame,
        yolo_init.trusted_mask,
        yolo_init.reference_depth,
    )
    yolo_init_stats = {
        **yolo_init.init_stats,
        'init_backup_shigurei_dir': str(init_backup_dir),
        'history_baseline': history_baseline,
        'debug_files': debug_files,
    }
    _write_status(
        json_path,
        task,
        'RUNNING',
        tracking_window=tracking_window,
        projection=yolo_init.projection,
        init=yolo_init_stats,
        init_backup_shigurei_dir=str(init_backup_dir),
        history_baseline=history_baseline,
        debug_files=debug_files,
        output_dir=str(output_dir),
    )

    candidate_start, confirmed, decisions, yolo_tracking, frames = _run_yolo_primary_tracking(cache, yolo_init, end)
    if settings.FULL_OUTPUT:
        debug_files.update(_save_debug_masks(task_timestamp, yolo_init.trusted_mask, yolo_init.reference_depth, debug_projected_mask))
        debug_files['decisions_path'] = _save_decisions_debug(task_timestamp, {'frames': [d.to_dict() for d in decisions], 'yolo_tracking': yolo_tracking})

    if confirmed is None or candidate_start is None:
        return _write_not_taken(
            json_path,
            task,
            tracking_window=tracking_window,
            projection=yolo_init.projection,
            init=yolo_init_stats,
            decisions=decisions,
            debug_files=debug_files,
            output_dir=output_dir,
            extra={'yolo_tracking': yolo_tracking},
        )

    result_frames, candidate_full, confirmed_full, rgb_window = _rgb_window_for_taken_result(cache, candidate_start, confirmed, yolo_init.init_frame)
    yolo_tracking = {**yolo_tracking, 'rgb_result_window': rgb_window}
    return _write_taken(
        json_path,
        task,
        frames=result_frames,
        candidate_start=candidate_full,
        confirmed=confirmed_full,
        init_frame=yolo_init.init_frame,
        trusted=yolo_init.trusted_mask,
        tracking_window=tracking_window,
        projection=yolo_init.projection,
        init_stats=yolo_init_stats,
        decisions=decisions,
        debug_files=debug_files,
        output_dir=output_dir,
        extra={'yolo_tracking': yolo_tracking},
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('Usage: python run_taken_object_detection_from_json.py <task_meta.json>', file=sys.stderr)
        return 2
    try:
        result = run_taken_object_detection(argv[1])
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        print('[OK] taken_object_detection')
        return 0
    except Exception as exc:
        try:
            json_path = resolve_task_json_path(argv[1])
            task = load_task_json(json_path)
            _write_status(json_path, task, 'INIT_FAILED', reason='exception', error_message=str(exc))
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
