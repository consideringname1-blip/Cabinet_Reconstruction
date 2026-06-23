from __future__ import annotations

import base64
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import SHIGURE_HISTORY_CACHE_ROOT, TAKEN_OBJECT_OUTPUT_ROOT, UPLOAD_FOLDER
from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS, quat_xyzw_to_rotation_matrix
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


def _write_status(json_path: Path, task: dict[str, Any], status: str, **fields: Any) -> None:
    payload = dict(task.get('TakenObjectDetection') or {})
    payload.update(fields)
    payload['status'] = status
    payload['updated_at'] = _utc_now()
    task['TakenObjectDetection'] = _jsonable(payload)
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


def _resolve_upload_file(name: str | None) -> Path | None:
    if not name:
        return None
    path = Path(str(name))
    if path.is_file():
        return path
    candidate = UPLOAD_FOLDER / path.name
    return candidate if candidate.is_file() else None


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


def _camera_matrix_from_info(camera_info: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(camera_info, Mapping):
        return None
    raw = camera_info.get('k') or camera_info.get('K') or camera_info.get('camera_matrix')
    if raw is None:
        return None
    try:
        matrix = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if not np.isfinite(matrix).all() or matrix[0, 0] == 0 or matrix[1, 1] == 0:
        return None
    return matrix


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
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    center_marker_cv = basis @ center_aruco.reshape(3)
    center_camera = marker_rotation @ center_marker_cv + marker_translation.reshape(3)
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
        'object_center_marker_opencv': center_marker_cv.tolist(),
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


def _init_yolo_primary(cache: ShigureRgbdCache, task: Mapping[str, Any], metadata: list[CachedSampleMetadata], first_after: CachedSampleMetadata, capture_seconds: float) -> tuple[YoloInitResult | None, dict[str, Any]]:
    shape = None
    if first_after.camera_info:
        h = int(first_after.camera_info.get('height') or 0)
        w = int(first_after.camera_info.get('width') or 0)
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
    frames = list(cache.iter_samples(start=scan_start, end=end))
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
    old = task.get('ModelEventTracking') if isinstance(task.get('ModelEventTracking'), Mapping) else {}
    if old.get('current_support_mask_path'):
        candidates.append(('legacy_support_mask', Path(str(old.get('current_support_mask_path')))))
    sam3 = task.get('sam3Name') if isinstance(task.get('sam3Name'), Mapping) else {}
    if sam3.get('mask'):
        resolved = _resolve_upload_file(str(sam3.get('mask')))
        if resolved:
            candidates.append(('sam3_mask_resized_fallback', resolved))

    for source, raw_path in candidates:
        path = raw_path if raw_path.is_file() else _resolve_upload_file(str(raw_path))
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


def _save_debug_masks(output_dir: Path, trusted: np.ndarray, reference_depth: np.ndarray, projected: np.ndarray) -> dict[str, str]:
    debug_dir = output_dir / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    projected_path = debug_dir / 'projected_mask.png'
    trusted_path = debug_dir / 'trusted_mask.png'
    reference_path = debug_dir / 'reference_depth_m.npy'
    Image.fromarray(projected.astype(np.uint8) * 255).save(projected_path)
    Image.fromarray(trusted.astype(np.uint8) * 255).save(trusted_path)
    np.save(reference_path, reference_depth.astype(np.float32))
    return {
        'projected_mask_path': str(projected_path),
        'trusted_mask_path': str(trusted_path),
        'reference_depth_m_path': str(reference_path),
    }


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
    _write_status(json_path, task, 'TAKEN', **payload)
    return {'status': 'TAKEN', 'result_timestamp': result_timestamp, 'backup_shigurei_dir': str(backup_dir)}


def _run_legacy_projected_mask_detection(json_path: Path, task: dict[str, Any], cache: ShigureRgbdCache, output_dir: Path, *, capture_seconds: float, capture_source: str | None, start: RosStamp, end: RosStamp, fallback_reason: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
        debug_files = _save_debug_masks(output_dir, trusted, reference_depth, projected_mask)
        _write_json(output_dir / 'decisions.json', {'frames': [d.to_dict() for d in decisions]})

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
    output_dir = TAKEN_OBJECT_OUTPUT_ROOT / task_id
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

    yolo_init, yolo_info = _init_yolo_primary(cache, task, metadata, first_after, capture_seconds)
    if yolo_init is None:
        fallback_reason = {'mode': 'yolo_primary', 'reason': yolo_info.get('reason'), 'details': yolo_info}
        if settings.ENABLE_LEGACY_FALLBACK:
            return _run_legacy_projected_mask_detection(json_path, task, cache, output_dir, capture_seconds=capture_seconds, capture_source=capture_source, start=start, end=end, fallback_reason=fallback_reason)
        _write_status(json_path, task, 'INIT_FAILED', reason=yolo_info.get('reason'), tracking_window=tracking_window, projection=yolo_info.get('projection'), init=yolo_info, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': yolo_info.get('reason')}

    tracking_window['mode'] = 'yolo_primary'
    tracking_window['yolo_init_soft_timeout_seconds'] = settings.YOLO_INIT_SOFT_TIMEOUT_SECONDS
    tracking_window['yolo_init_hard_timeout_seconds'] = settings.YOLO_INIT_HARD_TIMEOUT_SECONDS
    init_backup_dir = _backup_sample(task, yolo_init.init_frame, output_dir, kind='yolo_init')
    yolo_init_stats = {
        **yolo_init.init_stats,
        'init_backup_shigurei_dir': str(init_backup_dir),
    }
    _write_status(
        json_path,
        task,
        'RUNNING',
        tracking_window=tracking_window,
        projection=yolo_init.projection,
        init=yolo_init_stats,
        init_backup_shigurei_dir=str(init_backup_dir),
        output_dir=str(output_dir),
    )

    candidate_start, confirmed, decisions, yolo_tracking, frames = _run_yolo_primary_tracking(cache, yolo_init, end)
    debug_files: dict[str, str] = {}
    if settings.FULL_OUTPUT:
        debug_files = _save_debug_masks(output_dir, yolo_init.trusted_mask, yolo_init.reference_depth, yolo_init.trusted_mask)
        _write_json(output_dir / 'decisions.json', {'frames': [d.to_dict() for d in decisions], 'yolo_tracking': yolo_tracking})

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

    return _write_taken(
        json_path,
        task,
        frames=frames,
        candidate_start=candidate_start,
        confirmed=confirmed,
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
