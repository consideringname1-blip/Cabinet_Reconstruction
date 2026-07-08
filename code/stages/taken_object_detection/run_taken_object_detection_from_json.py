from __future__ import annotations

import base64
import json
import math
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
from spatial_transforms import (
    camera_info_image_shape,
    camera_matrix_from_info,
    project_aruco_points_to_shigure_pixels,
)
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


def _parse_float_array(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f'{name} expected {size} values, got {array.size}')
    return array


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


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


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


def _model_bounds_corners_aruco(task: Mapping[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
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


def _circle_mask(image_shape: tuple[int, int], center_xy: tuple[float, float], radius_px: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image_shape
    cx, cy = center_xy
    radius = max(1.0, float(radius_px))
    x0 = int(max(0, math.floor(cx - radius)))
    y0 = int(max(0, math.floor(cy - radius)))
    x1 = int(min(w, math.ceil(cx + radius)))
    y1 = int(min(h, math.ceil(cy + radius)))
    image = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return np.asarray(image, dtype=np.uint8) > 0, (x0, y0, x1, y1)


def _project_model_diag_circle_to_shigure(task: Mapping[str, Any], camera_info: Mapping[str, Any] | None, image_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    camera_matrix = camera_matrix_from_info(camera_info)
    if camera_matrix is None:
        return None, {'source': 'model_diag_circle_projection', 'reason': 'camera_matrix_missing'}
    marker_pose = _load_marker_pose_cv()
    if marker_pose is None:
        return None, {'source': 'model_diag_circle_projection', 'reason': 'marker_pose_missing'}
    corners_aruco, corners_info = _model_bounds_corners_aruco(task)
    if corners_aruco is None:
        return None, {'source': 'model_diag_circle_projection', **corners_info}
    center_aruco, center_source = _object_center_aruco(task)
    if center_aruco is None:
        return None, {'source': 'model_diag_circle_projection', 'reason': center_source, 'corners': corners_info}

    marker_rotation, marker_translation, marker_path = marker_pose
    points_camera, pixels, visible = project_aruco_points_to_shigure_pixels(
        corners_aruco.reshape(-1, 3),
        marker_rotation,
        marker_translation,
        camera_matrix,
    )
    if not np.any(visible):
        return None, {
            'source': 'model_diag_circle_projection',
            'reason': 'projected_bounds_behind_camera',
            'camera_xyz_m': points_camera.astype(float).tolist(),
        }

    visible_points = points_camera[visible]
    visible_pixels = pixels[visible]
    h, w = image_shape

    center_camera_points, center_pixels, center_visible = project_aruco_points_to_shigure_pixels(
        center_aruco.reshape(1, 3),
        marker_rotation,
        marker_translation,
        camera_matrix,
    )
    center_camera = center_camera_points.reshape(3)
    center_z = float(center_camera[2])
    if not bool(center_visible[0]) or not np.isfinite(center_pixels[0]).all():
        return None, {
            'source': 'model_diag_circle_projection',
            'reason': 'projected_center_behind_camera',
            'camera_xyz_m': center_camera.astype(float).tolist(),
        }
    center_pixel = (float(center_pixels[0, 0]), float(center_pixels[0, 1]))
    if center_pixel[0] < 0 or center_pixel[1] < 0 or center_pixel[0] >= w or center_pixel[1] >= h:
        return None, {
            'source': 'model_diag_circle_projection',
            'reason': 'projected_center_outside_image',
            'center_pixel_xy': [float(center_pixel[0]), float(center_pixel[1])],
            'image_shape': [h, w],
        }
    if not np.isfinite(center_z) or center_z <= 1.0e-6:
        return None, {
            'source': 'model_diag_circle_projection',
            'reason': 'projected_center_depth_invalid',
            'center_depth_m': center_z,
        }

    min_corner = np.nanmin(corners_aruco, axis=0)
    max_corner = np.nanmax(corners_aruco, axis=0)
    box_diag_m = float(np.linalg.norm(max_corner - min_corner))
    if not np.isfinite(box_diag_m) or box_diag_m <= 0.0:
        return None, {'source': 'model_diag_circle_projection', 'reason': 'model_bounds_diagonal_invalid'}
    radius_m = box_diag_m * 0.5
    corner_distances = np.linalg.norm(visible_pixels - np.asarray(center_pixel, dtype=np.float64).reshape(1, 2), axis=1)
    finite_corner_distances = corner_distances[np.isfinite(corner_distances)]
    if finite_corner_distances.size:
        radius_px = float(np.nanmax(finite_corner_distances))
        radius_source = 'projected_bounds_corners'
    else:
        focal = max(abs(float(camera_matrix[0, 0])), abs(float(camera_matrix[1, 1])))
        radius_px = float(focal * radius_m / center_z)
        radius_source = 'focal_length_radius_approximation'
    circle_mask, circle_bbox = _circle_mask(image_shape, center_pixel, radius_px)
    projected_pixels = int(np.count_nonzero(circle_mask))
    if projected_pixels <= 0:
        return None, {
            'source': 'model_diag_circle_projection',
            'reason': 'projected_diag_circle_outside_image',
            'center_pixel_xy': [float(center_pixel[0]), float(center_pixel[1])],
            'radius_px': radius_px,
            'image_shape': [h, w],
        }
    z_values = visible_points[:, 2]
    info = {
        'source': 'model_diag_circle_projection',
        'reason': 'projected_diag_circle_ready',
        'corners': corners_info,
        'center_source': center_source,
        'marker_pose_path': str(marker_path),
        'object_center_aruco': center_aruco.astype(float).tolist(),
        'center_pixel_xy': [float(center_pixel[0]), float(center_pixel[1])],
        'center_depth_m': center_z if np.isfinite(center_z) else None,
        'box_depth_min_m': float(np.nanmin(z_values)),
        'box_depth_max_m': float(np.nanmax(z_values)),
        'box_diag_m': box_diag_m,
        'circle_radius_m': radius_m,
        'circle_radius_px': radius_px,
        'circle_radius_source': radius_source,
        'bbox_xyxy': [int(v) for v in circle_bbox],
        'camera_matrix': camera_matrix.astype(float).tolist(),
        'image_shape': [h, w],
        'projected_pixels': projected_pixels,
    }
    return circle_mask, info


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


def _score_model_diag_circle_candidate(obs: YoloObjectObservation, sample: CachedRgbdSample, diag_circle_mask: np.ndarray, projection: Mapping[str, Any]) -> dict[str, Any]:
    center_depth = projection.get('center_depth_m')
    depth_diff = None
    if obs.median_depth_m is not None and center_depth is not None and np.isfinite(float(center_depth)):
        depth_diff = abs(float(obs.median_depth_m) - float(center_depth))
    overlap_pixels = int(np.count_nonzero(obs.mask & diag_circle_mask))
    inside_ratio = overlap_pixels / max(1, int(obs.mask_pixels))
    reject_reasons: list[str] = []
    if inside_ratio < settings.MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO:
        reject_reasons.append('mask_not_enough_inside_model_diag_circle')
    if depth_diff is None:
        reject_reasons.append('depth_unavailable')
    elif depth_diff > settings.MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M:
        reject_reasons.append('depth_too_different_from_model_center')
    return {
        'accepted': not reject_reasons,
        'reject_reasons': reject_reasons,
        'depth_diff_m': depth_diff,
        'max_depth_diff_m': settings.MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M,
        'mask_inside_diag_circle_pixels': overlap_pixels,
        'mask_inside_diag_circle_ratio': inside_ratio,
        'min_mask_inside_diag_circle_ratio': settings.MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO,
        'rgb_signature': _mask_rgb_stats(sample, obs.mask),
        'observation': obs,
    }


def _candidate_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in item.items() if k != 'observation'}
    obs = item.get('observation')
    if isinstance(obs, YoloObjectObservation):
        payload['observation'] = obs.to_dict()
    return payload


def _select_model_diag_circle_yolo_candidate(cache: ShigureRgbdCache, event: YoloEvent, sample: CachedRgbdSample, shape: tuple[int, int], diag_circle_mask: np.ndarray, projection: Mapping[str, Any]) -> tuple[YoloObjectObservation | None, dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for obj in event.payload.get('objects') or []:
        if not isinstance(obj, Mapping):
            continue
        obs = _observation_from_object(cache, event, obj, shape)
        if obs is None:
            continue
        scored.append(_score_model_diag_circle_candidate(obs, sample, diag_circle_mask, projection))
    if not scored:
        return None, {'reason': 'no_yolo_object_with_mask'}
    accepted = [item for item in scored if item.get('accepted')]
    if accepted:
        accepted.sort(
            key=lambda item: (
                -int(item['observation'].mask_pixels),
                -float(item.get('mask_inside_diag_circle_ratio') or 0.0),
                float(item.get('depth_diff_m') if item.get('depth_diff_m') is not None else math.inf),
            )
        )
    scored.sort(
        key=lambda item: (
            not bool(item.get('accepted')),
            -int(item['observation'].mask_pixels),
            -float(item.get('mask_inside_diag_circle_ratio') or 0.0),
            float(item.get('depth_diff_m') if item.get('depth_diff_m') is not None else math.inf),
        )
    )
    best = accepted[0] if accepted else scored[0]
    reason = 'matched_model_diag_circle_shigure_mask' if accepted else 'best_model_diag_circle_candidate_rejected'
    return (best['observation'] if accepted else None), {
        'reason': reason,
        'candidate_scope': 'shigure_object_masks_inside_projected_model_diag_circle_and_depth_near_model_center',
        'selection_policy': 'largest_accepted_mask_after_diag_circle_ratio_and_depth_filter',
        'candidate_count': len(scored),
        'accepted_count': len(accepted),
        'best': _candidate_payload(best),
        'top_candidates': [_candidate_payload(item) for item in scored[:8]],
    }


def _init_model_diag_circle_primary(cache: ShigureRgbdCache, task: Mapping[str, Any], metadata: list[CachedSampleMetadata], first_after: CachedSampleMetadata, capture_seconds: float) -> tuple[YoloInitResult | None, dict[str, Any]]:
    first_sample = cache.get_sample(first_after.stamp, mode='nearest')
    if first_sample is None:
        return None, {'mode': 'model_diag_circle_yolo_mask_init', 'reason': 'first_frame_unavailable'}
    shape = camera_info_image_shape(first_after.camera_info or first_sample.camera_info) or first_sample.depth.shape[:2]
    diag_circle_mask, projection = _project_model_diag_circle_to_shigure(task, first_after.camera_info or first_sample.camera_info, shape)
    init_info: dict[str, Any] = {
        'mode': 'model_diag_circle_yolo_mask_init',
        'projection': projection,
        'metadata_frame_count': len(metadata),
        'reference_source': 'selected_shigure_yolo_mask',
        'candidate_scope': 'shigure_object_masks_inside_projected_model_diag_circle_and_depth_near_model_center',
    }
    if diag_circle_mask is None:
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
        record: dict[str, Any] = {'stamp': sample.stamp.to_dict()}
        selected_obs, candidate_info = _select_model_diag_circle_yolo_candidate(cache, event, sample, shape, diag_circle_mask, projection)
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
            projected_mask=diag_circle_mask,
        ), init_stats
    reason = 'no_post_capture_yolo_events' if not post_events else 'no_model_diag_circle_candidate_mask'
    if checked:
        last = checked[-1]
        reason = str((last.get('candidate_match') or {}).get('reason') or reason)
    return None, {**init_info, 'reason': reason, 'checked_init_frames': checked[:25]}

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
    projected_path = debug_dir / '07_taken_init_projected_diag_circle_mask.png'
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
    _draw_debug_bbox(draw, projection.get('bbox_xyxy'), 'projected model diag circle', (255, 210, 0), width=3)
    if isinstance(selected_observation, Mapping):
        _draw_debug_bbox(draw, selected_observation.get('bbox_xyxy'), f'shigure mask id {object_id}', (0, 190, 255), width=3)
    image.save(overlay_path)

    return {
        'init_projected_diag_circle_mask_path': str(projected_path),
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
    result_frame = candidate_start
    rgb_backtrack = {
        'status': 'disabled_use_depth_taken_frame',
        'reason': 'taken_subject_crop_requires_event_frame_not_quiet_pre_take_frame',
        'selected_stamp': result_frame.stamp.to_dict(),
        'depth_taken_timestamp': candidate_start.stamp.to_dict(),
        'depth_confirm_timestamp': confirmed.stamp.to_dict(),
        'init_timestamp': init_frame.stamp.to_dict(),
        'window_frame_count': len(frames),
    }
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

    if settings.TRACKING_MODE != 'model_diag_circle':
        _write_status(
            json_path,
            task,
            'INIT_FAILED',
            reason='unsupported_tracking_mode',
            tracking_window=tracking_window,
            init={'requested_tracking_mode': settings.TRACKING_MODE, 'supported_modes': ['model_diag_circle']},
            output_dir=str(output_dir),
        )
        return {'status': 'INIT_FAILED', 'reason': 'unsupported_tracking_mode'}

    yolo_init, yolo_info = _init_model_diag_circle_primary(cache, task, metadata, first_after, capture_seconds)
    if yolo_init is None:
        _write_status(json_path, task, 'INIT_FAILED', reason=yolo_info.get('reason'), tracking_window=tracking_window, projection=yolo_info.get('projection'), init=yolo_info, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': yolo_info.get('reason')}

    tracking_window['mode'] = yolo_init.init_stats.get('mode') or settings.TRACKING_MODE
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
