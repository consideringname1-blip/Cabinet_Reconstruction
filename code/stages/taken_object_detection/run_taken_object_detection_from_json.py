from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
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
from stages.shigure_history.cache import CachedRgbdSample, RosStamp, ShigureRgbdCache, sample_key
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


def _backup_sample(task: Mapping[str, Any], sample: CachedRgbdSample, output_root: Path) -> Path:
    task_name = str(task.get('task_name') or task.get('task_id') or 'task').strip()
    key = sample_key(sample.stamp)
    backup_dir = output_root / f'{task_name}_{key}'
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
    elif sample.yolo_path and sample.yolo_path.is_file():
        shutil.copy2(sample.yolo_path, backup_dir / 'active_objects.json')
    marker_pose = _find_marker_pose_path()
    if marker_pose is not None:
        shutil.copy2(marker_pose, backup_dir / 'marker_6d_pose.json')
    _write_json(
        backup_dir / 'meta.json',
        {
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
    frames = list(cache.iter_samples(start=start, end=end))
    tracking_window = {
        'capture_time_source': capture_source,
        'capture_time_seconds': capture_seconds,
        'tracking_start_seconds': capture_seconds,
        'tracking_end_seconds': capture_seconds + settings.TRACKING_DURATION_SECONDS,
        'input_frame_count': len(frames),
    }
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
    if projected_mask is None:
        _write_status(json_path, task, 'INIT_FAILED', reason='projected_mask_missing', tracking_window=tracking_window, projection=projection, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': 'projected_mask_missing'}

    init_deadline = frames[0].stamp.seconds + min(settings.INIT_TIMEOUT_SECONDS, settings.INIT_STABLE_WINDOW_SECONDS)
    init_frames = [frame for frame in frames if frame.stamp.seconds <= init_deadline]
    trusted, reference_depth, init_end_frame, init_stats = _init_trusted_mask(init_frames, projected_mask)
    if trusted is None or reference_depth is None or init_end_frame is None:
        _write_status(json_path, task, 'INIT_FAILED', reason=init_stats.get('reason'), tracking_window=tracking_window, projection=projection, init=init_stats, output_dir=str(output_dir))
        return {'status': 'INIT_FAILED', 'reason': init_stats.get('reason')}

    _write_status(
        json_path,
        task,
        'RUNNING',
        tracking_window=tracking_window,
        projection=projection,
        init=init_stats,
        output_dir=str(output_dir),
    )

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
        _write_status(
            json_path,
            task,
            'NOT_TAKEN',
            result_timestamp=None,
            backup_shigurei_dir=None,
            tracking_window=tracking_window,
            projection=projection,
            init=init_stats,
            checked_frame_count=len(decisions),
            debug_files=debug_files,
            output_dir=str(output_dir),
        )
        return {'status': 'NOT_TAKEN', 'checked_frame_count': len(decisions)}

    used_full_occlusion = full_occlusion_start is not None and full_occlusion_start.stamp.seconds <= candidate_start.stamp.seconds
    result_frame = full_occlusion_start if used_full_occlusion else candidate_start
    assert result_frame is not None
    rgb_backtrack = {'status': 'skipped_full_occlusion'}
    if not used_full_occlusion:
        result_frame, rgb_backtrack = _backtrack_rgb_frame(frames, candidate_start, init_end_frame, trusted)
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
        'full_occlusion_start_timestamp': full_occlusion_start.stamp.to_dict() if full_occlusion_start else None,
        'used_full_occlusion_start': used_full_occlusion,
        'rgb_backtrack': rgb_backtrack,
        'checked_frame_count': len(decisions),
        'debug_files': debug_files,
        'output_dir': str(output_dir),
    }
    _write_status(json_path, task, 'TAKEN', **payload)
    return {'status': 'TAKEN', 'result_timestamp': result_timestamp, 'backup_shigurei_dir': str(backup_dir)}


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
