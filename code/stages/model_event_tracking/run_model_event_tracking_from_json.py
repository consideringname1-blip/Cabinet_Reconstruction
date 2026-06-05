from __future__ import annotations

import fcntl
import json
import os
import socket
import shutil
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from config import MODEL_EVENT_OUTPUT_ROOT
from task_json import load_task_json, resolve_task_json_path, save_task_json

from stages.model_event_tracking.cache import ShigureHistoryCache
from stages.model_event_tracking import settings
from stages.model_event_tracking.geometry import load_camera_matrix, project_model_bounds_to_shigurei
from stages.model_event_tracking.output_paths import (
    task_output_dir as model_event_task_output_dir,
    task_output_name,
)
from stages.model_event_tracking.schemas import ShigureFrame, to_jsonable
from stages.model_event_tracking.tracker import read_depth_image_m


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_id(task: dict[str, Any]) -> str:
    value = str(task.get("task_id") or "").strip()
    if not value:
        raise ValueError("task_id is missing")
    return value


def _exclusive_tracking_run(function: Any) -> Any:
    @wraps(function)
    def wrapped(json_path_arg: str | Path) -> dict[str, Any]:
        json_path = resolve_task_json_path(json_path_arg)
        task = load_task_json(json_path)
        task_id = _task_id(task)
        output_name = task_output_name(json_path, fallback=task_id)
        output_dir = model_event_task_output_dir(
            MODEL_EVENT_OUTPUT_ROOT,
            task_id=task_id,
            json_path=json_path,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = output_dir / ".tracking.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                _write_status(
                    json_path,
                    task,
                    "already_running",
                    reason="model event tracking is already running for this task",
                    task_output_name=output_name,
                    task_output_dir=str(output_dir),
                )
                return {
                    "status": "already_running",
                    "task_id": task_id,
                    "task_output_name": output_name,
                }
            try:
                return function(json_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return wrapped


def _write_status(json_path: Path, task: dict[str, Any], status: str, **fields: Any) -> None:
    payload = dict(task.get("ModelEventTracking") or {})
    if status != "failed" and "error_message" not in fields:
        payload.pop("error_message", None)
    if "reason" not in fields:
        payload.pop("reason", None)
    payload.update(fields)
    payload["status"] = status
    payload["updated_at"] = utc_now()
    task["ModelEventTracking"] = to_jsonable(payload)
    save_task_json(json_path, task)


def _model_bounds_ready(task: dict[str, Any]) -> bool:
    bounds = task.get("ModelBounds") or {}
    return isinstance(bounds, dict) and bounds.get("status") == "ready" and bool(bounds.get("corners_aruco"))


def _marker_search_roots() -> list[Path]:
    configured = str(os.environ.get("MODEL_EVENT_MARKER_POSE_SEARCH_ROOTS") or "").strip()
    if configured:
        return [Path(value) for value in configured.split(os.pathsep) if value.strip()]
    return [
        CODE_ROOT / ".test" / "marker",
        CODE_ROOT.parent / ".test" / "fusion_runs",
    ]


def _latest_historical_marker_pose() -> Path | None:
    candidates: list[Path] = []
    for root in _marker_search_roots():
        if root.is_file() and root.name == "marker_6d_pose.json":
            candidates.append(root)
            continue
        if root.is_dir():
            candidates.extend(path for path in root.rglob("marker_6d_pose.json") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _select_marker_pose(frames: list[ShigureFrame]) -> tuple[Path | None, str | None]:
    env_path = str(os.environ.get("MODEL_EVENT_MARKER_POSE_JSON") or "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path, "env:MODEL_EVENT_MARKER_POSE_JSON"
    for frame in frames:
        if frame.marker_pose_path and frame.marker_pose_path.is_file():
            return frame.marker_pose_path, "cached_frame"
    marker_pose = _latest_historical_marker_pose()
    if marker_pose is not None:
        return marker_pose, "historical_marker_pose"
    return None, None


def _first_camera_info(frames: list[ShigureFrame]) -> Path | None:
    for frame in frames:
        if frame.camera_info_path and frame.camera_info_path.is_file():
            return frame.camera_info_path
    return None


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_projection_debug_image(
    frame: ShigureFrame,
    projected_box: Any,
    output_dir: Path,
) -> Path | None:
    if frame.rgb_path is None or not frame.rgb_path.is_file():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "projection_start.jpg"
    with Image.open(frame.rgb_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = [float(value) for value in projected_box.bbox_xyxy]
    draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=4)
    points = [(float(x), float(y)) for x, y in np.asarray(projected_box.pixel_points, dtype=np.float64)]
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for a, b in edges:
        if a < len(points) and b < len(points):
            draw.line((points[a], points[b]), fill=(64, 220, 255), width=3)
    draw.ellipse((x0 - 4, y0 - 4, x0 + 4, y0 + 4), fill=(255, 255, 0))
    canvas.save(target, quality=95)
    return target


def _parse_iso_timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        tz_pos = min([pos for pos in (tail.find("+"), tail.find("-")) if pos >= 0], default=-1)
        if tz_pos >= 0:
            fraction, tz = tail[:tz_pos], tail[tz_pos:]
        else:
            fraction, tz = tail, ""
        text = f"{head}.{fraction[:6]}{tz}"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
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
    pv = task.get("PVCamera") if isinstance(task.get("PVCamera"), dict) else {}
    seconds = _parse_iso_timestamp_seconds(pv.get("time"))
    if seconds is not None:
        return seconds, "PVCamera.time"
    device = task.get("device") if isinstance(task.get("device"), dict) else {}
    seconds = _parse_iso_timestamp_seconds(device.get("time"))
    if seconds is not None:
        return seconds, "device.time"
    seconds = _parse_iso_timestamp_seconds(task.get("server_received_utc"))
    if seconds is not None:
        return seconds, "server_received_utc"
    return None, None


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _is_enabled_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _select_tracking_frames(task: dict[str, Any], frames: list[ShigureFrame]) -> tuple[list[ShigureFrame], dict[str, Any]]:
    metadata: dict[str, Any] = {"input_frame_count": len(frames)}
    selected = list(frames)
    capture_seconds, capture_source = _task_capture_time_seconds(task)
    settle_seconds = max(0.0, _float_env("MODEL_EVENT_CAPTURE_SETTLE_SECONDS", 2.0))
    metadata["capture_time_source"] = capture_source
    metadata["capture_time_seconds"] = capture_seconds
    metadata["capture_settle_seconds"] = settle_seconds

    if _is_enabled_env("MODEL_EVENT_START_AFTER_CAPTURE", True) and capture_seconds is not None:
        after_capture = [frame for frame in selected if frame.stamp.seconds >= capture_seconds]
        after_settle = [frame for frame in after_capture if frame.stamp.seconds >= capture_seconds + settle_seconds]
        if len(after_settle) >= 2:
            selected = after_settle
            metadata["window_start_reason"] = "capture_time_plus_settle"
        else:
            selected = after_capture
            metadata["window_start_reason"] = "capture_time_without_full_settle"
        metadata["post_capture_frame_count"] = len(after_capture)
        metadata["post_settle_frame_count"] = len(after_settle)

    max_post_capture_seconds = max(0.0, _float_env("MODEL_EVENT_MAX_POST_CAPTURE_SECONDS", 60.0))
    if max_post_capture_seconds > 0.0 and selected:
        window_origin_seconds = capture_seconds if capture_seconds is not None else selected[0].stamp.seconds
        selected = [
            frame
            for frame in selected
            if frame.stamp.seconds <= window_origin_seconds + max_post_capture_seconds
        ]
        metadata["max_post_capture_seconds"] = max_post_capture_seconds

    selected = _limited_frames(selected)
    metadata["selected_frame_count"] = len(selected)
    if selected:
        metadata["selected_start_stamp"] = selected[0].stamp.to_dict()
        metadata["selected_end_stamp"] = selected[-1].stamp.to_dict()
    return selected, metadata


def _sample_frames_by_interval(frames: list[ShigureFrame], sample_hz: float) -> list[ShigureFrame]:
    if not frames:
        return []
    interval = 1.0 / max(0.01, float(sample_hz))
    sampled = [frames[0]]
    last_seconds = frames[0].stamp.seconds
    for frame in frames[1:]:
        if frame.stamp.seconds - last_seconds + 1.0e-6 < interval:
            continue
        sampled.append(frame)
        last_seconds = frame.stamp.seconds
    return sampled


def _box_depth_profile(frame: ShigureFrame, projected_box: Any) -> dict[str, Any]:
    if frame.depth_path is None or not frame.depth_path.is_file():
        return {"usable": False, "reason": "depth_missing"}
    depth_m = read_depth_image_m(frame.depth_path)
    height, width = depth_m.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in projected_box.bbox_xyxy]
    ix0 = min(max(int(np.floor(x0)), 0), max(0, width - 1))
    iy0 = min(max(int(np.floor(y0)), 0), max(0, height - 1))
    ix1 = min(max(int(np.ceil(x1)), ix0 + 1), width)
    iy1 = min(max(int(np.ceil(y1)), iy0 + 1), height)
    crop = np.asarray(depth_m[iy0:iy1, ix0:ix1], dtype=np.float64)
    valid_mask = np.isfinite(crop) & (crop > 0.0)
    valid = crop[valid_mask]
    if valid.size == 0:
        return {
            "usable": False,
            "reason": "no_valid_depth_in_box",
            "bbox_xyxy": [ix0, iy0, ix1, iy1],
        }

    corner_depths = np.asarray(projected_box.corners_camera_m, dtype=np.float64)[:, 2]
    front_depth_m = float(np.nanmin(corner_depths))
    back_depth_m = float(np.nanmax(corner_depths))
    foreground_limit_m = front_depth_m - settings.OCCLUSION_FRONT_MARGIN_M
    model_min_m = front_depth_m - settings.OCCLUSION_FRONT_MARGIN_M
    model_max_m = back_depth_m + settings.OCCLUSION_MODEL_DEPTH_BAND_M
    foreground_mask = valid_mask & (crop < foreground_limit_m)
    foreground_ratio = float(np.count_nonzero(foreground_mask)) / float(valid.size)
    model_depth_ratio = float(np.count_nonzero((valid >= model_min_m) & (valid <= model_max_m))) / float(valid.size)
    return {
        "usable": True,
        "bbox_xyxy": [ix0, iy0, ix1, iy1],
        "valid_depth_pixels": int(valid.size),
        "median_depth_m": float(np.nanmedian(valid)),
        "front_depth_m": front_depth_m,
        "back_depth_m": back_depth_m,
        "foreground_ratio": foreground_ratio,
        "foreground_present": foreground_ratio >= settings.OCCLUSION_FOREGROUND_PRESENCE_RATIO,
        "model_depth_ratio": model_depth_ratio,
        "heavily_occluded": foreground_ratio >= settings.OCCLUSION_FOREGROUND_RATIO,
        "_depth_crop": crop,
        "_valid_mask": valid_mask,
        "_foreground_mask": foreground_mask,
    }


def _public_depth_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if not key.startswith("_")}


def _foreground_change_ratio(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    if not previous.get("usable") or not current.get("usable"):
        return None
    previous_foreground = np.asarray(previous["_foreground_mask"], dtype=bool)
    current_foreground = np.asarray(current["_foreground_mask"], dtype=bool)
    if previous_foreground.shape != current_foreground.shape:
        return None
    union = previous_foreground | current_foreground
    union_pixels = int(np.count_nonzero(union))
    if union_pixels == 0:
        return 0.0

    occupancy_changed = previous_foreground ^ current_foreground
    both_foreground = previous_foreground & current_foreground
    previous_depth = np.asarray(previous["_depth_crop"], dtype=np.float64)
    current_depth = np.asarray(current["_depth_crop"], dtype=np.float64)
    depth_changed = (
        both_foreground
        & np.isfinite(previous_depth)
        & np.isfinite(current_depth)
        & (np.abs(current_depth - previous_depth) >= settings.OCCLUSION_FOREGROUND_DEPTH_CHANGE_M)
    )
    return float(np.count_nonzero((occupancy_changed | depth_changed) & union)) / float(union_pixels)


def _prefilter_tracking_frames(
    frames: list[ShigureFrame],
    projected_box: Any,
) -> tuple[list[ShigureFrame], list[dict[str, Any]], dict[str, Any]]:
    sampled = _sample_frames_by_interval(frames, settings.MASK_ATTEMPT_HZ)
    profiles = [_box_depth_profile(frame, projected_box) for frame in sampled]
    reference_index = None
    foreground_stable_frames = 0
    previous_usable: dict[str, Any] | None = None

    for index, profile in enumerate(profiles):
        if not profile.get("usable"):
            foreground_stable_frames = 0
            previous_usable = None
            continue

        change_ratio = (
            _foreground_change_ratio(previous_usable, profile)
            if previous_usable is not None
            else None
        )
        profile["foreground_change_ratio"] = change_ratio
        foreground_present = bool(profile.get("foreground_present"))
        if not foreground_present:
            foreground_stable_frames = 0
        elif change_ratio is not None and change_ratio <= settings.OCCLUSION_FOREGROUND_CHANGE_RATIO:
            foreground_stable_frames += 1
        else:
            foreground_stable_frames = 0
        profile["foreground_stable_frames"] = foreground_stable_frames

        model_supported = (
            float(profile.get("model_depth_ratio") or 0.0)
            >= settings.REFERENCE_MODEL_DEPTH_RATIO
        )
        foreground_ready = (
            not foreground_present
            or foreground_stable_frames >= max(1, settings.OCCLUSION_STABLE_FRAMES)
        )
        profile["reference_model_supported"] = model_supported
        profile["reference_foreground_ready"] = foreground_ready
        profile["reference_eligible"] = model_supported and foreground_ready
        if profile["reference_eligible"]:
            reference_index = index
            break
        previous_usable = profile

    public_profiles = [_public_depth_profile(profile) for profile in profiles]
    metadata: dict[str, Any] = {
        "input_frame_count": len(frames),
        "sample_hz": settings.MASK_ATTEMPT_HZ,
        "sampled_frame_count": len(sampled),
        "foreground_presence_ratio": settings.OCCLUSION_FOREGROUND_PRESENCE_RATIO,
        "foreground_change_ratio": settings.OCCLUSION_FOREGROUND_CHANGE_RATIO,
        "foreground_stable_frames_required": settings.OCCLUSION_STABLE_FRAMES,
        "heavily_occluded_frame_count": sum(bool(item.get("heavily_occluded")) for item in profiles),
        "foreground_present_frame_count": sum(bool(item.get("foreground_present")) for item in profiles),
        "unusable_depth_frame_count": sum(not bool(item.get("usable")) for item in profiles),
        "reference_index": reference_index,
        "reference_scan": public_profiles[: reference_index + 1 if reference_index is not None else None],
    }
    if reference_index is None:
        return [], [], metadata

    # Occlusion only delays initialization. Once SAM3 starts, every sampled
    # frame remains in chronological order so a hand crossing the box and the
    # subsequent object motion are not removed from the video.
    selected_frames = sampled[reference_index:]
    selected_profiles = public_profiles[reference_index:]
    metadata["selected_frame_count"] = len(selected_frames)
    metadata["skipped_after_reference"] = 0
    if selected_frames:
        metadata["selected_start_stamp"] = selected_frames[0].stamp.to_dict()
        metadata["selected_end_stamp"] = selected_frames[-1].stamp.to_dict()
        metadata["reference_depth_profile"] = selected_profiles[0]
    return selected_frames, selected_profiles, metadata


def _prepare_video_output_dir(task_output_dir: Path) -> tuple[Path, int]:
    runs_root = task_output_dir / "sam3_video_runs"
    removed_run_count = 0
    if runs_root.is_dir():
        removed_run_count = sum(1 for path in runs_root.iterdir() if path.is_dir())
        shutil.rmtree(runs_root)
    video_output_dir = runs_root / "current"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    return video_output_dir, removed_run_count


def _frame_request_payload(frame: ShigureFrame, depth_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "stamp": frame.stamp.to_dict(),
        "rgb_path": str(frame.rgb_path) if frame.rgb_path is not None else None,
        "depth_path": str(frame.depth_path) if frame.depth_path is not None else None,
        "camera_info_path": str(frame.camera_info_path) if frame.camera_info_path is not None else None,
        "people_path": str(frame.people_path) if frame.people_path is not None else None,
        "marker_pose_path": str(frame.marker_pose_path) if frame.marker_pose_path is not None else None,
        "depth_profile": depth_profile,
    }


def _request_video_tracker(socket_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No response from SAM3 video tracker worker: {socket_path}")
    response = json.loads(b"".join(chunks).decode("utf-8").splitlines()[0])
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "SAM3 video tracker worker failed"))
    return response.get("result") or {}


def _limited_frames(frames: list[ShigureFrame]) -> list[ShigureFrame]:
    try:
        limit = int(os.environ.get("MODEL_EVENT_MAX_REPLAY_FRAMES", "0"))
    except Exception:
        limit = 0
    if limit > 0 and len(frames) > limit:
        return frames[-limit:]
    return frames


@_exclusive_tracking_run
def run_model_event_tracking(json_path_arg: str | Path) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = _task_id(task)

    if not _model_bounds_ready(task):
        _write_status(json_path, task, "skipped", reason="model bounds are not ready")
        return {"status": "skipped", "reason": "model bounds are not ready"}

    socket_value = str(os.environ.get("SAM3_VIDEO_TRACKER_WORKER_SOCKET") or "").strip()
    if not socket_value:
        _write_status(json_path, task, "skipped", reason="SAM3_VIDEO_TRACKER_WORKER_SOCKET is not set")
        return {"status": "skipped", "reason": "SAM3_VIDEO_TRACKER_WORKER_SOCKET is not set"}
    socket_path = Path(socket_value)

    cache = ShigureHistoryCache(settings.SHIGURE_EVENT_CACHE_ROOT)
    cached_frames = list(cache.iter_frames())
    cached_frame_count = len(cached_frames)
    valid_frames = [
        frame
        for frame in cached_frames
        if frame.rgb_path and frame.depth_path and frame.rgb_path.is_file() and frame.depth_path.is_file()
    ]
    frames, tracking_window = _select_tracking_frames(task, valid_frames)
    if len(frames) < 2:
        _write_status(
            json_path,
            task,
            "skipped",
            reason="not enough cached Shigurei frames after capture window selection",
            cache_root=str(settings.SHIGURE_EVENT_CACHE_ROOT),
            cached_frame_count=cached_frame_count,
            valid_frame_count=len(valid_frames),
            frame_count=len(frames),
            tracking_window=tracking_window,
        )
        return {"status": "skipped", "reason": "not enough cached Shigurei frames", "frame_count": len(frames)}

    marker_pose, marker_pose_source = _select_marker_pose(frames)
    if marker_pose is None:
        _write_status(
            json_path,
            task,
            "skipped",
            reason=(
                "no Shigurei marker_pose found; set MODEL_EVENT_MARKER_POSE_JSON "
                "or add marker_6d_pose.json under MODEL_EVENT_MARKER_POSE_SEARCH_ROOTS"
            ),
            cache_root=str(settings.SHIGURE_EVENT_CACHE_ROOT),
            cached_frame_count=cached_frame_count,
            valid_frame_count=len(valid_frames),
            frame_count=len(frames),
            tracking_window=tracking_window,
            marker_pose_search_roots=[str(path) for path in _marker_search_roots()],
        )
        return {"status": "skipped", "reason": "marker_pose missing", "frame_count": len(frames)}

    camera_info = _first_camera_info(frames)
    if camera_info is None:
        _write_status(json_path, task, "skipped", reason="cached frames do not contain camera_info")
        return {"status": "skipped", "reason": "camera_info missing"}

    first_rgb = frames[0].rgb_path
    assert first_rgb is not None
    image_size = _image_size(first_rgb)
    camera_matrix = load_camera_matrix(camera_info)
    projected_box = project_model_bounds_to_shigurei(
        task,
        marker_pose,
        camera_matrix,
        image_size=image_size,
        padding_px=float(os.environ.get("MODEL_EVENT_PROJECTED_BOX_PADDING_PX", "4")),
    )

    output_name = task_output_name(json_path, fallback=task_id)
    task_output_dir = model_event_task_output_dir(
        MODEL_EVENT_OUTPUT_ROOT,
        task_id=task_id,
        json_path=json_path,
    )
    debug_projection_path = _write_projection_debug_image(
        frames[0],
        projected_box,
        task_output_dir / "debug",
    )

    window_frames = list(frames)
    frames, depth_profiles, depth_prefilter = _prefilter_tracking_frames(window_frames, projected_box)
    tracking_window["depth_prefilter"] = depth_prefilter
    if len(frames) < 2:
        status = "target_occluded" if depth_prefilter.get("reference_index") is None else "skipped"
        reason = (
            "no reference frame after projected-box foreground became stable or cleared"
            if status == "target_occluded"
            else "not enough frames after projected-box foreground prefilter"
        )
        _write_status(
            json_path,
            task,
            status,
            reason=reason,
            cache_root=str(settings.SHIGURE_EVENT_CACHE_ROOT),
            cached_frame_count=cached_frame_count,
            valid_frame_count=len(valid_frames),
            frame_count=len(frames),
            tracking_window=tracking_window,
            marker_pose_path=str(marker_pose),
            marker_pose_source=marker_pose_source,
            camera_info_path=str(camera_info),
            projected_box=projected_box.to_dict(),
            debug_projection_path=str(debug_projection_path),
            task_output_name=output_name,
            task_output_dir=str(task_output_dir),
        )
        return {"status": status, "reason": reason, "frame_count": len(frames)}

    video_output_dir, removed_run_count = _prepare_video_output_dir(task_output_dir)
    tracking_window["removed_previous_sam3_run_dirs"] = removed_run_count
    tracking_window["task_output_name"] = output_name
    request = {
        "action": "track_video",
        "task_id": task_id,
        "task_output_name": output_name,
        "frames": [_frame_request_payload(frame, profile) for frame, profile in zip(frames, depth_profiles)],
        "contact_frames": [
            {
                "stamp": frame.stamp.to_dict(),
                "people_path": str(frame.people_path) if frame.people_path is not None else None,
            }
            for frame in window_frames
            if frame.people_path is not None
        ],
        "box_xyxy": list(projected_box.bbox_xyxy),
        "projected_box": projected_box.to_dict(),
        "camera_matrix": np.asarray(camera_matrix, dtype=np.float64).tolist(),
        "image_size": list(image_size),
        "output_dir": str(video_output_dir),
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": False,
        "retry_box_prompt_on_empty": True,
    }

    _write_status(
        json_path,
        task,
        "running",
        cache_root=str(settings.SHIGURE_EVENT_CACHE_ROOT),
        cached_frame_count=cached_frame_count,
        valid_frame_count=len(valid_frames),
        frame_count=len(frames),
        tracking_window=tracking_window,
        marker_pose_path=str(marker_pose),
        marker_pose_source=marker_pose_source,
        camera_info_path=str(camera_info),
        projected_box=projected_box.to_dict(),
        debug_projection_path=str(debug_projection_path) if debug_projection_path is not None else None,
        task_output_name=output_name,
        task_output_dir=str(task_output_dir),
    )

    result = _request_video_tracker(socket_path, request)
    sam3_video_diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
    status = str(result.get("status") or "no_event")
    decisions = list(result.get("decisions") or [])
    event_record = result.get("event_record")
    tracked_frame_count = int(result.get("mask_count") or 0)

    task = load_task_json(json_path)
    _write_status(
        json_path,
        task,
        status,
        cached_frame_count=cached_frame_count,
        valid_frame_count=len(valid_frames),
        frame_count=len(frames),
        tracking_window=tracking_window,
        tracked_frame_count=tracked_frame_count,
        debug_projection_path=str(debug_projection_path) if debug_projection_path is not None else None,
        checked_frame_count=len(decisions),
        task_output_name=output_name,
        task_output_dir=str(task_output_dir),
        sam3_video_output_dir=str(video_output_dir),
        sam3_video_diagnostics=sam3_video_diagnostics,
        event_record=event_record,
        last_decision=decisions[-1] if decisions else None,
    )
    return {
        "status": status,
        "frame_count": len(frames),
        "tracked_frame_count": tracked_frame_count,
        "event_record": event_record,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python run_model_event_tracking_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2
    try:
        result = run_model_event_tracking(argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[OK] model_event_tracking")
        return 0
    except Exception as exc:
        try:
            json_path = resolve_task_json_path(argv[1])
            task = load_task_json(json_path)
            _write_status(json_path, task, "failed", error_message=str(exc))
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
