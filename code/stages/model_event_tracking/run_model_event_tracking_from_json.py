from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from config import (
    MODEL_EVENT_OUTPUT_ROOT,
    MODEL_EVENT_TRACKING_POLL_INTERVAL_SEC,
    MODEL_EVENT_TRACKING_TIMEOUT_SEC,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json

from stages.model_event_tracking import settings
from stages.model_event_tracking.cache import ShigureHistoryCache
from stages.model_event_tracking.event_store import persist_taken_away_event
from stages.model_event_tracking.geometry import load_camera_matrix, project_model_bounds_to_shigurei
from stages.model_event_tracking.model_depth import (
    DepthFrameDecision,
    DynamicDepthMaskTracker,
    calibrate_depth_bias,
    render_model_depth_template,
)
from stages.model_event_tracking.output_paths import (
    task_output_dir as model_event_task_output_dir,
    task_output_name,
)
from stages.model_event_tracking.people import find_hand_contacts
from stages.model_event_tracking.schemas import (
    HandContact,
    MovementDecision,
    ProjectedBox,
    RosStamp,
    ShigureFrame,
    to_jsonable,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_id(task: dict[str, Any]) -> str:
    value = str(task.get("task_id") or "").strip()
    if not value:
        raise ValueError("task_id is missing")
    return value


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


def _exclusive_tracking_run(function: Any) -> Any:
    @wraps(function)
    def wrapped(json_path_arg: str | Path) -> dict[str, Any]:
        json_path = resolve_task_json_path(json_path_arg)
        task = load_task_json(json_path)
        task_id = _task_id(task)
        output_dir = model_event_task_output_dir(
            MODEL_EVENT_OUTPUT_ROOT,
            task_id=task_id,
            json_path=json_path,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / ".tracking.lock").open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "already_running", "task_id": task_id}
            try:
                return function(json_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return wrapped


def _model_ready(task: dict[str, Any]) -> bool:
    bounds = task.get("ModelBounds")
    blender = task.get("Blender")
    return (
        isinstance(bounds, dict)
        and bounds.get("status") == "ready"
        and bool(bounds.get("corners_aruco"))
        and isinstance(blender, dict)
        and bool(blender.get("fbx"))
    )


def _marker_search_roots() -> list[Path]:
    configured = str(os.environ.get("MODEL_EVENT_MARKER_POSE_SEARCH_ROOTS") or "").strip()
    if configured:
        return [Path(value) for value in configured.split(os.pathsep) if value.strip()]
    return [CODE_ROOT / ".test" / "marker", CODE_ROOT.parent / ".test" / "fusion_runs"]


def _select_marker_pose(frames: list[ShigureFrame]) -> tuple[Path | None, str | None]:
    configured = str(os.environ.get("MODEL_EVENT_MARKER_POSE_JSON") or "").strip()
    if configured and Path(configured).is_file():
        return Path(configured), "env:MODEL_EVENT_MARKER_POSE_JSON"
    for frame in frames:
        if frame.marker_pose_path and frame.marker_pose_path.is_file():
            return frame.marker_pose_path, "cached_frame"
    candidates: list[Path] = []
    for root in _marker_search_roots():
        if root.is_file() and root.name == "marker_6d_pose.json":
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(root.rglob("marker_6d_pose.json"))
    if not candidates:
        return None, None
    return max(candidates, key=lambda path: path.stat().st_mtime), "historical_marker_pose"


def _first_camera_info(frames: list[ShigureFrame]) -> Path | None:
    for frame in frames:
        if frame.camera_info_path and frame.camera_info_path.is_file():
            return frame.camera_info_path
    return None


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
        timezone_pos = min(
            [position for position in (tail.find("+"), tail.find("-")) if position >= 0],
            default=-1,
        )
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


def _select_tracking_frames(
    task: dict[str, Any],
    frames: list[ShigureFrame],
) -> tuple[list[ShigureFrame], dict[str, Any]]:
    capture_seconds, source = _task_capture_time_seconds(task)
    settle_seconds = max(0.0, float(os.environ.get("MODEL_EVENT_CAPTURE_SETTLE_SECONDS", "2.0")))
    timeout_seconds = max(0.0, float(MODEL_EVENT_TRACKING_TIMEOUT_SEC))
    selected = list(frames)
    forced_start = _stamp_seconds_from_env(os.environ.get("MODEL_EVENT_DEBUG_FRAME_START"))
    forced_end = _stamp_seconds_from_env(os.environ.get("MODEL_EVENT_DEBUG_FRAME_END"))
    tracking_origin_seconds = None
    if forced_start is not None or forced_end is not None:
        selected = [
            frame
            for frame in frames
            if (forced_start is None or frame.stamp.seconds + 1.0e-9 >= forced_start)
            and (forced_end is None or frame.stamp.seconds - 1.0e-9 <= forced_end)
        ]
        tracking_origin_seconds = forced_start if forced_start is not None else (selected[0].stamp.seconds if selected else None)
    elif capture_seconds is not None:
        tracking_origin_seconds = capture_seconds + settle_seconds
        selected = [frame for frame in selected if frame.stamp.seconds >= tracking_origin_seconds]
        if timeout_seconds > 0.0:
            selected = [
                frame
                for frame in selected
                if frame.stamp.seconds <= tracking_origin_seconds + timeout_seconds
            ]
    elif selected:
        tracking_origin_seconds = selected[0].stamp.seconds
        if timeout_seconds > 0.0:
            selected = [
                frame
                for frame in selected
                if frame.stamp.seconds <= tracking_origin_seconds + timeout_seconds
            ]
    try:
        limit = int(os.environ.get("MODEL_EVENT_MAX_REPLAY_FRAMES", "0"))
    except ValueError:
        limit = 0
    if limit > 0:
        selected = selected[:limit]
    metadata: dict[str, Any] = {
        "input_frame_count": len(frames),
        "capture_time_source": source,
        "capture_time_seconds": capture_seconds,
        "capture_settle_seconds": settle_seconds,
        "tracking_origin_seconds": tracking_origin_seconds,
        "timeout_seconds": timeout_seconds,
        "forced_debug_start_seconds": forced_start,
        "forced_debug_end_seconds": forced_end,
        "selected_frame_count": len(selected),
    }
    if selected:
        metadata["selected_start_stamp"] = selected[0].stamp.to_dict()
        metadata["selected_end_stamp"] = selected[-1].stamp.to_dict()
    return selected, metadata


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_projection_debug(
    frame: ShigureFrame,
    projected_box: ProjectedBox,
    model_mask: np.ndarray,
    output_path: Path,
) -> Path | None:
    if frame.rgb_path is None or not frame.rgb_path.is_file():
        return None
    with Image.open(frame.rgb_path) as image:
        canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(model_mask, dtype=bool)
    if mask.shape == canvas.shape[:2]:
        canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array([40, 235, 100])).astype(np.uint8)
    rendered = Image.fromarray(canvas)
    draw = ImageDraw.Draw(rendered)
    points = [(float(x), float(y)) for x, y in projected_box.pixel_points]
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        draw.line((points[start], points[end]), fill=(40, 220, 255), width=3)
    draw.rectangle(projected_box.bbox_xyxy, outline=(255, 64, 64), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path, quality=95)
    return output_path



def _is_enabled_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _stamp_seconds_from_env(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value).strip().rstrip("/").rsplit("/", 1)[-1]
    if "_" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    sec_text, nsec_text = text.split("_", 1)
    try:
        return float(int(sec_text)) + float(int(nsec_text)) * 1.0e-9
    except ValueError:
        return None


def _blend_debug_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if mask.shape != canvas.shape[:2] or not np.any(mask):
        return
    rgb = np.asarray(color, dtype=np.float32)
    canvas[mask] = ((1.0 - alpha) * canvas[mask].astype(np.float32) + alpha * rgb).astype(np.uint8)


def _write_frame_debug(
    frame: ShigureFrame,
    decision: DepthFrameDecision,
    *,
    model_mask: np.ndarray,
    output_dir: Path,
    index: int,
) -> Path | None:
    if frame.rgb_path is None or not frame.rgb_path.is_file():
        return None
    with Image.open(frame.rgb_path) as image:
        canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    _blend_debug_mask(canvas, np.asarray(model_mask, dtype=bool), (40, 235, 100), 0.32)
    _blend_debug_mask(canvas, np.asarray(decision.unoccluded_mask, dtype=bool), (255, 225, 40), 0.55)
    _blend_debug_mask(canvas, np.asarray(decision.removed_mask, dtype=bool), (255, 60, 60), 0.70)
    rendered = Image.fromarray(canvas)
    draw = ImageDraw.Draw(rendered)
    label = (
        f"{decision.status} removed={decision.removed_ratio:.3f} "
        f"visible={decision.evaluable_ratio:.3f}"
    )
    draw.rectangle((8, 8, 620, 38), fill=(0, 0, 0))
    draw.text((14, 14), label, fill=(255, 255, 255))
    name = f"{index:03d}_{decision.timestamp.sec}_{decision.timestamp.nanosec:09d}_{decision.status}_r{decision.removed_ratio:.3f}.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    rendered.save(path, quality=95)
    return path



def _previous_frame_before_timestamp(
    frames: list[ShigureFrame],
    timestamp: RosStamp | None,
) -> ShigureFrame | None:
    if timestamp is None:
        return None
    previous: ShigureFrame | None = None
    for frame in frames:
        if frame.stamp.seconds < timestamp.seconds:
            previous = frame
            continue
        break
    return previous


def _load_rgb_array(frame: ShigureFrame) -> np.ndarray | None:
    if frame.rgb_path is None or not frame.rgb_path.is_file():
        return None
    try:
        with Image.open(frame.rgb_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32)
    except Exception:
        return None


def _rgb_motion_ratio(
    frame: ShigureFrame,
    reference_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    rgb = _load_rgb_array(frame)
    if rgb is None or rgb.shape[:2] != reference_rgb.shape[:2] or mask.shape != reference_rgb.shape[:2]:
        return 0.0, 0
    diff = np.mean(np.abs(rgb - reference_rgb), axis=2)
    changed = mask & (diff >= settings.RGB_MOTION_DIFF_THRESHOLD)
    changed_pixels = int(np.count_nonzero(changed))
    return changed_pixels / max(1, int(np.count_nonzero(mask))), changed_pixels


def _select_rgb_motion_start_frame(
    frames: list[ShigureFrame],
    *,
    depth_start: DepthFrameDecision,
    depth_confirm: DepthFrameDecision,
    model_mask: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[ShigureFrame | None, ShigureFrame | None, dict[str, Any]]:
    end_seconds = depth_start.timestamp.seconds
    start_seconds = end_seconds - max(0.0, settings.RGB_MOTION_LOOKBACK_SECONDS)
    window = [
        frame
        for frame in frames
        if start_seconds - 1.0e-9 <= frame.stamp.seconds <= end_seconds + 1.0e-9
        and frame.rgb_path
        and frame.rgb_path.is_file()
    ]
    support = np.asarray(support_mask, dtype=bool)
    full = np.asarray(model_mask, dtype=bool)
    if support.shape == full.shape and np.count_nonzero(support) >= settings.RGB_MOTION_MIN_MASK_PIXELS:
        motion_mask = support
        mask_source = "depth_support_mask"
    else:
        motion_mask = full
        mask_source = "model_mask"
    mask_pixels = int(np.count_nonzero(motion_mask))
    metadata: dict[str, Any] = {
        "method": "rgb_motion_start_after_depth_confirmation",
        "lookback_seconds": settings.RGB_MOTION_LOOKBACK_SECONDS,
        "baseline_frames": settings.RGB_MOTION_BASELINE_FRAMES,
        "diff_threshold": settings.RGB_MOTION_DIFF_THRESHOLD,
        "start_ratio": settings.RGB_MOTION_START_RATIO,
        "quiet_ratio": settings.RGB_MOTION_QUIET_RATIO,
        "confirm_frames": settings.RGB_MOTION_CONFIRM_FRAMES,
        "display_offset_frames": settings.RGB_MOTION_DISPLAY_OFFSET_FRAMES,
        "mask_source": mask_source,
        "mask_pixels": mask_pixels,
        "depth_start_timestamp": depth_start.timestamp.to_dict(),
        "depth_confirm_timestamp": depth_confirm.timestamp.to_dict(),
        "window_frame_count": len(window),
    }
    if mask_pixels < settings.RGB_MOTION_MIN_MASK_PIXELS:
        metadata["status"] = "fallback_mask_too_small"
        return None, None, metadata
    rgb_frames: list[tuple[ShigureFrame, np.ndarray]] = []
    for frame in window:
        rgb = _load_rgb_array(frame)
        if rgb is not None and rgb.shape[:2] == motion_mask.shape:
            rgb_frames.append((frame, rgb))
    if not rgb_frames:
        metadata["status"] = "fallback_no_rgb_frames"
        return None, None, metadata
    baseline_count = max(1, min(settings.RGB_MOTION_BASELINE_FRAMES, len(rgb_frames)))
    reference = np.median(
        np.stack([rgb for _frame, rgb in rgb_frames[:baseline_count]], axis=0),
        axis=0,
    ).astype(np.float32)
    ratios: list[dict[str, Any]] = []
    for frame, _rgb in rgb_frames:
        ratio, changed_pixels = _rgb_motion_ratio(frame, reference, motion_mask)
        ratios.append(
            {
                "timestamp": frame.stamp.to_dict(),
                "ratio": ratio,
                "changed_pixels": changed_pixels,
            }
        )
    metadata["reference_timestamps"] = [frame.stamp.to_dict() for frame, _rgb in rgb_frames[:baseline_count]]
    metadata["ratios"] = ratios
    confirm = max(1, int(settings.RGB_MOTION_CONFIRM_FRAMES))
    start_index: int | None = None
    for index in range(0, len(ratios)):
        if index + confirm > len(ratios):
            break
        run = ratios[index : index + confirm]
        if not all(item["ratio"] >= settings.RGB_MOTION_START_RATIO for item in run):
            continue
        start_index = index
        break
    if start_index is None:
        metadata["status"] = "fallback_no_rgb_motion_start"
        best_index = max(range(len(ratios)), key=lambda idx: ratios[idx]["ratio"])
        metadata["best_ratio"] = ratios[best_index]
        return None, None, metadata
    display_index = min(
        len(rgb_frames) - 1,
        max(0, start_index + int(settings.RGB_MOTION_DISPLAY_OFFSET_FRAMES)),
    )
    metadata["status"] = "found"
    metadata["motion_start_index"] = start_index
    metadata["display_index"] = display_index
    metadata["motion_start_ratio"] = ratios[start_index]["ratio"]
    metadata["display_ratio"] = ratios[display_index]["ratio"]
    return rgb_frames[start_index][0], rgb_frames[display_index][0], metadata


def _people_frame_has_id(frame: ShigureFrame, people_id: str) -> bool:
    if frame.people_path is None or not frame.people_path.is_file() or not people_id:
        return False
    try:
        with frame.people_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return False
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict):
        message = payload if isinstance(payload, dict) else {}
    for person in message.get("pose_key_points_list") or []:
        if isinstance(person, dict) and str(person.get("people_id") or "") == people_id:
            return True
    return False


def _nearby_skeleton_frame(
    frames: list[ShigureFrame],
    center_frame: ShigureFrame,
    contact: HandContact | None,
    *,
    radius_frames: int = 5,
) -> ShigureFrame | None:
    if contact is None or not contact.people_id:
        return None
    try:
        center_index = next(index for index, frame in enumerate(frames) if frame.stamp == center_frame.stamp)
    except StopIteration:
        return None
    radius = max(0, int(radius_frames))
    indices = range(max(0, center_index - radius), min(len(frames), center_index + radius + 1))
    ordered = sorted(indices, key=lambda index: (abs(index - center_index), index))
    for index in ordered:
        frame = frames[index]
        if _people_frame_has_id(frame, contact.people_id):
            return frame
    return None


def _closest_wrist_frame(
    frames: list[ShigureFrame],
    projected_box: ProjectedBox,
    event_seconds: float,
) -> tuple[ShigureFrame | None, HandContact | None]:
    start_seconds = event_seconds - settings.HAND_LOOKBACK_SECONDS
    inside: list[tuple[ShigureFrame, HandContact]] = []
    nearest: list[tuple[ShigureFrame, HandContact]] = []
    for frame in frames:
        if frame.stamp.seconds < start_seconds or frame.stamp.seconds > event_seconds:
            continue
        if frame.people_path is None or not frame.people_path.is_file():
            continue
        for contact in find_hand_contacts(frame.people_path, projected_box):
            nearest.append((frame, contact))
            if contact.inside_box:
                inside.append((frame, contact))
    if inside:
        return min(inside, key=lambda item: item[0].stamp.seconds)
    if nearest:
        frame, contact = min(nearest, key=lambda item: (item[1].distance_m, -item[1].score))
        if contact.distance_m <= settings.HAND_NEAREST_MAX_DISTANCE_M:
            return frame, contact
    return None, None


@_exclusive_tracking_run
def run_model_event_tracking(json_path_arg: str | Path) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = _task_id(task)
    if not _model_ready(task):
        _write_status(json_path, task, "skipped", reason="trusted FBX/model bounds are not ready")
        return {"status": "skipped", "reason": "model is not ready"}

    cache = ShigureHistoryCache(settings.SHIGURE_EVENT_CACHE_ROOT)
    cached_frames = list(cache.iter_frames())
    valid_frames = [
        frame
        for frame in cached_frames
        if frame.rgb_path
        and frame.depth_path
        and frame.rgb_path.is_file()
        and frame.depth_path.is_file()
    ]
    frames, tracking_window = _select_tracking_frames(task, valid_frames)
    initial_wait_started = time.monotonic()
    initial_wait_timeout = max(0.0, float(MODEL_EVENT_TRACKING_TIMEOUT_SEC))
    while len(frames) < settings.MODEL_DEPTH_TAKEN_AWAY_STABLE_FRAMES:
        if (
            initial_wait_timeout > 0.0
            and time.monotonic() - initial_wait_started >= initial_wait_timeout
        ):
            reason = "timed out waiting for initial cached Shigurei frames"
            _write_status(
                json_path,
                task,
                "timeout",
                reason=reason,
                cached_frame_count=len(cached_frames),
                valid_frame_count=len(valid_frames),
                frame_count=len(frames),
                tracking_window=tracking_window,
                timeout_seconds=initial_wait_timeout,
            )
            return {"status": "timeout", "reason": reason, "frame_count": len(frames)}
        time.sleep(max(0.05, float(MODEL_EVENT_TRACKING_POLL_INTERVAL_SEC)))
        cached_frames = list(cache.iter_frames())
        valid_frames = [
            frame
            for frame in cached_frames
            if frame.rgb_path
            and frame.depth_path
            and frame.rgb_path.is_file()
            and frame.depth_path.is_file()
        ]
        frames, tracking_window = _select_tracking_frames(task, valid_frames)

    marker_pose, marker_source = _select_marker_pose(frames)
    camera_info = _first_camera_info(frames)
    if marker_pose is None or camera_info is None:
        reason = "marker_pose missing" if marker_pose is None else "camera_info missing"
        _write_status(json_path, task, "skipped", reason=reason, frame_count=len(frames))
        return {"status": "skipped", "reason": reason, "frame_count": len(frames)}

    first_rgb = frames[0].rgb_path
    assert first_rgb is not None
    with Image.open(first_rgb) as image:
        image_size = image.size
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
    shutil.rmtree(task_output_dir / "taken_away", ignore_errors=True)
    shutil.rmtree(task_output_dir / "depth_tracking", ignore_errors=True)
    depth_output_dir = task_output_dir / "debug" / "depth_tracking"
    shutil.rmtree(depth_output_dir, ignore_errors=True)
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    frame_debug_dir = (
        depth_output_dir / "frames"
        if _is_enabled_env("MODEL_EVENT_DEBUG_EVERY_FRAME", False)
        else None
    )
    if frame_debug_dir is not None:
        frame_debug_dir.mkdir(parents=True, exist_ok=True)

    _write_status(
        json_path,
        task,
        "running",
        detector="fbx_depth_template",
        cached_frame_count=len(cached_frames),
        valid_frame_count=len(valid_frames),
        frame_count=len(frames),
        tracking_window=tracking_window,
        marker_pose_path=str(marker_pose),
        marker_pose_source=marker_source,
        camera_info_path=str(camera_info),
        projected_box=projected_box.to_dict(),
        task_output_name=output_name,
        task_output_dir=str(task_output_dir),
    )

    fallback_back_depth = float(np.max(projected_box.corners_camera_m[:, 2]))
    template = render_model_depth_template(
        task=task,
        marker_pose_path=marker_pose,
        camera_matrix=camera_matrix,
        image_size=image_size,
        bbox_xyxy=projected_box.bbox_xyxy,
        fallback_back_depth_m=fallback_back_depth,
        output_dir=depth_output_dir / "template",
    )
    depth_bias_m, tracking_mask, calibration = calibrate_depth_bias(frames, template)
    Image.fromarray(tracking_mask.astype(np.uint8) * 255).save(
        depth_output_dir / "reference_observed_mask.png"
    )
    debug_path = _write_projection_debug(
        frames[0],
        projected_box,
        tracking_mask,
        depth_output_dir / "projection_and_model_mask.jpg",
    )

    mask_tracker = DynamicDepthMaskTracker(
        template,
        initial_support_mask=tracking_mask,
        depth_bias_m=depth_bias_m,
    )
    support_mask_path = depth_output_dir / "current_support_mask.png"
    unoccluded_mask_path = depth_output_dir / "current_unoccluded_mask.png"

    def save_current_masks() -> None:
        Image.fromarray(mask_tracker.support_mask.astype(np.uint8) * 255).save(
            support_mask_path
        )
        Image.fromarray(
            mask_tracker.current_unoccluded_mask.astype(np.uint8) * 255
        ).save(unoccluded_mask_path)

    decisions: list[DepthFrameDecision] = []
    processed_frames: list[ShigureFrame] = []
    candidate_start: DepthFrameDecision | None = None
    candidate_count = 0
    confirmed: DepthFrameDecision | None = None
    last_processed_frame: ShigureFrame | None = None
    timed_out = False
    timeout_seconds = max(0.0, float(MODEL_EVENT_TRACKING_TIMEOUT_SEC))
    poll_interval_seconds = max(0.05, float(MODEL_EVENT_TRACKING_POLL_INTERVAL_SEC))
    tracking_origin_seconds = tracking_window.get("tracking_origin_seconds")
    if tracking_origin_seconds is None:
        tracking_origin_seconds = frames[0].stamp.seconds
    stream_deadline_seconds = (
        float(tracking_origin_seconds) + timeout_seconds if timeout_seconds > 0.0 else None
    )

    def process_frame(frame: ShigureFrame) -> None:
        nonlocal candidate_start, candidate_count, confirmed, last_processed_frame
        decision = mask_tracker.update(
            frame,
            allow_support_update=candidate_count == 0,
        )
        decisions.append(decision)
        processed_frames.append(frame)
        last_processed_frame = frame
        if frame_debug_dir is not None:
            _write_frame_debug(
                frame,
                decision,
                model_mask=template.mask,
                output_dir=frame_debug_dir,
                index=len(decisions) - 1,
            )
        if confirmed is not None:
            return
        if decision.candidate:
            if candidate_count == 0:
                candidate_start = decision
            candidate_count += 1
            if candidate_count >= settings.MODEL_DEPTH_TAKEN_AWAY_STABLE_FRAMES:
                confirmed = decision
        else:
            candidate_count = 0
            candidate_start = None

    for frame in frames:
        process_frame(frame)
        if confirmed is not None and frame_debug_dir is None:
            break

    save_current_masks()

    initial_stream_elapsed = 0.0
    if last_processed_frame is not None:
        initial_stream_elapsed = max(
            0.0,
            last_processed_frame.stamp.seconds - float(tracking_origin_seconds),
        )
    wall_deadline = (
        time.monotonic() + max(0.0, timeout_seconds - initial_stream_elapsed)
        if timeout_seconds > 0.0
        else None
    )
    last_status_update = 0.0

    while confirmed is None and frame_debug_dir is None:
        if timeout_seconds > 0.0:
            stream_expired = (
                last_processed_frame is not None
                and stream_deadline_seconds is not None
                and last_processed_frame.stamp.seconds >= stream_deadline_seconds
            )
            wall_expired = wall_deadline is not None and time.monotonic() >= wall_deadline
            if stream_expired or wall_expired:
                timed_out = True
                break

        new_frames = list(
            cache.iter_frames_after(
                last_processed_frame.stamp if last_processed_frame is not None else None
            )
        )
        accepted_new_frame = False
        for frame in new_frames:
            if not (
                frame.rgb_path
                and frame.depth_path
                and frame.rgb_path.is_file()
                and frame.depth_path.is_file()
            ):
                continue
            if frame.stamp.seconds < float(tracking_origin_seconds):
                continue
            if (
                stream_deadline_seconds is not None
                and frame.stamp.seconds > stream_deadline_seconds
            ):
                timed_out = True
                break
            process_frame(frame)
            accepted_new_frame = True
            if confirmed is not None and frame_debug_dir is None:
                break
        if confirmed is not None or timed_out:
            break

        now = time.monotonic()
        if accepted_new_frame and now - last_status_update >= 5.0:
            save_current_masks()
            current = decisions[-1]
            task = load_task_json(json_path)
            _write_status(
                json_path,
                task,
                "running",
                detector="fbx_depth_template",
                frame_count=len(processed_frames),
                checked_frame_count=len(decisions),
                last_frame_stamp=current.timestamp.to_dict(),
                support_pixels=current.support_pixels,
                unoccluded_pixels=current.evaluable_pixels,
                removed_ratio=current.removed_ratio,
                timeout_seconds=timeout_seconds,
                current_support_mask_path=str(support_mask_path),
                current_unoccluded_mask_path=str(unoccluded_mask_path),
                task_output_name=output_name,
                task_output_dir=str(task_output_dir),
            )
            last_status_update = now
        if not accepted_new_frame:
            time.sleep(poll_interval_seconds)

    save_current_masks()

    diagnostics = {
        "detector": "fbx_depth_template",
        "template": template.metadata,
        "depth_bias_m": depth_bias_m,
        "calibration": calibration,
        "tracking": {
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "tracking_origin_seconds": tracking_origin_seconds,
            "stream_deadline_seconds": stream_deadline_seconds,
            "timed_out": timed_out,
            "followed_live_cache": len(processed_frames) > len(frames),
            "initial_frame_count": len(frames),
            "processed_frame_count": len(processed_frames),
            "final_support_pixels": int(np.count_nonzero(mask_tracker.support_mask)),
            "final_unoccluded_pixels": int(
                np.count_nonzero(mask_tracker.current_unoccluded_mask)
            ),
        },
        "thresholds": {
            "removal_reference_surface": "front",
            "occlusion_margin_m": settings.MODEL_DEPTH_OCCLUSION_MARGIN_M,
            "removal_margin_m": settings.MODEL_DEPTH_REMOVAL_MARGIN_M,
            "removed_ratio": settings.MODEL_DEPTH_REMOVED_RATIO,
            "max_present_ratio": settings.MODEL_DEPTH_MAX_PRESENT_RATIO,
            "min_evaluable_ratio": settings.MODEL_DEPTH_MIN_EVALUABLE_RATIO,
            "stable_frames": settings.MODEL_DEPTH_TAKEN_AWAY_STABLE_FRAMES,
            "present_tolerance_m": settings.MODEL_DEPTH_PRESENT_TOLERANCE_M,
            "reference_observed_pixels": int(np.count_nonzero(tracking_mask)),
        },
        "checked_frame_count": len(decisions),
        "frames": [decision.to_dict() for decision in decisions],
    }
    diagnostics_path = depth_output_dir / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics)

    event_record = None
    status = "timeout" if timed_out else "no_event"
    if confirmed is not None and candidate_start is not None:
        event_frame = next(
            frame
            for frame in processed_frames
            if frame.stamp == candidate_start.timestamp
        )
        rgb_start_frame, rgb_display_frame, rgb_motion_metadata = _select_rgb_motion_start_frame(
            processed_frames,
            depth_start=candidate_start,
            depth_confirm=confirmed,
            model_mask=template.mask,
            support_mask=tracking_mask,
        )
        display_frame = (
            rgb_display_frame
            or _previous_frame_before_timestamp(processed_frames, candidate_start.timestamp)
            or event_frame
        )
        event_timestamp = (rgb_start_frame or display_frame).stamp
        wrist_frame, hand_contact = _closest_wrist_frame(
            processed_frames,
            projected_box,
            event_timestamp.seconds,
        )
        if hand_contact is None:
            wrist_frame, hand_contact = _closest_wrist_frame(
                processed_frames,
                projected_box,
                candidate_start.timestamp.seconds,
            )
        skeleton_frame = _nearby_skeleton_frame(
            processed_frames,
            display_frame,
            hand_contact,
            radius_frames=settings.SKELETON_SEARCH_RADIUS_FRAMES,
        )
        evidence_frame = replace(display_frame, marker_pose_path=marker_pose)
        skeleton_frame = (
            replace(skeleton_frame, marker_pose_path=marker_pose)
            if skeleton_frame is not None
            else None
        )
        movement = MovementDecision(
            status="taken_away",
            moved=True,
            occluded=False,
            stable_in_place=False,
            should_stop_tracking=True,
            reason=(
                f"depth confirmed removal at {candidate_start.timestamp.sec}_"
                f"{candidate_start.timestamp.nanosec:09d}: "
                f"{candidate_start.removed_ratio:.3f} of the current unoccluded "
                f"model support was deeper than the rendered front surface plus "
                f"{settings.MODEL_DEPTH_REMOVAL_MARGIN_M:.3f} m for "
                f"{settings.MODEL_DEPTH_TAKEN_AWAY_STABLE_FRAMES} frames; "
                f"event image was selected by RGB motion start status "
                f"{rgb_motion_metadata.get('status')}"
            ),
            timestamp=event_timestamp,
            trigger_contact=hand_contact,
            depth_delta_m=candidate_start.median_removal_excess_m,
            overlap_pixels=candidate_start.evaluable_pixels,
            visible_area_ratio=candidate_start.evaluable_ratio,
            area_ratio=candidate_start.removed_ratio,
            mask_iou=0.0,
            movement_candidate_frames=settings.MODEL_DEPTH_TAKEN_AWAY_STABLE_FRAMES,
            depth_decision_timestamp=candidate_start.timestamp,
            rgb_motion_start_timestamp=(rgb_start_frame.stamp if rgb_start_frame is not None else None),
            display_timestamp=display_frame.stamp,
            rgb_motion_score=rgb_motion_metadata.get("motion_start_ratio"),
            rgb_motion_metadata=rgb_motion_metadata,
        )
        event_record = persist_taken_away_event(
            task_id=task_id,
            frame=evidence_frame,
            decision=movement,
            hand_contact=hand_contact,
            mask=candidate_start.removed_mask,
            model_mask=template.mask,
            trusted_mask=candidate_start.unoccluded_mask,
            projected_box=projected_box.to_dict(),
            people_frame=skeleton_frame,
            task_output_name=output_name,
            replace=True,
        )
        status = "taken_away"

    task = load_task_json(json_path)
    _write_status(
        json_path,
        task,
        status,
        detector="fbx_depth_template",
        cached_frame_count=len(cached_frames),
        valid_frame_count=len(valid_frames),
        frame_count=len(processed_frames),
        initial_frame_count=len(frames),
        checked_frame_count=len(decisions),
        tracking_window=tracking_window,
        marker_pose_path=str(marker_pose),
        marker_pose_source=marker_source,
        camera_info_path=str(camera_info),
        projected_box=projected_box.to_dict(),
        debug_projection_path=str(debug_path) if debug_path else None,
        depth_template_dir=str(template.output_dir),
        depth_diagnostics_path=str(diagnostics_path),
        depth_bias_m=depth_bias_m,
        timeout_seconds=timeout_seconds,
        timed_out=timed_out,
        final_support_pixels=int(np.count_nonzero(mask_tracker.support_mask)),
        final_unoccluded_pixels=int(
            np.count_nonzero(mask_tracker.current_unoccluded_mask)
        ),
        current_support_mask_path=str(support_mask_path),
        current_unoccluded_mask_path=str(unoccluded_mask_path),
        event_record=event_record.to_dict() if event_record else None,
        task_output_name=output_name,
        task_output_dir=str(task_output_dir),
    )
    return {
        "status": status,
        "frame_count": len(processed_frames),
        "initial_frame_count": len(frames),
        "checked_frame_count": len(decisions),
        "event_record": event_record.to_dict() if event_record else None,
    }

def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python run_model_event_tracking_from_json.py <task_meta.json>", file=sys.stderr)
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
