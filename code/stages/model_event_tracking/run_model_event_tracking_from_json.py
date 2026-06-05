from __future__ import annotations

import json
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
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
from stages.model_event_tracking.event_store import persist_taken_away_event
from stages.model_event_tracking.geometry import load_camera_matrix, project_model_bounds_to_shigurei
from stages.model_event_tracking.movement import MaskDepthMovementTracker
from stages.model_event_tracking.people import choose_event_start_contact, find_hand_contacts
from stages.model_event_tracking.schemas import ShigureFrame, to_jsonable
from stages.model_event_tracking.settings import SHIGURE_EVENT_CACHE_ROOT
from stages.model_event_tracking.tracker import read_depth_image_m


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
        start_seconds = selected[0].stamp.seconds
        selected = [frame for frame in selected if frame.stamp.seconds <= start_seconds + max_post_capture_seconds]
        metadata["max_post_capture_seconds"] = max_post_capture_seconds

    selected = _limited_frames(selected)
    metadata["selected_frame_count"] = len(selected)
    if selected:
        metadata["selected_start_stamp"] = selected[0].stamp.to_dict()
        metadata["selected_end_stamp"] = selected[-1].stamp.to_dict()
    return selected, metadata


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

    cache = ShigureHistoryCache(SHIGURE_EVENT_CACHE_ROOT)
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
            cache_root=str(SHIGURE_EVENT_CACHE_ROOT),
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
            cache_root=str(SHIGURE_EVENT_CACHE_ROOT),
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

    run_id = uuid.uuid4().hex[:12]
    task_output_dir = Path(MODEL_EVENT_OUTPUT_ROOT) / task_id
    debug_projection_path = _write_projection_debug_image(
        frames[0],
        projected_box,
        task_output_dir / "debug",
    )
    video_output_dir = task_output_dir / "sam3_video_runs" / run_id
    request = {
        "action": "track_video",
        "task_id": task_id,
        "frames": [str(frame.rgb_path) for frame in frames],
        "box_xyxy": list(projected_box.bbox_xyxy),
        "image_size": list(image_size),
        "output_dir": str(video_output_dir),
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": False,
        "propagation_direction": "forward",
        "start_frame_index": 0,
    }

    _write_status(
        json_path,
        task,
        "running",
        cache_root=str(SHIGURE_EVENT_CACHE_ROOT),
        cached_frame_count=cached_frame_count,
        valid_frame_count=len(valid_frames),
        frame_count=len(frames),
        tracking_window=tracking_window,
        marker_pose_path=str(marker_pose),
        marker_pose_source=marker_pose_source,
        camera_info_path=str(camera_info),
        projected_box=projected_box.to_dict(),
        debug_projection_path=str(debug_projection_path) if debug_projection_path is not None else None,
    )

    result = _request_video_tracker(socket_path, request)
    sam3_video_diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
    masks = sorted(result.get("masks") or [], key=lambda item: int(item.get("frame_index", 0)))
    if not masks:
        task = load_task_json(json_path)
        _write_status(
            json_path,
            task,
            "no_masks",
            cached_frame_count=cached_frame_count,
            valid_frame_count=len(valid_frames),
            frame_count=len(frames),
            tracking_window=tracking_window,
            sam3_video_output_dir=str(video_output_dir),
            sam3_video_diagnostics=sam3_video_diagnostics,
            debug_projection_path=str(debug_projection_path) if debug_projection_path is not None else None,
        )
        return {"status": "no_masks", "frame_count": len(frames)}

    movement_tracker = MaskDepthMovementTracker()
    mask_by_index = {int(item["frame_index"]): item for item in masks if item.get("mask_path")}
    decisions: list[dict[str, Any]] = []
    event_record = None
    for frame_index in sorted(mask_by_index):
        if frame_index < 0 or frame_index >= len(frames):
            continue
        frame = frames[frame_index]
        mask_path = Path(mask_by_index[frame_index]["mask_path"])
        if not mask_path.is_file() or frame.depth_path is None:
            continue
        mask = np.asarray(Image.open(mask_path)) > 0
        depth_m = read_depth_image_m(frame.depth_path)
        hand_contact = None
        if frame.people_path and frame.people_path.is_file():
            contacts = find_hand_contacts(frame.people_path, projected_box)
            hand_contact = choose_event_start_contact(contacts)
        decision = movement_tracker.update(
            mask,
            depth_m,
            camera_matrix,
            timestamp=frame.stamp,
            hand_contact=hand_contact,
        )
        decisions.append(decision.to_dict())
        if decision.moved and decision.should_stop_tracking:
            event_record = persist_taken_away_event(
                task_id=task_id,
                frame=frame,
                decision=decision,
                hand_contact=hand_contact,
                mask=mask,
                projected_box=projected_box.to_dict(),
            )
            break

    task = load_task_json(json_path)
    if event_record is not None:
        status = "taken_away"
    elif isinstance(sam3_video_diagnostics, dict) and int(sam3_video_diagnostics.get("stream_saved_masks") or 0) == 0:
        status = "target_lost"
    else:
        status = "no_event"
    _write_status(
        json_path,
        task,
        status,
        cached_frame_count=cached_frame_count,
        valid_frame_count=len(valid_frames),
        frame_count=len(frames),
        tracking_window=tracking_window,
        tracked_frame_count=len(masks),
        debug_projection_path=str(debug_projection_path) if debug_projection_path is not None else None,
        checked_frame_count=len(decisions),
        sam3_video_output_dir=str(video_output_dir),
        sam3_video_diagnostics=sam3_video_diagnostics,
        event_record=event_record.to_dict() if event_record is not None else None,
        last_decision=decisions[-1] if decisions else None,
    )
    return {
        "status": status,
        "frame_count": len(frames),
        "tracked_frame_count": len(masks),
        "event_record": event_record.to_dict() if event_record is not None else None,
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
