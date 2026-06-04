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
from PIL import Image

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
    payload.update(fields)
    payload["status"] = status
    payload["updated_at"] = utc_now()
    task["ModelEventTracking"] = to_jsonable(payload)
    save_task_json(json_path, task)


def _model_bounds_ready(task: dict[str, Any]) -> bool:
    bounds = task.get("ModelBounds") or {}
    return isinstance(bounds, dict) and bounds.get("status") == "ready" and bool(bounds.get("corners_aruco"))


def _select_marker_pose(frames: list[ShigureFrame]) -> Path | None:
    env_path = str(os.environ.get("MODEL_EVENT_MARKER_POSE_JSON") or "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path
    for frame in frames:
        if frame.marker_pose_path and frame.marker_pose_path.is_file():
            return frame.marker_pose_path
    return None


def _first_camera_info(frames: list[ShigureFrame]) -> Path | None:
    for frame in frames:
        if frame.camera_info_path and frame.camera_info_path.is_file():
            return frame.camera_info_path
    return None


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


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
    frames = _limited_frames(list(cache.iter_frames()))
    frames = [frame for frame in frames if frame.rgb_path and frame.depth_path and frame.rgb_path.is_file() and frame.depth_path.is_file()]
    if len(frames) < 2:
        _write_status(
            json_path,
            task,
            "skipped",
            reason="not enough cached Shigurei frames",
            cache_root=str(SHIGURE_EVENT_CACHE_ROOT),
            frame_count=len(frames),
        )
        return {"status": "skipped", "reason": "not enough cached Shigurei frames", "frame_count": len(frames)}

    marker_pose = _select_marker_pose(frames)
    if marker_pose is None:
        _write_status(
            json_path,
            task,
            "skipped",
            reason="cached frames do not contain marker_pose; set MODEL_EVENT_MARKER_POSE_JSON or cache marker poses",
            cache_root=str(SHIGURE_EVENT_CACHE_ROOT),
            frame_count=len(frames),
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
    video_output_dir = Path(MODEL_EVENT_OUTPUT_ROOT) / task_id / "sam3_video_runs" / run_id
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
        frame_count=len(frames),
        marker_pose_path=str(marker_pose),
        camera_info_path=str(camera_info),
        projected_box=projected_box.to_dict(),
    )

    result = _request_video_tracker(socket_path, request)
    masks = sorted(result.get("masks") or [], key=lambda item: int(item.get("frame_index", 0)))
    if not masks:
        task = load_task_json(json_path)
        _write_status(
            json_path,
            task,
            "no_masks",
            frame_count=len(frames),
            sam3_video_output_dir=str(video_output_dir),
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
    status = "taken_away" if event_record is not None else "no_event"
    _write_status(
        json_path,
        task,
        status,
        frame_count=len(frames),
        tracked_frame_count=len(masks),
        checked_frame_count=len(decisions),
        sam3_video_output_dir=str(video_output_dir),
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
