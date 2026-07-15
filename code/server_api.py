"""Flask API entrypoint for task creation and status polling."""

import ipaddress
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from console_output_log import install_console_output_log
from artifact_layout import (
    SHIGURE_EVENT_ROOT,
    SHIGURE_TRACKING_BOX_SNAPSHOT_PATH,
    aruco_task_json_path,
    aruco_worker_frame_color,
    aruco_worker_frame_meta,
    ensure_aruco_task_dirs,
    ensure_artifact_roots,
    ensure_model_task_dirs,
    make_timestamp,
    model_debug_dir,
    model_result_dir,
    model_task_json_path,
    model_worker_dir,
    model_worker_file,
    sanitize_artifact_token,
)

install_console_output_log()

from depth_camera_config import get_depth_sensor_limits, normalize_depth_sensor_name
from config import MAX_REALTIME_MODEL_POSE_ITEMS
from task_worker import (
    STAGE_ORDER,
    activate_uploaded_task,
    get_task,
    reserve_uploading_task,
    start_worker,
)
from task_db import (
    get_enabled_aruco_markers,
    get_identity_sync_job,
    get_latest_aruco_reference,
    get_completed_tasks_for_startup,
    get_latest_shigure_canonical_event,
    list_live_display_object_states,
    list_object_lifecycle_history,
    get_task_by_task_id,
    sync_marker_registry_from_reference_folder,
    update_task_status,
    initialize_task_table,
)
from task_json import resolve_project_path, save_task_json
from spatial_transforms import (
    aruco_points_to_hololens,
    aruco_pose_to_hololens_pose,
    compose_pose_rt,
    invert_pose_rt,
    minimal_pose_payload,
    pose_to_rt,
    rt_to_pose,
)


app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


@app.after_request
def _log_http_failure(response):
    if response.status_code >= 400:
        print(f"[HTTP] {request.method} {request.path} failed: status={response.status_code}")
    return response


ensure_artifact_roots()
initialize_task_table()
start_worker()


PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction"
PURPOSE_ARUCO_REFERENCE = "aruco_reference"


def _write_atomic_bytes(path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as file:
        file.write(data)
    tmp_path.replace(path)


def _write_atomic_json(path, payload: dict) -> None:
    _write_atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _frame_artifact_timestamp(task_timestamp: str, index: int, frame: dict) -> str:
    raw = str(frame.get("time") or "").strip()
    if not raw:
        raise ValueError(f"PVCameraFramesJ[{index}].time is required")
    fallback = f"{task_timestamp}_{index:03d}"
    return sanitize_artifact_token(raw, fallback=fallback)


def _task_artifact_url(host: str, task_id: str, area: str, filename: str | None) -> str | None:
    if not filename:
        return None
    return f"{host}/task-artifacts/{task_id}/{area}/{filename}"


def _parse_binary_flag(value, field_name: str) -> bool:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"{field_name} must be 0 or 1")
    return text == "1"


def _strict_json_integer(value, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _strict_numeric_matrix(value, field_name: str, shape: tuple[int, int]) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{field_name} must be a {shape[0]}x{shape[1]} numeric matrix")
    normalized: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"{field_name} must be a {shape[0]}x{shape[1]} numeric matrix")
        normalized_row: list[float] = []
        for element in row:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise ValueError(f"{field_name} must contain only numbers")
            numeric = float(element)
            if not np.isfinite(numeric):
                raise ValueError(f"{field_name} must contain only finite numbers")
            normalized_row.append(numeric)
        normalized.append(normalized_row)
    return normalized


def _strict_numeric_vector(value, field_name: str, size: int) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field_name} must contain exactly {size} numbers")
    normalized: list[float] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise ValueError(f"{field_name} must contain only numbers")
        numeric = float(element)
        if not np.isfinite(numeric):
            raise ValueError(f"{field_name} must contain only finite numbers")
        normalized.append(numeric)
    return normalized


def _require_exact_multipart_keys(expected_form: set[str], expected_files: set[str]) -> None:
    if request.args:
        raise ValueError("query parameters are not supported")
    actual_form = set(request.form.keys())
    actual_files = set(request.files.keys())
    if actual_form != expected_form:
        missing = sorted(expected_form - actual_form)
        unexpected = sorted(actual_form - expected_form)
        raise ValueError(f"invalid form fields; missing={missing}, unsupported={unexpected}")
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"invalid upload files; missing={missing}, unsupported={unexpected}")
    for field_name in expected_form:
        if len(request.form.getlist(field_name)) != 1:
            raise ValueError(f"form field {field_name} must appear exactly once")
    for field_name in expected_files:
        if len(request.files.getlist(field_name)) != 1:
            raise ValueError(f"upload file {field_name} must appear exactly once")


def _require_json_transport() -> None:
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    if request.args or request.form or request.files:
        raise ValueError("JSON endpoints do not accept query, form, or file fields")


def _require_no_request_body() -> None:
    if request.form or request.files or request.get_data(cache=True):
        raise ValueError("request body is not supported")


def _require_exact_query(expected_keys: set[str]) -> None:
    actual_keys = set(request.args.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"invalid query fields; missing={missing}, unsupported={unexpected}")
    for key in expected_keys:
        if len(request.args.getlist(key)) != 1:
            raise ValueError(f"query field {key} must appear exactly once")


def _normalize_pv_frame(frame: dict, index: int, *, aruco_reference: bool) -> dict:
    field_name = f"PVCameraFramesJ[{index}]" if aruco_reference else "PVCameraJ"
    if not isinstance(frame, dict):
        raise ValueError(f"{field_name} must be a JSON object")

    expected_keys = {"width", "height", "k", "pose", "time"}
    actual_keys = set(frame)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"{field_name} is missing required fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{field_name} contains unsupported fields: {', '.join(unexpected)}")

    width = _strict_json_integer(frame["width"], f"{field_name}.width", minimum=1)
    height = _strict_json_integer(frame["height"], f"{field_name}.height", minimum=1)
    k = _strict_numeric_matrix(frame["k"], f"{field_name}.k", (3, 3))
    pose = _strict_numeric_matrix(frame["pose"], f"{field_name}.pose", (4, 4))
    frame_time = frame["time"]
    if not isinstance(frame_time, str) or not frame_time.strip():
        raise ValueError(f"{field_name}.time must be a non-empty string")

    normalized = {
        "width": width,
        "height": height,
        "k": k,
        "pose": pose,
        "time": frame_time.strip(),
    }
    if aruco_reference:
        normalized["frame_index"] = index
    return normalized


def _normalize_device(payload: dict, *, purpose: str) -> dict:
    expected_keys = {"startup_session_id"}
    if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
        expected_keys.add("ip")
    actual_keys = set(payload)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"deviceJ is missing required fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"deviceJ contains unsupported fields: {', '.join(unexpected)}")
    startup_session_id = payload["startup_session_id"]
    if not isinstance(startup_session_id, str) or not startup_session_id.strip():
        raise ValueError("deviceJ.startup_session_id must be a non-empty string")
    normalized = {"startup_session_id": startup_session_id.strip()}
    if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
        ip_text = payload["ip"]
        if not isinstance(ip_text, str) or not ip_text.strip():
            raise ValueError("deviceJ.ip must be a non-empty IPv4 address")
        try:
            ip_value = ipaddress.ip_address(ip_text.strip())
        except ValueError as exc:
            raise ValueError("deviceJ.ip must be a valid IPv4 address") from exc
        if ip_value.version != 4:
            raise ValueError("deviceJ.ip must be a valid IPv4 address")
        normalized["ip"] = str(ip_value)
    return normalized


def _normalize_purpose(value) -> str:
    purpose = str(value or "").strip()
    if not purpose:
        raise ValueError("purpose is required")
    if purpose not in {PURPOSE_OBJECT_RECONSTRUCTION, PURPOSE_ARUCO_REFERENCE}:
        raise ValueError(f"Unsupported purpose: {purpose}")
    return purpose


def _task_startup_session_id(task_data: dict | None = None, task_json: dict | None = None) -> str | None:
    task_json = task_json or ((task_data or {}).get("task_json") or {})
    value = (
        (task_data or {}).get("startup_session_id")
        or ((task_json.get("device") or {}).get("startup_session_id") if isinstance(task_json, dict) else None)
    )
    return str(value or "").strip() or None


def _load_latest_aruco_reference_pose(startup_session_id: str | None) -> dict | None:
    startup_session_id = str(startup_session_id or "").strip()
    if not startup_session_id:
        return None
    row = get_latest_aruco_reference(startup_session_id)
    if not row:
        return None
    return _load_marker_pose_json(row.get("marker_pose_json"))


def _response_startup_session_id(
    task_data: dict | None = None,
    task_json: dict | None = None,
    startup_session_id: str | None = None,
) -> str | None:
    _ = task_data, task_json
    return str(startup_session_id or "").strip() or None


def _hololens_current_pose_for_response(
    task_json: dict,
    *,
    task_data: dict | None = None,
    startup_session_id: str | None = None,
) -> dict | None:
    response_startup_session_id = _response_startup_session_id(task_data, task_json, startup_session_id)
    if not response_startup_session_id:
        return None

    object_aruco = task_json.get("object_aruco") if isinstance(task_json.get("object_aruco"), dict) else None
    if object_aruco is not None:
        aruco_reference = _load_latest_aruco_reference_pose(response_startup_session_id)
        if aruco_reference is not None:
            try:
                return aruco_pose_to_hololens_pose(object_aruco, aruco_reference)
            except Exception as exc:
                print(f"[WARN] pose response conversion failed for startup={response_startup_session_id}: {exc}")

    task_startup_session_id = _task_startup_session_id(task_data, task_json)
    if task_startup_session_id != response_startup_session_id:
        return None

    current = task_json.get("object_hololens_current")
    if current is None:
        current = task_json.get("object_hololens_original")
    if not isinstance(current, dict):
        return None
    try:
        return minimal_pose_payload(current, include_scale=True)
    except Exception:
        return None


def _build_model_key(task_id: str | None) -> str:
    task_id_text = str(task_id or "").strip()
    if not task_id_text:
        raise ValueError("task_id is required for model_instance")
    return task_id_text


def _stage_progress(status: str, purpose: str | None) -> dict:
    status_text = str(status or "").strip() or "pending"
    if purpose == PURPOSE_ARUCO_REFERENCE:
        stage_order = ["aruco_detect"]
    else:
        stage_order = list(STAGE_ORDER)

    stage_count = max(1, len(stage_order))
    if status_text == "pending":
        stage_index = 0
        stage_name = stage_order[0] if stage_order else "pending"
    elif status_text in stage_order:
        stage_index = stage_order.index(status_text)
        stage_name = status_text
    elif status_text in {"completed", "aruco_completed"}:
        stage_index = stage_count
        stage_name = status_text
    else:
        stage_index = 0
        stage_name = status_text

    progress = max(0.0, min(1.0, float(stage_index) / float(stage_count)))
    percent = int(round(progress * 100.0))
    return {
        "stage_name": stage_name,
        "stage_index": stage_index,
        "stage_count": stage_count,
        "progress": progress,
        "progress_text": f"{stage_name} {percent}%",
    }


def _display_identity_from_task_json(task_json: dict) -> dict:
    identity = task_json.get("DisplayIdentity")
    return identity if isinstance(identity, dict) else {}


def _public_sam3_spatial_box(value) -> dict | None:
    """Strip server-canonical geometry from the Unity preview payload."""

    if not isinstance(value, dict):
        return None
    server_only_keys = {
        "canonical_coordinate_space",
        "diagonal_m",
        "aruco_sync_source",
    }
    return {
        key: item
        for key, item in value.items()
        if key not in server_only_keys
        and not str(key).startswith("canonical_")
        and not str(key).startswith("aruco_")
        and not str(key).endswith("_aruco")
    }


def _build_model_instance(
    *,
    task_id: str,
    display_object_id: str,
    model_revision: int,
    fbx_url: str,
    pose: dict,
) -> dict:
    task_id = str(task_id or "").strip()
    display_object_id = str(display_object_id or "").strip()
    fbx_url = str(fbx_url or "").strip()
    if not task_id or not display_object_id or not fbx_url:
        raise ValueError("model_instance requires task_id, display_object_id, and fbx_url")
    if not isinstance(pose, dict):
        raise ValueError("model_instance.pose must be an object")
    instance = {
        "model_key": _build_model_key(task_id),
        "task_id": task_id,
        "display_object_id": display_object_id,
        "model_revision": _strict_json_integer(
            model_revision,
            "model_revision",
            minimum=1,
        ),
        "fbx_url": fbx_url,
        "pose": pose,
        "coordinate_space": "hololens_current_local",
    }
    return instance


def _build_pending_task_response(
    task_data: dict,
    *,
    position: int | None = None,
    startup_session_id: str | None = None,
) -> dict:
    task_json = task_data.get("task_json") or {}
    task_id = str(task_data.get("task_id") or "")
    status = str(task_data.get("status") or "pending")
    purpose = task_json.get("purpose")
    progress = _stage_progress(status, purpose)

    response = {
        "task_id": task_id,
        "status": status,
        "purpose": purpose,
        "terminal": False,
        "stage_runs": task_data.get("stage_runs") or [],
        "timing_events": task_data.get("timing_events") or [],
        "ai_model_timings": task_data.get("ai_model_timings") or [],
    }
    response.update(progress)
    if position is not None:
        response["position"] = int(position)

    spatial_box = _public_sam3_spatial_box(task_json.get("Sam3SpatialBox"))
    if spatial_box:
        response["sam3_spatial_box"] = spatial_box

    return response


def _task_fbx_url(host: str, task_id: str, task_json: dict) -> str | None:
    blender_info = task_json.get("Blender") or {}
    fbx_name = str(blender_info.get("fbx") or "").strip()
    task_timestamp = str(task_json.get("task_timestamp") or "").strip()
    if not task_timestamp:
        return None
    result_dir = model_result_dir(task_timestamp)
    if not fbx_name or blender_info.get("artifact_root") != "model_result":
        return None
    fbx_path = result_dir / fbx_name
    if not fbx_path.is_file():
        return None
    return _task_artifact_url(host, task_id, "result", fbx_name)

def _sanitize_depth_png(depth_png_bytes: bytes, sensor_name: str) -> tuple[bytes, dict]:
    limits = get_depth_sensor_limits(sensor_name)
    depth_png = np.frombuffer(depth_png_bytes, dtype=np.uint8)
    depth_image = cv2.imdecode(depth_png, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("depth_image is not a valid PNG")
    if depth_image.dtype != np.uint16:
        raise ValueError(f"depth_image must be uint16 depth, got {depth_image.dtype}")
    if depth_image.ndim != 2:
        raise ValueError(f"depth_image must be a single-channel image, got shape {depth_image.shape}")

    valid_mask = (
        (depth_image >= limits.min_depth_mm)
        & (depth_image <= limits.max_reliable_depth_mm)
    )
    sanitized_depth = np.where(valid_mask, depth_image, 0).astype(np.uint16)

    encoded_ok, encoded_png = cv2.imencode(".png", sanitized_depth)
    if not encoded_ok:
        raise ValueError(f"Failed to re-encode sanitized {limits.sensor} depth image")

    stats = {
        "sensor": limits.sensor,
        "width": int(depth_image.shape[1]),
        "height": int(depth_image.shape[0]),
        "raw_nonzero_pixels": int(np.count_nonzero(depth_image)),
        "valid_depth_pixels": int(valid_mask.sum()),
        "clipped_depth_pixels": int(np.count_nonzero(depth_image) - valid_mask.sum()),
        "min_depth_mm": int(limits.min_depth_mm),
        "max_reliable_depth_mm": int(limits.max_reliable_depth_mm),
        "input_png_bytes": int(len(depth_png_bytes)),
        "sanitized_png_bytes": int(encoded_png.size),
    }

    if limits.upload_guard_enabled and stats["valid_depth_pixels"] < int(limits.min_usable_depth_pixels):
        raise ValueError(
            f"{limits.sensor} depth is not usable. Keep the object within {limits.max_reliable_depth_mm / 1000.0:.1f} m."
        )
    if limits.upload_guard_enabled and stats["sanitized_png_bytes"] > int(limits.max_upload_png_bytes):
        raise ValueError(
            f"{limits.sensor} depth is still too large after range filtering. Keep the object within {limits.max_reliable_depth_mm / 1000.0:.1f} m."
        )

    return encoded_png.tobytes(), stats


def _build_completed_task_response(
    task_data: dict,
    *,
    host_override: str | None = None,
    startup_session_id: str | None = None,
) -> dict:
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
        "purpose": (task_data.get("task_json") or {}).get("purpose"),
        "terminal": True,
        "stage_runs": task_data.get("stage_runs") or [],
        "timing_events": task_data.get("timing_events") or [],
        "ai_model_timings": task_data.get("ai_model_timings") or [],
        "aruco_coordinate_synced": bool(task_data.get("aruco_coordinate_synced")),
    }
    task_json = task_data.get("task_json") or {}
    response["delivery_role"] = "capture_preview"
    identity_sync = task_json.get("ShigureIdentitySync")
    if isinstance(identity_sync, dict):
        # Auxiliary status only; it never promotes a HoloLens capture to live.
        response["shigure_identity_sync"] = dict(identity_sync)
    task_id = str(task_data.get("task_id") or "")
    host = (host_override or request.host_url).rstrip("/")
    sync_payload = response.get("shigure_identity_sync")
    if isinstance(sync_payload, dict):
        sync_job_id = str(sync_payload.get("sync_job_id") or "").strip().lower()
        if len(sync_job_id) == 32 and all(ch in "0123456789abcdef" for ch in sync_job_id):
            sync_payload["status_url"] = f"{host}/api/v2/identity-sync/{sync_job_id}"
    fbx_url = _task_fbx_url(host, task_id, task_json)
    if fbx_url:
        identity = _display_identity_from_task_json(task_json)
        display_object_id = str(identity.get("display_object_id") or "").strip()
        if not display_object_id:
            response["error"] = "completed task is missing DisplayIdentity.display_object_id"
            return response
        pose = _hololens_current_pose_for_response(
            task_json,
            task_data=task_data,
            startup_session_id=startup_session_id,
        )
        if pose is None:
            response["error"] = "current_session_pose_unavailable"
            return response
        try:
            model_revision = _strict_json_integer(
                identity.get("model_revision"),
                "model_revision",
                minimum=1,
            )
        except (TypeError, ValueError) as exc:
            response["error"] = f"invalid DisplayIdentity.model_revision: {exc}"
            return response
        response["model_instance"] = _build_model_instance(
            task_id=task_id,
            display_object_id=display_object_id,
            model_revision=model_revision,
            fbx_url=fbx_url,
            pose=pose,
        )
    else:
        response["error"] = "completed model FBX is missing"

    return response


def _build_aruco_completed_task_response(task_data: dict) -> dict:
    task_json = task_data.get("task_json") or {}
    startup_session_id = str(
        task_data.get("startup_session_id")
        or (task_json.get("device") or {}).get("startup_session_id")
        or ""
    ).strip()
    aruco_stage = (
        ((task_json.get("debug") or {}).get("pose_transform_stages") or {}).get("aruco_stage")
        or {}
    )
    latest_reference_row = get_latest_aruco_reference(startup_session_id) if startup_session_id else None
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
        "purpose": task_json.get("purpose"),
        "terminal": True,
        "stage_runs": task_data.get("stage_runs") or [],
        "timing_events": task_data.get("timing_events") or [],
        "ai_model_timings": task_data.get("ai_model_timings") or [],
        "startup_session_id": startup_session_id,
        "aruco_detected": latest_reference_row is not None,
        "aruco_reference_task_id": (latest_reference_row or {}).get("task_id"),
        "retro_synced_completed_task_count": _strict_json_integer(
            aruco_stage["retro_synced_completed_task_count"],
            "aruco_stage.retro_synced_completed_task_count",
        ),
        "coordinate_handling": "server_only_hololens_local_payloads",
    }
    return response


def _load_marker_pose_json(raw_json: str | None):
    if not raw_json:
        return None
    try:
        loaded = json.loads(raw_json)
        if not isinstance(loaded, dict):
            return None
        position = loaded.get("position")
        rotation = loaded.get("rotation_quaternion_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            return None
        if not isinstance(rotation, list) or len(rotation) != 4:
            return None
        return {
            "position": [float(v) for v in position],
            "rotation_quaternion_xyzw": [float(v) for v in rotation],
        }
    except Exception:
        return None

@app.route("/", methods=["GET"], strict_slashes=False)
def index():
    try:
        _require_exact_query(set())
        _require_no_request_body()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "status": "running",
            "endpoints": [
                "/generate",
                "/check-queue",
                "/aruco/latest-reference?startup_session_id=<startup_session_id>",
                "/aruco/markers",
                "/aruco/markers/sync",
            ],
        }
    )


@app.route("/task-artifacts/<task_id>/<area>/<path:filename>", methods=["GET"], strict_slashes=False)
def serve_task_artifact(task_id: str, area: str, filename: str):
    try:
        _require_exact_query(set())
        _require_no_request_body()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    task_record = get_task_by_task_id(task_id)
    if task_record is None:
        return jsonify({"error": "task_not_found"}), 404
    task_timestamp = str(task_record.get("task_timestamp") or "").strip()
    if not task_timestamp:
        return jsonify({"error": "task_timestamp_missing"}), 404
    area_roots = {
        "worker": model_worker_dir(task_timestamp),
        "result": model_result_dir(task_timestamp),
        "debug": model_debug_dir(task_timestamp),
    }
    root = area_roots.get(str(area or "").strip())
    if root is None:
        return jsonify({"error": "invalid_artifact_area"}), 400
    return send_from_directory(root, filename)


@app.route("/generate", methods=["POST"], strict_slashes=False)
def generate_model():
    try:
        def _parse_json_field(field_name: str) -> dict:
            raw = request.form.get(field_name, type=str)
            if not raw:
                raise ValueError(f"missing {field_name}")
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"{field_name} is not valid JSON: {exc}")
            if not isinstance(obj, dict):
                raise ValueError(f"{field_name} must be a JSON object")
            return obj

        def _parse_optional_json_array_field(field_name: str) -> list | None:
            raw = request.form.get(field_name, type=str)
            if not raw:
                return None
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"{field_name} is not valid JSON: {exc}")
            if not isinstance(obj, list):
                raise ValueError(f"{field_name} must be a JSON array")
            return obj

        def _read_upload_file(field_name: str) -> bytes:
            file_storage = request.files.get(field_name)
            if file_storage is None:
                raise ValueError(f"missing uploaded file {field_name}")
            data = file_storage.read()
            if not data:
                raise ValueError(f"uploaded file {field_name} is empty")
            return data

        purpose = _normalize_purpose(request.form.get("purpose"))
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            _require_exact_multipart_keys(
                {
                    "purpose",
                    "deviceJ",
                    "PVCameraJ",
                    "DepthCameraJ",
                    "SelectionBoxJ",
                    "force_new_3d_model",
                },
                {"pv_image", "depth_image"},
            )
        devj = _normalize_device(_parse_json_field("deviceJ"), purpose=purpose)
        startup_session_id = devj["startup_session_id"]

        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            if "force_new_3d_model" not in request.form or not str(
                request.form.get("force_new_3d_model") or ""
            ).strip():
                raise ValueError("force_new_3d_model is required for object_reconstruction")
            if "PVCameraFramesJ" in request.form:
                raise ValueError("PVCameraFramesJ is only valid for aruco_reference")
            force_new_3d_model = _parse_binary_flag(
                request.form["force_new_3d_model"],
                "force_new_3d_model",
            )
            normalized_pv_frames = [
                _normalize_pv_frame(
                    _parse_json_field("PVCameraJ"),
                    0,
                    aruco_reference=False,
                )
            ]
        else:
            if "force_new_3d_model" in request.form:
                raise ValueError("force_new_3d_model is only valid for object_reconstruction")
            if "PVCameraJ" in request.form:
                raise ValueError("PVCameraJ is only valid for object_reconstruction")
            pv_frames_input = _parse_optional_json_array_field("PVCameraFramesJ")
            if not pv_frames_input:
                raise ValueError("PVCameraFramesJ must contain at least one frame")
            normalized_pv_frames = [
                _normalize_pv_frame(frame, index, aruco_reference=True)
                for index, frame in enumerate(pv_frames_input)
            ]
            _require_exact_multipart_keys(
                {"purpose", "deviceJ", "PVCameraFramesJ"},
                {f"pv_image_{index}" for index in range(len(normalized_pv_frames))},
            )

        dj = None
        sbj = None
        top_left = None
        bottom_right = None
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            dj = _parse_json_field("DepthCameraJ")
            sbj = _parse_json_field("SelectionBoxJ")

            if set(dj) != {"pose", "sensor"}:
                raise ValueError("DepthCameraJ must contain exactly pose and sensor")
            requested_sensor = normalize_depth_sensor_name(dj["sensor"])
            dj = {
                "pose": _strict_numeric_matrix(dj["pose"], "DepthCameraJ.pose", (4, 4)),
                "sensor": requested_sensor,
            }

            if set(sbj) != {"top_left", "bottom_right"}:
                raise ValueError("SelectionBoxJ must contain exactly top_left and bottom_right")
            top_left = _strict_numeric_vector(sbj["top_left"], "SelectionBoxJ.top_left", 2)
            bottom_right = _strict_numeric_vector(sbj["bottom_right"], "SelectionBoxJ.bottom_right", 2)
            if any(value < 0.0 or value > 1.0 for value in top_left + bottom_right):
                raise ValueError("SelectionBoxJ coordinates must be in [0, 1]")
            if bottom_right[0] <= top_left[0] or bottom_right[1] <= top_left[1]:
                raise ValueError("SelectionBoxJ.bottom_right must be below and right of top_left")

        now_utc = datetime.now(timezone.utc)
        server_received_utc = now_utc.isoformat().replace("+00:00", "Z")
        base = make_timestamp(now_utc)
        reserved_task_id = None

        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            task_record = reserve_uploading_task(
                task_timestamp=base,
                startup_session_id=startup_session_id,
            )
            reserved_task_id = str(task_record["task_id"])
            ensure_model_task_dirs(base)
            meta_path = model_task_json_path(base)
        else:
            meta_path = aruco_task_json_path(base)
            task_record = reserve_uploading_task(
                task_timestamp=base,
                startup_session_id=startup_session_id,
                json_path=meta_path,
            )
            reserved_task_id = str(task_record["task_id"])
            ensure_aruco_task_dirs(base)

        color_path = None
        for index, frame in enumerate(normalized_pv_frames):
            field_name = "pv_image" if purpose == PURPOSE_OBJECT_RECONSTRUCTION else f"pv_image_{index}"
            pv_png_bytes = _read_upload_file(field_name)
            if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
                frame_color_path = model_worker_file(base, "input.color")
                frame["artifact_root"] = "model_worker"
            else:
                frame_timestamp = _frame_artifact_timestamp(base, index, frame)
                frame["artifact_root"] = "aruco_worker"
                frame["task_timestamp"] = base
                frame["artifact_timestamp"] = frame_timestamp
                frame_color_path = aruco_worker_frame_color(base, frame_timestamp)
            _write_atomic_bytes(frame_color_path, pv_png_bytes)
            frame["name"] = str(frame_color_path.name)
            frame["upload_field"] = field_name
            frame["png_bytes"] = int(len(pv_png_bytes))
            if purpose == PURPOSE_ARUCO_REFERENCE:
                frame_meta_path = aruco_worker_frame_meta(base, frame["artifact_timestamp"])
                _write_atomic_json(frame_meta_path, frame)
                frame["meta_name"] = str(frame_meta_path.name)
            if index == 0:
                color_path = frame_color_path

        depth_path = None
        depth_stats = None
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            depth_png_bytes = _read_upload_file("depth_image")
            depth_png_bytes, depth_stats = _sanitize_depth_png(depth_png_bytes, requested_sensor)
            depth_path = model_worker_file(base, "input.depth")
            _write_atomic_bytes(depth_path, depth_png_bytes)

        out_json = {
            "server_received_utc": server_received_utc,
            "task_name": base,
            "task_timestamp": base,
            "task_id": reserved_task_id,
            "purpose": purpose,
            "device": devj,
        }
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            first_frame = normalized_pv_frames[0]
            out_json["PVCamera"] = {
                "name": str(color_path.name),
                "width": first_frame["width"],
                "height": first_frame["height"],
                "k": first_frame["k"],
                "pose": first_frame["pose"],
                "time": first_frame["time"],
            }
            out_json["force_new_3d_model"] = bool(force_new_3d_model)
            out_json["DepthCamera"] = {
                "name": str(depth_path.name) if depth_path else None,
                "pose": dj.get("pose") if dj else None,
                "sensor": requested_sensor,
                "stats": depth_stats,
            }
            out_json["SelectionBox"] = {
                "top_left": top_left,
                "bottom_right": bottom_right,
            }
        else:
            out_json["PVCameraFrames"] = normalized_pv_frames

        save_task_json(meta_path, out_json)

        task_id = reserved_task_id
        activate_uploaded_task(task_id, task_json=out_json)

        return jsonify({"task_id": task_id})

    except Exception as exc:
        reserved_task_id = locals().get("reserved_task_id")
        if reserved_task_id:
            try:
                update_task_status(str(reserved_task_id), "upload_failed", error_message=str(exc))
            except Exception:
                pass
        print("[ERROR] /generate exception:", exc)
        return jsonify({"error": str(exc)}), 400


@app.route("/check-queue", methods=["POST"], strict_slashes=False)
def check_task_queue():
    try:
        _require_json_transport()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"task_ids", "startup_session_id"}:
            raise ValueError("request body must contain exactly task_ids and startup_session_id")
        task_ids = payload["task_ids"]
        if not isinstance(task_ids, list):
            raise ValueError("task_ids must be a JSON array")
        if not task_ids:
            raise ValueError("task_ids must not be empty")
        startup_session_value = payload["startup_session_id"]
        if not isinstance(startup_session_value, str):
            raise ValueError("startup_session_id must be a string")
        client_startup_session_id = startup_session_value.strip()
        if not client_startup_session_id:
            raise ValueError("startup_session_id is required")

        pending = []
        seen = set()
        for raw_task_id in task_ids:
            if not isinstance(raw_task_id, str) or not raw_task_id.strip():
                raise ValueError("every task_id must be a non-empty string")
            task_id = raw_task_id.strip()
            if task_id in seen:
                raise ValueError(f"duplicate task_id: {task_id}")
            seen.add(task_id)

            task_data = get_task(task_id)
            if not task_data:
                return jsonify(
                    {
                        "ready": True,
                        "task_id": task_id,
                        "status": "failed",
                        "purpose": None,
                        "terminal": True,
                        "task": {
                            "status": "failed",
                            "task_id": task_id,
                            "terminal": True,
                            "error": "Invalid task ID",
                        },
                    }
                )

            status = task_data["status"]
            task_json = task_data.get("task_json") or {}
            purpose = task_json.get("purpose")
            if status == "completed":
                response = _build_completed_task_response(
                    task_data,
                    startup_session_id=client_startup_session_id,
                )
            elif status == "aruco_completed":
                response = _build_aruco_completed_task_response(task_data)
            elif status in {"failed", "upload_failed"}:
                response = {
                    "status": status,
                    "task_id": task_data.get("task_id"),
                    "purpose": purpose,
                    "terminal": True,
                    "error": task_data.get("error_message") or "Unknown error",
                    "stage_runs": task_data.get("stage_runs") or [],
                    "timing_events": task_data.get("timing_events") or [],
                    "ai_model_timings": task_data.get("ai_model_timings") or [],
                }
            else:
                pending.append(
                    _build_pending_task_response(
                        task_data,
                        position=len(pending) + 1,
                        startup_session_id=client_startup_session_id,
                    )
                )
                continue

            response["terminal"] = True
            return jsonify(
                {
                    "ready": True,
                    "task_id": task_id,
                    "status": status,
                    "purpose": purpose,
                    "terminal": True,
                    "task": response,
                }
            )

        return jsonify({"ready": False, "pending": pending})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in check_task_queue: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/aruco/latest-reference", methods=["GET"], strict_slashes=False)
def latest_aruco_reference():
    try:
        _require_exact_query({"startup_session_id"})
        _require_no_request_body()
        startup_session_id = str(request.args.get("startup_session_id") or "").strip()
        if not startup_session_id:
            raise ValueError("startup_session_id is required")

        latest_reference_row = get_latest_aruco_reference(startup_session_id)
        if not latest_reference_row:
            return jsonify(
                {
                    "error": "No ArUco reference found for this startup session",
                    "startup_session_id": startup_session_id,
                }
            ), 404

        return jsonify(
            {
                "status": "aruco_reference",
                "startup_session_id": startup_session_id,
                "task_id": latest_reference_row.get("task_id"),
                "created_at": latest_reference_row.get("created_at"),
                "aruco_detected": True,
                "coordinate_handling": "server_only_hololens_local_payloads",
                "terminal": True,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in latest_aruco_reference: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/aruco/markers", methods=["GET"], strict_slashes=False)
def list_aruco_markers():
    try:
        _require_exact_query(set())
        _require_no_request_body()
        return jsonify({"markers": get_enabled_aruco_markers()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in list_aruco_markers: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/aruco/markers/sync", methods=["POST"], strict_slashes=False)
def sync_aruco_markers():
    try:
        _require_exact_query(set())
        _require_no_request_body()
        synced_count = sync_marker_registry_from_reference_folder()
        return jsonify({"synced_count": synced_count, "markers": get_enabled_aruco_markers()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in sync_aruco_markers: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route(
    "/api/v2/identity-sync/<sync_job_id>",
    methods=["GET"],
    strict_slashes=False,
)
def identity_sync_status_v2(sync_job_id: str):
    try:
        _require_no_request_body()
        if request.args:
            raise ValueError("identity-sync status does not accept query fields")
        sync_job_id = str(sync_job_id or "").strip().lower()
        if len(sync_job_id) != 32 or any(ch not in "0123456789abcdef" for ch in sync_job_id):
            raise ValueError("sync_job_id must be a 32-character lowercase hex UUID")
        row = get_identity_sync_job(sync_job_id)
        if row is None:
            return jsonify({"error": "identity sync job not found"}), 404
        raw_result = row.get("result_json")
        result = json.loads(raw_result) if isinstance(raw_result, str) and raw_result else None
        return jsonify(
            {
                "sync_job_id": sync_job_id,
                "kind": row.get("kind"),
                "status": row.get("status"),
                "display_object_id": row.get("display_object_id"),
                "result": result,
                "error_message": row.get("error_message"),
                "updated_at": row.get("updated_at"),
                "terminal": str(row.get("status")) in {"COMPLETED", "FAILED"},
            }
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in identity_sync_status_v2: {exc}")
        return jsonify({"error": str(exc)}), 500


def _json_column(value, field_name: str, expected_type: type):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain JSON")
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is invalid JSON: {exc}") from exc
    if not isinstance(loaded, expected_type):
        raise ValueError(f"{field_name} contains the wrong JSON type")
    return loaded


def _strict_current_pose(pose_aruco: dict, reference_pose: dict) -> dict:
    converted = aruco_pose_to_hololens_pose(pose_aruco, reference_pose)
    return minimal_pose_payload(converted, include_scale=False)


def _live_spatial_revision(presence_epoch, spatial_sequence) -> int:
    presence_epoch = _strict_json_integer(
        presence_epoch,
        "presence_epoch",
        minimum=0,
    )
    spatial_sequence = _strict_json_integer(
        spatial_sequence,
        "latest_spatial_observation_seq",
        minimum=0,
    )
    if spatial_sequence >= (1 << 32):
        raise ValueError("latest_spatial_observation_seq exceeds 32-bit component range")
    return max(1, (presence_epoch << 32) | spatial_sequence)


def _strict_current_box(
    corners_aruco,
    reference_pose: dict,
    *,
    revision: int,
    emit_no_box: bool = False,
) -> dict | None:
    if corners_aruco is None:
        if not emit_no_box:
            return None
        return {
            "status": "no_box",
            "coordinate_space": "hololens_current_local",
            "revision": max(1, int(revision)),
        }
    points = np.asarray(corners_aruco, dtype=np.float64)
    if points.shape != (8, 3) or not np.all(np.isfinite(points)):
        raise ValueError("spatial box must contain 8 finite ArUco points")
    current = aruco_points_to_hololens(points, reference_pose)
    return {
        "status": "ready",
        "coordinate_space": "hololens_current_local",
        "revision": max(1, int(revision)),
        "corners_hololens_current_local_m": [
            [float(component) for component in point] for point in current
        ],
    }


def _strict_current_skeleton(skeleton_aruco, reference_pose: dict) -> dict | None:
    if not isinstance(skeleton_aruco, dict):
        return None
    people_id = str(skeleton_aruco.get("people_id") or "").strip()
    raw_joints = skeleton_aruco.get("joints")
    if not people_id or not isinstance(raw_joints, list) or not raw_joints:
        return None

    prepared: list[tuple[str, list[float], float, bool]] = []
    for raw_joint in raw_joints:
        if not isinstance(raw_joint, dict):
            return None
        name = str(raw_joint.get("name") or "").strip()
        position = np.asarray(raw_joint.get("position"), dtype=np.float64)
        try:
            score = float(raw_joint.get("score"))
        except (TypeError, ValueError):
            return None
        valid = raw_joint.get("valid")
        if (
            not name
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.isfinite(score)
            or not isinstance(valid, bool)
        ):
            return None
        prepared.append(
            (
                name,
                [float(component) for component in position],
                score,
                valid,
            )
        )

    converted = aruco_points_to_hololens(
        [position for _, position, _, _ in prepared],
        reference_pose,
    )
    return {
        "people_id": people_id,
        "joints": [
            {
                "name": name,
                "position": [float(component) for component in converted[index]],
                "score": score,
                "valid": valid,
            }
            for index, (name, _position, score, valid) in enumerate(prepared)
        ],
    }


def _event_artifact_url(host: str, stored_path: str | None) -> str | None:
    raw = str(stored_path or "").strip()
    if not raw:
        return None
    try:
        resolved = resolve_project_path(raw, require_exists=True)
        relative = resolved.relative_to(SHIGURE_EVENT_ROOT.resolve())
    except (FileNotFoundError, ValueError):
        return None
    if len(relative.parts) != 2 or resolved.name != relative.parts[1]:
        return None
    event_directory = sanitize_artifact_token(relative.parts[0], fallback="invalid")
    filename = sanitize_artifact_token(relative.parts[1], fallback="invalid")
    if event_directory != relative.parts[0] or filename != relative.parts[1]:
        return None
    return f"{host}/shigure-event-artifacts/{event_directory}/{filename}"


@app.route(
    "/shigure-event-artifacts/<event_directory>/<filename>",
    methods=["GET"],
    strict_slashes=False,
)
def shigure_event_artifact(event_directory: str, filename: str):
    safe_event = sanitize_artifact_token(event_directory, fallback="invalid")
    safe_filename = sanitize_artifact_token(filename, fallback="invalid")
    if safe_event != event_directory or safe_filename != filename:
        return jsonify({"error": "invalid artifact path"}), 400
    return send_from_directory(SHIGURE_EVENT_ROOT / safe_event, safe_filename)


def _startup_relative_aruco_reference(
    startup_session_id: str,
    display_object_id: str | None = None,
) -> tuple[dict, str, str] | None:
    # Without a current ArMarker, use only a same-startup capture as the
    # temporary local anchor. This cannot authorize cross-startup history.
    for row in get_completed_tasks_for_startup(startup_session_id):
        task_id = str(row.get("task_id") or "").strip()
        task_data = get_task(task_id) if task_id else None
        task = task_data.get("task_json") if task_data else None
        if not isinstance(task, dict):
            continue
        identity = _display_identity_from_task_json(task)
        if (
            display_object_id
            and str(identity.get("display_object_id") or "").strip()
            != display_object_id
        ):
            continue
        object_aruco = task.get("object_aruco")
        object_local = task.get("object_hololens_current")
        if not isinstance(object_aruco, dict) or not isinstance(
            object_local, dict
        ):
            continue
        try:
            aruco_position, aruco_rotation, _ = pose_to_rt(
                object_aruco, "same_startup.object_aruco"
            )
            local_position, local_rotation, _ = pose_to_rt(
                object_local, "same_startup.object_hololens_current"
            )
            inverse_rotation, inverse_translation = invert_pose_rt(
                aruco_rotation, aruco_position
            )
            reference_rotation, reference_translation = compose_pose_rt(
                local_rotation,
                local_position,
                inverse_rotation,
                inverse_translation,
            )
            return (
                rt_to_pose(reference_rotation, reference_translation),
                task_id,
                str(row.get("created_at") or ""),
            )
        except Exception:
            continue
    return None


def _utc_datetime(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _live_snapshot_items(
    *,
    startup_session_id: str,
) -> tuple[list[dict], str, dict | None]:
    # Model and pose transport intentionally remains bounded at five. Spatial
    # boxes use the separate complete snapshot below and are never truncated.
    states = list_live_display_object_states(
        limit=MAX_REALTIME_MODEL_POSE_ITEMS
    )
    reference_row = get_latest_aruco_reference(startup_session_id)
    reference_pose = (
        _load_marker_pose_json(reference_row.get("marker_pose_json"))
        if reference_row is not None
        else None
    )
    if reference_row is not None and reference_pose is None:
        raise ValueError("current startup session ArUco reference is invalid")
    coordinate_epoch = (
        str(reference_row.get("task_id") or reference_row.get("id") or "").strip()
        if reference_row is not None
        else f"startup-local:{startup_session_id}"
    )
    if not coordinate_epoch:
        raise ValueError("current coordinate epoch is unavailable")

    host = request.host_url.rstrip("/")
    items: list[dict] = []
    for state in states:
        display_object_id = str(state.get("display_object_id") or "").strip()
        model_revision = int(state.get("active_model_revision") or 0)
        if not display_object_id or model_revision <= 0:
            continue

        active_task_id = str(
            state.get("active_model_task_id") or ""
        ).strip()
        active_task = get_task(active_task_id) if active_task_id else None
        selected_source = "hololens"
        selected_revision = int(
            state.get("latest_hololens_pose_revision") or 0
        )
        try:
            if reference_pose is None:
                if (
                    not active_task
                    or str(active_task.get("startup_session_id") or "")
                    != startup_session_id
                ):
                    continue
                active_task_json = active_task.get("task_json") or {}
                selected_pose = minimal_pose_payload(
                    active_task_json.get("object_hololens_current"),
                    include_scale=True,
                )
                selected_revision = max(1, selected_revision)
            else:
                hololens_pose_aruco = _json_column(
                    state.get("latest_hololens_pose_aruco_json"),
                    "latest_hololens_pose_aruco_json",
                    dict,
                )
                tracking_pose_aruco = _json_column(
                    state.get("latest_tracking_pose_aruco_json"),
                    "latest_tracking_pose_aruco_json",
                    dict,
                )
                selected_pose_aruco = hololens_pose_aruco
                if (
                    isinstance(tracking_pose_aruco, dict)
                    and int(
                        state.get("latest_tracking_model_revision") or 0
                    )
                    == model_revision
                ):
                    selected_source = "tracking"
                    selected_revision = int(
                        state.get("latest_tracking_pose_revision") or 0
                    )
                    selected_pose_aruco = tracking_pose_aruco
                if selected_pose_aruco is None or selected_revision <= 0:
                    continue
                selected_pose = _strict_current_pose(
                    selected_pose_aruco, reference_pose
                )
        except Exception as exc:
            print(
                f"[WARN] skipped invalid live pose for {display_object_id}: {exc}"
            )
            continue

        item = {
            "display_object_id": display_object_id,
            "model_revision": model_revision,
            "active_model_task_id": state.get("active_model_task_id"),
            "pose_source": selected_source,
            "pose_revision": selected_revision,
            "pose": selected_pose,
            "coordinate_space": "hololens_current_local",
            "presence": str(state.get("presence") or "UNKNOWN"),
            "presence_epoch": int(state.get("presence_epoch") or 0),
        }
        latest_event = get_latest_shigure_canonical_event(display_object_id)
        if latest_event is not None:
            item["tracking_status"] = latest_event.get("resolution_status")
            item["tracking_event_uid"] = latest_event.get("event_uid")

        if active_task and str(active_task.get("status") or "") == "completed":
            active_task_json = active_task.get("task_json") or {}
            active_identity = _display_identity_from_task_json(active_task_json)
            identity_matches_state = (
                str(active_identity.get("display_object_id") or "").strip()
                == display_object_id
                and int(active_identity.get("model_revision") or 0)
                == model_revision
            )
            fbx_url = _task_fbx_url(host, active_task_id, active_task_json)
            if identity_matches_state and fbx_url:
                item["model"] = _build_model_instance(
                    task_id=active_task_id,
                    display_object_id=display_object_id,
                    model_revision=model_revision,
                    fbx_url=fbx_url,
                    pose=selected_pose,
                )
        items.append(item)
    return items, coordinate_epoch, reference_pose


def _live_tracking_box_items(reference_pose: dict | None) -> list[dict]:
    # Best-effort relay of the latest complete Shigure object_tracking snapshot.
    # No freshness window, stability gate, identity binding, or whole-snapshot
    # validation is applied here. Per-item parsing only exists to perform the
    # required ArUco -> current HoloLens-local coordinate conversion.
    if reference_pose is None:
        return []
    try:
        payload = json.loads(
            SHIGURE_TRACKING_BOX_SNAPSHOT_PATH.read_text(encoding="utf-8")
        )
        raw_boxes = payload.get("boxes") if isinstance(payload, dict) else []
        if not isinstance(raw_boxes, list):
            raw_boxes = []
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"[WARN] failed to relay latest raw tracking boxes: {exc}")
        return []

    boxes: list[dict] = []
    for index, item in enumerate(raw_boxes):
        if not isinstance(item, dict):
            continue
        tracking_id = str(
            item.get("tracking_id")
            or item.get("raw_tracking_id")
            or ""
        ).strip()
        if not tracking_id:
            continue
        try:
            revision = max(1, int(item.get("revision") or index + 1))
            spatial_box = _strict_current_box(
                item.get("corners_aruco"),
                reference_pose,
                revision=revision,
            )
        except Exception as exc:
            print(
                "[WARN] skipped unconvertible relayed tracking box "
                f"{tracking_id}: {exc}"
            )
            continue
        if spatial_box is None:
            continue
        boxes.append(
            {
                "tracking_id": tracking_id,
                "revision": revision,
                "spatial_box": spatial_box,
            }
        )
    return boxes


@app.route(
    "/api/v2/shigure/object-tracking-boxes/latest",
    methods=["GET"],
    strict_slashes=False,
)
def latest_shigure_object_tracking_boxes():
    startup_session_id = str(
        request.args.get("startup_session_id") or ""
    ).strip()
    if not startup_session_id:
        return jsonify({"success": False, "error": "startup_session_id is required"}), 400

    reference_row = get_latest_aruco_reference(startup_session_id)
    reference_pose = (
        _load_marker_pose_json(reference_row.get("marker_pose_json"))
        if reference_row is not None
        else None
    )
    coordinate_epoch = (
        str(reference_row.get("task_id") or reference_row.get("id") or "").strip()
        if reference_row is not None
        else f"startup-local:{startup_session_id}"
    )
    boxes = _live_tracking_box_items(reference_pose)
    return jsonify(
        {
            "success": True,
            "startup_session_id": startup_session_id,
            "coordinate_space": "hololens_current_local",
            "coordinate_epoch": coordinate_epoch,
            "tracking_box_count": len(boxes),
            "tracking_boxes": boxes,
        }
    )


_LIVE_TRANSPORT_LOCK = threading.RLock()
_LIVE_TRANSPORT_SESSIONS: dict[str, dict[str, int]] = {}


def _accept_live_handshake(startup_session_id: str, request_generation: int) -> dict:
    with _LIVE_TRANSPORT_LOCK:
        existing = _LIVE_TRANSPORT_SESSIONS.get(startup_session_id)
        if existing is not None and request_generation < existing["request_generation"]:
            raise ValueError("stale request_generation")
        if existing is None:
            state = {"request_generation": request_generation, "mode_epoch": 1}
        elif request_generation > existing["request_generation"]:
            state = {
                "request_generation": request_generation,
                "mode_epoch": existing["mode_epoch"] + 1,
            }
        else:
            state = dict(existing)
        _LIVE_TRANSPORT_SESSIONS[startup_session_id] = state
        return dict(state)


def _live_handshake_status(startup_session_id: str) -> dict:
    with _LIVE_TRANSPORT_LOCK:
        state = _LIVE_TRANSPORT_SESSIONS.get(startup_session_id)
        if state is None:
            raise ValueError("live mode handshake is required")
        return dict(state)


def _live_transport_response(startup_session_id: str, state: dict):
    items, coordinate_epoch, reference_pose = _live_snapshot_items(
        startup_session_id=startup_session_id,
    )
    tracking_boxes = _live_tracking_box_items(reference_pose)
    return {
        "success": True,
        "startup_session_id": startup_session_id,
        "mode": "live",
        "mode_epoch": int(state["mode_epoch"]),
        "request_generation": int(state["request_generation"]),
        "coordinate_epoch": coordinate_epoch,
        "count": len(items),
        "items": items,
        "tracking_box_snapshot_complete": True,
        "tracking_box_count": len(tracking_boxes),
        "tracking_boxes": tracking_boxes,
    }


@app.route("/realtime-tracking/mode", methods=["POST"], strict_slashes=False)
def realtime_tracking_mode():
    try:
        _require_json_transport()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        expected_keys = {"startup_session_id", "mode", "request_generation"}
        if set(payload) != expected_keys:
            missing = sorted(expected_keys - set(payload))
            unexpected = sorted(set(payload) - expected_keys)
            raise ValueError(
                f"invalid request fields; missing={missing}, unsupported={unexpected}"
            )
        startup_session_id = payload["startup_session_id"]
        if not isinstance(startup_session_id, str) or not startup_session_id.strip():
            raise ValueError("startup_session_id is required")
        startup_session_id = startup_session_id.strip()
        if payload["mode"] != "live":
            raise ValueError("v2 only accepts mode=live; history is presentation-only")
        request_generation = _strict_json_integer(
            payload["request_generation"],
            "request_generation",
            minimum=0,
        )
        state = _accept_live_handshake(startup_session_id, request_generation)
        return jsonify(_live_transport_response(startup_session_id, state))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in realtime_tracking_mode: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/realtime-tracking/status", methods=["GET"], strict_slashes=False)
def realtime_tracking_status():
    try:
        _require_exact_query({"startup_session_id"})
        _require_no_request_body()
        startup_session_id = str(request.args.get("startup_session_id") or "").strip()
        if not startup_session_id:
            raise ValueError("startup_session_id is required")
        state = _live_handshake_status(startup_session_id)
        return jsonify(_live_transport_response(startup_session_id, state))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in realtime_tracking_status: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route(
    "/api/v2/display-objects/<display_object_id>/history",
    methods=["GET"],
    strict_slashes=False,
)
def display_object_history_v2(display_object_id: str):
    try:
        _require_no_request_body()
        expected = {"kind", "limit", "startup_session_id"}
        actual = set(request.args)
        if "before_cursor" in actual:
            expected.add("before_cursor")
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"invalid query fields; missing={missing}, unsupported={unexpected}"
            )
        for key in expected:
            if len(request.args.getlist(key)) != 1:
                raise ValueError(f"query field {key} must appear exactly once")
        if request.args["kind"] != "take_out":
            raise ValueError("kind must be take_out")
        limit = _strict_json_integer(
            int(request.args["limit"]),
            "limit",
            minimum=1,
        )
        if limit > 20:
            raise ValueError("limit must not exceed 20")
        startup_session_id = str(request.args["startup_session_id"]).strip()
        if not startup_session_id:
            raise ValueError("startup_session_id is required")
        before_cursor = request.args.get("before_cursor")
        before_id = None
        if before_cursor is not None:
            before_id = _strict_json_integer(
                int(before_cursor),
                "before_cursor",
                minimum=1,
            )

        reference_row = get_latest_aruco_reference(startup_session_id)
        same_startup_since = None
        if reference_row is not None:
            reference_pose = _load_marker_pose_json(
                reference_row.get("marker_pose_json")
            )
            if reference_pose is None:
                raise ValueError(
                    "current startup session ArUco reference is invalid"
                )
            coordinate_epoch = str(
                reference_row.get("task_id")
                or reference_row.get("id")
                or ""
            ).strip()
        else:
            fallback = _startup_relative_aruco_reference(
                startup_session_id,
                str(display_object_id or "").strip(),
            )
            if fallback is None:
                raise ValueError(
                    "no ArMarker and no same-startup object anchor"
                )
            reference_pose, _anchor_task_id, anchor_created_at = fallback
            # The derived reference maps canonical ArUco poses into this
            # startup's existing HoloLens-local frame. Match the live transport
            # epoch so Unity can safely apply the same-startup history pose.
            coordinate_epoch = f"startup-local:{startup_session_id}"
            same_startup_since = _utc_datetime(anchor_created_at)
        if not coordinate_epoch:
            raise ValueError("current coordinate epoch is unavailable")

        host = request.host_url.rstrip("/")
        history_event = None
        # The response contains one usable history event. Invalid or incomplete
        # rows must not form an artificial pagination wall: otherwise a client
        # using limit=1 can never reach the first valid row below 20 bad rows.
        scan_before_id = before_id
        page_size = 100
        while history_event is None:
            candidates = list_object_lifecycle_history(
                display_object_id,
                limit=page_size,
                before_id=scan_before_id,
                take_out_only=True,
            )
            if not candidates:
                break
            for event in candidates:
                if same_startup_since is not None:
                    event_time = _utc_datetime(event.get("occurred_at"))
                    if event_time is None or event_time < same_startup_since:
                        continue
                try:
                    pose_aruco = _json_column(
                        event.get("pose_aruco_json"),
                        "pose_aruco_json",
                        dict,
                    )
                    skeleton_aruco = _json_column(
                        event.get("skeleton_json"),
                        "skeleton_json",
                        dict,
                    )
                    if pose_aruco is None:
                        continue
                    scene_image_url = _event_artifact_url(
                        host,
                        event.get("scene_image_path"),
                    )
                    pose = _strict_current_pose(pose_aruco, reference_pose)
                    skeleton = (
                        _strict_current_skeleton(
                            skeleton_aruco, reference_pose
                        )
                        if skeleton_aruco is not None
                        else None
                    )
                    corners_aruco = _json_column(
                        event.get("spatial_box_corners_aruco_json"),
                        "spatial_box_corners_aruco_json",
                        list,
                    )
                    spatial_box = _strict_current_box(
                        corners_aruco,
                        reference_pose,
                        revision=int(event.get("presence_epoch") or 1),
                    )
                except Exception as exc:
                    print(
                        f"[WARN] skipped invalid history event "
                        f"{event.get('lifecycle_event_uid')}: {exc}"
                    )
                    continue
                evidence = (
                    {
                        "scene_image_url": scene_image_url,
                        "skeleton": skeleton,
                    }
                    if scene_image_url is not None and skeleton is not None
                    else None
                )
                history_event = {
                    "display_object_id": str(event["display_object_id"]),
                    "event_uid": str(event["lifecycle_event_uid"]),
                    "history_cursor": str(event["id"]),
                    "pose": pose,
                    "spatial_box": spatial_box,
                    "evidence": evidence,
                }
                break
            if history_event is not None or len(candidates) < page_size:
                break
            next_before_id = int(candidates[-1]["id"])
            if scan_before_id is not None and next_before_id >= scan_before_id:
                raise RuntimeError("history pagination did not advance")
            scan_before_id = next_before_id

        return jsonify(
            {
                "success": True,
                "coordinate_space": "hololens_current_local",
                "coordinate_epoch": coordinate_epoch,
                "history_event": history_event,
            }
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in display_object_history_v2: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500



@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("Starting Flask application...")
    app.run(host="0.0.0.0", port=7355, debug=False, use_reloader=False, threaded=True)
