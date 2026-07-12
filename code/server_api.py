"""Flask API entrypoint for task creation and status polling."""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import RLock

import cv2
import numpy as np
from flask import Flask, g, jsonify, request, send_from_directory

from console_output_log import install_console_output_log
from artifact_layout import (
    aruco_task_json_path,
    aruco_worker_frame_color,
    aruco_worker_frame_meta,
    ensure_aruco_task_dirs,
    ensure_artifact_roots,
    ensure_history_request_dirs,
    ensure_model_task_dirs,
    make_timestamp,
    model_debug_dir,
    model_result_dir,
    model_task_json_path,
    model_worker_dir,
    model_worker_file,
    sanitize_artifact_token,
    history_request_result_dir,
    history_request_worker_dir,
)

install_console_output_log()

from depth_camera_config import (
    DEPTH_SENSOR_AHAT,
    get_depth_sensor_limits,
    normalize_depth_sensor_name,
)
from task_worker import (
    STAGE_ORDER,
    activate_uploaded_task,
    get_latest_completed_task_data,
    get_task,
    reserve_uploading_task,
    start_worker,
)
from task_db import (
    commit_display_object_capture_state,
    create_history_placement_request,
    get_enabled_aruco_markers,
    get_latest_completed_tasks,
    get_latest_history_placement_request,
    get_latest_aruco_reference,
    get_latest_realtime_tracking_event,
    get_latest_ready_model_bounds,
    list_display_object_states,
    get_model_bounds_by_task_id,
    get_ready_model_bounds_in_range,
    get_task_by_task_id,
    sync_marker_registry_from_reference_folder,
    update_history_placement_request,
    update_task_status,
    initialize_task_table,
)
from realtime_tracking import MODE_HISTORY, MODE_LIVE, coordinator as realtime_tracking_coordinator
from model_bounds import decode_model_bounds_row, latest_bounds_for_ray, range_bounds_for_ray
from model_generation_common import resolve_model_generation_source, resolve_runtime_mesh_source
from stages.history_placement_restoration import settings as history_placement_settings
from stages.history_placement_restoration.run_history_placement_restoration_from_json import (
    prepare_shared_current_context,
    run_history_placement_restoration,
)
from task_json import resolve_task_json_path_from_record, save_task_json
from coordinate_systems import convert_hololens_pv_pose_matrix_to_unity_pose_components
from spatial_transforms import (
    aruco_points_to_hololens,
    aruco_pose_to_hololens_pose,
    minimal_pose_payload,
    resolve_hololens_original_pose,
)


app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


@app.before_request
def _log_request_start():
    g.request_started_at = time.perf_counter()
    try:
        form_keys = list(request.form.keys()) if request.form else []
        file_keys = list(request.files.keys()) if request.files else []
        json_payload = request.get_json(silent=True) if request.is_json else None
        json_keys = list(json_payload.keys()) if isinstance(json_payload, dict) else []
        print(
            f"[HTTP][REQ] {request.method} {request.path} "
            f"remote={request.remote_addr} args={dict(request.args)} "
            f"form_keys={form_keys} file_keys={file_keys} json_keys={json_keys}"
        )
    except Exception as exc:
        print(f"[HTTP][REQ_LOG_ERR] {request.method} {request.path}: {exc}")


@app.after_request
def _log_request_end(response):
    try:
        elapsed_ms = (time.perf_counter() - float(getattr(g, 'request_started_at', time.perf_counter()))) * 1000.0
        print(f"[HTTP][RESP] {request.method} {request.path} status={response.status_code} elapsed_ms={elapsed_ms:.1f}")
    except Exception as exc:
        print(f"[HTTP][RESP_LOG_ERR] {request.method} {request.path}: {exc}")
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
    raw = frame.get("time") or frame.get("timestamp") or frame.get("frame_timestamp")
    fallback = f"{task_timestamp}_{index:03d}"
    return sanitize_artifact_token(str(raw or ""), fallback=fallback)


def _task_artifact_url(host: str, task_id: str, area: str, filename: str | None) -> str | None:
    if not filename:
        return None
    return f"{host}/task-artifacts/{task_id}/{area}/{filename}"


def _model_file_url(host: str, task_id: str, source, filename: str | None) -> str | None:
    if not filename:
        return None
    folder = getattr(source, "folder", None)
    if folder == "model_worker":
        return _task_artifact_url(host, task_id, "worker", filename)
    if folder == "model_result":
        return _task_artifact_url(host, task_id, "result", filename)
    return None


def _is_truthy_query_value(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_unity_pv_pose_components(pose_value) -> tuple[list[float] | None, list[float] | None]:
    if pose_value is None:
        return None, None

    position, _rotation, quat_xyzw = convert_hololens_pv_pose_matrix_to_unity_pose_components(
        pose_value
    )
    return [float(v) for v in position], [float(v) for v in quat_xyzw]


def _normalize_purpose(value) -> str:
    purpose = str(value or "").strip()
    if not purpose:
        return PURPOSE_OBJECT_RECONSTRUCTION
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


def _hololens_original_pose_for_response(
    task_json: dict,
    *,
    task_data: dict | None = None,
    startup_session_id: str | None = None,
) -> dict | None:
    response_startup_session_id = _response_startup_session_id(task_data, task_json, startup_session_id)
    task_startup_session_id = _task_startup_session_id(task_data, task_json)
    if not response_startup_session_id or task_startup_session_id != response_startup_session_id:
        return None
    original = resolve_hololens_original_pose(task_json)
    if original is None:
        return None
    try:
        return minimal_pose_payload(original, include_scale=True)
    except Exception:
        return None


def _append_pose_fields(
    response: dict,
    task_json: dict,
    *,
    task_data: dict | None = None,
    startup_session_id: str | None = None,
) -> None:
    current_pose = _hololens_current_pose_for_response(
        task_json,
        task_data=task_data,
        startup_session_id=startup_session_id,
    )
    original_pose = _hololens_original_pose_for_response(
        task_json,
        task_data=task_data,
        startup_session_id=startup_session_id,
    )
    response["object_hololens_current"] = current_pose
    response["object_hololens_original"] = original_pose
    response["coordinate_space"] = "hololens_current_local" if current_pose else None


def _build_model_key(task_id: str | None, fbx_url: str) -> str:
    task_id_text = str(task_id or "").strip()
    if task_id_text:
        return task_id_text
    return str(fbx_url or "").strip()


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


def _append_display_identity_fields(payload: dict, task_json: dict) -> None:
    identity = _display_identity_from_task_json(task_json)
    payload["display_identity"] = identity or None
    if not identity:
        return
    payload["display_object_id"] = identity.get("display_object_id")
    payload["capture_instance_id"] = identity.get("capture_instance_id")


def _hololens_pose_key_for_public_response(key: str) -> str | None:
    if key == "pose_aruco":
        return "pose_hololens"
    if key.endswith("_pose_aruco"):
        return f"{key[:-len('_aruco')]}_hololens"
    if key == "pose_armarker":
        return "pose_hololens"
    if key.endswith("_pose_armarker"):
        return f"{key[:-len('_armarker')]}_hololens"
    return None


def _hololens_point_key_for_public_response(key: str) -> str | None:
    point_keys = {
        "object_center_aruco",
        "object_center_armarker",
        "aruco_position",
        "reference_aruco",
    }
    if key not in point_keys:
        return None
    if key.endswith("_aruco"):
        return f"{key[:-len('_aruco')]}_hololens"
    if key.endswith("_armarker"):
        return f"{key[:-len('_armarker')]}_hololens"
    if key.startswith("aruco_"):
        return f"hololens_{key[len('aruco_') :]}"
    return f"{key}_hololens"


def _public_spatial_payload(value, *, startup_session_id: str | None = None):
    aruco_reference = _load_latest_aruco_reference_pose(startup_session_id)

    def convert(item):
        if isinstance(item, dict):
            converted = {}
            for key, child in item.items():
                if key in {"aruco_reference", "object_aruco"}:
                    continue
                if key in {"aabb_min_aruco", "aabb_max_aruco", "corners_aruco", "local_up_aruco"}:
                    continue
                pose_key = _hololens_pose_key_for_public_response(str(key))
                if pose_key is not None:
                    if aruco_reference is not None and isinstance(child, dict):
                        try:
                            converted[pose_key] = aruco_pose_to_hololens_pose(child, aruco_reference)
                        except Exception as exc:
                            converted[f"{pose_key}_error"] = str(exc)
                    continue
                point_key = _hololens_point_key_for_public_response(str(key))
                if point_key is not None:
                    if aruco_reference is not None:
                        try:
                            point = aruco_points_to_hololens([child], aruco_reference)[0]
                            converted[point_key] = [float(v) for v in point]
                        except Exception as exc:
                            converted[f"{point_key}_error"] = str(exc)
                    continue
                if key == "coordinate_space" and child in {"aruco", "armarker"}:
                    converted[key] = "hololens_current_local" if aruco_reference is not None else None
                    continue
                converted[key] = convert(child)
            return converted
        if isinstance(item, list):
            return [convert(child) for child in item]
        return item

    return convert(value)


def _dedupe_display_object_models(models: list[dict], *, limit: int | None = None) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for model in models:
        identity = model.get("display_identity") if isinstance(model.get("display_identity"), dict) else {}
        display_object_id = str((identity or {}).get("display_object_id") or model.get("display_object_id") or "").strip()
        binding_status = str((identity or {}).get("binding_status") or "").strip()
        if display_object_id and binding_status != "unbound":
            key = f"display:{display_object_id}"
        else:
            key = f"task:{model.get('task_id') or model.get('id')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(model)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _is_hololens_uploaded_model_task(task_json: dict) -> bool:
    if not isinstance(task_json, dict):
        return False
    if str(task_json.get("purpose") or "").strip() != PURPOSE_OBJECT_RECONSTRUCTION:
        return False
    device = task_json.get("device") if isinstance(task_json.get("device"), dict) else {}
    device_type = str((device or {}).get("type") or "").strip().lower()
    if "hololens" not in device_type:
        return False
    return bool(task_json.get("PVCamera") or task_json.get("PVCameraFrames"))


def _display_object_selection_key(task_data: dict, task_json: dict) -> tuple[str, str | None]:
    identity = _display_identity_from_task_json(task_json)
    display_object_id = str(identity.get("display_object_id") or task_json.get("display_object_id") or "").strip()
    binding_status = str(identity.get("binding_status") or "").strip()
    if display_object_id and binding_status != "unbound":
        return f"display:{display_object_id}", display_object_id
    task_id = str(task_data.get("task_id") or task_json.get("task_id") or task_data.get("id") or "").strip()
    return f"task:{task_id}", None


def _select_latest_hololens_uploaded_model_tasks(
    rows: list[dict],
    *,
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    skipped: list[dict] = []
    seen_keys: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        task_data = get_task(task_id) if task_id else None
        if not task_data:
            skipped.append({"task_id": task_id, "reason": "task_not_found"})
            continue

        task_json = task_data.get("task_json") or {}
        if not _is_hololens_uploaded_model_task(task_json):
            skipped.append({"task_id": task_id, "reason": "not_hololens_uploaded_model"})
            continue

        selection_key, display_object_id = _display_object_selection_key(task_data, task_json)
        if selection_key in seen_keys:
            skipped.append(
                {
                    "task_id": task_id,
                    "reason": "older_duplicate_display_object",
                    "display_object_id": display_object_id,
                    "selection_key": selection_key,
                }
            )
            continue

        seen_keys.add(selection_key)
        task_data["selection_key"] = selection_key
        task_data["selection_display_object_id"] = display_object_id
        selected.append(task_data)
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped


def _history_model_selection_scan_limit(model_limit: int) -> int:
    if int(model_limit or 0) <= 0:
        return 0
    return max(50, int(model_limit) * 10)


def _include_duplicate_captures_requested() -> bool:
    return (
        _is_truthy_query_value(request.args.get("include_duplicate_captures"))
        or _is_truthy_query_value(request.args.get("include_duplicate_display_captures"))
    )


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
    task_data: dict,
    task_json: dict,
    fbx_url: str,
    *,
    startup_session_id: str | None = None,
    include_spatial_box: bool = False,
) -> dict:
    task_id = task_data.get("task_id")
    instance = {
        "model_key": _build_model_key(task_id, fbx_url),
        "task_id": task_id,
        "fbx_url": fbx_url,
    }
    _append_pose_fields(
        instance,
        task_json,
        task_data=task_data,
        startup_session_id=startup_session_id,
    )
    if include_spatial_box:
        spatial_box = _public_sam3_spatial_box(task_json.get("Sam3SpatialBox"))
        if spatial_box:
            instance["sam3_spatial_box"] = spatial_box
    _append_display_identity_fields(instance, task_json)
    identity = _display_identity_from_task_json(task_json)
    if identity.get("model_revision") is not None:
        instance["model_revision"] = int(identity.get("model_revision") or 0)
    if identity.get("hololens_pose_revision") is not None:
        instance["hololens_pose_revision"] = int(identity.get("hololens_pose_revision") or 0)
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
        response["model_instance"] = _build_model_instance(
            task_data,
            task_json,
            "",
            startup_session_id=startup_session_id,
            include_spatial_box=True,
        )

    return response


def _resolve_placement_status(task_data: dict, task_json: dict) -> str:
    if bool(task_data.get("aruco_coordinate_synced")) and task_json.get("object_aruco"):
        return "aruco_synced"
    if task_json.get("object_hololens_current"):
        return "hololens_local"
    return "missing_pose"


def _aabb_corners_from_min_max(min_corner: list | None, max_corner: list | None) -> list[list[float]] | None:
    if min_corner is None or max_corner is None:
        return None
    try:
        a = np.asarray(min_corner, dtype=np.float64).reshape(3)
        b = np.asarray(max_corner, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return [[float(x), float(y), float(z)] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])]


def _hololens_bounds_from_decoded(
    decoded: dict,
    task_json: dict,
    *,
    task_data: dict | None = None,
    startup_session_id: str | None = None,
) -> dict:
    response_startup_session_id = _response_startup_session_id(task_data, task_json, startup_session_id)
    aruco_reference = _load_latest_aruco_reference_pose(response_startup_session_id)
    if aruco_reference is None:
        return {}

    corners = decoded.get("corners_aruco") or _aabb_corners_from_min_max(
        decoded.get("aabb_min_aruco"),
        decoded.get("aabb_max_aruco"),
    )
    if not corners:
        return {}
    try:
        points = aruco_points_to_hololens(corners, aruco_reference)
    except Exception as exc:
        print(f"[WARN] bounds response conversion failed for startup={response_startup_session_id}: {exc}")
        return {}
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    return {
        "coordinate_space": "hololens_current_local",
        "aabb_min_hololens": [float(v) for v in min_corner],
        "aabb_max_hololens": [float(v) for v in max_corner],
        "corners_hololens": points.astype(float).tolist(),
    }


def _build_task_model_bounds_status(
    task_id: str,
    task_json: dict,
    *,
    task_data: dict | None = None,
    startup_session_id: str | None = None,
) -> dict:
    row = get_model_bounds_by_task_id(task_id) if task_id else None
    if row:
        decoded = decode_model_bounds_row(row)
        payload = {
            "status": decoded.get("status") or "missing",
            "coordinate_space": "hololens_current_local",
            "error_message": decoded.get("error_message"),
        }
        payload.update(
            _hololens_bounds_from_decoded(
                decoded,
                task_json,
                task_data=task_data,
                startup_session_id=startup_session_id,
            )
        )
        return payload

    model_bounds = task_json.get("ModelBounds")
    if isinstance(model_bounds, dict):
        payload = {
            "status": model_bounds.get("status") or "missing",
            "coordinate_space": "hololens_current_local",
            "error_message": model_bounds.get("error_message"),
        }
        payload.update(
            _hololens_bounds_from_decoded(
                {
                    "aabb_min_aruco": model_bounds.get("aabb_min_aruco"),
                    "aabb_max_aruco": model_bounds.get("aabb_max_aruco"),
                    "corners_aruco": model_bounds.get("corners_aruco"),
                },
                task_json,
                task_data=task_data,
                startup_session_id=startup_session_id,
            )
        )
        return payload

    if not task_json.get("object_aruco"):
        return {
            "status": "pending_reference",
            "coordinate_space": "hololens_current_local",
            "error_message": "object ArUco pose is not available on the server yet",
        }

    return {"status": "missing", "coordinate_space": "hololens_current_local"}


def _source_url_if_present(host: str, task_id: str, source, filename: str | None, file_path) -> str | None:
    if not filename or not file_path or not file_path.exists():
        return None
    return _model_file_url(host, task_id, source, filename)


def _build_bounds_download_urls(task_id: str, task_json: dict, fbx_name: str | None) -> dict:
    urls = {}
    host = request.host_url.rstrip("/")

    try:
        generated_source = resolve_model_generation_source(task_json, require_mtl_image=False)
    except Exception:
        generated_source = None
    try:
        runtime_source = resolve_runtime_mesh_source(task_json, require_mtl_image=False)
    except Exception:
        runtime_source = None

    entries = []
    if generated_source is not None:
        entries.extend(
            [
                ("mesh", generated_source, generated_source.mesh, generated_source.mesh_path),
                ("mtl", generated_source, generated_source.mtl, generated_source.mtl_path),
                ("image", generated_source, generated_source.image, generated_source.image_path),
            ]
        )
    if runtime_source is not None:
        entries.extend(
            [
                ("runtime_mesh", runtime_source, runtime_source.mesh, runtime_source.mesh_path),
                ("runtime_mtl", runtime_source, runtime_source.mtl, runtime_source.mtl_path),
                ("runtime_image", runtime_source, runtime_source.image, runtime_source.image_path),
            ]
        )

    for key, source, filename, file_path in entries:
        url = _source_url_if_present(host, task_id, source, filename, file_path)
        if url:
            urls[key] = url

    blender_info = task_json.get("Blender") or {}
    task_timestamp = str(task_json.get("task_timestamp") or "").strip()
    if fbx_name and blender_info.get("artifact_root") == "model_result" and task_timestamp:
        fbx_path = model_result_dir(task_timestamp) / fbx_name
        if fbx_path.exists():
            urls["fbx"] = _task_artifact_url(host, task_id, "result", fbx_name)

    return urls

def _build_model_bounds_response(
    row: dict,
    hit_result: dict | None = None,
    *,
    startup_session_id: str | None = None,
) -> dict:
    decoded = decode_model_bounds_row(row)
    task_id = str(decoded.get("task_id") or "")
    task_data = get_task(task_id) if task_id else None
    task_json = (task_data or {}).get("task_json") or {}
    fbx_name = decoded.get("fbx_name") or (task_json.get("Blender") or {}).get("fbx")
    download_urls = _build_bounds_download_urls(task_id, task_json, fbx_name)
    fbx_url = download_urls.get("fbx")

    model = {
        "id": decoded.get("id"),
        "task_id": task_id,
        "status": decoded.get("status"),
        "model_name": decoded.get("model_name"),
        "fbx_name": fbx_name,
        "uploaded_at": decoded.get("uploaded_at"),
        "coordinate_space": "hololens_current_local",
        "source_model_path": decoded.get("source_model_path"),
        "error_message": decoded.get("error_message"),
        "download_urls": download_urls,
    }
    model.update(
        _hololens_bounds_from_decoded(
            decoded,
            task_json,
            task_data=task_data,
            startup_session_id=startup_session_id,
        )
    )
    _append_pose_fields(
        model,
        task_json,
        task_data=task_data,
        startup_session_id=startup_session_id,
    )
    _append_display_identity_fields(model, task_json)
    if fbx_url:
        model["fbx_url"] = fbx_url
        model["model_instance"] = _build_model_instance(
            task_data or {"task_id": task_id},
            task_json,
            fbx_url,
            startup_session_id=startup_session_id,
        )
    if hit_result:
        model["hit_distance_m"] = hit_result.get("hit_distance_m")
        if hit_result.get("hit_point_hololens") is not None:
            model["hit_point_hololens"] = hit_result.get("hit_point_hololens")

    return model


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
    task_id = str(task_data.get("task_id") or "")
    response["placement_status"] = _resolve_placement_status(task_data, task_json)
    response["model_bounds"] = _build_task_model_bounds_status(
        task_id,
        task_json,
        task_data=task_data,
        startup_session_id=startup_session_id,
    )
    response["model_generation"] = task_json.get("ModelGeneration") or None
    response["display_identity"] = task_json.get("DisplayIdentity") or None
    if response["display_identity"]:
        response["display_object_id"] = response["display_identity"].get("display_object_id")
        response["capture_instance_id"] = response["display_identity"].get("capture_instance_id")
        response["model_revision"] = int(response["display_identity"].get("model_revision") or 0)
        response["hololens_pose_revision"] = int(response["display_identity"].get("hololens_pose_revision") or 0)
    response_startup_session_id = _response_startup_session_id(
        task_data,
        task_json,
        startup_session_id,
    )
    response["history_placement_restoration"] = _public_spatial_payload(
        task_json.get("HistoryPlacementRestoration") or None,
        startup_session_id=response_startup_session_id,
    )
    auxiliary_outputs = task_data.get("auxiliary_outputs") if isinstance(task_data.get("auxiliary_outputs"), dict) else {}
    contact_body_branch = (
        auxiliary_outputs.get("shigure_contact_body")
        if isinstance(auxiliary_outputs.get("shigure_contact_body"), dict)
        else {}
    )
    response["taken_object_detection"] = _public_spatial_payload(
        contact_body_branch.get("TakenObjectDetection") or task_json.get("TakenObjectDetection") or None,
        startup_session_id=response_startup_session_id,
    )
    response["sam3d_body_mesh"] = _public_spatial_payload(
        contact_body_branch.get("SAM3DBodyMesh") or task_json.get("SAM3DBodyMesh") or None,
        startup_session_id=response_startup_session_id,
    )
    response["auxiliary_jobs"] = task_data.get("auxiliary_jobs") or []
    try:
        generated_source = resolve_model_generation_source(task_json, require_mtl_image=True)
    except Exception as exc:
        response["error"] = str(exc)
        return response

    try:
        runtime_source = resolve_runtime_mesh_source(task_json, require_mtl_image=True)
    except Exception:
        runtime_source = None
    blender_info = task_json.get("Blender") or {}
    fbx_name = blender_info.get("fbx")
    if fbx_name and blender_info.get("artifact_root") == "model_result" and task_json.get("task_timestamp"):
        fbx_path = model_result_dir(str(task_json.get("task_timestamp"))) / fbx_name
    else:
        fbx_path = None

    _append_pose_fields(
        response,
        task_json,
        task_data=task_data,
        startup_session_id=startup_session_id,
    )

    if not generated_source.mesh_path.exists():
        response["error"] = f"{generated_source.source_stage} obj not found on disk"
        return response

    if not generated_source.mtl_path or not generated_source.mtl_path.exists():
        response["error"] = f"{generated_source.source_stage} mtl not found on disk"
        return response

    if not generated_source.image_path or not generated_source.image_path.exists():
        response["error"] = f"{generated_source.source_stage} image not found on disk"
        return response

    host = (host_override or request.host_url).rstrip("/")
    response.update(
        {
            "mesh_url": _model_file_url(host, task_id, generated_source, generated_source.mesh),
            "mtl_url": _model_file_url(host, task_id, generated_source, generated_source.mtl),
            "image_url": _model_file_url(host, task_id, generated_source, generated_source.image),
        }
    )
    if generated_source.video and generated_source.video_path and generated_source.video_path.exists():
        response["video_url"] = _task_artifact_url(host, task_id, "debug", generated_source.video)

    taken_payload = response.get("taken_object_detection") if isinstance(response.get("taken_object_detection"), dict) else {}
    if (taken_payload or {}).get("artifact_root") == "model_result" and (taken_payload or {}).get("status") == "TAKEN":
        taken_urls = {}
        for payload_key, url_key in (
            ("result_rgb", "result_rgb_url"),
            ("result_depth", "result_depth_url"),
            ("camera_info", "camera_info_url"),
            ("active_objects", "active_objects_url"),
            ("marker_pose", "marker_pose_url"),
        ):
            url = _task_artifact_url(host, task_id, "result", taken_payload.get(payload_key))
            if url:
                taken_urls[url_key] = url
        if taken_urls:
            response["taken_object_detection_urls"] = taken_urls

    body_payload = response.get("sam3d_body_mesh") if isinstance(response.get("sam3d_body_mesh"), dict) else {}
    if (body_payload or {}).get("selected_person_fbx_folder") == "model_result" and (body_payload or {}).get("status") == "SUCCESS":
        body_urls = {}
        for payload_key, url_key in (
            ("selected_person_fbx_path", "selected_person_fbx_url"),
            ("selected_person_obj_path", "selected_person_obj_url"),
            ("people_json_path", "people_url"),
            ("subject_crop_path", "subject_crop_url"),
        ):
            value = str(body_payload.get(payload_key) or "").strip()
            if value:
                url = _task_artifact_url(host, task_id, "result", value.rsplit("/", 1)[-1])
                if url:
                    body_urls[url_key] = url
        if body_urls:
            response["sam3d_body_mesh_urls"] = body_urls
    if (
        runtime_source is not None
        and runtime_source.mtl_path is not None
        and runtime_source.image_path is not None
        and runtime_source.mesh_path.exists()
        and runtime_source.mtl_path.exists()
        and runtime_source.image_path.exists()
    ):
        response.update(
            {
                "runtime_mesh_url": _model_file_url(host, task_id, runtime_source, runtime_source.mesh),
                "runtime_mtl_url": _model_file_url(host, task_id, runtime_source, runtime_source.mtl),
                "runtime_image_url": _model_file_url(host, task_id, runtime_source, runtime_source.image),
                "runtime_mesh": runtime_source.payload,
            }
        )
    if fbx_path and fbx_path.exists():
        fbx_url = _task_artifact_url(host, task_id, "result", fbx_name)
        response["fbx_url"] = fbx_url
        response["model_instance"] = _build_model_instance(
            task_data,
            task_json,
            fbx_url,
            startup_session_id=startup_session_id,
        )

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
        "retro_synced_completed_task_count": _safe_int(
            aruco_stage.get("retro_synced_completed_task_count") or 0
        ),
        "coordinate_handling": "server_only_hololens_local_payloads",
    }
    latest_completed_model = get_latest_completed_task_data(
        startup_session_id=startup_session_id,
        require_aruco_coordinate_synced=True,
        history_offset=0,
        attempt_sync=False,
    )
    response["latest_completed_model_available"] = latest_completed_model is not None
    if latest_completed_model:
        response["latest_completed_model_task_id"] = latest_completed_model.get("task_id")
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
    return jsonify(
        {
            "status": "running",
            "endpoints": [
                "/generate",
                "/check-queue",
                "/latest-completed-task-ids?limit=5",
                "/model-bounds/latest?limit=5",
                "/model-bounds/latest?limit=5&include_duplicate_captures=1",
                "/model-bounds/range?start=<uploaded_at>&end=<uploaded_at>",
                "/spatial-query/ray",
                "/spatial-query/ray-range",
                "/history-placement-restoration/start",
                "/aruco/latest-reference?startup_session_id=<startup_session_id>",
                "/aruco/markers",
                "/aruco/markers/sync",
            ],
        }
    )


@app.route("/task-artifacts/<task_id>/<area>/<path:filename>", methods=["GET"], strict_slashes=False)
def serve_task_artifact(task_id: str, area: str, filename: str):
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
        print("[RECV] form keys:", list(request.form.keys()))
        print("[RECV] files:", list(request.files.keys()))

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
        devj = _parse_json_field("deviceJ")
        force_new_3d_model = _is_truthy_query_value(
            request.form.get("force_new_3d_model")
            or request.form.get("forceNew3DModel")
            or devj.get("force_new_3d_model")
            or devj.get("forceNew3DModel")
        )
        pv_frames_input = _parse_optional_json_array_field("PVCameraFramesJ")

        if purpose == PURPOSE_OBJECT_RECONSTRUCTION or pv_frames_input is None:
            pvj = _parse_json_field("PVCameraJ")
            pv_frames_input = [pvj]
        else:
            if not pv_frames_input:
                raise ValueError("PVCameraFramesJ must contain at least one frame")
            pvj = pv_frames_input[0] if pv_frames_input else None
            if not isinstance(pvj, dict):
                raise ValueError("PVCameraFramesJ must contain JSON objects")

        normalized_pv_frames = []
        for index, frame in enumerate(pv_frames_input):
            if not isinstance(frame, dict):
                raise ValueError(f"PVCameraFramesJ[{index}] must be a JSON object")
            frame_pose = frame.get("pose")
            frame_position, frame_rotation_quaternion_xyzw = _extract_unity_pv_pose_components(frame_pose)
            normalized_pv_frames.append(
                {
                    "frame_index": int(frame.get("index", index)),
                    "width": frame.get("width", 0),
                    "height": frame.get("height", 0),
                    "k": frame.get("k"),
                    "pose": frame_pose,
                    "position": frame_position,
                    "rotation_quaternion_xyzw": frame_rotation_quaternion_xyzw,
                    "time": frame.get("time", ""),
                    "device_pose": frame.get("device_pose"),
                    "device_rotation": frame.get("device_rotation"),
                }
            )
        if not normalized_pv_frames:
            raise ValueError("at least one PV camera frame is required")

        dj = None
        sbj = None
        top_left = None
        bottom_right = None
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            dj = _parse_json_field("DepthCameraJ")
            sbj = _parse_json_field("SelectionBoxJ")

            requested_sensor = normalize_depth_sensor_name(dj.get("sensor") or DEPTH_SENSOR_AHAT)

            top_left = sbj.get("top_left")
            bottom_right = sbj.get("bottom_right")

            if not (isinstance(top_left, list) and len(top_left) == 2):
                raise ValueError("SelectionBoxJ.top_left must be a list of length 2")

            if not (isinstance(bottom_right, list) and len(bottom_right) == 2):
                raise ValueError("SelectionBoxJ.bottom_right must be a list of length 2")

        now_utc = datetime.now(timezone.utc)
        server_received_utc = now_utc.isoformat().replace("+00:00", "Z")
        base = make_timestamp(now_utc)
        reserved_task_id = None

        startup_session_id = str(devj.get("startup_session_id") or "").strip() or None
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
            if field_name not in request.files and index == 0:
                field_name = "pv_image"
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
            "artifact_schema_version": 1,
            "purpose": purpose,
            "device": {
                "type": devj.get("type", ""),
                "ip": devj.get("ip", ""),
                "time": devj.get("time", ""),
                "pose": devj.get("pose"),
                "startup_session_id": devj.get("startup_session_id", ""),
            },
            "PVCamera": {
                "name": str(color_path.name) if color_path else normalized_pv_frames[0].get("name"),
                "width": normalized_pv_frames[0].get("width", 0),
                "height": normalized_pv_frames[0].get("height", 0),
                "k": normalized_pv_frames[0].get("k"),
                "pose": normalized_pv_frames[0].get("pose"),
                "position": normalized_pv_frames[0].get("position"),
                "rotation_quaternion_xyzw": normalized_pv_frames[0].get("rotation_quaternion_xyzw"),
            },
            "PVCameraFrames": normalized_pv_frames,
        }
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
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


def _model_is_ready_for_runtime_download(task_data: dict) -> bool:
    task_json = task_data.get("task_json") or {}
    if (task_json.get("purpose") or PURPOSE_OBJECT_RECONSTRUCTION) != PURPOSE_OBJECT_RECONSTRUCTION:
        return False
    status = str(task_data.get("status") or "")
    if status in {"pending", "uploading", "upload_failed", "failed", "completed", "aruco_completed"}:
        return False
    try:
        if STAGE_ORDER.index(status) < STAGE_ORDER.index("model_bounds"):
            return False
    except ValueError:
        return False
    blender_info = task_json.get("Blender") or {}
    fbx_name = str(blender_info.get("fbx") or "").strip()
    if not fbx_name or blender_info.get("artifact_root") != "model_result":
        return False
    task_timestamp = str(task_json.get("task_timestamp") or task_data.get("task_timestamp") or "").strip()
    if not task_timestamp:
        return False
    if not (model_result_dir(task_timestamp) / fbx_name).exists():
        return False
    return bool(task_json.get("object_hololens_current") or task_json.get("object_aruco"))


def _build_model_ready_task_response(task_data: dict, *, startup_session_id: str | None = None) -> dict:
    response = _build_completed_task_response(task_data, startup_session_id=startup_session_id)
    response["status"] = "model_ready"
    response["terminal"] = False
    response["model_ready"] = True
    return response


@app.route("/check-queue", methods=["POST"], strict_slashes=False)
def check_task_queue():
    try:
        payload = request.get_json(silent=True) or {}
        task_ids = payload.get("task_ids")
        client_startup_session_id = str(payload.get("startup_session_id") or "").strip() or None
        if not isinstance(task_ids, list):
            return jsonify({"error": "task_ids must be a JSON array"}), 400
        if not client_startup_session_id:
            return jsonify({"error": "startup_session_id is required"}), 400

        pending = []
        seen = set()
        for raw_task_id in task_ids:
            task_id = str(raw_task_id or "").strip()
            if not task_id or task_id in seen:
                continue
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
            elif status == "failed":
                response = {
                    "status": "failed",
                    "task_id": task_data.get("task_id"),
                    "purpose": purpose,
                    "terminal": True,
                    "error": task_data.get("error_message") or "Unknown error",
                    "stage_runs": task_data.get("stage_runs") or [],
                    "timing_events": task_data.get("timing_events") or [],
                    "ai_model_timings": task_data.get("ai_model_timings") or [],
                }
            else:
                if _model_is_ready_for_runtime_download(task_data):
                    response = _build_model_ready_task_response(
                        task_data,
                        startup_session_id=client_startup_session_id,
                    )
                    return jsonify(
                        {
                            "ready": True,
                            "task_id": task_id,
                            "status": "model_ready",
                            "purpose": purpose,
                            "terminal": False,
                            "task": response,
                        }
                    )
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
    except Exception as exc:
        print(f"Error in check_task_queue: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/aruco/latest-reference", methods=["GET"], strict_slashes=False)
def latest_aruco_reference():
    try:
        startup_session_id = str(request.args.get("startup_session_id") or "").strip()
        if not startup_session_id:
            return jsonify({"error": "Missing startup_session_id parameter"}), 400

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
    except Exception as exc:
        print(f"Error in latest_aruco_reference: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/aruco/markers", methods=["GET"], strict_slashes=False)
def list_aruco_markers():
    try:
        return jsonify({"markers": get_enabled_aruco_markers()})
    except Exception as exc:
        print(f"Error in list_aruco_markers: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/aruco/markers/sync", methods=["POST"], strict_slashes=False)
def sync_aruco_markers():
    try:
        synced_count = sync_marker_registry_from_reference_folder()
        return jsonify({"synced_count": synced_count, "markers": get_enabled_aruco_markers()})
    except Exception as exc:
        print(f"Error in sync_aruco_markers: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/latest-completed-task-ids", methods=["GET"], strict_slashes=False)
def latest_completed_task_ids():
    try:
        startup_session_id = str(request.args.get("startup_session_id") or "").strip() or None
        limit = max(1, request.args.get("limit", default=5, type=int) or 5)
        require_aruco_coordinate_synced = _is_truthy_query_value(
            request.args.get("require_aruco_coordinate_synced")
        )
        scan_limit = _history_model_selection_scan_limit(limit)
        rows = get_latest_completed_tasks(
            startup_session_id=startup_session_id,
            require_aruco_coordinate_synced=require_aruco_coordinate_synced,
            limit=scan_limit,
        )
        selected_rows, skipped_rows = _select_latest_hololens_uploaded_model_tasks(rows, limit=limit)
        return jsonify(
            {
                "success": True,
                "count": len(selected_rows),
                "raw_count": len(rows),
                "skipped_count": len(skipped_rows),
                "deduped_by_display_object": True,
                "task_ids": [
                    {
                        "task_id": row.get("task_id"),
                        "purpose": PURPOSE_OBJECT_RECONSTRUCTION,
                        "status": row.get("status"),
                        "aruco_coordinate_synced": bool(row.get("aruco_coordinate_synced")),
                        "display_object_id": row.get("selection_display_object_id"),
                    }
                    for row in selected_rows
                    if row.get("task_id")
                ],
            }
        )
    except Exception as exc:
        print(f"Error in latest_completed_task_ids: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500



def _prepare_spatial_query_payload(payload: dict) -> tuple[dict, str | None]:
    startup_session_id = str(
        payload.get("startup_session_id")
        or request.args.get("startup_session_id")
        or ""
    ).strip() or None
    if payload.get("origin_hololens") is None or payload.get("direction_hololens") is None:
        raise ValueError("origin_hololens and direction_hololens are required for public spatial queries")
    if not startup_session_id:
        raise ValueError("startup_session_id is required for HoloLens-coordinate spatial queries")
    aruco_reference = _load_latest_aruco_reference_pose(startup_session_id)
    if aruco_reference is None:
        raise ValueError("No ArUco reference found for this startup session")
    converted = dict(payload)
    converted["aruco_reference"] = aruco_reference
    return converted, startup_session_id

@app.route("/model-bounds/latest", methods=["GET"], strict_slashes=False)
def model_bounds_latest():
    try:
        limit = max(1, request.args.get("limit", default=5, type=int) or 5)
        include_duplicates = _include_duplicate_captures_requested()
        startup_session_id = str(request.args.get("startup_session_id") or "").strip() or None
        if not startup_session_id:
            return jsonify({"success": False, "error": "startup_session_id is required"}), 400
        fetch_limit = limit if include_duplicates else min(max(limit * 4, limit), 50)
        rows = get_latest_ready_model_bounds(fetch_limit)
        models = [
            _build_model_bounds_response(row, startup_session_id=startup_session_id)
            for row in rows
        ]
        bounds = models if include_duplicates else _dedupe_display_object_models(models, limit=limit)
        return jsonify(
            {
                "success": True,
                "count": len(bounds),
                "raw_count": len(models),
                "deduped_by_display_object": not include_duplicates,
                "bounds": bounds,
            }
        )
    except Exception as exc:
        print(f"Error in model_bounds_latest: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/model-bounds/range", methods=["GET"], strict_slashes=False)
def model_bounds_range():
    try:
        start = str(request.args.get("start") or "").strip()
        end = str(request.args.get("end") or "").strip()
        limit = max(1, request.args.get("limit", default=50, type=int) or 50)
        include_duplicates = _include_duplicate_captures_requested()
        startup_session_id = str(request.args.get("startup_session_id") or "").strip() or None
        if not startup_session_id:
            return jsonify({"success": False, "error": "startup_session_id is required"}), 400
        rows = get_ready_model_bounds_in_range(start, end, limit=limit)
        models = [
            _build_model_bounds_response(row, startup_session_id=startup_session_id)
            for row in rows
        ]
        bounds = models if include_duplicates else _dedupe_display_object_models(models, limit=limit)
        return jsonify(
            {
                "success": True,
                "count": len(bounds),
                "raw_count": len(models),
                "deduped_by_display_object": not include_duplicates,
                "start": start,
                "end": end,
                "bounds": bounds,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in model_bounds_range: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/spatial-query/ray", methods=["POST"], strict_slashes=False)
def spatial_query_ray():
    try:
        payload = request.get_json(silent=True) or {}
        query_payload, startup_session_id = _prepare_spatial_query_payload(payload)
        _rows, result = latest_bounds_for_ray(query_payload)
        if not result.get("hit"):
            return jsonify(
                {
                    "success": True,
                    "hit": False,
                    "candidates_checked": result.get("candidates_checked", 0),
                    "max_distance_m": result.get("max_distance_m"),
                    "coordinate_space": result.get("coordinate_space"),
                }
            )

        return jsonify(
            {
                "success": True,
                "hit": True,
                "candidates_checked": result.get("candidates_checked", 0),
                "coordinate_space": result.get("coordinate_space"),
                "hit_point_hololens": result.get("hit_point_hololens"),
                "model": _build_model_bounds_response(
                    result["row"],
                    result,
                    startup_session_id=startup_session_id,
                ),
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in spatial_query_ray: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/spatial-query/ray-range", methods=["POST"], strict_slashes=False)
def spatial_query_ray_range():
    try:
        payload = request.get_json(silent=True) or {}
        query_payload, startup_session_id = _prepare_spatial_query_payload(payload)
        _rows, result = range_bounds_for_ray(query_payload)
        if not result.get("hit"):
            return jsonify(
                {
                    "success": True,
                    "hit": False,
                    "candidates_checked": result.get("candidates_checked", 0),
                    "max_distance_m": result.get("max_distance_m"),
                    "coordinate_space": result.get("coordinate_space"),
                }
            )

        return jsonify(
            {
                "success": True,
                "hit": True,
                "candidates_checked": result.get("candidates_checked", 0),
                "coordinate_space": result.get("coordinate_space"),
                "hit_point_hololens": result.get("hit_point_hololens"),
                "model": _build_model_bounds_response(
                    result["row"],
                    result,
                    startup_session_id=startup_session_id,
                ),
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in spatial_query_ray_range: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500



@app.route("/history-placement-restoration/start", methods=["POST"], strict_slashes=False)
def history_placement_restoration_start():
    request_id = str(uuid.uuid4())
    request_timestamp = make_timestamp()
    try:
        payload = request.get_json(silent=True) or {}
        task_id = str(payload.get("task_id") or "").strip()
        startup_session_id = str(payload.get("startup_session_id") or "").strip() or None
        if not startup_session_id:
            return jsonify({"success": False, "error": "startup_session_id is required"}), 400
        target_time = str(payload.get("target_time") or "").strip() or None
        raw_limit = payload.get("model_limit", payload.get("limit", history_placement_settings.DEFAULT_MODEL_LIMIT))
        try:
            model_limit = int(raw_limit)
        except Exception:
            model_limit = history_placement_settings.DEFAULT_MODEL_LIMIT
        model_limit = max(0, model_limit)

        create_history_placement_request(
            request_id=request_id,
            request_timestamp=request_timestamp,
            startup_session_id=startup_session_id,
            target_time=target_time,
            model_limit=model_limit,
        )
        ensure_history_request_dirs(request_timestamp)
        request_worker_dir = history_request_worker_dir(request_timestamp)
        request_result_dir = history_request_result_dir(request_timestamp)
        request_worker_dir.mkdir(parents=True, exist_ok=True)
        request_result_dir.mkdir(parents=True, exist_ok=True)
        save_task_json(
            request_worker_dir / "01_request.json",
            {
                "request_id": request_id,
                "request_timestamp": request_timestamp,
                "task_id": task_id or None,
                "startup_session_id": startup_session_id,
                "target_time": target_time,
                "model_limit": model_limit,
                "source": "api_button",
            },
        )

        selection_skipped: list[dict] = []
        if task_id:
            task_data = get_task(task_id)
            if not task_data:
                update_history_placement_request(request_id, status="failed", error_message="task_id not found")
                return jsonify({"success": False, "error": "task_id not found", "task_id": task_id}), 404
            rows, selection_skipped = _select_latest_hololens_uploaded_model_tasks([task_data], limit=1)
            if not rows:
                reason = (selection_skipped[0] or {}).get("reason") if selection_skipped else "not_trackable"
                update_history_placement_request(request_id, status="failed", error_message=str(reason))
                return jsonify({"success": False, "error": str(reason), "task_id": task_id}), 400
        else:
            scan_limit = _history_model_selection_scan_limit(model_limit)
            raw_rows = get_latest_completed_tasks(
                startup_session_id=None,
                require_aruco_coordinate_synced=True,
                limit=scan_limit,
            )
            rows, selection_skipped = _select_latest_hololens_uploaded_model_tasks(
                raw_rows,
                limit=None if model_limit <= 0 else model_limit,
            )
            if startup_session_id:
                selection_skipped = [
                    {
                        "reason": "startup_session_not_used_for_history_model_selection",
                        "startup_session_id": startup_session_id,
                        "selection_scope": "global_latest_completed_per_display_object",
                    }
                ] + selection_skipped

        selection_policy = (
            "specific_task"
            if task_id
            else "hololens_uploaded_latest_completed_per_display_object_global_across_startup_sessions"
        )
        selected_tasks = [
            {
                "item_index": index,
                "task_id": row.get("task_id"),
                "task_timestamp": row.get("task_timestamp"),
                "status": row.get("status"),
                "display_object_id": row.get("selection_display_object_id"),
                "selection_key": row.get("selection_key"),
                "selection_reason": selection_policy,
            }
            for index, row in enumerate(rows)
        ]
        save_task_json(
            request_worker_dir / "01_selected_model_tasks.json",
            {
                "items": selected_tasks,
                "skipped": selection_skipped,
                "selection_policy": selection_policy,
            },
        )

        shared_history_context = {"_current_lock": RLock(), "_cache_lock": RLock()}
        configured_parallel_workers = max(1, int(getattr(history_placement_settings, "PARALLEL_WORKERS", 8) or 1))
        max_workers = max(1, min(len(rows), configured_parallel_workers, 8))
        save_task_json(
            request_worker_dir / "02_parallel_execution.json",
            {
                "selected_count": len(rows),
                "configured_parallel_workers": configured_parallel_workers,
                "max_total_workers_cap": 8,
                "max_workers": max_workers,
                "schedule": "item_worker_threads_plus_main_thread_shared_current_prewarm",
            },
        )

        request_host = request.host_url.rstrip("/")

        def _run_history_item(index: int, row: dict) -> tuple[int, dict]:
            row_task_id = str(row.get("task_id") or "")
            try:
                json_path = resolve_task_json_path_from_record(row)
            except Exception as exc:
                error_payload = {"item_index": index, "task_id": row_task_id, "success": False, "error": str(exc)}
                save_task_json(request_result_dir / f"02_result_{index:03d}_summary.json", error_payload)
                return index, error_payload
            try:
                item_work_dir = request_worker_dir / f"02_result_{index:03d}_working"
                item_work_dir.mkdir(parents=True, exist_ok=True)
                result = run_history_placement_restoration(
                    json_path,
                    target_time=target_time,
                    request_source="api_button",
                    artifact_output_dir=item_work_dir,
                    shared_current_context=shared_history_context,
                )
                save_task_json(
                    request_worker_dir / f"02_result_{index:03d}_working_state.json",
                    {
                        "item_index": index,
                        "task_id": row_task_id,
                        "task_timestamp": row.get("task_timestamp"),
                        "artifact_output_dir": str(item_work_dir),
                        "status": result.get("status"),
                        "parallel_max_workers": max_workers,
                    },
                )
                result_payload = {
                    "item_index": index,
                    "task_id": row_task_id,
                    "task_timestamp": row.get("task_timestamp"),
                    "success": True,
                    "status": result.get("status"),
                    "history_placement_restoration": _public_spatial_payload(
                        result.get("payload"),
                        startup_session_id=startup_session_id,
                    ),
                }
                task_response = get_task(row_task_id) if row_task_id else None
                if task_response:
                    model_payload = _build_completed_task_response(
                        task_response,
                        host_override=request_host,
                        startup_session_id=startup_session_id,
                    )
                    for key in (
                        "fbx_url",
                        "model_instance",
                        "object_hololens_current",
                        "object_hololens_original",
                        "coordinate_space",
                        "taken_object_detection",
                        "taken_object_detection_urls",
                        "sam3d_body_mesh",
                        "sam3d_body_mesh_urls",
                    ):
                        if model_payload.get(key) is not None:
                            result_payload[key] = model_payload.get(key)
                    if model_payload.get("error"):
                        result_payload["model_payload_error"] = model_payload.get("error")
                save_task_json(request_result_dir / f"02_result_{index:03d}_summary.json", result_payload)
                return index, result_payload
            except Exception as exc:
                error_payload = {"item_index": index, "task_id": row_task_id, "success": False, "error": str(exc)}
                save_task_json(request_result_dir / f"02_result_{index:03d}_summary.json", error_payload)
                return index, error_payload

        results_by_index = {}
        prewarm_json_path = None
        prewarm_error = None
        for row in rows:
            try:
                prewarm_json_path = resolve_task_json_path_from_record(row)
                break
            except Exception as exc:
                prewarm_error = str(exc)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_history_item, index, row): index for index, row in enumerate(rows)}
            if prewarm_json_path is not None:
                try:
                    prewarm_payload = prepare_shared_current_context(
                        prewarm_json_path,
                        target_time=target_time,
                        shared_current_context=shared_history_context,
                    )
                except Exception as exc:
                    prewarm_payload = {"success": False, "error": str(exc)}
            else:
                prewarm_payload = {"success": False, "error": prewarm_error or "no_resolvable_task_json"}
            save_task_json(request_worker_dir / "02_shared_current_prewarm.json", prewarm_payload)

            for future in as_completed(futures):
                index = futures[future]
                try:
                    result_index, result_payload = future.result()
                except Exception as exc:
                    row = rows[index]
                    row_task_id = str(row.get("task_id") or "")
                    result_index = index
                    result_payload = {"item_index": index, "task_id": row_task_id, "success": False, "error": str(exc)}
                    save_task_json(request_result_dir / f"02_result_{index:03d}_summary.json", result_payload)
                results_by_index[result_index] = result_payload

        results = [results_by_index[index] for index in range(len(rows)) if index in results_by_index]

        response_payload = {
            "success": True,
            "request_id": request_id,
            "request_timestamp": request_timestamp,
            "count": len(results),
            "model_limit": model_limit,
            "unlimited": model_limit == 0,
            "results": results,
        }
        save_task_json(request_result_dir / "02_response.json", response_payload)
        save_task_json(request_result_dir / "02_unity_display.json", response_payload)
        update_history_placement_request(
            request_id,
            status="completed",
            selected_task_count=len(rows),
            result_count=len(results),
        )
        return jsonify(response_payload)
    except Exception as exc:
        try:
            update_history_placement_request(request_id, status="failed", error_message=str(exc))
        except Exception:
            pass
        print(f"Error in history_placement_restoration_start: {exc}")
        return jsonify({"success": False, "request_id": request_id, "error": str(exc)}), 500


@app.route("/history-placement-restoration/latest", methods=["GET"], strict_slashes=False)
def history_placement_restoration_latest():
    try:
        latest = get_latest_history_placement_request("completed")
        if not latest:
            return jsonify({"success": False, "error": "history_placement_request_not_found"}), 404
        request_timestamp = str(latest.get("request_timestamp") or "").strip()
        response_path = history_request_result_dir(request_timestamp) / "02_response.json"
        if not response_path.is_file():
            return jsonify({"success": False, "error": "history_placement_response_missing"}), 404
        with response_path.open("r", encoding="utf-8") as file:
            return jsonify(json.load(file))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


def _json_column(value):
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _backfill_display_object_states() -> None:
    """Populate revision state for models created before the new registry."""

    rows = get_latest_completed_tasks(limit=50)
    for row in reversed(rows):
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        task_data = get_task(task_id)
        task_json = (task_data or {}).get("task_json") or {}
        identity = task_json.get("DisplayIdentity") if isinstance(task_json.get("DisplayIdentity"), dict) else {}
        display_object_id = str(identity.get("display_object_id") or task_json.get("display_object_id") or "").strip()
        pose_aruco = task_json.get("object_aruco") if isinstance(task_json.get("object_aruco"), dict) else None
        if not display_object_id or pose_aruco is None:
            continue
        historical = task_json.get("HistoricalModelMatch") if isinstance(task_json.get("HistoricalModelMatch"), dict) else {}
        reused = bool(historical.get("reuse_model"))
        selected_model_task_id = str(historical.get("selected_model_task_id") or "").strip() or None
        try:
            commit_display_object_capture_state(
                display_object_id=display_object_id,
                capture_task_id=task_id,
                pose_aruco=pose_aruco,
                captured_at=str(task_json.get("server_received_utc") or row.get("created_at") or ""),
                generated_new_model=not reused,
                active_model_task_id=selected_model_task_id if reused else task_id,
            )
        except Exception as exc:
            print(f"[WARN] display state backfill failed for {task_id}: {exc}")


def _cached_model_revisions(payload: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    raw = payload.get("cached_models")
    items = raw if isinstance(raw, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        display_object_id = str(item.get("display_object_id") or "").strip()
        if not display_object_id:
            continue
        try:
            result[display_object_id] = int(item.get("model_revision") or 0)
        except Exception:
            result[display_object_id] = 0
    return result


def _tracking_mode_items(
    *,
    startup_session_id: str,
    mode: str,
    cached_revisions: dict[str, int] | None = None,
) -> tuple[list[dict], str | None]:
    states = list_display_object_states(limit=5)
    if not states:
        _backfill_display_object_states()
        states = list_display_object_states(limit=5)

    reference_row = get_latest_aruco_reference(startup_session_id) if startup_session_id else None
    reference_pose = _load_marker_pose_json((reference_row or {}).get("marker_pose_json")) if reference_row else None
    coordinate_epoch = str((reference_row or {}).get("task_id") or (reference_row or {}).get("id") or "").strip() or None
    host = request.host_url.rstrip("/")
    cached_revisions = cached_revisions or {}
    items: list[dict] = []
    for state in states:
        display_object_id = str(state.get("display_object_id") or "")
        model_revision = int(state.get("active_model_revision") or 0)
        hololens_pose_aruco = _json_column(state.get("latest_hololens_pose_aruco_json"))
        tracking_pose_aruco = _json_column(state.get("latest_tracking_pose_aruco_json"))
        hololens_pose = None
        tracking_pose = None
        if reference_pose is not None and isinstance(hololens_pose_aruco, dict):
            try:
                hololens_pose = aruco_pose_to_hololens_pose(hololens_pose_aruco, reference_pose)
            except Exception as exc:
                print(f"[WARN] HoloLens snapshot conversion failed for {display_object_id}: {exc}")
        tracking_model_revision = int(state.get("latest_tracking_model_revision") or 0)
        if (
            reference_pose is not None
            and isinstance(tracking_pose_aruco, dict)
            and tracking_model_revision == model_revision
        ):
            try:
                tracking_pose = aruco_pose_to_hololens_pose(tracking_pose_aruco, reference_pose)
            except Exception as exc:
                print(f"[WARN] tracking snapshot conversion failed for {display_object_id}: {exc}")

        selected_source = "hololens"
        selected_pose = hololens_pose
        selected_revision = int(state.get("latest_hololens_pose_revision") or 0)
        if mode == MODE_LIVE and tracking_pose is not None:
            selected_source = "tracking"
            selected_pose = tracking_pose
            selected_revision = int(state.get("latest_tracking_pose_revision") or 0)

        item = {
            "display_object_id": display_object_id,
            "model_revision": model_revision,
            "active_model_task_id": state.get("active_model_task_id"),
            "latest_hololens_pose": hololens_pose,
            "hololens_pose_revision": int(state.get("latest_hololens_pose_revision") or 0),
            "latest_tracking_pose": tracking_pose,
            "tracking_pose_revision": int(state.get("latest_tracking_pose_revision") or 0),
            "tracking_model_revision": tracking_model_revision,
            "pose_source": selected_source,
            "source_pose_revision": selected_revision,
            "pose": selected_pose,
            "coordinate_space": "hololens_current_local" if selected_pose is not None else None,
            "latest_body_revision": int(state.get("latest_body_revision") or 0),
            "body_revision": int(state.get("latest_body_revision") or 0),
            "latest_body_task_id": state.get("latest_body_task_id"),
        }
        latest_tracking_event = get_latest_realtime_tracking_event(display_object_id)
        if latest_tracking_event is not None:
            tracking_event_sequence = int(latest_tracking_event.get("id") or 0)
            item["tracking_status"] = latest_tracking_event.get("status")
            item["tracking_reason"] = latest_tracking_event.get("reason")
            item["tracking_event_sequence"] = tracking_event_sequence
            item["event_sequence"] = tracking_event_sequence
            item["tracking_observation_seq"] = int(
                latest_tracking_event.get("observation_seq") or 0
            )
        if cached_revisions.get(display_object_id) != model_revision:
            active_task_id = str(state.get("active_model_task_id") or "").strip()
            active_task = get_task(active_task_id) if active_task_id else None
            if active_task and str(active_task.get("status") or "") == "completed":
                model_payload = _build_completed_task_response(
                    active_task,
                    host_override=host,
                    startup_session_id=startup_session_id,
                )
                model_entry = {
                    key: model_payload.get(key)
                    for key in ("task_id", "fbx_url")
                    if model_payload.get(key) is not None
                }
                model_entry["display_object_id"] = display_object_id
                model_entry["model_revision"] = model_revision
                model_instance = model_payload.get("model_instance")
                if isinstance(model_instance, dict):
                    model_instance = dict(model_instance)
                    model_instance["display_object_id"] = display_object_id
                    model_instance["model_revision"] = model_revision
                    model_entry["model_instance"] = model_instance
                item["model"] = model_entry
        body_revision = int(state.get("latest_body_revision") or 0)
        body_task_id = str(state.get("latest_body_task_id") or "").strip()
        if body_revision > 0 and body_task_id:
            body_task = get_task(body_task_id)
            if body_task and str(body_task.get("status") or "") == "completed":
                body_payload = _build_completed_task_response(
                    body_task,
                    host_override=host,
                    startup_session_id=startup_session_id,
                )
                body_evidence = {
                    key: body_payload.get(key)
                    for key in (
                        "task_id",
                        "display_object_id",
                        "model_instance",
                        "taken_object_detection",
                        "taken_object_detection_urls",
                        "sam3d_body_mesh",
                        "sam3d_body_mesh_urls",
                    )
                    if body_payload.get(key) is not None
                }
                item["body_evidence"] = body_evidence
                item["latest_body"] = body_evidence
                item["evidence"] = body_evidence
        items.append(item)
    return items, coordinate_epoch


@app.route("/realtime-tracking/mode", methods=["POST"], strict_slashes=False)
def realtime_tracking_mode():
    try:
        payload = request.get_json(silent=True) or {}
        startup_session_id = str(payload.get("startup_session_id") or "").strip()
        if not startup_session_id:
            return jsonify({"success": False, "error": "startup_session_id is required"}), 400
        requested_mode = str(payload.get("mode") or MODE_LIVE).strip().lower()
        request_generation = int(payload.get("request_generation") or 0)
        mode_state = realtime_tracking_coordinator.set_mode(
            startup_session_id=startup_session_id,
            mode=requested_mode,
            request_generation=request_generation,
        )
        items, coordinate_epoch = _tracking_mode_items(
            startup_session_id=startup_session_id,
            mode=str(mode_state["mode"]),
            cached_revisions=_cached_model_revisions(payload),
        )
        return jsonify(
            {
                "success": True,
                **mode_state,
                "coordinate_epoch": coordinate_epoch,
                "count": len(items),
                "items": items,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in realtime_tracking_mode: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/realtime-tracking/status", methods=["GET"], strict_slashes=False)
def realtime_tracking_status():
    try:
        startup_session_id = str(request.args.get("startup_session_id") or "").strip()
        mode_state = realtime_tracking_coordinator.mode_status(startup_session_id)
        items, coordinate_epoch = _tracking_mode_items(
            startup_session_id=startup_session_id,
            mode=str(mode_state["mode"]),
        )
        return jsonify(
            {
                "success": True,
                **mode_state,
                "coordinate_epoch": coordinate_epoch,
                "count": len(items),
                "items": items,
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print(app.url_map)
    print("Starting Flask application...")
    app.run(host="0.0.0.0", port=7355, debug=False, use_reloader=False, threaded=True)
