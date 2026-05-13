"""Flask API entrypoint for task creation and status polling."""

import json
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from config import (
    AHAT_ENABLE_UPLOAD_GUARD,
    AHAT_MAX_RELIABLE_DEPTH_MM,
    AHAT_MAX_UPLOAD_PNG_BYTES,
    AHAT_MIN_DEPTH_MM,
    AHAT_MIN_USABLE_DEPTH_PIXELS,
    AHAT_SENSOR_NAME,
    BLENDER_FBX_DIR,
    FOLDER_MAP,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_VIDEOS,
    RUNTIME_MESH_OUTPUT_ROOT,
    UPLOAD_FOLDER,
)
from task_worker import (
    create_task,
    get_latest_completed_task_data,
    get_task,
    start_worker,
)
from task_db import (
    get_enabled_aruco_markers,
    get_latest_completed_tasks,
    get_latest_aruco_reference,
    get_latest_ready_model_bounds,
    get_model_bounds_by_task_id,
    get_ready_model_bounds_in_range,
    sync_marker_registry_from_reference_folder,
)
from model_bounds import decode_model_bounds_row, latest_bounds_for_ray, range_bounds_for_ray
from task_json import save_task_json
from unity_coordinate_utils import convert_hololens_pv_pose_matrix_to_unity_pose_components


app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


start_worker()


PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction"
PURPOSE_ARUCO_REFERENCE = "aruco_reference"


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


def _append_pose_fields(response: dict, task_json: dict) -> None:
    for key in ("object_world", "object_aruco", "aruco_reference", "debug"):
        value = task_json.get(key)
        response[key] = value if value else None


def _build_model_key(task_id: str | None, fbx_url: str) -> str:
    task_id_text = str(task_id or "").strip()
    if task_id_text:
        return task_id_text
    return str(fbx_url or "").strip()


def _build_model_instance(task_data: dict, task_json: dict, fbx_url: str) -> dict:
    task_id = task_data.get("task_id")
    return {
        "model_key": _build_model_key(task_id, fbx_url),
        "task_id": task_id,
        "fbx_url": fbx_url,
        "object_world": task_json.get("object_world") or None,
        "object_aruco": task_json.get("object_aruco") or None,
        "aruco_reference": task_json.get("aruco_reference") or None,
    }


def _resolve_placement_status(task_data: dict, task_json: dict) -> str:
    if bool(task_data.get("aruco_coordinate_synced")) and task_json.get("object_aruco"):
        return "aruco_synced"
    if task_json.get("object_world"):
        return "world_temporary"
    return "missing_pose"


def _build_task_model_bounds_status(task_id: str, task_json: dict) -> dict:
    row = get_model_bounds_by_task_id(task_id) if task_id else None
    if row:
        decoded = decode_model_bounds_row(row)
        return {
            "status": decoded.get("status") or "missing",
            "coordinate_space": decoded.get("coordinate_space") or "aruco",
            "aruco_reference_task_id": decoded.get("aruco_reference_task_id"),
            "object_aruco": decoded.get("object_aruco"),
            "aabb_min_aruco": decoded.get("aabb_min_aruco"),
            "aabb_max_aruco": decoded.get("aabb_max_aruco"),
            "corners_aruco": decoded.get("corners_aruco"),
            "error_message": decoded.get("error_message"),
        }

    model_bounds = task_json.get("ModelBounds")
    if isinstance(model_bounds, dict):
        return model_bounds

    if not task_json.get("object_aruco"):
        return {
            "status": "pending_reference",
            "coordinate_space": "aruco",
            "error_message": "object_aruco is missing; wait for a valid ArUco reference",
        }

    return {"status": "missing", "coordinate_space": "aruco"}


def _url_for_file_if_present(folder: str, filename: str | None, file_path) -> str | None:
    if not filename or not file_path or not file_path.exists():
        return None
    host = request.host_url.rstrip("/")
    return f"{host}/files/{folder}/{filename}"


def _build_bounds_download_urls(task_json: dict, fbx_name: str | None) -> dict:
    instantmesh_info = task_json.get("InstantMesh") or {}
    runtime_mesh_info = task_json.get("RuntimeMesh") or {}
    urls = {}

    mesh_name = instantmesh_info.get("mesh")
    mtl_name = instantmesh_info.get("mtl")
    image_name = instantmesh_info.get("image")
    runtime_mesh_name = runtime_mesh_info.get("mesh")
    runtime_mtl_name = runtime_mesh_info.get("mtl")
    runtime_image_name = runtime_mesh_info.get("image")

    for key, folder, filename, path in (
        ("mesh", "meshes", mesh_name, INSTANTMESH_OUTPUT_MESHES / mesh_name if mesh_name else None),
        ("mtl", "meshes", mtl_name, INSTANTMESH_OUTPUT_MESHES / mtl_name if mtl_name else None),
        ("image", "meshes", image_name, INSTANTMESH_OUTPUT_MESHES / image_name if image_name else None),
        (
            "runtime_mesh",
            "runtime_meshes",
            runtime_mesh_name,
            RUNTIME_MESH_OUTPUT_ROOT / runtime_mesh_name if runtime_mesh_name else None,
        ),
        (
            "runtime_mtl",
            "runtime_meshes",
            runtime_mtl_name,
            RUNTIME_MESH_OUTPUT_ROOT / runtime_mtl_name if runtime_mtl_name else None,
        ),
        (
            "runtime_image",
            "runtime_meshes",
            runtime_image_name,
            RUNTIME_MESH_OUTPUT_ROOT / runtime_image_name if runtime_image_name else None,
        ),
        ("fbx", "fbx", fbx_name, BLENDER_FBX_DIR / fbx_name if fbx_name else None),
    ):
        url = _url_for_file_if_present(folder, filename, path)
        if url:
            urls[key] = url

    return urls


def _build_model_bounds_response(row: dict, hit_result: dict | None = None) -> dict:
    decoded = decode_model_bounds_row(row)
    task_id = str(decoded.get("task_id") or "")
    task_data = get_task(task_id) if task_id else None
    task_json = (task_data or {}).get("task_json") or {}
    fbx_name = decoded.get("fbx_name") or (task_json.get("Blender") or {}).get("fbx")
    object_aruco = decoded.get("object_aruco") or task_json.get("object_aruco") or None
    download_urls = _build_bounds_download_urls(task_json, fbx_name)
    fbx_url = download_urls.get("fbx")

    model = {
        "id": decoded.get("id"),
        "task_id": task_id,
        "status": decoded.get("status"),
        "model_name": decoded.get("model_name"),
        "fbx_name": fbx_name,
        "uploaded_at": decoded.get("uploaded_at"),
        "coordinate_space": decoded.get("coordinate_space") or "aruco",
        "aruco_reference_task_id": decoded.get("aruco_reference_task_id"),
        "object_aruco": object_aruco,
        "aabb_min_aruco": decoded.get("aabb_min_aruco"),
        "aabb_max_aruco": decoded.get("aabb_max_aruco"),
        "corners_aruco": decoded.get("corners_aruco"),
        "source_model_path": decoded.get("source_model_path"),
        "error_message": decoded.get("error_message"),
        "download_urls": download_urls,
    }
    if fbx_url:
        model["fbx_url"] = fbx_url
        model["model_instance"] = {
            "model_key": _build_model_key(task_id, fbx_url),
            "task_id": task_id,
            "fbx_url": fbx_url,
            "object_world": task_json.get("object_world") or None,
            "object_aruco": object_aruco,
            "aruco_reference": None,
        }

    if hit_result:
        model["hit_distance_m"] = hit_result.get("hit_distance_m")
        model["hit_point_aruco"] = hit_result.get("hit_point_aruco")

    return model


def _sanitize_ahat_depth_png(depth_png_bytes: bytes) -> tuple[bytes, dict]:
    depth_png = np.frombuffer(depth_png_bytes, dtype=np.uint8)
    depth_image = cv2.imdecode(depth_png, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("depth_image is not a valid PNG")
    if depth_image.dtype != np.uint16:
        raise ValueError(f"depth_image must be uint16 depth, got {depth_image.dtype}")
    if depth_image.ndim != 2:
        raise ValueError(f"depth_image must be a single-channel image, got shape {depth_image.shape}")

    valid_mask = (depth_image >= AHAT_MIN_DEPTH_MM) & (depth_image <= AHAT_MAX_RELIABLE_DEPTH_MM)
    sanitized_depth = np.where(valid_mask, depth_image, 0).astype(np.uint16)

    encoded_ok, encoded_png = cv2.imencode(".png", sanitized_depth)
    if not encoded_ok:
        raise ValueError("Failed to re-encode sanitized AHAT depth image")

    stats = {
        "width": int(depth_image.shape[1]),
        "height": int(depth_image.shape[0]),
        "raw_nonzero_pixels": int(np.count_nonzero(depth_image)),
        "valid_depth_pixels": int(valid_mask.sum()),
        "clipped_depth_pixels": int(np.count_nonzero(depth_image) - valid_mask.sum()),
        "min_depth_mm": int(AHAT_MIN_DEPTH_MM),
        "max_reliable_depth_mm": int(AHAT_MAX_RELIABLE_DEPTH_MM),
        "input_png_bytes": int(len(depth_png_bytes)),
        "sanitized_png_bytes": int(encoded_png.size),
    }

    if AHAT_ENABLE_UPLOAD_GUARD and stats["valid_depth_pixels"] < int(AHAT_MIN_USABLE_DEPTH_PIXELS):
        raise ValueError(
            f"AHAT depth is not usable. Move the object closer and keep it within {AHAT_MAX_RELIABLE_DEPTH_MM / 1000.0:.1f} m."
        )
    if AHAT_ENABLE_UPLOAD_GUARD and stats["sanitized_png_bytes"] > int(AHAT_MAX_UPLOAD_PNG_BYTES):
        raise ValueError(
            f"AHAT depth is still too large after near-range filtering. Move the object closer and keep it within {AHAT_MAX_RELIABLE_DEPTH_MM / 1000.0:.1f} m."
        )

    return encoded_png.tobytes(), stats


def _build_completed_task_response(task_data: dict) -> dict:
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
        "purpose": (task_data.get("task_json") or {}).get("purpose"),
        "terminal": True,
        "stage_runs": task_data.get("stage_runs") or [],
        "aruco_coordinate_synced": bool(task_data.get("aruco_coordinate_synced")),
    }
    task_json = task_data.get("task_json") or {}
    task_id = str(task_data.get("task_id") or "")
    response["placement_status"] = _resolve_placement_status(task_data, task_json)
    response["model_bounds"] = _build_task_model_bounds_status(task_id, task_json)

    instantmesh_info = task_json.get("InstantMesh") or {}
    runtime_mesh_info = task_json.get("RuntimeMesh") or {}
    blender_info = task_json.get("Blender") or {}

    mesh_name = instantmesh_info.get("mesh")
    mtl_name = instantmesh_info.get("mtl")
    image_name = instantmesh_info.get("image")
    video_name = instantmesh_info.get("video")
    runtime_mesh_name = runtime_mesh_info.get("mesh")
    runtime_mtl_name = runtime_mesh_info.get("mtl")
    runtime_image_name = runtime_mesh_info.get("image")
    fbx_name = blender_info.get("fbx")

    _append_pose_fields(response, task_json)

    mesh_path = INSTANTMESH_OUTPUT_MESHES / mesh_name if mesh_name else None
    mtl_path = INSTANTMESH_OUTPUT_MESHES / mtl_name if mtl_name else None
    image_path = INSTANTMESH_OUTPUT_MESHES / image_name if image_name else None
    video_path = INSTANTMESH_OUTPUT_VIDEOS / video_name if video_name else None
    runtime_mesh_path = RUNTIME_MESH_OUTPUT_ROOT / runtime_mesh_name if runtime_mesh_name else None
    runtime_mtl_path = RUNTIME_MESH_OUTPUT_ROOT / runtime_mtl_name if runtime_mtl_name else None
    runtime_image_path = RUNTIME_MESH_OUTPUT_ROOT / runtime_image_name if runtime_image_name else None
    fbx_path = BLENDER_FBX_DIR / fbx_name if fbx_name else None

    if not mesh_path or not mesh_path.exists():
        response["error"] = "InstantMesh obj not found on disk"
        return response

    if not mtl_path or not mtl_path.exists():
        response["error"] = "InstantMesh mtl not found on disk"
        return response

    if not image_path or not image_path.exists():
        response["error"] = "InstantMesh image not found on disk"
        return response

    host = request.host_url.rstrip("/")
    response.update(
        {
            "mesh_url": f"{host}/files/meshes/{mesh_name}",
            "mtl_url": f"{host}/files/meshes/{mtl_name}",
            "image_url": f"{host}/files/meshes/{image_name}",
        }
    )
    if video_path and video_path.exists():
        response["video_url"] = f"{host}/files/videos/{video_name}"
    if (
        runtime_mesh_path
        and runtime_mesh_path.exists()
        and runtime_mtl_path
        and runtime_mtl_path.exists()
        and runtime_image_path
        and runtime_image_path.exists()
    ):
        response.update(
            {
                "runtime_mesh_url": f"{host}/files/runtime_meshes/{runtime_mesh_name}",
                "runtime_mtl_url": f"{host}/files/runtime_meshes/{runtime_mtl_name}",
                "runtime_image_url": f"{host}/files/runtime_meshes/{runtime_image_name}",
                "runtime_mesh": runtime_mesh_info,
            }
        )
    if fbx_path and fbx_path.exists():
        fbx_url = f"{host}/files/fbx/{fbx_name}"
        response["fbx_url"] = fbx_url
        response["model_instance"] = _build_model_instance(task_data, task_json, fbx_url)

    return response


def _build_aruco_completed_task_response(task_data: dict) -> dict:
    task_json = task_data.get("task_json") or {}
    startup_session_id = str(
        task_data.get("startup_session_id")
        or (task_json.get("device") or {}).get("startup_session_id")
        or ""
    ).strip()
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
        "purpose": task_json.get("purpose"),
        "terminal": True,
        "stage_runs": task_data.get("stage_runs") or [],
    }
    _append_pose_fields(response, task_json)

    aruco_stage = (
        ((task_json.get("debug") or {}).get("pose_transform_stages") or {}).get("aruco_stage")
        or {}
    )
    response["retro_synced_completed_task_count"] = _safe_int(
        aruco_stage.get("retro_synced_completed_task_count") or 0
    )

    if not response.get("aruco_reference"):
        latest_reference_row = get_latest_aruco_reference(startup_session_id) if startup_session_id else None
        latest_reference_task_id = str((latest_reference_row or {}).get("task_id") or "")
        if latest_reference_row and latest_reference_task_id == str(task_data.get("task_id") or ""):
            response["aruco_reference"] = _load_marker_pose_json(latest_reference_row.get("marker_pose_json"))

    response["aruco_detected"] = bool(response.get("aruco_reference"))
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
                "/model-bounds/range?start=<uploaded_at>&end=<uploaded_at>",
                "/spatial-query/ray",
                "/spatial-query/ray-range",
                "/aruco/latest-reference?startup_session_id=<startup_session_id>",
                "/aruco/markers",
                "/aruco/markers/sync",
                "/files/<folder>/<filename>",
            ],
        }
    )


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

            requested_sensor = str(dj.get("sensor") or AHAT_SENSOR_NAME).strip().upper()
            if requested_sensor != AHAT_SENSOR_NAME:
                raise ValueError(f"Only {AHAT_SENSOR_NAME} depth uploads are supported in this build")

            top_left = sbj.get("top_left")
            bottom_right = sbj.get("bottom_right")

            if not (isinstance(top_left, list) and len(top_left) == 2):
                raise ValueError("SelectionBoxJ.top_left must be a list of length 2")

            if not (isinstance(bottom_right, list) and len(bottom_right) == 2):
                raise ValueError("SelectionBoxJ.bottom_right must be a list of length 2")

        now_utc = datetime.now(timezone.utc)
        server_received_utc = now_utc.isoformat().replace("+00:00", "Z")
        base = now_utc.strftime("%Y%m%d_%H%M%S_%fZ")

        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

        color_path = None
        for index, frame in enumerate(normalized_pv_frames):
            field_name = "pv_image" if purpose == PURPOSE_OBJECT_RECONSTRUCTION else f"pv_image_{index}"
            if field_name not in request.files and index == 0:
                field_name = "pv_image"
            pv_png_bytes = _read_upload_file(field_name)
            suffix = "color" if purpose == PURPOSE_OBJECT_RECONSTRUCTION else f"color_{index:03d}"
            frame_color_path = UPLOAD_FOLDER / f"{base}_{suffix}.png"
            with open(frame_color_path, "wb") as f:
                f.write(pv_png_bytes)
            frame["name"] = str(frame_color_path.name)
            frame["upload_field"] = field_name
            frame["png_bytes"] = int(len(pv_png_bytes))
            if index == 0:
                color_path = frame_color_path

        depth_path = None
        depth_stats = None
        if purpose == PURPOSE_OBJECT_RECONSTRUCTION:
            depth_png_bytes = _read_upload_file("depth_image")
            depth_png_bytes, depth_stats = _sanitize_ahat_depth_png(depth_png_bytes)
            depth_path = UPLOAD_FOLDER / f"{base}_depth.png"
            with open(depth_path, "wb") as f:
                f.write(depth_png_bytes)

        out_json = {
            "server_received_utc": server_received_utc,
            "task_name": base,
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
            out_json["DepthCamera"] = {
                "name": str(depth_path.name) if depth_path else None,
                "pose": dj.get("pose") if dj else None,
                "sensor": AHAT_SENSOR_NAME,
                "stats": depth_stats,
            }
            out_json["SelectionBox"] = {
                "top_left": top_left,
                "bottom_right": bottom_right,
            }

        meta_path = UPLOAD_FOLDER / f"{base}_meta.json"
        save_task_json(meta_path, out_json)

        task_id = create_task(meta_path)

        return jsonify({"task_id": task_id})

    except Exception as exc:
        print("[ERROR] /generate exception:", exc)
        return jsonify({"error": str(exc)}), 400


@app.route("/check-queue", methods=["POST"], strict_slashes=False)
def check_task_queue():
    try:
        payload = request.get_json(silent=True) or {}
        task_ids = payload.get("task_ids")
        if not isinstance(task_ids, list):
            return jsonify({"error": "task_ids must be a JSON array"}), 400

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
                response = _build_completed_task_response(task_data)
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
                }
            else:
                pending.append(
                    {
                        "task_id": task_id,
                        "status": status,
                        "purpose": purpose,
                    }
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
        aruco_reference = _load_marker_pose_json(
            latest_reference_row.get("marker_pose_json") if latest_reference_row else None
        )
        if not aruco_reference:
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
                "aruco_reference": aruco_reference,
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
        rows = get_latest_completed_tasks(
            startup_session_id=startup_session_id,
            require_aruco_coordinate_synced=require_aruco_coordinate_synced,
            limit=limit,
        )
        return jsonify(
            {
                "success": True,
                "count": len(rows),
                "task_ids": [
                    {
                        "task_id": row.get("task_id"),
                        "purpose": PURPOSE_OBJECT_RECONSTRUCTION,
                        "status": row.get("status"),
                        "aruco_coordinate_synced": bool(row.get("aruco_coordinate_synced")),
                    }
                    for row in rows
                    if row.get("task_id")
                ],
            }
        )
    except Exception as exc:
        print(f"Error in latest_completed_task_ids: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/model-bounds/latest", methods=["GET"], strict_slashes=False)
def model_bounds_latest():
    try:
        limit = max(1, request.args.get("limit", default=5, type=int) or 5)
        rows = get_latest_ready_model_bounds(limit)
        return jsonify(
            {
                "success": True,
                "count": len(rows),
                "bounds": [_build_model_bounds_response(row) for row in rows],
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
        rows = get_ready_model_bounds_in_range(start, end, limit=limit)
        return jsonify(
            {
                "success": True,
                "count": len(rows),
                "start": start,
                "end": end,
                "bounds": [_build_model_bounds_response(row) for row in rows],
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
        _rows, result = latest_bounds_for_ray(payload)
        if not result.get("hit"):
            return jsonify(
                {
                    "success": True,
                    "hit": False,
                    "candidates_checked": result.get("candidates_checked", 0),
                    "max_distance_m": result.get("max_distance_m"),
                }
            )

        return jsonify(
            {
                "success": True,
                "hit": True,
                "candidates_checked": result.get("candidates_checked", 0),
                "model": _build_model_bounds_response(result["row"], result),
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
        _rows, result = range_bounds_for_ray(payload)
        if not result.get("hit"):
            return jsonify(
                {
                    "success": True,
                    "hit": False,
                    "candidates_checked": result.get("candidates_checked", 0),
                    "max_distance_m": result.get("max_distance_m"),
                }
            )

        return jsonify(
            {
                "success": True,
                "hit": True,
                "candidates_checked": result.get("candidates_checked", 0),
                "model": _build_model_bounds_response(result["row"], result),
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print(f"Error in spatial_query_ray_range: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/files/<path:folder>/<filename>", strict_slashes=False)
def serve_file(folder, filename):
    file_dir = FOLDER_MAP.get(folder)
    if file_dir is None:
        return jsonify({"error": "Invalid folder"}), 400
    file_path = file_dir / filename
    if not file_path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    return send_from_directory(str(file_dir), filename)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print(app.url_map)
    print("Starting Flask application...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
