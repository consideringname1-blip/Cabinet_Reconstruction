"""Flask API entrypoint for task creation and status polling."""

import json
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
    UPLOAD_FOLDER,
)
from task_worker import (
    create_task,
    get_current_task_id,
    get_latest_completed_task_data,
    get_queue_snapshot,
    get_task,
    start_worker,
)
from task_db import (
    get_enabled_aruco_markers,
    get_latest_aruco_reference,
    sync_marker_registry_from_reference_folder,
)
from task_json import save_task_json
from unity_coordinate_utils import convert_hololens_pv_pose_matrix_to_unity_pose_components


app = Flask(__name__)


start_worker()


TERMINAL_STATUSES = frozenset({"completed", "aruco_completed", "failed"})
PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction"
PURPOSE_ARUCO_REFERENCE = "aruco_reference"


def _is_truthy_query_value(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


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
    for key in ("object", "object_world", "object_aruco", "aruco_reference", "debug"):
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
    }
    task_json = task_data.get("task_json") or {}

    instantmesh_info = task_json.get("InstantMesh") or {}
    blender_info = task_json.get("Blender") or {}

    mesh_name = instantmesh_info.get("mesh")
    mtl_name = instantmesh_info.get("mtl")
    image_name = instantmesh_info.get("image")
    video_name = instantmesh_info.get("video")
    fbx_name = blender_info.get("fbx")

    _append_pose_fields(response, task_json)

    mesh_path = INSTANTMESH_OUTPUT_MESHES / mesh_name if mesh_name else None
    mtl_path = INSTANTMESH_OUTPUT_MESHES / mtl_name if mtl_name else None
    image_path = INSTANTMESH_OUTPUT_MESHES / image_name if image_name else None
    video_path = INSTANTMESH_OUTPUT_VIDEOS / video_name if video_name else None
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
    if fbx_path and fbx_path.exists():
        fbx_url = f"{host}/files/fbx/{fbx_name}"
        response["fbx_url"] = fbx_url
        response["model_instance"] = _build_model_instance(task_data, task_json, fbx_url)

    return response


def _build_aruco_completed_task_response(task_data: dict) -> dict:
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
        "purpose": (task_data.get("task_json") or {}).get("purpose"),
        "terminal": True,
    }
    _append_pose_fields(response, task_data.get("task_json") or {})
    return response


def _load_marker_pose_json(raw_json: str | None):
    if not raw_json:
        return None
    try:
        return json.loads(raw_json)
    except Exception:
        return None

@app.route("/", methods=["GET"], strict_slashes=False)
def index():
    return jsonify(
        {
            "status": "running",
            "endpoints": [
                "/generate",
                "/check/<task_id>",
                "/check?task_id=<task_id>",
                "/latest-completed?history_offset=<0-4>",
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
                "rotation": devj.get("rotation"),
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
            "object": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
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


@app.route("/check/<task_id>", methods=["GET"], strict_slashes=False)
def check_task(task_id):
    try:
        task_data = get_task(task_id)
        if not task_data:
            return jsonify({"error": "Invalid task ID"}), 404

        status = task_data["status"]
        response = {"status": status}

        if status == "completed":
            response = _build_completed_task_response(task_data)
        elif status == "aruco_completed":
            response = _build_aruco_completed_task_response(task_data)

        elif status == "failed":
            response["error"] = task_data.get("error_message") or "Unknown error"
        else:
            current = get_current_task_id()
            queue_snapshot = get_queue_snapshot()
            if task_id == current:
                response["position"] = 0
                response["message"] = "Currently processing"
            elif task_id in queue_snapshot:
                response["position"] = queue_snapshot.index(task_id) + 1
                response["message"] = f"In queue, position {response['position']}"

        response["terminal"] = status in TERMINAL_STATUSES
        return jsonify(response)
    except Exception as exc:
        print(f"Error in check_task: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/check", methods=["GET"], strict_slashes=False)
def check_task_query():
    task_id = request.args.get("task_id")
    if not task_id:
        return jsonify({"error": "Missing task_id parameter"}), 400
    return check_task(task_id)


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


@app.route("/latest-completed", methods=["GET"], strict_slashes=False)
def latest_completed_task():
    try:
        startup_session_id = str(request.args.get("startup_session_id") or "").strip() or None
        history_offset = max(0, request.args.get("history_offset", default=0, type=int) or 0)
        require_aruco_coordinate_synced = _is_truthy_query_value(
            request.args.get("require_aruco_coordinate_synced")
        )
        task_data = get_latest_completed_task_data(
            startup_session_id=startup_session_id,
            require_aruco_coordinate_synced=require_aruco_coordinate_synced,
            history_offset=history_offset,
        )
        fallback_from_other_session = False
        if not task_data and startup_session_id and require_aruco_coordinate_synced:
            task_data = get_latest_completed_task_data(
                startup_session_id=None,
                require_aruco_coordinate_synced=True,
                history_offset=history_offset,
            )
            fallback_from_other_session = task_data is not None
        if not task_data:
            if startup_session_id:
                error_message = "No completed task found for this startup session"
                if require_aruco_coordinate_synced:
                    error_message = (
                        "No completed task with synced ArUco coordinates found for this startup session"
                    )
                return jsonify(
                    {
                        "error": error_message,
                        "startup_session_id": startup_session_id,
                        "history_offset": history_offset,
                    }
                ), 404
            error_message = "No completed task found"
            if require_aruco_coordinate_synced:
                error_message = "No completed task with synced ArUco coordinates found"
            return jsonify({"error": error_message}), 404

        response = _build_completed_task_response(task_data)
        response["history_offset"] = history_offset
        if startup_session_id and require_aruco_coordinate_synced:
            latest_reference_row = get_latest_aruco_reference(startup_session_id)
            current_aruco_reference = _load_marker_pose_json(
                latest_reference_row.get("marker_pose_json") if latest_reference_row else None
            )
            if current_aruco_reference:
                response["aruco_reference"] = current_aruco_reference
                if isinstance(response.get("model_instance"), dict):
                    response["model_instance"]["aruco_reference"] = current_aruco_reference
            if fallback_from_other_session:
                response["fallback_from_other_startup_session"] = True
                response["source_startup_session_id"] = task_data.get("startup_session_id")
        return jsonify(response)
    except Exception as exc:
        print(f"Error in latest_completed_task: {exc}")
        return jsonify({"error": str(exc)}), 500


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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
