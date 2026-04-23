"""Flask API entrypoint for task creation and status polling."""

import base64
import json
from datetime import datetime, timezone

import numpy as np

from flask import Flask, jsonify, request, send_from_directory

from config import (
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
from task_json import save_task_json


app = Flask(__name__)


start_worker()


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero-length quaternion")
    return q / n


def _rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")

    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return _normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def _extract_flipped_pv_pose_components(pose_value) -> tuple[list[float] | None, list[float] | None]:
    if pose_value is None:
        return None, None

    pose = np.asarray(pose_value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("PVCamera.pose must be 4x4")

    position = pose[3, :3].astype(np.float64)
    position[2] *= -1.0

    rotation = pose[:3, :3].astype(np.float64)
    quat_xyzw = _rotation_matrix_to_quat_xyzw(rotation)
    quat_xyzw[2] *= -1.0
    quat_xyzw = _normalize_quat_xyzw(quat_xyzw)

    return [float(v) for v in position], [float(v) for v in quat_xyzw]


def _append_pose_fields(response: dict, task_json: dict) -> None:
    for key in ("object", "object_world", "object_aruco", "aruco_reference", "debug"):
        value = task_json.get(key)
        response[key] = value if value else None


def _build_completed_task_response(task_data: dict) -> dict:
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
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
        response["fbx_url"] = f"{host}/files/fbx/{fbx_name}"

    return response


def _build_aruco_completed_task_response(task_data: dict) -> dict:
    response = {
        "status": task_data["status"],
        "task_id": task_data.get("task_id"),
    }
    _append_pose_fields(response, task_data.get("task_json") or {})
    return response

@app.route("/", methods=["GET"], strict_slashes=False)
def index():
    return jsonify(
        {
            "status": "running",
            "endpoints": [
                "/generate",
                "/check/<task_id>",
                "/check?task_id=<task_id>",
                "/latest-completed",
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

        def _b64_to_bytes(value: str) -> bytes:
            if not isinstance(value, str) or len(value) == 0:
                raise ValueError("image base64 is empty")
            if "," in value and value.strip().lower().startswith("data:"):
                value = value.split(",", 1)[1]
            try:
                return base64.b64decode(value, validate=False)
            except Exception as exc:
                raise ValueError(f"invalid base64 image: {exc}")

        pvj = _parse_json_field("PVCameraJ")
        dj = _parse_json_field("DepthCameraJ")
        devj = _parse_json_field("deviceJ")
        sbj = _parse_json_field("SelectionBoxJ")
        pv_position, pv_rotation_quaternion_xyzw = _extract_flipped_pv_pose_components(pvj.get("pose"))

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

        pv_png_bytes = _b64_to_bytes(pvj.get("image", ""))
        color_path = UPLOAD_FOLDER / f"{base}_color.png"
        with open(color_path, "wb") as f:
            f.write(pv_png_bytes)

        depth_path = None
        depth_b64 = dj.get("image", "")
        if isinstance(depth_b64, str) and depth_b64:
            depth_png_bytes = _b64_to_bytes(depth_b64)
            depth_path = UPLOAD_FOLDER / f"{base}_depth.png"
            with open(depth_path, "wb") as f:
                f.write(depth_png_bytes)

        out_json = {
            "server_received_utc": server_received_utc,
            "task_name": base,
            "device": {
                "type": devj.get("type", ""),
                "ip": devj.get("ip", ""),
                "time": devj.get("time", ""),
                "pose": devj.get("pose"),
                "rotation": devj.get("rotation"),
                "startup_session_id": devj.get("startup_session_id", ""),
            },
            "PVCamera": {
                "name": str(color_path.name),
                "width": pvj.get("width", 0),
                "height": pvj.get("height", 0),
                "k": pvj.get("k"),
                "pose": pvj.get("pose"),
                "position": pv_position,
                "rotation_quaternion_xyzw": pv_rotation_quaternion_xyzw,
            },
            "DepthCamera": {
                "name": str(depth_path.name) if depth_path else None,
                "pose": dj.get("pose"),
                "sensor": "AHAT",
            },
            "SelectionBox": {
                "top_left": top_left,
                "bottom_right": bottom_right,
            },
            "object": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
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


@app.route("/latest-completed", methods=["GET"], strict_slashes=False)
def latest_completed_task():
    try:
        task_data = get_latest_completed_task_data()
        if not task_data:
            return jsonify({"error": "No completed task found"}), 404

        response = _build_completed_task_response(task_data)
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
