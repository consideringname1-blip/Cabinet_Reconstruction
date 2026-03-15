import base64
import json
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory

from config import (
    BLENDER_FBX_DIR,
    FOLDER_MAP,
    INSTANTMESH_OUTPUT_MESHES,
    UPLOAD_FOLDER,
)
from task_worker import (
    create_task,
    get_current_task_id,
    get_queue_snapshot,
    get_task,
    start_worker,
)


app = Flask(__name__)


start_worker()

@app.route("/", methods=["GET"], strict_slashes=False)
def index():
    return jsonify(
        {
            "status": "running",
            "endpoints": [
                "/generate",
                "/check/<task_id>",
                "/check?task_id=<task_id>",
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
            },
            "PVCamera": {
                "name": str(color_path.name),
                "width": pvj.get("width", 0),
                "height": pvj.get("height", 0),
                "k": pvj.get("k"),
                "pose": pvj.get("pose"),
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
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
            f.write("\n")

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
        task_json = task_data.get("task_json") or {}

        if status == "completed":
            instantmesh_info = (task_json.get("InstantMesh") or {})
            blender_info = (task_json.get("Blender") or {})

            mesh_name = instantmesh_info.get("mesh")
            mtl_name = instantmesh_info.get("mtl")
            image_name = instantmesh_info.get("image")
            fbx_name = blender_info.get("fbx")

            object_info = task_json.get("object")
            response["pose"] = object_info if object_info else None

            mesh_path = INSTANTMESH_OUTPUT_MESHES / mesh_name if mesh_name else None
            mtl_path = INSTANTMESH_OUTPUT_MESHES / mtl_name if mtl_name else None
            image_path = INSTANTMESH_OUTPUT_MESHES / image_name if image_name else None
            fbx_path = BLENDER_FBX_DIR / fbx_name if fbx_name else None

            if not mesh_path or not mesh_path.exists():
                response["error"] = "InstantMesh obj not found on disk"
            elif not mtl_path or not mtl_path.exists():
                response["error"] = "InstantMesh mtl not found on disk"
            elif not image_path or not image_path.exists():
                response["error"] = "InstantMesh image not found on disk"
            else:
                host = request.host_url.rstrip("/")
                response.update(
                    {
                        "mesh_url": f"{host}/files/meshes/{mesh_name}",
                        "mtl_url": f"{host}/files/meshes/{mtl_name}",
                        "image_url": f"{host}/files/meshes/{image_name}",
                    }
                )
                if fbx_path and fbx_path.exists():
                    response["fbx_url"] = f"{host}/files/fbx/{fbx_name}"

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
