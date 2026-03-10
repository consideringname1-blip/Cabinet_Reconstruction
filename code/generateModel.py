# generateModel.py
import json
import os
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory

from config import (
    UPLOAD_FOLDER,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    BLENDER_FBX_DIR,
    FOLDER_MAP,
    HOLOLENS2_PY,
    HOLOLENS2_CONVERT,
    HOLOLENS2_CONVERT_DIR,
)
from pose_utils import rotate_vec_by_quat
from task_worker import (
    start_worker,
    create_task,
    get_task,
    get_current_task_id,
    get_queue_snapshot,
    find_task_file,
)

import base64
import subprocess

app = Flask(__name__)

# 启动后台任务线程
start_worker()


def _safe_timestr(dt: datetime) -> str:
    """
    文件名安全的 UTC 精确时间字符串：YYYYMMDD_HHMMSS_ffffffZ
    """
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S_%fZ")


def _parse_json_maybe(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _pick_first_file(files, keys):
    for k in keys:
        if k in files and files[k] and getattr(files[k], "filename", "") != "":
            return k, files[k]
    return None, None


@app.route("/", methods=["GET"], strict_slashes=False)
def index():
    return jsonify({
        "status": "running",
        "endpoints": [
            "/generate",
            "/check/<task_id>",
            "/check?task_id=<task_id>",
            "/files/<folder>/<filename>",
        ],
    })


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
            except Exception as e:
                raise ValueError(f"{field_name} is not valid JSON: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"{field_name} must be a JSON object")
            return obj

        def _b64_to_bytes(s: str) -> bytes:
            if not isinstance(s, str) or len(s) == 0:
                raise ValueError("image base64 is empty")
            # 兼容 data URI：data:image/png;base64,....
            if "," in s and s.strip().lower().startswith("data:"):
                s = s.split(",", 1)[1]
            try:
                return base64.b64decode(s, validate=False)
            except Exception as e:
                raise ValueError(f"invalid base64 image: {e}")

        # =========================
        # 1) 读取四段 JSON
        # =========================
        pvj = _parse_json_field("PVCameraJ")
        dj = _parse_json_field("DepthCameraJ")
        devj = _parse_json_field("deviceJ")
        sbj = _parse_json_field("SelectionBoxJ")

        # =========================
        # 1) 简单检查SelectionBoxJ
        # =========================
        top_left = sbj.get("top_left", None)
        bottom_right = sbj.get("bottom_right", None)

        if not (isinstance(top_left, list) and len(top_left) == 2):
            raise ValueError("SelectionBoxJ.top_left must be a list of length 2")

        if not (isinstance(bottom_right, list) and len(bottom_right) == 2):
            raise ValueError("SelectionBoxJ.bottom_right must be a list of length 2")

        # =========================
        # 2) 服务器接收时间：用于 JSON 字段 & 文件命名
        # =========================
        now_utc = datetime.now(timezone.utc)
        server_received_utc = now_utc.isoformat().replace("+00:00", "Z")
        # 文件名安全版本（不要直接用 server_received_utc 里的冒号）
        base = now_utc.strftime("%Y%m%d_%H%M%S_%fZ")

        task_id = str(uuid.uuid4())

        # 确保目录存在
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

        # =========================
        # 3) 保存 PV / Depth 图片（base64 -> png）
        # =========================
        pv_png_bytes = _b64_to_bytes(pvj.get("image", ""))
        color_path = UPLOAD_FOLDER / f"{base}_color.png"
        with open(color_path, "wb") as f:
            f.write(pv_png_bytes)

        depth_path = None
        depth_b64 = dj.get("image", "")
        if isinstance(depth_b64, str) and len(depth_b64) > 0:
            depth_png_bytes = _b64_to_bytes(depth_b64)
            depth_path = UPLOAD_FOLDER / f"{base}_depth.png"
            with open(depth_path, "wb") as f:
                f.write(depth_png_bytes)

        # =========================
        # 4) 组装“服务端保存格式”（object 默认值）
        # =========================
        out_json = {
            "task_id": task_id,
            "server_received_utc": server_received_utc,
            "task_name": base,

            "device": {
                "type": devj.get("type", ""),
                "ip": devj.get("ip", ""),
                "time": devj.get("time", ""),
                "pose": devj.get("pose", None),           # [x,y,z]
                "rotation": devj.get("rotation", None),   # [qx,qy,qz,qw]
            },

            "PVCamera": {
                "name": str(color_path.name),
                "width": pvj.get("width", 0),
                "height": pvj.get("height", 0),
                "k": pvj.get("k", None),
                "pose": pvj.get("pose", None),
            },

            "DepthCamera": {
                "name": str(depth_path.name) if depth_path else None,
                "pose": dj.get("pose", None),
                "sensor": "AHAT",
            },

            "SelectionBox": {
                "top_left": top_left,
                "bottom_right": bottom_right,
            },

            "object": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0, 0],
                "scale": [1.0, 1.0, 1.0],
            },

        }

        # 保存 meta（建议也存一份，方便你调试/复现）
        meta_path = UPLOAD_FOLDER / f"{base}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)


        # =========================
        # 5) 处理 hololens2深度图对齐
        # =========================
        subprocess.run(
            [HOLOLENS2_PY, str(HOLOLENS2_CONVERT), str(meta_path)],
            cwd=str(HOLOLENS2_CONVERT_DIR),
            check=True,
        )
        #断点，注意现在还没有读取json文件内容


        # =========================
        # 6) 入队 worker（InstantMesh 用 PV 彩图）
        # =========================
        center_depth = 1.0  # 你现在协议没给，就先默认；后面你可改成从 Depth 推/算
        create_task(
            task_id=task_id,
            upload_path=color_path,
            center_depth=center_depth,
            device_pose=devj,      # 如果 worker 不用也无所谓
            object_pose=None,
            # 如果 create_task 支持，你可以加：
            # depth_path=depth_path,
            # meta_path=meta_path,
            # meta=out_json,
        )

        return jsonify({
            "task_id": task_id,
            "server_received_utc": server_received_utc,
        })

    except Exception as e:
        print("[ERROR] /generate exception:", e)
        return jsonify({"error": str(e)}), 400



@app.route("/check/<task_id>", methods=["GET"], strict_slashes=False)
def check_task(task_id):
    try:
        task_data = get_task(task_id)
        if not task_data:
            return jsonify({"error": "Invalid task ID"}), 404

        status = task_data["status"]
        response = {"status": status}

        if status == "completed":
            meshes_dir = INSTANTMESH_OUTPUT_MESHES
            images_dir = INSTANTMESH_OUTPUT_IMAGES
            mesh_path = find_task_file(meshes_dir, task_id, "obj")
            mtl_path = find_task_file(meshes_dir, task_id, "mtl")
            image_path = find_task_file(images_dir, task_id, "png")
            fbx_path = find_task_file(BLENDER_FBX_DIR, task_id, "fbx")

            pose = task_data.get("object_pose")
            response["pose"] = pose if pose else None

            if not mesh_path or not mtl_path or not image_path:
                response["error"] = "Output files not found on disk"
            else:
                host = request.host_url.rstrip("/")
                response.update({
                    "mesh_url": f"{host}/files/meshes/{os.path.basename(mesh_path)}",
                    "mtl_url": f"{host}/files/meshes/{os.path.basename(mtl_path)}",
                    "image_url": f"{host}/files/images/{os.path.basename(image_path)}",
                })
                if fbx_path:
                    response["fbx_url"] = f"{host}/files/fbx/{os.path.basename(fbx_path)}"

        elif status == "failed":
            response["error"] = task_data.get("error", "Unknown error")
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
    except Exception as e:
        print(f"Error in check_task: {e}")
        return jsonify({"error": str(e)}), 500


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
