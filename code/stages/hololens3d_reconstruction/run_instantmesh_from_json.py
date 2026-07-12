import json
import shutil
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import _bootstrap
from PIL import Image

from artifact_layout import (
    model_debug_file,
    model_worker_dir,
    model_worker_file,
)
from config import TASK_DEBUG_OUTPUT_ENABLE
from path_config import IMESH_PY, INSTANTMESH_CONFIG, INSTANTMESH_DIR, INSTANTMESH_RUN_PY
from settings import (
    ENABLE_INSTANTMESH_VIDEO_OUTPUT,
    INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO,
    INSTANTMESH_CLEAN_COMPONENT_MIN_FACES,
    INSTANTMESH_CLEAN_ENABLE,
)
from mesh_obj_utils import clean_obj_connected_components
from stages.hololens3d_reconstruction.model_generation_common import (
    BACKEND_INSTANTMESH,
    build_model_generation_payload,
)
from stage_common import ensure_file, load_stage_task, resolve_python
from subprocess_stream import stream_command
from task_json import load_task_json, resolve_task_json_path, save_task_json


def create_white_background_image(source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(source_path).convert("RGBA")
    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(white_background, image).convert("RGB")
    merged.save(target_path)
    return target_path


def rewrite_obj_mtl_reference(obj_path: Path, mtl_name: str) -> None:
    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.startswith("mtllib "):
            rewritten.append(f"mtllib {mtl_name}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(0, f"mtllib {mtl_name}")
    obj_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def rewrite_mtl_texture_reference(mtl_path: Path, texture_name: str) -> None:
    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.strip().startswith("map_Kd "):
            rewritten.append(f"map_Kd {texture_name}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.append(f"map_Kd {texture_name}")
    mtl_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def run_instantmesh(json_path: Path, task: dict) -> None:
    sam3_name = task.get("sam3Name") or {}

    sam3_color_name = sam3_name.get("color")
    if not sam3_color_name:
        raise ValueError("sam3Name.color is missing")

    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise ValueError("task_timestamp is required for InstantMesh artifacts")
    sam3_color_path = ensure_file(model_worker_file(task_timestamp, "sam3.color"), "SAM3 color image")
    prepared_target_path = model_worker_file(task_timestamp, "generation.instantmesh_input")
    prepared_input_path = create_white_background_image(
        source_path=sam3_color_path,
        target_path=prepared_target_path,
    )
    backend_output_root = model_worker_dir(task_timestamp) / "03_instantmesh_backend"
    backend_mesh_root = backend_output_root / "instant-mesh-large" / "meshes"
    backend_video_root = backend_output_root / "instant-mesh-large" / "videos"
    video_output_enabled = bool(ENABLE_INSTANTMESH_VIDEO_OUTPUT and TASK_DEBUG_OUTPUT_ENABLE)

    try:
        import os
        imesh_python = resolve_python(IMESH_PY)
        imesh_bin = str(Path(imesh_python).resolve().parent)

        env = os.environ.copy()
        env["PATH"] = imesh_bin + os.pathsep + env.get("PATH", "")

        stream_command(
            [
                imesh_python,
                str(INSTANTMESH_RUN_PY),
                str(INSTANTMESH_CONFIG),
                str(prepared_input_path),
                "--output_path",
                str(backend_output_root),
                "--export_texmap",
                # "--no_rembg",
            ]
            + (["--save_video"] if video_output_enabled else []),
            cwd=INSTANTMESH_DIR,
            env=env,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    output_stem = prepared_input_path.stem
    mesh_name = f"{output_stem}.obj"
    mtl_name = f"{output_stem}.mtl"
    image_name = f"{output_stem}.png"
    video_name = f"{output_stem}.mp4" if video_output_enabled else None

    mesh_path = ensure_file(backend_mesh_root / mesh_name, "InstantMesh obj")
    ensure_file(backend_mesh_root / mtl_name, "InstantMesh mtl")
    ensure_file(backend_mesh_root / image_name, "InstantMesh texture image")
    if video_name is not None:
        ensure_file(backend_video_root / video_name, "InstantMesh video")

    raw_mesh_name = mesh_name
    cleanup_info = None
    if bool(INSTANTMESH_CLEAN_ENABLE):
        clean_mesh_name = f"{Path(mesh_name).stem}_clean.obj"
        cleanup_info = clean_obj_connected_components(
            mesh_path,
            backend_mesh_root / clean_mesh_name,
            min_face_ratio=float(INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO),
            min_faces=int(INSTANTMESH_CLEAN_COMPONENT_MIN_FACES),
        )
        if int(cleanup_info.get("removed_faces") or 0) > 0:
            mesh_name = clean_mesh_name

    artifact_root = "model_worker"
    model_video_name = video_name

    raw_obj_path = model_worker_file(task_timestamp, "generation.instantmesh_raw_obj")
    model_obj_path = model_worker_file(task_timestamp, "model.source_obj")
    model_mtl_path = model_worker_file(task_timestamp, "model.source_mtl")
    model_texture_path = model_worker_file(task_timestamp, "model.source_texture")
    for target in (raw_obj_path, model_obj_path, model_mtl_path, model_texture_path):
        target.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(backend_mesh_root / raw_mesh_name, raw_obj_path)
    shutil.copy2(backend_mesh_root / mesh_name, model_obj_path)
    shutil.copy2(backend_mesh_root / mtl_name, model_mtl_path)
    shutil.copy2(backend_mesh_root / image_name, model_texture_path)
    rewrite_obj_mtl_reference(raw_obj_path, model_mtl_path.name)
    rewrite_obj_mtl_reference(model_obj_path, model_mtl_path.name)
    rewrite_mtl_texture_reference(model_mtl_path, model_texture_path.name)

    if video_name:
        video_target = model_debug_file(task_timestamp, "generation.instantmesh_video")
        video_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backend_video_root / video_name, video_target)
        model_video_name = video_target.name

    model_mesh_name = model_obj_path.name
    model_mtl_name = model_mtl_path.name
    model_image_name = model_texture_path.name
    raw_mesh_name = raw_obj_path.name

    details = {
        "video_render_enabled": bool(video_output_enabled),
    }
    if cleanup_info is not None:
        details["raw_mesh"] = raw_mesh_name
        details["cleanup"] = cleanup_info
    task["ModelGeneration"] = build_model_generation_payload(
        backend=BACKEND_INSTANTMESH,
        mesh=model_mesh_name,
        mtl=model_mtl_name,
        image=model_image_name,
        artifact_root=artifact_root,
        video=model_video_name,
        extra=details,
    )
    save_task_json(json_path, task)


WORKER_RESPONSE_ENCODING = "utf-8"


def _read_socket_json(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    if not chunks:
        raise RuntimeError("empty InstantMesh worker request")
    raw = b"".join(chunks).splitlines()[0]
    return json.loads(raw.decode(WORKER_RESPONSE_ENCODING))


def _send_socket_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode(WORKER_RESPONSE_ENCODING))


def _request_json_paths(request: dict[str, Any]) -> list[Path]:
    raw_paths = request.get("json_paths")
    if raw_paths is None:
        raw_path = request.get("json_path")
        if raw_path is None:
            raise ValueError("InstantMesh worker request requires json_path or json_paths")
        raw_paths = [raw_path]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("json_paths must be a non-empty list")
    return [resolve_task_json_path(str(raw_path)) for raw_path in raw_paths]


def run_socket_server(socket_path: Path) -> None:
    socket_path = socket_path.expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(8)
    print(f"[InstantMesh worker] listening: {socket_path}", flush=True)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = _read_socket_json(conn)
                    if request.get("action") == "shutdown":
                        _send_socket_json(conn, {"ok": True, "shutdown": True})
                        break
                    results = []
                    for json_path in _request_json_paths(request):
                        task = load_task_json(json_path)
                        run_instantmesh(json_path, task)
                        results.append({"json_path": str(json_path), "ok": True})
                    _send_socket_json(conn, {"ok": True, "results": results})
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    _send_socket_json(conn, {"ok": False, "error": str(exc)})
    finally:
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--socket-server":
        run_socket_server(Path(sys.argv[2]))
        return 0

    try:
        json_path, task = load_stage_task(
            sys.argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_instantmesh_from_json.py <task_meta.json or filename>",
            stage_name="model_generation",
        )
        run_instantmesh(json_path, task)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
