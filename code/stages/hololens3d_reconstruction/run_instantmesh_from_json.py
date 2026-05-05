import subprocess
import sys
from pathlib import Path

import _bootstrap
from PIL import Image

from config import (
    ENABLE_INSTANTMESH_VIDEO_OUTPUT,
    IMESH_PY,
    INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO,
    INSTANTMESH_CLEAN_COMPONENT_MIN_FACES,
    INSTANTMESH_CLEAN_ENABLE,
    INSTANTMESH_CONFIG,
    INSTANTMESH_DIR,
    INSTANTMESH_INPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_VIDEOS,
    INSTANTMESH_RUN_PY,
    OUTPUT_ROOT,
    SAM3_OUTPUT_ROOT,
)
from mesh_obj_utils import clean_obj_connected_components
from stage_common import ensure_file, load_stage_task, resolve_python
from task_json import save_task_json


def create_white_background_image(source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(source_path).convert("RGBA")
    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(white_background, image).convert("RGB")
    merged.save(target_path)
    return target_path


def run_instantmesh(json_path: Path, task: dict) -> None:
    sam3_name = task.get("sam3Name") or {}

    sam3_color_name = sam3_name.get("color")
    if not sam3_color_name:
        raise ValueError("sam3Name.color is missing")

    sam3_color_path = ensure_file(SAM3_OUTPUT_ROOT / sam3_color_name, "SAM3 color image")
    prepared_input_path = create_white_background_image(
        source_path=sam3_color_path,
        target_path=INSTANTMESH_INPUT_ROOT / sam3_color_name,
    )

    try:
        import os
        imesh_python = resolve_python(IMESH_PY)
        imesh_bin = str(Path(imesh_python).resolve().parent)

        env = os.environ.copy()
        env["PATH"] = imesh_bin + os.pathsep + env.get("PATH", "")

        result = subprocess.run(
            [
                imesh_python,
                str(INSTANTMESH_RUN_PY),
                str(INSTANTMESH_CONFIG),
                str(prepared_input_path),
                "--output_path",
                str(OUTPUT_ROOT),
                "--export_texmap",
                # "--no_rembg",
            ]
            + (["--save_video"] if ENABLE_INSTANTMESH_VIDEO_OUTPUT else []),
            cwd=str(INSTANTMESH_DIR),
            env=env,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr or exc.stdout or str(exc)) from exc

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    output_stem = prepared_input_path.stem
    mesh_name = f"{output_stem}.obj"
    mtl_name = f"{output_stem}.mtl"
    image_name = f"{output_stem}.png"
    video_name = f"{output_stem}.mp4" if ENABLE_INSTANTMESH_VIDEO_OUTPUT else None

    mesh_path = ensure_file(INSTANTMESH_OUTPUT_MESHES / mesh_name, "InstantMesh obj")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / mtl_name, "InstantMesh mtl")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / image_name, "InstantMesh texture image")
    if video_name is not None:
        ensure_file(INSTANTMESH_OUTPUT_VIDEOS / video_name, "InstantMesh video")

    raw_mesh_name = mesh_name
    cleanup_info = None
    if bool(INSTANTMESH_CLEAN_ENABLE):
        clean_mesh_name = f"{Path(mesh_name).stem}_clean.obj"
        cleanup_info = clean_obj_connected_components(
            mesh_path,
            INSTANTMESH_OUTPUT_MESHES / clean_mesh_name,
            min_face_ratio=float(INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO),
            min_faces=int(INSTANTMESH_CLEAN_COMPONENT_MIN_FACES),
        )
        if int(cleanup_info.get("removed_faces") or 0) > 0:
            mesh_name = clean_mesh_name

    instantmesh_payload = {
        "mesh": mesh_name,
        "mtl": mtl_name,
        "image": image_name,
        "video": video_name,
        "video_render_enabled": bool(ENABLE_INSTANTMESH_VIDEO_OUTPUT),
    }
    if cleanup_info is not None:
        instantmesh_payload["raw_mesh"] = raw_mesh_name
        instantmesh_payload["cleanup"] = cleanup_info
    task["InstantMesh"] = instantmesh_payload
    save_task_json(json_path, task)


def main() -> int:
    try:
        json_path, task = load_stage_task(
            sys.argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_instantmesh_from_json.py <task_meta.json or filename>",
            stage_name="instantmesh",
        )
        run_instantmesh(json_path, task)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
