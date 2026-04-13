import subprocess
import sys
from pathlib import Path

from _bootstrap import CODE_ROOT
from PIL import Image

from config import (
    ENABLE_INSTANTMESH_VIDEO_OUTPUT,
    IMESH_PY,
    INSTANTMESH_CONFIG,
    INSTANTMESH_DIR,
    INSTANTMESH_INPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_VIDEOS,
    INSTANTMESH_RUN_PY,
    OUTPUT_ROOT,
    SAM3_OUTPUT_ROOT,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


def _resolve_python(python_path: str) -> str:
    return python_path or sys.executable


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def create_white_background_image(source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(source_path).convert("RGBA")
    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(white_background, image).convert("RGB")
    merged.save(target_path)
    return target_path


def run_instantmesh(json_path: Path) -> None:
    task = load_task_json(json_path)
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
        imesh_python = _resolve_python(IMESH_PY)
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

    ensure_file(INSTANTMESH_OUTPUT_MESHES / mesh_name, "InstantMesh obj")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / mtl_name, "InstantMesh mtl")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / image_name, "InstantMesh texture image")
    if video_name is not None:
        ensure_file(INSTANTMESH_OUTPUT_VIDEOS / video_name, "InstantMesh video")

    task["InstantMesh"] = {
        "mesh": mesh_name,
        "mtl": mtl_name,
        "image": image_name,
        "video": video_name,
        "video_render_enabled": bool(ENABLE_INSTANTMESH_VIDEO_OUTPUT),
    }
    save_task_json(json_path, task)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python code/stages/run_instantmesh_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_task_json_path(sys.argv[1])
    ensure_file(json_path, "JSON file")
    try:
        run_instantmesh(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
