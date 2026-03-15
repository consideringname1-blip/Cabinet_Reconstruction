import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from config import (
    IMESH_PY,
    INSTANTMESH_CONFIG,
    INSTANTMESH_DIR,
    INSTANTMESH_INPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_RUN_PY,
    OUTPUT_ROOT,
    SAM3_OUTPUT_ROOT,
)


def _resolve_python(python_path: str) -> str:
    return python_path or sys.executable


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, data: dict) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


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
    task = load_json(json_path)
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
        result = subprocess.run(
            [
                _resolve_python(IMESH_PY),
                str(INSTANTMESH_RUN_PY),
                str(INSTANTMESH_CONFIG),
                str(prepared_input_path),
                "--output_path",
                str(OUTPUT_ROOT),
                "--save_video",
                "--export_texmap",
                "--no_rembg",
            ],
            cwd=str(INSTANTMESH_DIR),
            check=True,
            capture_output=True,
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

    ensure_file(INSTANTMESH_OUTPUT_MESHES / mesh_name, "InstantMesh obj")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / mtl_name, "InstantMesh mtl")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / image_name, "InstantMesh texture image")

    task["InstantMesh"] = {
        "mesh": mesh_name,
        "mtl": mtl_name,
        "image": image_name,
    }
    save_json(json_path, task)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python run_instantmesh_from_json.py /path/to/task.json", file=sys.stderr)
        return 2

    json_path = Path(sys.argv[1]).expanduser().resolve()
    ensure_file(json_path, "JSON file")
    try:
        run_instantmesh(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
