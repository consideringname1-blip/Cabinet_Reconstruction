import sys
from pathlib import Path

import bpy

THIS_FILE = Path(__file__).resolve()
STAGES_DIR = THIS_FILE.parent
CODE_ROOT = STAGES_DIR.parent

if str(STAGES_DIR) not in sys.path:
    sys.path.insert(0, str(STAGES_DIR))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import BLENDER_FBX_DIR, INSTANTMESH_OUTPUT_MESHES
from task_json import load_task_json, resolve_task_json_path, save_task_json

def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def fix_mtl_texture_name(mtl_path: Path, texture_name: str) -> None:
    lines = []
    with mtl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("map_Kd"):
                lines.append(f"map_Kd {texture_name}\n")
            else:
                lines.append(line)

    with mtl_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def get_imported_mesh_objects() -> list:
    return [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]


def _parse_blender_scale(task: dict) -> tuple[float, float, float]:
    object_info = task.get("object") or {}
    scale = object_info.get("scale") or [1.0, 1.0, 1.0]
    if len(scale) != 3:
        raise ValueError("object.scale must have 3 values")
    return tuple(float(v) for v in scale)


def apply_object_transform(objects: list, task: dict) -> None:
    scale = _parse_blender_scale(task)

    for obj in objects:
        obj.scale = scale


def export_fbx_from_json(json_path: Path) -> Path:
    task = load_task_json(json_path)
    instantmesh_info = task.get("InstantMesh") or {}

    mesh_name = instantmesh_info.get("mesh")
    mtl_name = instantmesh_info.get("mtl")
    image_name = instantmesh_info.get("image")

    if not mesh_name or not mtl_name or not image_name:
        raise ValueError("InstantMesh.mesh / mtl / image is missing")

    mesh_path = ensure_file(INSTANTMESH_OUTPUT_MESHES / mesh_name, "InstantMesh obj")
    mtl_path = ensure_file(INSTANTMESH_OUTPUT_MESHES / mtl_name, "InstantMesh mtl")
    ensure_file(INSTANTMESH_OUTPUT_MESHES / image_name, "InstantMesh texture image")

    fbx_path = BLENDER_FBX_DIR / f"{mesh_path.stem}.fbx"
    BLENDER_FBX_DIR.mkdir(parents=True, exist_ok=True)

    fix_mtl_texture_name(mtl_path, image_name)

    clean_scene()
    bpy.ops.wm.obj_import(filepath=str(mesh_path))

    imported_objects = get_imported_mesh_objects()
    if not imported_objects:
        raise RuntimeError("No mesh object was imported into Blender")

    apply_object_transform(imported_objects, task)

    bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        embed_textures=True,
        path_mode="COPY",
        axis_forward="-Z",
        axis_up="Y",
        bake_space_transform=True,
    )

    if not fbx_path.is_file():
        raise RuntimeError(f"FBX export failed: {fbx_path}")

    task["Blender"] = {"fbx": fbx_path.name}
    save_task_json(json_path, task)
    return fbx_path


def main() -> int:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    if len(argv) != 1:
        print(
            "Usage: blender --background --python code/stages/convert_obj_to_fbx.py -- <task_meta.json or filename>",
            file=sys.stderr,
        )
        return 2

    json_path = resolve_task_json_path(argv[0])
    ensure_file(json_path, "JSON file")
    try:
        export_fbx_from_json(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
