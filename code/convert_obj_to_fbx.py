import json
import sys
from pathlib import Path

import bpy


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.append(str(CODE_ROOT))

from config import BLENDER_FBX_DIR, INSTANTMESH_OUTPUT_MESHES


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, data: dict) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def apply_object_transform(objects: list, object_info: dict) -> None:
    position = object_info.get("position") or [0.0, 0.0, 0.0]
    rotation = object_info.get("rotation") or [0.0, 0.0, 0.0, 1.0]
    scale = object_info.get("scale") or [1.0, 1.0, 1.0]

    if len(position) != 3:
        raise ValueError("object.position must have 3 values")
    if len(rotation) != 4:
        raise ValueError("object.rotation must have 4 values")
    if len(scale) != 3:
        raise ValueError("object.scale must have 3 values")

    x, y, z = [float(v) for v in position]
    qx, qy, qz, qw = [float(v) for v in rotation]
    sx, sy, sz = [float(v) for v in scale]

    for obj in objects:
        obj.location = (x, y, z)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = (qw, qx, qy, qz)
        obj.scale = (sx, sy, sz)


def export_fbx_from_json(json_path: Path) -> Path:
    task = load_json(json_path)
    instantmesh_info = task.get("InstantMesh") or {}
    object_info = task.get("object") or {}

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
    bpy.ops.import_scene.obj(filepath=str(mesh_path), use_image_search=True)

    imported_objects = get_imported_mesh_objects()
    if not imported_objects:
        raise RuntimeError("No mesh object was imported into Blender")

    apply_object_transform(imported_objects, object_info)

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
    save_json(json_path, task)
    return fbx_path


def main() -> int:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    if len(argv) != 1:
        print(
            "Usage: blender --background --python convert_obj_to_fbx.py -- /path/to/task.json",
            file=sys.stderr,
        )
        return 2

    json_path = Path(argv[0]).expanduser().resolve()
    ensure_file(json_path, "JSON file")
    try:
        export_fbx_from_json(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
