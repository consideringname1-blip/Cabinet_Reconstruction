import sys
from pathlib import Path

import bpy

from blender_common import clean_scene, ensure_file
from config import BLENDER_FBX_DIR, INSTANTMESH_OUTPUT_MESHES
from stage_common import parse_blender_stage_args
from task_json import load_task_json, resolve_task_json_path, save_task_json

# Keep Blender-side axis settings local to this script so Blender's bundled
# Python does not need to import the heavier server-side alignment module.
FBX_CONVERT_OBJ_IMPORT_FORWARD_AXIS = "NEGATIVE_Z"
FBX_CONVERT_OBJ_IMPORT_UP_AXIS = "Y"
FBX_EXPORT_FORWARD_AXIS = "-Z"
FBX_EXPORT_UP_AXIS = "Y"


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

    clean_scene(purge_orphans=False)
    bpy.ops.wm.obj_import(
        filepath=str(mesh_path),
        forward_axis=FBX_CONVERT_OBJ_IMPORT_FORWARD_AXIS,
        up_axis=FBX_CONVERT_OBJ_IMPORT_UP_AXIS,
    )

    imported_objects = get_imported_mesh_objects()
    if not imported_objects:
        raise RuntimeError("No mesh object was imported into Blender")

    apply_object_transform(imported_objects, task)

    bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        embed_textures=True,
        path_mode="COPY",
        axis_forward=FBX_EXPORT_FORWARD_AXIS,
        axis_up=FBX_EXPORT_UP_AXIS,
        bake_space_transform=True,
    )

    if not fbx_path.is_file():
        raise RuntimeError(f"FBX export failed: {fbx_path}")

    task["Blender"] = {
        "fbx": fbx_path.name,
        "obj_import_axes": {
            "forward": FBX_CONVERT_OBJ_IMPORT_FORWARD_AXIS,
            "up": FBX_CONVERT_OBJ_IMPORT_UP_AXIS,
        },
        "fbx_export_axes": {
            "forward": FBX_EXPORT_FORWARD_AXIS,
            "up": FBX_EXPORT_UP_AXIS,
        },
    }
    save_task_json(json_path, task)
    return fbx_path


def main() -> int:
    try:
        argv = parse_blender_stage_args(
            sys.argv,
            usage=(
                "Usage: blender --background --python "
                "code/stages/hololens3d_reconstruction/convert_obj_to_fbx.py -- <task_meta.json or filename>"
            ),
            expected_count=1,
        )
    except SystemExit as exc:
        return int(exc.code)

    json_path = ensure_file(resolve_task_json_path(argv[0]), "JSON file")
    try:
        export_fbx_from_json(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
