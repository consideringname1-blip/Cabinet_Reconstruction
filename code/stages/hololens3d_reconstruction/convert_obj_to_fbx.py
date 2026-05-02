import sys
from pathlib import Path

_BOOTSTRAP_ROOTS = (
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
)
for _bootstrap_root in _BOOTSTRAP_ROOTS:
    _bootstrap_root_str = str(_bootstrap_root)
    if _bootstrap_root_str not in sys.path:
        sys.path.insert(0, _bootstrap_root_str)

import _bootstrap
import bpy

from blender_common import clean_scene, ensure_file
from config import BLENDER_FBX_DIR, INSTANTMESH_OUTPUT_MESHES, RUNTIME_MESH_OUTPUT_ROOT
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
    object_info = task.get("object_world")
    if not isinstance(object_info, dict):
        raise ValueError("object_world is missing")
    scale = object_info.get("scale")
    if not isinstance(scale, list) or len(scale) != 3:
        raise ValueError("object_world.scale must have 3 values")
    return tuple(float(v) for v in scale)


def apply_object_transform(objects: list, task: dict) -> None:
    scale = _parse_blender_scale(task)

    for obj in objects:
        obj.scale = scale


def _resolve_fbx_source(task: dict) -> tuple[str, Path, Path, str]:
    runtime_mesh_info = task.get("RuntimeMesh") or {}
    runtime_mesh_name = runtime_mesh_info.get("mesh")
    runtime_mtl_name = runtime_mesh_info.get("mtl")
    runtime_image_name = runtime_mesh_info.get("image")
    if runtime_mesh_name and runtime_mtl_name and runtime_image_name:
        return (
            "RuntimeMesh",
            ensure_file(RUNTIME_MESH_OUTPUT_ROOT / runtime_mesh_name, "RuntimeMesh obj"),
            ensure_file(RUNTIME_MESH_OUTPUT_ROOT / runtime_mtl_name, "RuntimeMesh mtl"),
            str(runtime_image_name),
        )

    instantmesh_info = task.get("InstantMesh") or {}
    mesh_name = instantmesh_info.get("mesh")
    mtl_name = instantmesh_info.get("mtl")
    image_name = instantmesh_info.get("image")

    if not mesh_name or not mtl_name or not image_name:
        raise ValueError("RuntimeMesh or InstantMesh mesh / mtl / image is missing")

    return (
        "InstantMesh",
        ensure_file(INSTANTMESH_OUTPUT_MESHES / mesh_name, "InstantMesh obj"),
        ensure_file(INSTANTMESH_OUTPUT_MESHES / mtl_name, "InstantMesh mtl"),
        str(image_name),
    )


def export_fbx_from_json(json_path: Path) -> Path:
    task = load_task_json(json_path)
    source_stage, mesh_path, mtl_path, image_name = _resolve_fbx_source(task)
    ensure_file(mesh_path.with_name(image_name), f"{source_stage} texture image")

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
        "source_stage": source_stage,
        "source_mesh": mesh_path.name,
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
