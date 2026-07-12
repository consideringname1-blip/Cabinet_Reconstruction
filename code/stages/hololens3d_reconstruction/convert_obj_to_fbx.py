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

from artifact_layout import model_result_file
from blender_common import clean_scene, ensure_file
from blender_mesh_postprocess import (
    count_mesh_objects,
    select_objects,
)
from stages.hololens3d_reconstruction.model_generation_common import (
    ModelFileSource,
    resolve_runtime_mesh_source,
)
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
    object_info = task.get("object_hololens_original")
    if not isinstance(object_info, dict):
        raise ValueError("object_hololens_original is missing")
    scale = object_info.get("scale")
    if not isinstance(scale, list) or len(scale) != 3:
        raise ValueError("object_hololens_original.scale must have 3 values")
    return tuple(float(v) for v in scale)


def apply_object_transform(objects: list, task: dict) -> None:
    scale = _parse_blender_scale(task)

    for obj in objects:
        obj.scale = scale


def _resolve_fbx_source(task: dict) -> ModelFileSource:
    source = resolve_runtime_mesh_source(task, require_mtl_image=True)
    if source is None:
        raise ValueError("RuntimeMesh is missing; run the runtime_mesh stage before Blender export")
    if not source.mtl or not source.image or source.mtl_path is None or source.image_path is None:
        raise ValueError(f"{source.source_stage}.mesh / mtl / image is missing")
    return source


def _import_obj_mtl_png_source(source: ModelFileSource) -> tuple[list, dict]:
    mesh_path = ensure_file(source.mesh_path, f"{source.source_stage} obj")
    mtl_path = ensure_file(source.mtl_path, f"{source.source_stage} mtl")
    image_path = ensure_file(source.image_path, f"{source.source_stage} texture image")
    fix_mtl_texture_name(mtl_path, image_path.name)

    clean_scene(purge_orphans=False)
    bpy.ops.wm.obj_import(
        filepath=str(mesh_path),
        forward_axis=FBX_CONVERT_OBJ_IMPORT_FORWARD_AXIS,
        up_axis=FBX_CONVERT_OBJ_IMPORT_UP_AXIS,
    )
    imported_objects = get_imported_mesh_objects()
    if not imported_objects:
        raise RuntimeError("No mesh object was imported into Blender")

    source_info = {
        "source_format": "obj_mtl_png",
        "source_mesh": mesh_path.name,
        "source_texture": image_path.name,
    }
    axis_contract = str(source.payload.get("axis_contract") or "").strip()
    if axis_contract:
        source_info["axis_contract"] = axis_contract
    axis_transform = str(source.payload.get("axis_transform") or "").strip()
    if axis_transform:
        source_info["axis_transform"] = axis_transform
    return imported_objects, source_info


def _import_runtime_mesh_source(source: ModelFileSource) -> tuple[list, dict]:
    return _import_obj_mtl_png_source(source)


def _prepare_runtime_mesh_fbx_meshes(objects: list) -> tuple[list, dict]:
    vertices, faces = count_mesh_objects(objects)
    return objects, {
        "source_already_runtime_axis_normalized": True,
        "vertices": int(vertices),
        "faces": int(faces),
    }


def _export_fbx(objects: list, fbx_path: Path) -> None:
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_objects(objects)
    if not selected:
        raise RuntimeError("No mesh objects are selected for FBX export")
    bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        use_selection=True,
        object_types={"MESH"},
        embed_textures=True,
        path_mode="COPY",
        axis_forward=FBX_EXPORT_FORWARD_AXIS,
        axis_up=FBX_EXPORT_UP_AXIS,
        bake_space_transform=True,
        colors_type="SRGB",
        prioritize_active_color=True,
    )


def export_fbx_from_json(json_path: Path) -> Path:
    task = load_task_json(json_path)
    source = _resolve_fbx_source(task)

    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise ValueError("task_timestamp is required for FBX artifacts")
    fbx_path = model_result_file(task_timestamp, "model.final_fbx")
    fbx_path.parent.mkdir(parents=True, exist_ok=True)

    imported_objects, source_info = _import_runtime_mesh_source(source)
    processed_objects, postprocess_info = _prepare_runtime_mesh_fbx_meshes(imported_objects)

    apply_object_transform(processed_objects, task)
    _export_fbx(processed_objects, fbx_path)

    if not fbx_path.is_file():
        raise RuntimeError(f"FBX export failed: {fbx_path}")

    task["Blender"] = {
        "backend": source.backend,
        "fbx": fbx_path.name,
        "artifact_root": "model_result",
        "source_mesh": source.mesh,
        **source_info,
        "postprocess": postprocess_info,
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
