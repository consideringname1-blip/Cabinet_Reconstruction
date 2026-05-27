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
from blender_mesh_postprocess import (
    apply_decimate_to_objects,
    bake_vertex_color_sources_to_targets,
    clean_connected_components,
    count_mesh_objects,
    duplicate_mesh_objects,
    ensure_source_materials,
    repair_black_or_transparent_faces,
    select_objects,
    smart_unwrap_objects,
)
from config import (
    BLENDER_FBX_DIR,
    MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO,
    MODEL_FBX_CLEAN_COMPONENT_MIN_FACES,
    MODEL_FBX_CLEAN_ENABLE,
    MODEL_FBX_DECIMATE_RATIO,
    SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD,
    SAM3D_OBJECTS_DECIMATE_ENABLE,
    SAM3D_OBJECTS_FBX_DECIMATE_RATIO,
    SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO,
    SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD,
    SAM3D_OBJECTS_POSTPROCESS_BAKE_MARGIN_PX,
    SAM3D_OBJECTS_POSTPROCESS_TEXTURE_SIZE,
    SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN,
    SAM3D_OBJECTS_REPAIR_BLACK_FACES,
)
from model_generation_common import (
    MODEL_STAGE_INSTANTMESH,
    MODEL_STAGE_RUNTIME_MESH,
    MODEL_STAGE_SAM3D_OBJECTS,
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


def _import_instantmesh_source(source: ModelFileSource) -> tuple[list, dict]:
    return _import_obj_mtl_png_source(source)


def _import_runtime_mesh_source(source: ModelFileSource) -> tuple[list, dict]:
    return _import_obj_mtl_png_source(source)


def _apply_imported_glb_scale(objects: list) -> None:
    for obj in objects:
        if obj.type != "MESH":
            continue
        select_objects([obj], active=obj)
        try:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        except Exception:
            pass


def _import_sam3d_source(source: ModelFileSource) -> tuple[list, dict]:
    raw_glb_name = str(source.payload.get("raw_glb") or "").strip()
    if not raw_glb_name:
        raise ValueError("SAM3DObjects.raw_glb is missing; regenerate the SAM3D stage")
    raw_glb_path = ensure_file(source.root / raw_glb_name, "SAM3D Objects raw GLB")

    clean_scene(purge_orphans=True)
    bpy.ops.import_scene.gltf(filepath=str(raw_glb_path))
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not imported_objects:
        raise RuntimeError(f"No mesh objects were imported from {raw_glb_path}")
    _apply_imported_glb_scale(imported_objects)
    ensure_source_materials(imported_objects)
    return imported_objects, {
        "source_format": "glb",
        "source_mesh": source.mesh,
        "fbx_source_mesh": raw_glb_path.name,
    }


def _import_source_objects(source: ModelFileSource) -> tuple[list, dict]:
    if source.source_stage == MODEL_STAGE_RUNTIME_MESH:
        return _import_runtime_mesh_source(source)
    if source.source_stage == MODEL_STAGE_INSTANTMESH:
        return _import_instantmesh_source(source)
    if source.source_stage == MODEL_STAGE_SAM3D_OBJECTS:
        return _import_sam3d_source(source)
    raise ValueError(f"Unsupported FBX source stage: {source.source_stage}")


def _clean_fbx_source_geometry(objects: list, *, repair_black_faces: bool) -> tuple[list, dict]:
    original_vertices, original_faces = count_mesh_objects(objects)
    stats = {
        "original_vertices": int(original_vertices),
        "original_faces": int(original_faces),
    }

    objects, black_repair = repair_black_or_transparent_faces(
        objects,
        enabled=bool(repair_black_faces),
        rgb_threshold=float(SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD),
        alpha_threshold=float(SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD),
        max_repair_ratio=float(SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO),
    )
    stats["black_repair"] = black_repair

    objects, component_cleanup = clean_connected_components(
        objects,
        enabled=bool(MODEL_FBX_CLEAN_ENABLE),
        min_face_ratio=float(MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO),
        min_faces=int(MODEL_FBX_CLEAN_COMPONENT_MIN_FACES),
    )
    stats["component_cleanup"] = component_cleanup
    if not objects:
        raise RuntimeError("FBX postprocess removed all mesh geometry")
    return objects, stats


def _prepare_runtime_mesh_fbx_meshes(objects: list) -> tuple[list, dict]:
    vertices, faces = count_mesh_objects(objects)
    return objects, {
        "source_already_runtime_axis_normalized": True,
        "vertices": int(vertices),
        "faces": int(faces),
    }


def _prepare_instantmesh_fbx_meshes(objects: list) -> tuple[list, dict]:
    objects, stats = _clean_fbx_source_geometry(objects, repair_black_faces=False)
    objects, decimate = apply_decimate_to_objects(
        objects,
        ratio=float(MODEL_FBX_DECIMATE_RATIO),
        modifier_prefix="runtime_fbx_decimate",
    )
    stats["decimate"] = decimate
    stats["vertices"] = int(decimate.get("vertices") or 0)
    stats["faces"] = int(decimate.get("faces") or 0)
    if not objects:
        raise RuntimeError("FBX postprocess removed all mesh geometry")
    return objects, stats


def _prepare_sam3d_fbx_meshes(source_objects: list) -> tuple[list, list, dict]:
    source_objects, stats = _clean_fbx_source_geometry(
        source_objects,
        repair_black_faces=bool(SAM3D_OBJECTS_REPAIR_BLACK_FACES),
    )
    source_vertices, source_faces = count_mesh_objects(source_objects)
    stats["bake_source_vertices"] = int(source_vertices)
    stats["bake_source_faces"] = int(source_faces)

    target_objects = duplicate_mesh_objects(source_objects, suffix="_fbx_low")
    if not target_objects:
        raise RuntimeError("No SAM3D low-resolution target meshes could be created")

    target_objects, decimate = apply_decimate_to_objects(
        target_objects,
        ratio=float(SAM3D_OBJECTS_FBX_DECIMATE_RATIO),
        modifier_prefix="sam3d_runtime_fbx_decimate",
        enabled=bool(SAM3D_OBJECTS_DECIMATE_ENABLE),
    )
    stats["decimate"] = decimate
    stats["vertices"] = int(decimate.get("vertices") or 0)
    stats["faces"] = int(decimate.get("faces") or 0)
    if not target_objects:
        raise RuntimeError("FBX postprocess removed all mesh geometry")
    return source_objects, target_objects, stats


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

    fbx_path = BLENDER_FBX_DIR / f"{Path(source.mesh).stem}.fbx"
    BLENDER_FBX_DIR.mkdir(parents=True, exist_ok=True)

    imported_objects, source_info = _import_source_objects(source)
    if source.source_stage == MODEL_STAGE_RUNTIME_MESH:
        processed_objects, postprocess_info = _prepare_runtime_mesh_fbx_meshes(imported_objects)
    elif source.source_stage == MODEL_STAGE_SAM3D_OBJECTS:
        bake_source_objects, processed_objects, postprocess_info = _prepare_sam3d_fbx_meshes(imported_objects)
        smart_unwrap_objects(processed_objects, island_margin=float(SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN))
        postprocess_info["color_texture_bake"] = bake_vertex_color_sources_to_targets(
            bake_source_objects,
            processed_objects,
            output_dir=BLENDER_FBX_DIR,
            texture_stem=Path(source.mesh).stem,
            texture_size=int(SAM3D_OBJECTS_POSTPROCESS_TEXTURE_SIZE),
            margin_px=int(SAM3D_OBJECTS_POSTPROCESS_BAKE_MARGIN_PX),
        )
    else:
        processed_objects, postprocess_info = _prepare_instantmesh_fbx_meshes(imported_objects)

    apply_object_transform(processed_objects, task)
    _export_fbx(processed_objects, fbx_path)

    if not fbx_path.is_file():
        raise RuntimeError(f"FBX export failed: {fbx_path}")

    task["Blender"] = {
        "fbx": fbx_path.name,
        "source_stage": source.source_stage,
        "source_mesh": source.mesh,
        "source_mesh_folder": source.folder,
        "source_backend": source.backend,
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
