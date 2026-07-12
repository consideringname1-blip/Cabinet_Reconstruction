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

from artifact_layout import model_worker_file
from blender_common import clean_scene, ensure_file
from blender_mesh_postprocess import (
    clean_connected_components,
    ensure_source_materials,
    repair_black_or_transparent_faces,
    select_objects,
)
from settings import (
    MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO,
    MODEL_FBX_CLEAN_COMPONENT_MIN_FACES,
    MODEL_FBX_CLEAN_ENABLE,
    RUNTIME_MESH_BAKE_MARGIN_PX,
    RUNTIME_MESH_DECIMATE_RATIO,
    RUNTIME_MESH_TEXTURE_SIZE,
    RUNTIME_MESH_UV_ISLAND_MARGIN,
    SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD,
    SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO,
    SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD,
    SAM3D_OBJECTS_REPAIR_BLACK_FACES,
)
from stages.hololens3d_reconstruction.model_generation_common import BACKEND_SAM3D_OBJECTS, ModelFileSource, resolve_model_generation_source
from stage_common import parse_blender_stage_args
from task_json import load_task_json, resolve_task_json_path, save_task_json


def _ratio_label(ratio: float) -> str:
    return f"{int(round(float(ratio) * 100)):02d}pct"


def _count_mesh(obj) -> tuple[int, int]:
    return len(obj.data.vertices), len(obj.data.polygons)


def _import_obj(obj_path: Path):
    clean_scene(purge_orphans=True)
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected 1 mesh object in {obj_path.name}, got {len(meshes)}")
    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _import_obj_into_scene(obj_path: Path):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh object was imported from {obj_path.name}")
    if len(meshes) > 1:
        selected = select_objects(meshes, active=meshes[0])
        if not selected:
            raise RuntimeError(f"No mesh object was selected from {obj_path.name}")
        bpy.ops.object.join()
        meshes = [bpy.context.view_layer.objects.active]
    return meshes[0]


def _apply_imported_glb_scale(objects: list) -> None:
    for obj in objects:
        if obj.type != "MESH":
            continue
        select_objects([obj], active=obj)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _import_sam3d_color_sources(source: ModelFileSource) -> tuple[list, dict]:
    raw_glb_name = str(source.payload.get("raw_glb") or "").strip()
    if not raw_glb_name:
        raise ValueError("ModelGeneration.raw_glb is required for sam3d_objects")
    raw_glb_path = ensure_file(source.root / raw_glb_name, "SAM3D Objects raw GLB")

    clean_scene(purge_orphans=True)
    bpy.ops.import_scene.gltf(filepath=str(raw_glb_path))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects were imported from {raw_glb_path}")
    _apply_imported_glb_scale(objects)
    ensure_source_materials(objects)
    objects, black_repair = repair_black_or_transparent_faces(
        objects,
        enabled=bool(SAM3D_OBJECTS_REPAIR_BLACK_FACES),
        rgb_threshold=float(SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD),
        alpha_threshold=float(SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD),
        max_repair_ratio=float(SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO),
    )
    objects, component_cleanup = clean_connected_components(
        objects,
        enabled=bool(MODEL_FBX_CLEAN_ENABLE),
        min_face_ratio=float(MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO),
        min_faces=int(MODEL_FBX_CLEAN_COMPONENT_MIN_FACES),
    )
    if not objects:
        raise RuntimeError("SAM3D Objects color-source cleanup removed all geometry")
    ensure_source_materials(objects)
    return objects, {
        "raw_glb": raw_glb_name,
        "black_repair": black_repair,
        "component_cleanup": component_cleanup,
    }


def _duplicate_mesh(obj, name: str):
    duplicate = obj.copy()
    duplicate.data = obj.data.copy()
    duplicate.animation_data_clear()
    duplicate.name = name
    duplicate.data.name = f"{name}_mesh"
    bpy.context.collection.objects.link(duplicate)
    return duplicate


def _apply_decimate(obj, ratio: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    modifier = obj.modifiers.new(name=f"runtime_decimate_{_ratio_label(ratio)}", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = float(ratio)
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def _smart_unwrap(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=1.15192,
        island_margin=float(RUNTIME_MESH_UV_ISLAND_MARGIN),
        area_weight=0.0,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def _make_bake_material(name: str, image):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.image = image
    if bsdf is not None:
        material.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
    nodes.active = image_node
    return material


def _bake_texture(high_objects: list, low, texture_path: Path) -> None:
    high_objects = [obj for obj in high_objects if obj is not None and obj.type == "MESH"]
    if not high_objects:
        raise RuntimeError("No high-resolution mesh is available for runtime texture baking")
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new(
        name=texture_path.stem,
        width=int(RUNTIME_MESH_TEXTURE_SIZE),
        height=int(RUNTIME_MESH_TEXTURE_SIZE),
        alpha=False,
        float_buffer=False,
    )

    material = _make_bake_material(f"{texture_path.stem}_mat", image)
    low.data.materials.clear()
    low.data.materials.append(material)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = 0.04
    bpy.context.scene.render.bake.max_ray_distance = 0.12
    bpy.context.scene.render.bake.margin = int(RUNTIME_MESH_BAKE_MARGIN_PX)
    bpy.context.scene.render.bake.use_clear = True

    bpy.ops.object.select_all(action="DESELECT")
    for high in high_objects:
        high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})

    image.filepath_raw = str(texture_path)
    image.file_format = "PNG"
    image.save()


def _export_obj(obj, obj_path: Path) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        path_mode="RELATIVE",
    )


def _fix_mtl_texture(mtl_path: Path, texture_name: str) -> None:
    if not mtl_path.is_file():
        raise FileNotFoundError(f"Runtime mesh mtl not found after export: {mtl_path}")

    lines = mtl_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if not line.strip().startswith("map_Kd ")]
    fixed = []
    inserted = False
    for line in lines:
        fixed.append(line)
        if line.startswith("Kd "):
            fixed.append(f"map_Kd {texture_name}")
            inserted = True
    if not inserted:
        fixed.append(f"map_Kd {texture_name}")
    mtl_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")


def build_runtime_mesh_from_json(json_path: Path) -> dict:
    task = load_task_json(json_path)
    source = resolve_model_generation_source(task, require_mtl_image=True)
    if not source.mtl or not source.image or source.mtl_path is None or source.image_path is None:
        raise ValueError(f"{source.source_stage}.mesh / mtl / image is missing")

    source_mesh = ensure_file(source.mesh_path, f"{source.source_stage} obj")
    source_mtl = ensure_file(source.mtl_path, f"{source.source_stage} mtl")
    ensure_file(source.image_path, f"{source.source_stage} texture image")
    _fix_mtl_texture(source_mtl, str(source.image))

    ratio = float(RUNTIME_MESH_DECIMATE_RATIO)
    texture_size = int(RUNTIME_MESH_TEXTURE_SIZE)
    ratio_label = _ratio_label(ratio)
    output_stem = f"{source_mesh.stem}_runtime_{ratio_label}_{texture_size}"
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise ValueError("task_timestamp is required for runtime mesh artifacts")
    output_obj = model_worker_file(task_timestamp, "model.runtime_obj")
    output_mtl = model_worker_file(task_timestamp, "model.runtime_mtl")
    output_texture = model_worker_file(task_timestamp, "model.runtime_texture")
    output_obj.parent.mkdir(parents=True, exist_ok=True)

    backend_detail = {}
    if source.backend == BACKEND_SAM3D_OBJECTS:
        high_objects, backend_detail = _import_sam3d_color_sources(source)
        low_source = _import_obj_into_scene(source_mesh)
        original_vertices, original_faces = _count_mesh(low_source)
        low = _duplicate_mesh(low_source, output_stem)
    else:
        high = _import_obj(source_mesh)
        high_objects = [high]
        original_vertices, original_faces = _count_mesh(high)
        low = _duplicate_mesh(high, output_stem)
    _apply_decimate(low, ratio)
    _smart_unwrap(low)
    _bake_texture(high_objects, low, output_texture)
    _export_obj(low, output_obj)
    _fix_mtl_texture(output_mtl, output_texture.name)
    output_vertices, output_faces = _count_mesh(low)

    for output_path, label in (
        (output_obj, "runtime mesh obj"),
        (output_mtl, "runtime mesh mtl"),
        (output_texture, "runtime mesh texture"),
    ):
        ensure_file(output_path, label)

    runtime_mesh = {
        "backend": source.backend,
        "mesh": output_obj.name,
        "mtl": output_mtl.name,
        "image": output_texture.name,
        "artifact_root": "model_worker",
        "source_mesh": source.mesh,
        "decimate_ratio": ratio,
        "texture_size": texture_size,
        "bake_margin_px": int(RUNTIME_MESH_BAKE_MARGIN_PX),
        "uv_island_margin": float(RUNTIME_MESH_UV_ISLAND_MARGIN),
        "original_vertices": int(original_vertices),
        "original_faces": int(original_faces),
        "vertices": int(output_vertices),
        "faces": int(output_faces),
        **backend_detail,
    }
    task["RuntimeMesh"] = runtime_mesh
    save_task_json(json_path, task)
    return runtime_mesh

def main() -> int:
    try:
        argv = parse_blender_stage_args(
            sys.argv,
            usage=(
                "Usage: blender --background --python "
                "code/stages/hololens3d_reconstruction/bake_runtime_mesh.py -- <task_meta.json or filename>"
            ),
            expected_count=1,
        )
    except SystemExit as exc:
        return int(exc.code)

    json_path = ensure_file(resolve_task_json_path(argv[0]), "JSON file")
    try:
        runtime_mesh = build_runtime_mesh_from_json(json_path)
        print(f"[RuntimeMesh] {runtime_mesh}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
