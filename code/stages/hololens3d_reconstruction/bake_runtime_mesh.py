import shutil
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
from config import RUNTIME_MESH_OUTPUT_ROOT
from settings import (
    RUNTIME_MESH_BAKE_MARGIN_PX,
    RUNTIME_MESH_DECIMATE_RATIO,
    RUNTIME_MESH_TEXTURE_SIZE,
    RUNTIME_MESH_UV_ISLAND_MARGIN,
)
from model_generation_common import MODEL_STAGE_SAM3D_OBJECTS, ModelFileSource, resolve_model_generation_source
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


def _bake_texture(high, low, texture_path: Path) -> None:
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


def _reuse_runtime_ready_sam3d_mesh(
    json_path: Path,
    task: dict,
    source: ModelFileSource,
) -> dict:
    if not source.mtl or not source.image or source.mtl_path is None or source.image_path is None:
        raise ValueError(f"{source.source_stage}.mesh / mtl / image is missing")

    source_mesh = ensure_file(source.mesh_path, f"{source.source_stage} processed obj")
    source_mtl = ensure_file(source.mtl_path, f"{source.source_stage} processed mtl")
    source_texture = ensure_file(source.image_path, f"{source.source_stage} processed texture")

    RUNTIME_MESH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_obj = RUNTIME_MESH_OUTPUT_ROOT / source.mesh
    output_mtl = RUNTIME_MESH_OUTPUT_ROOT / source.mtl
    output_texture = RUNTIME_MESH_OUTPUT_ROOT / source.image

    shutil.copy2(source_mesh, output_obj)
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(source_texture, output_texture)
    _fix_mtl_texture(output_mtl, output_texture.name)
    postprocess = source.payload.get("postprocess") if isinstance(source.payload.get("postprocess"), dict) else {}
    runtime_mesh = {
        "mesh": output_obj.name,
        "mtl": output_mtl.name,
        "image": output_texture.name,
        "source_stage": source.source_stage,
        "source_backend": source.backend,
        "source_mesh_folder": source.folder,
        "source_mesh": source.mesh,
        "source_mtl": source.mtl,
        "source_image": source.image,
        "reused_processed_mesh": True,
        "decimate_ratio": postprocess.get("decimate_ratio"),
        "texture_size": postprocess.get("texture_size"),
        "bake_margin_px": postprocess.get("bake_margin_px"),
        "uv_island_margin": postprocess.get("uv_island_margin"),
        "original_vertices": postprocess.get("original_vertices"),
        "original_faces": postprocess.get("original_faces"),
        "joined_vertices": postprocess.get("joined_vertices"),
        "joined_faces": postprocess.get("joined_faces"),
        "outer_vertices": postprocess.get("outer_vertices"),
        "outer_faces": postprocess.get("outer_faces"),
        "vertices": postprocess.get("vertices"),
        "faces": postprocess.get("faces"),
    }
    task["RuntimeMesh"] = runtime_mesh
    save_task_json(json_path, task)
    return runtime_mesh


def build_runtime_mesh_from_json(json_path: Path) -> dict:
    task = load_task_json(json_path)
    source = resolve_model_generation_source(task, require_mtl_image=True)
    if not source.mtl or not source.image or source.mtl_path is None or source.image_path is None:
        raise ValueError(f"{source.source_stage}.mesh / mtl / image is missing")

    if source.source_stage == MODEL_STAGE_SAM3D_OBJECTS and bool(source.payload.get("runtime_ready")):
        return _reuse_runtime_ready_sam3d_mesh(json_path, task, source)

    source_mesh = ensure_file(source.mesh_path, f"{source.source_stage} obj")
    source_mtl = ensure_file(source.mtl_path, f"{source.source_stage} mtl")
    ensure_file(source.image_path, f"{source.source_stage} texture image")
    _fix_mtl_texture(source_mtl, str(source.image))

    ratio = float(RUNTIME_MESH_DECIMATE_RATIO)
    texture_size = int(RUNTIME_MESH_TEXTURE_SIZE)
    ratio_label = _ratio_label(ratio)
    output_stem = f"{source_mesh.stem}_runtime_{ratio_label}_{texture_size}"
    output_obj = RUNTIME_MESH_OUTPUT_ROOT / f"{output_stem}.obj"
    output_mtl = RUNTIME_MESH_OUTPUT_ROOT / f"{output_stem}.mtl"
    output_texture = RUNTIME_MESH_OUTPUT_ROOT / f"{output_stem}.png"
    RUNTIME_MESH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    high = _import_obj(source_mesh)
    original_vertices, original_faces = _count_mesh(high)
    low = _duplicate_mesh(high, output_stem)
    _apply_decimate(low, ratio)
    _smart_unwrap(low)
    _bake_texture(high, low, output_texture)
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
        "mesh": output_obj.name,
        "mtl": output_mtl.name,
        "image": output_texture.name,
        "source_stage": source.source_stage,
        "source_backend": source.backend,
        "source_mesh_folder": source.folder,
        "source_mesh": source.mesh,
        "source_mtl": source.mtl,
        "source_image": source.image,
        "decimate_ratio": ratio,
        "texture_size": texture_size,
        "bake_margin_px": int(RUNTIME_MESH_BAKE_MARGIN_PX),
        "uv_island_margin": float(RUNTIME_MESH_UV_ISLAND_MARGIN),
        "original_vertices": int(original_vertices),
        "original_faces": int(original_faces),
        "vertices": int(output_vertices),
        "faces": int(output_faces),
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
