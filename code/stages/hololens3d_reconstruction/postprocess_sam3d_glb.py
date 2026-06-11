from __future__ import annotations

import json
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
    clean_connected_components,
    count_mesh_objects,
    ensure_source_materials,
    make_placeholder_texture,
    repair_black_or_transparent_faces,
    select_objects,
    smart_unwrap_objects,
)
from settings import (
    MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO,
    MODEL_FBX_CLEAN_COMPONENT_MIN_FACES,
    MODEL_FBX_CLEAN_ENABLE,
    SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD,
    SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO,
    SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD,
    SAM3D_OBJECTS_DECIMATE_ENABLE,
    SAM3D_OBJECTS_POSTPROCESS_DECIMATE_RATIO,
    SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN,
    SAM3D_OBJECTS_REPAIR_BLACK_FACES,
)
from stage_common import parse_blender_stage_args


def _import_glb(glb_path: Path) -> list:
    clean_scene(purge_orphans=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects were imported from {glb_path}")
    return meshes


def _export_obj(objects: list, obj_path: Path) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_objects(objects)
    if not selected:
        raise RuntimeError("No SAM3D mesh objects available for OBJ export")
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        export_colors=True,
        path_mode="RELATIVE",
    )


def _fix_mtl_texture(mtl_path: Path, texture_name: str) -> None:
    if not mtl_path.is_file():
        raise FileNotFoundError(f"SAM3D postprocess mtl not found after export: {mtl_path}")

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
        fixed.append("Kd 1.000000 1.000000 1.000000")
        fixed.append(f"map_Kd {texture_name}")
    mtl_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")


def postprocess_sam3d_glb(
    raw_glb_path: Path,
    obj_path: Path,
    mtl_path: Path,
    texture_path: Path,
    stats_path: Path,
) -> dict:
    raw_glb_path = ensure_file(raw_glb_path, "SAM3D raw GLB")
    source_objects = _import_glb(raw_glb_path)
    ensure_source_materials(source_objects)
    original_vertices, original_faces = count_mesh_objects(source_objects)

    source_objects, black_repair = repair_black_or_transparent_faces(
        source_objects,
        enabled=bool(SAM3D_OBJECTS_REPAIR_BLACK_FACES),
        rgb_threshold=float(SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD),
        alpha_threshold=float(SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD),
        max_repair_ratio=float(SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO),
    )
    source_objects, component_cleanup = clean_connected_components(
        source_objects,
        enabled=bool(MODEL_FBX_CLEAN_ENABLE),
        min_face_ratio=float(MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO),
        min_faces=int(MODEL_FBX_CLEAN_COMPONENT_MIN_FACES),
    )

    ratio = float(SAM3D_OBJECTS_POSTPROCESS_DECIMATE_RATIO)
    source_objects, decimate = apply_decimate_to_objects(
        source_objects,
        ratio=ratio,
        modifier_prefix="sam3d_geometry_decimate",
        enabled=bool(SAM3D_OBJECTS_DECIMATE_ENABLE),
    )
    if not source_objects:
        raise RuntimeError("SAM3D postprocess removed all mesh geometry")

    smart_unwrap_objects(source_objects, island_margin=float(SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN))
    _export_obj(source_objects, obj_path)
    make_placeholder_texture(texture_path)
    _fix_mtl_texture(mtl_path, texture_path.name)

    for output_path, label in (
        (obj_path, "SAM3D processed obj"),
        (mtl_path, "SAM3D processed mtl"),
        (texture_path, "SAM3D processed placeholder texture"),
    ):
        ensure_file(output_path, label)

    stats = {
        "raw_format": "glb",
        "final_format": "obj_mtl_png_geometry_only",
        "texture_baked": False,
        "decimate_ratio": ratio,
        "original_vertices": int(original_vertices),
        "original_faces": int(original_faces),
        "black_repair": black_repair,
        "component_cleanup": component_cleanup,
        "decimate": decimate,
        "uv_unwrapped": True,
        "uv_island_margin": float(SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN),
        "vertices": int(decimate.get("vertices") or 0),
        "faces": int(decimate.get("faces") or 0),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    try:
        argv = parse_blender_stage_args(
            sys.argv,
            usage=(
                "Usage: blender --background --python "
                "code/stages/hololens3d_reconstruction/postprocess_sam3d_glb.py -- "
                "<raw.glb> <output.obj> <output.mtl> <output.png> <stats.json>"
            ),
            expected_count=5,
        )
    except SystemExit as exc:
        return int(exc.code)

    try:
        stats = postprocess_sam3d_glb(
            Path(argv[0]),
            Path(argv[1]),
            Path(argv[2]),
            Path(argv[3]),
            Path(argv[4]),
        )
        print(f"[SAM3DObjectsPostprocess] {stats}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
