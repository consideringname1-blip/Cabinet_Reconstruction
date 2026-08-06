#!/usr/bin/env python3
"""Decimate one colored PLY mesh in Blender and export a colored PLY preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy


def blender_args() -> list[str]:
    import sys

    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-triangles", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(blender_args())

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.ply_import(
        filepath=str(args.input),
        merge_verts=False,
        import_colors="SRGB",
        import_attributes=True,
    )
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"PLY import did not produce a mesh: {args.input}")

    source_vertices = len(obj.data.vertices)
    source_triangles = len(obj.data.polygons)
    ratio = min(1.0, args.target_triangles / max(source_triangles, 1))
    if ratio < 1.0:
        modifier = obj.modifiers.new(name="preview_decimate", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.ply_export(
        filepath=str(args.output),
        export_selected_objects=True,
        apply_modifiers=True,
        export_uv=True,
        export_normals=True,
        export_colors="SRGB",
        export_attributes=True,
        export_triangulated_mesh=True,
        ascii_format=False,
    )
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "backend": f"Blender {bpy.app.version_string} Decimate COLLAPSE",
        "source_vertices": source_vertices,
        "source_triangles": source_triangles,
        "target_triangles": args.target_triangles,
        "ratio": ratio,
        "preview_vertices": len(obj.data.vertices),
        "preview_triangles": len(obj.data.polygons),
        "vertex_color_attributes": list(obj.data.color_attributes.keys()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
