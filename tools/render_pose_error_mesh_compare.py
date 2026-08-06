#!/usr/bin/env python3
"""Render baseline and perturbed TSDF meshes from one fixed Blender view."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((args.run_root / "EXPERIMENT_REPORT.json").read_text())
    plane = report["dominant_plane"]
    center = Vector(plane["center"])
    normal = Vector(plane["normal"]).normalized()
    world_up = Vector((0.0, 1.0, 0.0))
    tangent = normal.cross(world_up).normalized()
    view_direction = (normal * 1.0 + tangent * 0.58 + world_up * 0.18).normalized()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 820
    scene.render.resolution_y = 820
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.curvature_ridge_factor = 2.0
    scene.display.shading.curvature_valley_factor = 1.5
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (0.96, 0.96, 0.96)
    scene.render.film_transparent = False

    bpy.ops.object.camera_add(location=center + view_direction * 1.7)
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.05
    camera.data.lens = 50
    look_at(camera, center)
    scene.camera = camera

    scenarios = {
        "hololens_pose_baseline": (0.26, 0.52, 0.82, 1.0),
        "arkit_like_pose": (0.90, 0.36, 0.22, 1.0),
    }
    for name, color in scenarios.items():
        mesh_path = args.run_root / name / "cabinet_tsdf_mesh.ply"
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
        obj = bpy.context.object
        obj.name = name
        material = bpy.data.materials.new(name=f"{name}_material")
        material.diffuse_color = color
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = False
        scene.render.filepath = str(args.output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    main()
