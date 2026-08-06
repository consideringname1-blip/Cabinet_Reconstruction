#!/usr/bin/env python3
"""Blender background renderer for an articulated GLB preview."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    low = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return low, high


def gltf_axis_to_blender(axis: Vector) -> Vector:
    """Convert a glTF/HoloLens-world direction after Blender's glTF import.

    Blender converts glTF Y-up to Blender Z-up as
    (x, y, z) -> (x, -z, y), so post-import displacement needs the same map.
    """
    return Vector((axis.x, -axis.z, axis.y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drawer-axis", type=float, nargs=3)
    parser.add_argument("--drawer-displacement-m", type=float, default=0.0)
    parser.add_argument("--render-engine", choices=["eevee", "cycles"], default="eevee")
    args, _ = parser.parse_known_args(
        __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if args.drawer_displacement_m:
        if args.drawer_axis is None:
            parser.error("--drawer-axis is required with --drawer-displacement-m")
        direction = gltf_axis_to_blender(Vector(args.drawer_axis)).normalized()
        moving = [obj for obj in meshes if "drawer_moving" in obj.name.lower()]
        if not moving:
            raise RuntimeError("No drawer_moving mesh was found in the GLB")
        for obj in moving:
            obj.location += direction * args.drawer_displacement_m

    def assign_material(obj, color):
        material = bpy.data.materials.new(name=f"preview_{obj.name}")
        material.diffuse_color = color
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.72
        obj.data.materials.clear()
        obj.data.materials.append(material)

    for obj in meshes:
        name = obj.name.lower()
        if "drawer_moving" in name:
            assign_material(obj, (0.95, 0.22, 0.045, 1.0))
        elif "joint_axis" in name:
            assign_material(obj, (0.03, 0.8, 0.12, 1.0))
        else:
            assign_material(obj, (0.23, 0.34, 0.46, 1.0))
    low, high = world_bounds(meshes)
    center = (low + high) / 2
    radius = max(high.x - low.x, high.y - low.y, high.z - low.z) / 2

    bpy.ops.object.camera_add(location=center + Vector((1.65, -2.1, 1.35)) * radius)
    camera = bpy.context.object
    camera.data.lens = 52
    look_at(camera, center)
    bpy.context.scene.camera = camera

    for location, energy, size in [
        (center + Vector((2.0, -1.2, 2.4)) * radius, 120, 3.0),
        (center + Vector((-1.8, -0.8, 1.0)) * radius, 80, 2.2),
        (center + Vector((0.3, 2.0, 0.5)) * radius, 60, 2.0),
    ]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size * radius
        look_at(light, center)

    scene = bpy.context.scene
    if args.render_engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 16
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 850
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.025, 0.025, 0.035)
    scene.render.filepath = str(args.output.resolve())
    scene.view_settings.look = "AgX - Medium High Contrast"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(args.output)


if __name__ == "__main__":
    main()
