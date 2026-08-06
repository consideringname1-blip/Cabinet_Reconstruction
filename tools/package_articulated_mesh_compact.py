#!/usr/bin/env python3
"""Create a compact GLB/URDF preview while preserving full NKSR PLY meshes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def simplify(path: Path, target_triangles: int) -> tuple[o3d.geometry.TriangleMesh, dict]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    original_vertices = len(mesh.vertices)
    original_triangles = len(mesh.triangles)
    if original_triangles > target_triangles:
        preview = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    else:
        preview = mesh
    preview.compute_vertex_normals()
    return preview, {
        "source": str(path),
        "original_vertices": int(original_vertices),
        "original_triangles": int(original_triangles),
        "preview_vertices": int(len(preview.vertices)),
        "preview_triangles": int(len(preview.triangles)),
    }


def to_trimesh(mesh: o3d.geometry.TriangleMesh, fallback_rgba: list[int]) -> trimesh.Trimesh:
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) == len(mesh.vertices):
        rgba = np.concatenate(
            [np.clip(colors, 0.0, 1.0) * 255.0, np.full((len(colors), 1), 255.0)],
            axis=1,
        ).astype(np.uint8)
    else:
        rgba = np.tile(np.asarray(fallback_rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
        vertex_colors=rgba,
        process=False,
    )


def add_axis(scene: trimesh.Scene, axis: np.ndarray, center: np.ndarray) -> None:
    length = 0.55
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    transform[:3, 3] = center
    cylinder = trimesh.creation.cylinder(radius=0.006, height=length, sections=20)
    cylinder.apply_transform(transform)
    cylinder.visual.face_colors = [0, 255, 80, 255]
    cone = trimesh.creation.cone(radius=0.018, height=0.055, sections=20)
    transform[:3, 3] = center + axis * (length / 2)
    cone.apply_transform(transform)
    cone.visual.face_colors = [0, 255, 80, 255]
    scene.add_geometry(cylinder, geom_name="joint_axis", node_name="joint_axis")
    scene.add_geometry(cone, geom_name="joint_axis_arrow", node_name="joint_axis_arrow")


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(destination)
    if destination.exists():
        raise FileExistsError(destination)
    os.symlink(source.resolve(), destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-mesh", type=Path, required=True)
    parser.add_argument("--moving-mesh", type=Path, required=True)
    parser.add_argument("--static-full-mesh", type=Path)
    parser.add_argument("--moving-full-mesh", type=Path)
    parser.add_argument("--axis", type=float, nargs=3, required=True)
    parser.add_argument("--travel-m", type=float, required=True)
    parser.add_argument("--canonical-state", choices=["open", "closed"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--static-preview-triangles", type=int, default=400000)
    parser.add_argument("--moving-preview-triangles", type=int, default=200000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_full_mesh = args.static_full_mesh or args.static_mesh
    moving_full_mesh = args.moving_full_mesh or args.moving_mesh
    axis = np.asarray(args.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    static_preview, static_stats = simplify(args.static_mesh, args.static_preview_triangles)
    moving_preview, moving_stats = simplify(args.moving_mesh, args.moving_preview_triangles)
    static_tm = to_trimesh(static_preview, [150, 150, 150, 255])
    moving_tm = to_trimesh(moving_preview, [210, 150, 70, 255])

    static_obj = args.output_dir / "cabinet_static_preview.obj"
    moving_obj = args.output_dir / "drawer_moving_preview.obj"
    static_tm.export(static_obj)
    moving_tm.export(moving_obj)
    safe_symlink(static_full_mesh, args.output_dir / "cabinet_static_full_nksr.ply")
    safe_symlink(moving_full_mesh, args.output_dir / "drawer_moving_full_nksr.ply")

    scene = trimesh.Scene()
    scene.add_geometry(static_tm, geom_name="cabinet_static", node_name="cabinet_static")
    scene.add_geometry(moving_tm, geom_name="drawer_moving", node_name="drawer_moving")
    add_axis(scene, axis, moving_tm.vertices.mean(axis=0))
    glb_path = args.output_dir / f"{args.model_name}.glb"
    scene.export(glb_path)

    if args.canonical_state == "open":
        lower, upper = -abs(args.travel_m), 0.0
        convention = "q=0 is open; negative q closes the drawer"
    else:
        lower, upper = 0.0, abs(args.travel_m)
        convention = "q=0 is closed; positive q opens the drawer"
    axis_text = " ".join(f"{value:.12g}" for value in axis)
    urdf_text = f"""<?xml version="1.0"?>
<robot name="{args.model_name}">
  <link name="cabinet_static">
    <visual><geometry><mesh filename="cabinet_static_preview.obj"/></geometry></visual>
    <collision><geometry><mesh filename="cabinet_static_preview.obj"/></geometry></collision>
  </link>
  <link name="drawer_moving">
    <visual><geometry><mesh filename="drawer_moving_preview.obj"/></geometry></visual>
    <collision><geometry><mesh filename="drawer_moving_preview.obj"/></geometry></collision>
  </link>
  <joint name="drawer_prismatic" type="prismatic">
    <parent link="cabinet_static"/>
    <child link="drawer_moving"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="{axis_text}"/>
    <limit lower="{lower:.12g}" upper="{upper:.12g}" effort="100" velocity="1"/>
  </joint>
</robot>
"""
    urdf_path = args.output_dir / f"{args.model_name}.urdf"
    urdf_path.write_text(urdf_text, encoding="utf-8")
    manifest = {
        "model_name": args.model_name,
        "joint_type": "prismatic",
        "axis_world": axis.tolist(),
        "travel_m": abs(args.travel_m),
        "canonical_state": args.canonical_state,
        "joint_state_convention": convention,
        "authoritative_full_resolution_meshes": {
            "static": "cabinet_static_full_nksr.ply",
            "moving": "drawer_moving_full_nksr.ply",
            "static_source": str(static_full_mesh.resolve()),
            "moving_source": str(moving_full_mesh.resolve()),
        },
        "preview_decimation": {
            "purpose": "interactive visualization and URDF loading only; full PLY meshes are not modified",
            "static": static_stats,
            "moving": moving_stats,
        },
        "outputs": {
            "glb": str(glb_path),
            "urdf": str(urdf_path),
            "static_preview_obj": str(static_obj),
            "moving_preview_obj": str(moving_obj),
        },
    }
    manifest_path = args.output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
