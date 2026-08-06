#!/usr/bin/env python3
"""Package separate static/moving meshes as GLB plus a prismatic URDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_mesh()
    return loaded


def axis_geometry(axis: np.ndarray, center: np.ndarray, length: float = 0.55) -> tuple:
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    transform[:3, 3] = center
    cylinder = trimesh.creation.cylinder(radius=0.006, height=length, sections=20)
    cylinder.apply_transform(transform)
    cylinder.visual.face_colors = [0, 255, 80, 255]
    cone = trimesh.creation.cone(radius=0.018, height=0.055, sections=20)
    cone_transform = transform.copy()
    cone_transform[:3, 3] = center + axis * (length / 2)
    cone.apply_transform(cone_transform)
    cone.visual.face_colors = [0, 255, 80, 255]
    return cylinder, cone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-mesh", type=Path, required=True)
    parser.add_argument("--moving-mesh", type=Path, required=True)
    parser.add_argument("--axis", type=float, nargs=3, required=True)
    parser.add_argument("--travel-m", type=float, required=True)
    parser.add_argument("--canonical-state", choices=["open", "closed"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    axis = np.asarray(args.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    static = as_mesh(args.static_mesh)
    moving = as_mesh(args.moving_mesh)
    static_obj = args.output_dir / "cabinet_static.obj"
    moving_obj = args.output_dir / "drawer_moving.obj"
    static.export(static_obj)
    moving.export(moving_obj)

    scene = trimesh.Scene()
    scene.add_geometry(static, geom_name="cabinet_static", node_name="cabinet_static")
    scene.add_geometry(moving, geom_name="drawer_moving", node_name="drawer_moving")
    center = moving.vertices.mean(axis=0)
    cylinder, cone = axis_geometry(axis, center)
    scene.add_geometry(cylinder, geom_name="joint_axis", node_name="joint_axis")
    scene.add_geometry(cone, geom_name="joint_axis_arrow", node_name="joint_axis_arrow")
    glb_path = args.output_dir / f"{args.model_name}.glb"
    scene.export(glb_path)

    if args.canonical_state == "open":
        lower, upper = -abs(args.travel_m), 0.0
        state_note = "q=0 is open; negative q closes the drawer"
    else:
        lower, upper = 0.0, abs(args.travel_m)
        state_note = "q=0 is closed; positive q opens the drawer"
    axis_text = " ".join(f"{value:.12g}" for value in axis)
    urdf = f"""<?xml version="1.0"?>
<robot name="{args.model_name}">
  <link name="cabinet_static">
    <visual><geometry><mesh filename="cabinet_static.obj"/></geometry></visual>
    <collision><geometry><mesh filename="cabinet_static.obj"/></geometry></collision>
  </link>
  <link name="drawer_moving">
    <visual><geometry><mesh filename="drawer_moving.obj"/></geometry></visual>
    <collision><geometry><mesh filename="drawer_moving.obj"/></geometry></collision>
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
    urdf_path.write_text(urdf, encoding="utf-8")
    manifest = {
        "model_name": args.model_name,
        "joint_type": "prismatic",
        "axis_world": axis.tolist(),
        "travel_m": abs(args.travel_m),
        "canonical_state": args.canonical_state,
        "joint_state_convention": state_note,
        "static_source_mesh": str(args.static_mesh),
        "moving_source_mesh": str(args.moving_mesh),
        "outputs": {
            "glb_preview_with_axis": str(glb_path),
            "urdf": str(urdf_path),
            "static_obj": str(static_obj),
            "moving_obj": str(moving_obj),
        },
        "note": "GLB contains separate named nodes and an axis arrow; URDF contains the actual prismatic joint.",
    }
    manifest_path = args.output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
