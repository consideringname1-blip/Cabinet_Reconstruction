#!/usr/bin/env python3
"""Load the generated GLB/URDF packages and record structural validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from yourdfpy import URDF


def validate(package_dir: Path) -> dict:
    manifest = json.loads((package_dir / "model_manifest.json").read_text())
    glb_path = Path(manifest["outputs"]["glb"])
    urdf_path = Path(manifest["outputs"]["urdf"])
    glb = trimesh.load(glb_path, process=False)
    if not isinstance(glb, trimesh.Scene):
        raise TypeError(f"Expected Scene from {glb_path}")
    robot = URDF.load(urdf_path.resolve(), mesh_dir=package_dir.resolve())
    joints = robot.robot.joints
    if len(joints) != 1:
        raise ValueError(f"Expected one joint, found {len(joints)}")
    joint = joints[0]
    bounds = np.asarray(robot.scene.bounds)
    result = {
        "package_dir": str(package_dir),
        "glb": {
            "path": str(glb_path),
            "file_bytes": glb_path.stat().st_size,
            "geometry_names": sorted(glb.geometry.keys()),
            "node_names": sorted(glb.graph.nodes_geometry),
            "bounds_finite": bool(np.isfinite(glb.bounds).all()),
        },
        "urdf": {
            "path": str(urdf_path),
            "file_bytes": urdf_path.stat().st_size,
            "links": [link.name for link in robot.robot.links],
            "scene_geometry_names": sorted(robot.scene.geometry.keys()),
            "joint_name": joint.name,
            "joint_type": joint.type,
            "joint_axis": np.asarray(joint.axis).tolist(),
            "joint_limit": [float(joint.limit.lower), float(joint.limit.upper)],
            "bounds_m": bounds.tolist(),
            "bounds_finite": bool(np.isfinite(bounds).all()),
        },
        "full_mesh_symlinks_resolve": {
            name: (package_dir / name).is_symlink() and (package_dir / name).resolve().is_file()
            for name in ("cabinet_static_full_nksr.ply", "drawer_moving_full_nksr.ply")
        },
    }
    result["valid"] = (
        result["glb"]["bounds_finite"]
        and result["urdf"]["bounds_finite"]
        and result["urdf"]["joint_type"] == "prismatic"
        and all(result["full_mesh_symlinks_resolve"].values())
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--package-dir", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.package_dir:
        packages = args.package_dir
    elif args.run_root:
        packages = [
            args.run_root / "official/articulated_model_prismatic",
            args.run_root
            / "extended/prismatic_open_fusion_gt_plane_axis/articulated_model",
        ]
    else:
        parser.error("provide --package-dir or --run-root")
    result = {"packages": [validate(path) for path in packages]}
    result["all_valid"] = all(item["valid"] for item in result["packages"])
    output = args.output or args.run_root / "validation/articulated_package_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
