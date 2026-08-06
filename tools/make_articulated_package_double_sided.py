#!/usr/bin/env python3
"""Create robust double-sided GLB and URDF visualization assets without changing source meshes."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A


def read_glb(path: Path) -> tuple[dict, list[tuple[int, bytes]]]:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION or total != len(data):
        raise ValueError(f"Invalid GLB header: {path}")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    tree = None
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + length]
        offset += length
        if kind == JSON_CHUNK:
            tree = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        else:
            chunks.append((kind, payload))
    if tree is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    return tree, chunks


def write_glb(path: Path, tree: dict, chunks: list[tuple[int, bytes]]) -> None:
    json_bytes = json.dumps(tree, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    packed = [struct.pack("<II", len(json_bytes), JSON_CHUNK) + json_bytes]
    for kind, payload in chunks:
        packed.append(struct.pack("<II", len(payload), kind) + payload)
    body = b"".join(packed)
    path.write_bytes(struct.pack("<III", GLB_MAGIC, GLB_VERSION, 12 + len(body)) + body)


def make_glb_double_sided(source: Path, output: Path) -> dict:
    tree, chunks = read_glb(source)
    materials = tree.setdefault("materials", [])
    if materials:
        for material in materials:
            material["doubleSided"] = True
    else:
        materials.append(
            {
                "name": "two_sided_vertex_color",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.78,
                },
            }
        )
    assigned = 0
    for mesh in tree.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "material" not in primitive:
                primitive["material"] = 0
            assigned += 1
    write_glb(output, tree, chunks)
    check, _ = read_glb(output)
    assert check["materials"] and all(x.get("doubleSided") is True for x in check["materials"])
    return {
        "source": str(source),
        "output": str(output),
        "materials": len(check["materials"]),
        "primitives": assigned,
        "all_materials_double_sided": True,
        "vertex_color_attributes_preserved": all(
            "COLOR_0" in primitive.get("attributes", {})
            for mesh in check.get("meshes", [])
            for primitive in mesh.get("primitives", [])
        ),
    }


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    return mesh


def doubled_visual_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.visual.vertex_colors)
    count = len(vertices)
    doubled = trimesh.Trimesh(
        vertices=np.concatenate([vertices, vertices], axis=0),
        faces=np.concatenate([faces, faces[:, ::-1] + count], axis=0),
        vertex_normals=np.concatenate([normals, -normals], axis=0),
        vertex_colors=np.concatenate([colors, colors], axis=0),
        process=False,
    )
    return doubled


def make_urdf_double_sided(package_dir: Path, model_name: str) -> dict:
    static_source = package_dir / "cabinet_static_preview.obj"
    moving_source = package_dir / "drawer_moving_preview.obj"
    static_output = package_dir / "cabinet_static_preview_double_sided.obj"
    moving_output = package_dir / "drawer_moving_preview_double_sided.obj"
    stats = {}
    for label, source, output in [
        ("static", static_source, static_output),
        ("moving", moving_source, moving_output),
    ]:
        mesh = load_mesh(source)
        doubled = doubled_visual_mesh(mesh)
        doubled.export(output)
        stats[label] = {
            "source": str(source),
            "output": str(output),
            "source_vertices": int(len(mesh.vertices)),
            "source_triangles": int(len(mesh.faces)),
            "visual_vertices": int(len(doubled.vertices)),
            "visual_triangles": int(len(doubled.faces)),
        }

    source_urdf = package_dir / f"{model_name}.urdf"
    output_urdf = package_dir / f"{model_name}_double_sided.urdf"
    text = source_urdf.read_text(encoding="utf-8")
    text = text.replace(
        "<visual><geometry><mesh filename=\"cabinet_static_preview.obj\"/></geometry></visual>",
        "<visual><geometry><mesh filename=\"cabinet_static_preview_double_sided.obj\"/></geometry></visual>",
    )
    text = text.replace(
        "<visual><geometry><mesh filename=\"drawer_moving_preview.obj\"/></geometry></visual>",
        "<visual><geometry><mesh filename=\"drawer_moving_preview_double_sided.obj\"/></geometry></visual>",
    )
    output_urdf.write_text(text, encoding="utf-8")
    stats["source_urdf"] = str(source_urdf)
    stats["output_urdf"] = str(output_urdf)
    stats["collision_meshes_unchanged"] = True
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    source_glb = args.package_dir / f"{args.model_name}.glb"
    output_glb = args.package_dir / f"{args.model_name}_double_sided.glb"
    report = {
        "reason": "Open reconstructed sheets have inconsistent orientation and glTF defaults to back-face culling.",
        "geometry_policy": "Authoritative NKSR PLY and original package are unchanged.",
        "glb": make_glb_double_sided(source_glb, output_glb),
        "urdf": make_urdf_double_sided(args.package_dir, args.model_name),
    }
    report_path = args.package_dir / "double_sided_export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
