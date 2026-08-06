#!/usr/bin/env python3
"""Embed a looping prismatic open/close animation in an existing binary glTF."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT = 5126


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION or total != len(data):
        raise ValueError(f"Invalid GLB header: {path}")
    offset = 12
    tree = None
    binary = None
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + length]
        offset += length
        if kind == JSON_CHUNK:
            tree = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif kind == BIN_CHUNK:
            if binary is not None:
                raise ValueError("Multiple BIN chunks are not supported")
            binary = payload
        else:
            raise ValueError(f"Unsupported GLB chunk type: {kind:#x}")
    if tree is None or binary is None:
        raise ValueError("GLB must contain JSON and BIN chunks")
    return tree, binary


def write_glb(path: Path, tree: dict, binary: bytes) -> None:
    binary += b"\x00" * ((-len(binary)) % 4)
    tree["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(tree, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    body = (
        struct.pack("<II", len(json_bytes), JSON_CHUNK)
        + json_bytes
        + struct.pack("<II", len(binary), BIN_CHUNK)
        + binary
    )
    path.write_bytes(struct.pack("<III", GLB_MAGIC, GLB_VERSION, 12 + len(body)) + body)


def append_f32(binary: bytes, values: np.ndarray) -> tuple[bytes, int, int]:
    binary += b"\x00" * ((-len(binary)) % 4)
    offset = len(binary)
    payload = np.asarray(values, dtype="<f4").tobytes()
    return binary + payload, offset, len(payload)


def find_node(tree: dict, name: str) -> int:
    matches = [index for index, node in enumerate(tree.get("nodes", [])) if node.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one node named {name!r}; found {len(matches)}")
    return matches[0]


def add_animation(
    tree: dict,
    binary: bytes,
    node_index: int,
    axis: np.ndarray,
    travel: float,
    open_seconds: float,
    hold_seconds: float,
    close_seconds: float,
    name: str,
) -> tuple[dict, bytes, dict]:
    node = tree["nodes"][node_index]
    if "matrix" in node:
        raise ValueError("Animated node uses a matrix; decompose it before adding TRS animation")
    base = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    times = np.asarray(
        [
            0.0,
            open_seconds,
            open_seconds + hold_seconds,
            open_seconds + hold_seconds + close_seconds,
            open_seconds + 2.0 * hold_seconds + close_seconds,
        ],
        dtype=np.float32,
    )
    closed = base
    opened = base + axis.astype(np.float32) * np.float32(travel)
    translations = np.stack([closed, opened, opened, closed, closed], axis=0)

    binary, time_offset, time_length = append_f32(binary, times)
    time_view = len(tree.setdefault("bufferViews", []))
    tree["bufferViews"].append(
        {"buffer": 0, "byteOffset": time_offset, "byteLength": time_length}
    )
    binary, translation_offset, translation_length = append_f32(binary, translations)
    translation_view = len(tree["bufferViews"])
    tree["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": translation_offset,
            "byteLength": translation_length,
        }
    )

    time_accessor = len(tree.setdefault("accessors", []))
    tree["accessors"].append(
        {
            "bufferView": time_view,
            "componentType": FLOAT,
            "count": len(times),
            "type": "SCALAR",
            "min": [float(times.min())],
            "max": [float(times.max())],
        }
    )
    translation_accessor = len(tree["accessors"])
    tree["accessors"].append(
        {
            "bufferView": translation_view,
            "componentType": FLOAT,
            "count": len(translations),
            "type": "VEC3",
            "min": translations.min(axis=0).astype(float).tolist(),
            "max": translations.max(axis=0).astype(float).tolist(),
        }
    )

    animation = {
        "name": name,
        "samplers": [
            {
                "input": time_accessor,
                "output": translation_accessor,
                "interpolation": "LINEAR",
            }
        ],
        "channels": [
            {
                "sampler": 0,
                "target": {"node": node_index, "path": "translation"},
            }
        ],
        "extras": {
            "jointType": "prismatic",
            "axisInParentGltfCoordinates": axis.astype(float).tolist(),
            "travelMeters": float(travel),
            "closedAtSeconds": [0.0, float(times[-2]), float(times[-1])],
            "openAtSeconds": [float(times[1]), float(times[2])],
        },
    }
    tree.setdefault("animations", []).append(animation)
    return tree, binary, {
        "animation_name": name,
        "node_index": node_index,
        "node_name": node.get("name"),
        "axis_in_parent_gltf_coordinates": axis.astype(float).tolist(),
        "travel_m": float(travel),
        "keyframe_times_s": times.astype(float).tolist(),
        "keyframe_translations": translations.astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", default="drawer_moving")
    parser.add_argument("--axis", type=float, nargs=3, required=True)
    parser.add_argument("--travel-m", type=float, required=True)
    parser.add_argument("--open-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--close-seconds", type=float, default=2.0)
    parser.add_argument("--animation-name", default="drawer_open_close")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Refusing to overwrite the source GLB")
    if min(args.open_seconds, args.hold_seconds, args.close_seconds) <= 0:
        raise ValueError("Animation durations must be positive")
    axis = np.asarray(args.axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Axis must be finite and non-zero")
    axis /= norm
    travel = abs(float(args.travel_m))

    tree, binary = read_glb(args.input)
    node_index = find_node(tree, args.node)
    existing_names = {animation.get("name") for animation in tree.get("animations", [])}
    if args.animation_name in existing_names:
        raise ValueError(f"Animation {args.animation_name!r} already exists")
    tree, binary, report = add_animation(
        tree,
        binary,
        node_index,
        axis,
        travel,
        args.open_seconds,
        args.hold_seconds,
        args.close_seconds,
        args.animation_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_glb(args.output, tree, binary)

    check_tree, _ = read_glb(args.output)
    check_animation = check_tree["animations"][-1]
    if check_animation["channels"][0]["target"] != {
        "node": node_index,
        "path": "translation",
    }:
        raise RuntimeError("Animation target validation failed")
    report.update(
        {
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "coordinate_note": (
                "The axis is stored in the drawer node's glTF parent coordinates. "
                "Blender imports this direction as (x, -z, y)."
            ),
            "validation": "passed",
        }
    )
    report_path = args.report or args.output.with_suffix(".animation_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
