#!/usr/bin/env python3
"""Project a reconstructed surface into source PV frames for alignment QA."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-ply", type=Path, required=True)
    parser.add_argument("--source-pinhole-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads((args.sequence_dir / "frame_mapping.json").read_text())
    poses = load_odometry(args.source_pinhole_dir / "odometry.log")
    fx, fy, cx, cy = np.loadtxt(args.source_pinhole_dir / "calibration.txt").reshape(-1)[:4]
    surface = o3d.io.read_point_cloud(str(args.surface_ply))
    world = np.asarray(surface.points)
    colors = np.asarray(surface.colors)
    if len(colors) != len(world):
        colors = np.full((len(world), 3), 0.8)

    frame_ids = [0, 12, 24, 36, 48, 60, 72, 82]
    tiles = []
    for local_index in frame_ids:
        source_index = int(mapping[local_index]["source_index"])
        bgr = cv2.imread(mapping[local_index]["rgb"], cv2.IMREAD_COLOR)
        height, width = bgr.shape[:2]
        camera = (world - poses[source_index, :3, 3]) @ poses[source_index, :3, :3]
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > 0.2) & (z < 4.0)
        camera, z, color = camera[valid], z[valid], colors[valid]
        u = np.rint(fx * camera[:, 0] / z + cx).astype(np.int64)
        v = np.rint(fy * camera[:, 1] / z + cy).astype(np.int64)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z, color = u[inside], v[inside], z[inside], color[inside]
        order = np.argsort(z)[::-1]
        render = np.zeros_like(bgr)
        occupancy = np.zeros((height, width), dtype=np.uint8)
        render[v[order], u[order]] = np.rint(color[order, ::-1] * 255).astype(np.uint8)
        occupancy[v, u] = 255
        render = cv2.dilate(render, np.ones((2, 2), np.uint8))
        occupancy = cv2.dilate(occupancy, np.ones((2, 2), np.uint8))
        blended = bgr.copy()
        mask = occupancy > 0
        blended[mask] = cv2.addWeighted(bgr, 0.4, render, 0.6, 0)[mask]
        cv2.putText(
            blended,
            f"local {local_index} source {source_index}",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)))

    width, height = tiles[0].size
    canvas = Image.new("RGB", (width * 4, height * 2), "black")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 4) * width, (index // 4) * height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=94)


if __name__ == "__main__":
    main()
