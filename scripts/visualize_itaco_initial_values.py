#!/usr/bin/env python3
"""Visualize the saved initial values for the current iTACO cabinet run.

This script is diagnostic only. It reads the preserved official-compatible
inputs/coarse outputs and writes new figures under ``diagnostics`` without
modifying any pipeline artifact.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image


DEFAULT_RUN_ROOT = Path(
    "/workspace_whz/data/output/itaco_hololens_gt/"
    "2026-07-30-002840_revolute_interior_001"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render iTACO initial inputs, camera trajectory, and coarse axes."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Official forward-view local index; negative indices follow Python convention.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cloud-points", type=int, default=60_000)
    parser.add_argument("--axis-length-m", type=float, default=0.55)
    parser.add_argument("--uid-alpha", type=float, default=0.48)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def one_mapping(run_root: Path, prefix: str) -> tuple[Path, list[dict]]:
    candidates = sorted((run_root / "inputs").glob(f"{prefix}*/frame_mapping.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {prefix} frame mapping, found {len(candidates)}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0], json.loads(candidates[0].read_text(encoding="utf-8"))


def numeric_suffix(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def uid_color(uid: int) -> tuple[float, float, float]:
    hue = (0.08 + uid * 0.618033988749895) % 1.0
    saturation = 0.72 + 0.18 * ((uid % 3) / 2.0)
    return colorsys.hsv_to_rgb(hue, saturation, 0.95)


def load_autoseg(mask_path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with np.load(mask_path) as archive:
        if "a" not in archive.files:
            raise KeyError(f"{mask_path} does not contain key 'a'")
        raw = archive["a"]
    original_shape = raw.shape
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 3:
        raise ValueError(f"Expected AutoSeg shape (D,H,W) or (D,1,H,W), got {original_shape}")
    return raw.astype(bool), original_shape


def uid_overlay(rgb: np.ndarray, masks: np.ndarray, alpha: float) -> np.ndarray:
    if rgb.shape[:2] != masks.shape[1:]:
        raise ValueError(f"RGB/mask size mismatch: {rgb.shape[:2]} versus {masks.shape[1:]}")
    result = rgb.astype(np.float32).copy()
    for uid, mask in enumerate(masks):
        if mask.any():
            color = np.asarray(uid_color(uid), dtype=np.float32) * 255.0
            result[mask] = (1.0 - alpha) * result[mask] + alpha * color
    return np.clip(result, 0, 255).astype(np.uint8)


def equal_3d_limits(axis: plt.Axes, points: np.ndarray, padding_fraction: float = 0.08) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * (0.5 + padding_fraction), 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def camera_poses(coarse_root: Path) -> tuple[list[Path], np.ndarray]:
    paths = sorted((coarse_root / "cam_pose").glob("cam2label_*.npy"), key=numeric_suffix)
    poses = np.stack([np.load(path) for path in paths])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected coarse camera-pose shape: {poses.shape}")
    return paths, poses


def draw_camera_trajectory(axis: plt.Axes, poses: np.ndarray) -> None:
    centers = poses[:, :3, 3]
    frame_ids = np.arange(len(centers))
    axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="#d6d6d6", linewidth=1.5)
    scatter = axis.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2], c=frame_ids,
        cmap="viridis", s=24, depthshade=False,
    )
    stride = max(1, len(poses) // 7)
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))) * 0.12, 0.02)
    for pose in poses[::stride]:
        origin = pose[:3, 3]
        forward = pose[:3, 2]
        axis.quiver(*origin, *forward, length=scale, normalize=True, color="#ff9f1c", linewidth=1.2)
    axis.scatter(*centers[0], color="#22c55e", s=70, marker="o", label="start")
    axis.scatter(*centers[-1], color="#ef4444", s=70, marker="X", label="end")
    equal_3d_limits(axis, centers)
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z (m)")
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=8)
    plt.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02, label="forward frame index")


def render_initial_values(
    output_path: Path,
    rgb: np.ndarray,
    moving_mask: np.ndarray,
    masks: np.ndarray,
    poses: np.ndarray,
    frame_index: int,
    source_index: int,
    autoseg_reverse_index: int,
    uid_alpha: float,
) -> None:
    overlay = uid_overlay(rgb, masks, uid_alpha)
    figure = plt.figure(figsize=(17, 12), dpi=150)
    rgb_axis = figure.add_subplot(2, 2, 1)
    moving_axis = figure.add_subplot(2, 2, 2)
    uid_axis = figure.add_subplot(2, 2, 3)
    camera_axis = figure.add_subplot(2, 2, 4, projection="3d")

    rgb_axis.imshow(rgb)
    rgb_axis.set_title(f"RGB — forward frame {frame_index}, source frame {source_index}")
    rgb_axis.axis("off")

    moving_axis.imshow(rgb)
    moving_layer = np.ma.masked_where(~moving_mask, moving_mask.astype(np.float32))
    moving_axis.imshow(moving_layer, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.72)
    moving_axis.set_title(
        "MonST3R-slot moving map (saved binary)\n"
        f"HoloLens-derived substitution; coverage={moving_mask.mean() * 100:.2f}%"
    )
    moving_axis.axis("off")

    uid_axis.imshow(overlay)
    for uid, mask in enumerate(masks):
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            continue
        uid_axis.text(
            float(np.median(xx)), float(np.median(yy)), str(uid),
            ha="center", va="center", fontsize=7, weight="bold",
            color=uid_color(uid),
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.84, "pad": 1.0},
        )
    uid_axis.set_title(
        f"AutoSeg/SAM2 legacy UID overlay — D={masks.shape[0]}\n"
        f"final-output mask_{autoseg_reverse_index:03d}.npz; UIDs 0–{masks.shape[0] - 1}"
    )
    uid_axis.axis("off")

    draw_camera_trajectory(camera_axis, poses)
    camera_axis.set_title(
        "Initial camera trajectory — coarse cam2label\n"
        "saved HoloLens camera-to-world substitution"
    )
    figure.suptitle("iTACO saved initial values — current cabinet run", fontsize=17, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def cloud_for_display(
    cloud_path: Path, max_points: int, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) == 0:
        raise RuntimeError(f"Empty point cloud: {cloud_path}")
    if colors.shape != points.shape:
        colors = np.full(points.shape, 0.58, dtype=np.float64)
    robust_low, robust_high = np.percentile(points, [1.0, 99.0], axis=0)
    robust = np.all((points >= robust_low) & (points <= robust_high), axis=1)
    display_ids = np.flatnonzero(robust)
    if len(display_ids) > max_points:
        rng = np.random.default_rng(random_seed)
        display_ids = np.sort(rng.choice(display_ids, max_points, replace=False))
    return points[display_ids], colors[display_ids], robust_low, robust_high


def axis_line(origin: np.ndarray, direction: np.ndarray, length: float) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    return np.stack([origin - 0.5 * length * direction, origin + 0.5 * length * direction])


def draw_axis_cloud_panel(
    axis3d: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray,
    direction: np.ndarray,
    origin: np.ndarray,
    length: float,
    axis_color: str,
    title: str,
) -> np.ndarray:
    line = axis_line(origin, direction, length)
    axis3d.scatter(
        points[:, 0], points[:, 1], points[:, 2], s=0.28,
        c=np.clip(colors, 0.0, 1.0), alpha=0.24, depthshade=False, rasterized=True,
    )
    axis3d.plot(line[:, 0], line[:, 1], line[:, 2], color=axis_color, linewidth=4)
    start = origin - 0.36 * length * direction
    axis3d.quiver(
        *start, *direction, length=0.72 * length, normalize=True,
        color=axis_color, linewidth=3, arrow_length_ratio=0.12,
    )
    axis3d.scatter(*origin, color=axis_color, s=55, marker="o", depthshade=False)
    equal_3d_limits(axis3d, np.vstack([points, line]))
    axis3d.set_xlabel("world X (m)")
    axis3d.set_ylabel("world Y (m)")
    axis3d.set_zlabel("world Z (m)")
    axis3d.view_init(elev=24, azim=-58)
    axis3d.set_title(title)
    return line


def tube_samples(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float = 0.004,
    longitudinal: int = 160,
    radial: int = 10,
) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(direction, helper)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(direction, basis_u)
    along = np.linspace(-0.5 * length, 0.5 * length, longitudinal)
    angles = np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)
    return np.asarray(
        [
            origin + distance * direction + radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
            for distance in along
            for angle in angles
        ]
    )


def render_initial_axes(
    image_path: Path,
    ply_path: Path,
    cloud_path: Path,
    coarse_root: Path,
    max_cloud_points: int,
    axis_length: float,
    random_seed: int,
) -> dict:
    points, colors, robust_low, robust_high = cloud_for_display(
        cloud_path, max_cloud_points, random_seed
    )
    prismatic_axis = np.load(coarse_root / "prismatic/joint_axis.npy").astype(np.float64)
    prismatic_pos = np.load(coarse_root / "prismatic/joint_pos.npy").astype(np.float64)
    revolute_axis = np.load(coarse_root / "revolute/joint_axis.npy").astype(np.float64)
    revolute_pos = np.load(coarse_root / "revolute/joint_pos.npy").astype(np.float64)
    prismatic_axis /= np.linalg.norm(prismatic_axis)
    revolute_axis /= np.linalg.norm(revolute_axis)

    # Prismatic origins are not physically identifiable. Preserve the saved zero
    # position in the manifest, but anchor its saved direction at the cloud median
    # so the direction can actually be inspected over the object.
    prismatic_display_anchor = np.median(points, axis=0)
    nearest_revolute_distance = float(np.min(np.linalg.norm(points - revolute_pos, axis=1)))

    figure = plt.figure(figsize=(16, 8), dpi=150)
    prismatic_panel = figure.add_subplot(1, 2, 1, projection="3d")
    revolute_panel = figure.add_subplot(1, 2, 2, projection="3d")
    draw_axis_cloud_panel(
        prismatic_panel, points, colors, prismatic_axis, prismatic_display_anchor,
        axis_length, "#00e5ff",
        "Initial prismatic direction (coarse)\n"
        "display anchor = cloud median; saved joint_pos=[0,0,0] is arbitrary",
    )
    draw_axis_cloud_panel(
        revolute_panel, points, colors, revolute_axis, revolute_pos,
        axis_length, "#ff3b30",
        "Initial revolute axis (coarse)\n"
        f"exact saved joint_pos; nearest displayed cloud={nearest_revolute_distance:.3f} m",
    )
    figure.suptitle(
        "Official-compatible coarse joint candidates over saved cabinet surface\n"
        "cyan = prismatic direction; red = revolute axis",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)

    prismatic_tube = tube_samples(prismatic_display_anchor, prismatic_axis, axis_length)
    revolute_tube = tube_samples(revolute_pos, revolute_axis, axis_length)
    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(np.vstack([points, prismatic_tube, revolute_tube]))
    combined.colors = o3d.utility.Vector3dVector(
        np.vstack(
            [
                colors,
                np.tile(np.array([[0.0, 0.90, 1.0]]), (len(prismatic_tube), 1)),
                np.tile(np.array([[1.0, 0.23, 0.19]]), (len(revolute_tube), 1)),
            ]
        )
    )
    if not o3d.io.write_point_cloud(str(ply_path), combined, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {ply_path}")

    return {
        "surface_display_point_count": int(len(points)),
        "surface_robust_bounds_1_99_percentile": {
            "low": robust_low.tolist(),
            "high": robust_high.tolist(),
        },
        "prismatic": {
            "saved_axis_unit": prismatic_axis.tolist(),
            "saved_joint_pos": prismatic_pos.tolist(),
            "display_anchor": prismatic_display_anchor.tolist(),
            "display_anchor_policy": "cloud coordinate-wise median because prismatic origin is arbitrary",
        },
        "revolute": {
            "saved_axis_unit": revolute_axis.tolist(),
            "saved_joint_pos_and_display_anchor": revolute_pos.tolist(),
            "nearest_display_cloud_distance_m": nearest_revolute_distance,
        },
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "diagnostics/initial_values"
    )
    if not 0.0 <= args.uid_alpha <= 1.0:
        raise ValueError("--uid-alpha must be in [0,1]")
    if args.max_cloud_points <= 0 or args.axis_length_m <= 0:
        raise ValueError("--max-cloud-points and --axis-length-m must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_mapping_path, forward_mapping = one_mapping(run_root, "interaction_forward_")
    reverse_mapping_path, reverse_mapping = one_mapping(run_root, "autoseg_reverse_")
    frame_count = len(forward_mapping)
    frame_index = args.frame_index if args.frame_index >= 0 else frame_count + args.frame_index
    if not 0 <= frame_index < frame_count:
        raise IndexError(f"frame-index {args.frame_index} resolves to {frame_index}, outside 0..{frame_count - 1}")
    source_index = int(forward_mapping[frame_index]["source_index"])
    reverse_lookup = {int(item["source_index"]): int(item["local_index"]) for item in reverse_mapping}
    if source_index not in reverse_lookup:
        raise RuntimeError(f"Source frame {source_index} is absent from {reverse_mapping_path}")
    reverse_index = reverse_lookup[source_index]

    official = run_root / "official"
    rgb_path = require_file(official / f"view/rgb/{frame_index:06d}.jpg")
    moving_map_path = require_file(
        official / f"preprocess/monst3r/dynamic_mask_{frame_index}.png"
    )
    autoseg_path = require_file(
        official / f"preprocess/video_segment_reverse/small/final-output/mask_{reverse_index:03d}.npz"
    )
    coarse_root = official / "prediction/coarse_prediction/monst3r/0"
    cloud_path = require_file(official / "view/surface/surface.ply")
    native_manifest_path = require_file(official / "native_gt_view_manifest.json")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    moving_mask = np.asarray(Image.open(moving_map_path).convert("L")) > 0
    masks, original_mask_shape = load_autoseg(autoseg_path)
    pose_paths, poses = camera_poses(coarse_root)
    if not (rgb.shape[:2] == moving_mask.shape == masks.shape[1:]):
        raise ValueError(
            f"Panel shape mismatch: rgb={rgb.shape[:2]}, moving={moving_mask.shape}, masks={masks.shape[1:]}"
        )
    if len(poses) != frame_count:
        raise ValueError(f"Camera/frame count mismatch: {len(poses)} versus {frame_count}")

    initial_values_path = output_dir / "itaco_initial_values_2x2.png"
    axes_image_path = output_dir / "itaco_coarse_initial_axes_over_point_cloud.png"
    axes_ply_path = output_dir / "itaco_coarse_initial_axes_over_point_cloud.ply"
    render_initial_values(
        initial_values_path, rgb, moving_mask, masks, poses,
        frame_index, source_index, reverse_index, args.uid_alpha,
    )
    axes_report = render_initial_axes(
        axes_image_path, axes_ply_path, cloud_path, coarse_root,
        args.max_cloud_points, args.axis_length_m, args.random_seed,
    )

    native_manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "kind": "non-official diagnostic visualization of preserved official-compatible inputs and coarse outputs",
        "run_root": str(run_root),
        "command": [sys.executable, *sys.argv],
        "frame": {
            "forward_local_index": frame_index,
            "source_index": source_index,
            "autoseg_reverse_local_index": reverse_index,
            "forward_mapping": str(forward_mapping_path.resolve()),
            "reverse_mapping": str(reverse_mapping_path.resolve()),
        },
        "inputs": {
            "rgb": str(rgb_path.resolve()),
            "monst3r_slot_moving_map": str(moving_map_path.resolve()),
            "autoseg_final_mask": str(autoseg_path.resolve()),
            "coarse_camera_pose_files": [str(path.resolve()) for path in pose_paths],
            "coarse_root": str(coarse_root.resolve()),
            "surface_point_cloud": str(cloud_path.resolve()),
            "native_view_manifest": str(native_manifest_path.resolve()),
        },
        "explicit_substitutions": native_manifest.get("explicit_substitutions", {}),
        "autoseg": {
            "mask_key": "a",
            "original_shape": list(original_mask_shape),
            "tracked_segment_count_D": int(masks.shape[0]),
            "uid_range": [0, int(masks.shape[0] - 1)],
            "nonempty_uid_count_selected_frame": int(
                np.count_nonzero(masks.reshape(masks.shape[0], -1).any(axis=1))
            ),
            "legacy_uid_note": "UID denotes final-output layer index, not stage-1.5 persistent proposal_id",
            "overlay_alpha": args.uid_alpha,
        },
        "moving_map": {
            "encoding": "saved binary PNG",
            "coverage_fraction_selected_frame": float(moving_mask.mean()),
            "provenance_note": "stored in the MonST3R-compatible slot; this run substituted HoloLens metric depth/odometry",
        },
        "camera": {
            "pose_shape": list(poses.shape),
            "pose_convention": "saved coarse cam2label / camera-to-world-compatible 4x4 matrices",
        },
        "axes": axes_report,
        "visualization_parameters": {
            "random_seed": args.random_seed,
            "max_cloud_points": args.max_cloud_points,
            "axis_length_m": args.axis_length_m,
            "surface_display_filter": "inside per-axis 1st..99th percentile bounds",
        },
        "outputs": {
            "initial_values_2x2": str(initial_values_path.resolve()),
            "coarse_axes_image": str(axes_image_path.resolve()),
            "coarse_axes_point_cloud": str(axes_ply_path.resolve()),
        },
    }
    manifest_path = output_dir / "itaco_initial_values_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"D={masks.shape[0]}")
    print(initial_values_path.resolve())
    print(axes_image_path.resolve())
    print(axes_ply_path.resolve())
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
