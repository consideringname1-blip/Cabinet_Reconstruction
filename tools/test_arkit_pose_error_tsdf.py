#!/usr/bin/env python3
"""Isolate ARKit-scale pose error in RGB-D/TSDF fusion.

This is a diagnostic experiment, not an official iTACO stage.  It keeps the
HoloLens world PLY observations, RGB images, deterministic cabinet masks, and
TSDF settings fixed.  Only camera-to-world poses change:

1. the recorded HoloLens pose baseline;
2. a time-correlated synthetic perturbation calibrated to the previously
   measured iPhone ARKit static-pair ICP residual scale.

The result measures dominant-plane thickness (layering) and mesh-normal /
triangle degradation (the "hedgehog" appearance).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arkit-diagnostic", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=5)
    parser.add_argument("--frame-end", type=int, default=165)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--tsdf-voxel", type=float, default=0.004)
    parser.add_argument("--tsdf-trunc", type=float, default=0.024)
    parser.add_argument("--surface-voxel", type=float, default=0.004)
    parser.add_argument(
        "--world-aabb",
        default="-1.25,-0.35,-1.635,-0.98,-2.10,-1.05",
        help="Baseline-world cabinet ROI: xmin,xmax,ymin,ymax,zmin,zmax",
    )
    return parser.parse_args()


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def load_depth_timestamps(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def load_rgb_paths(pinhole: Path) -> list[Path]:
    paths = []
    for line in (pinhole / "rgb.txt").read_text().splitlines():
        if not line.strip():
            continue
        relative = line.split(maxsplit=1)[1].replace("\\", "/")
        paths.append(pinhole / relative)
    return paths


def project_world_depth(
    world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    camera = (world - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z >= depth_min) & (z <= depth_max)
    camera = camera[valid]
    z = camera[:, 2]
    x = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
    y = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    flat = np.full(height * width, np.inf, dtype=np.float64)
    for u in (np.floor(x), np.ceil(x)):
        for v in (np.floor(y), np.ceil(y)):
            ui = u.astype(np.int64)
            vi = v.astype(np.int64)
            inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            np.minimum.at(flat, vi[inside] * width + ui[inside], z[inside])
    depth = flat.reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def cabinet_mask(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Deterministic white-object mask, identical for every pose scenario."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = (
        (hsv[..., 2] >= 145)
        & (hsv[..., 1] <= 105)
        & (depth > 0.25)
        & (depth < 2.5)
    ).astype(np.uint8)
    height, width = candidate.shape
    candidate[: max(3, height // 40)] = 0
    candidate[:, : max(3, width // 40)] = 0
    candidate[:, width - max(3, width // 40) :] = 0
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return candidate.astype(bool)
    center = np.array([width * 0.5, height * 0.52])
    best_label = None
    best_score = -np.inf
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 180:
            continue
        distance = np.linalg.norm((centroids[label] - center) / np.array([width, height]))
        score = math.log(area) - 3.2 * distance
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        return candidate.astype(bool)
    mask = labels == best_label
    mask = binary_fill_holes(mask)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    return mask.astype(bool)


def depth_to_camera_points(
    depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.column_stack((x, y, z))
    colors = rgb[v, u].astype(np.float64) / 255.0
    return points, colors


def world_aabb_mask(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    camera = np.column_stack(
        (
            (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        )
    )
    world = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    inside = (
        (world[:, 0] >= bounds[0])
        & (world[:, 0] <= bounds[1])
        & (world[:, 1] >= bounds[2])
        & (world[:, 1] <= bounds[3])
        & (world[:, 2] >= bounds[4])
        & (world[:, 2] <= bounds[5])
    )
    mask = np.zeros((height, width), dtype=bool)
    mask[v[inside], u[inside]] = True
    return mask


def pose_error_stats(reference: np.ndarray, perturbed: np.ndarray) -> dict:
    absolute_translation = np.linalg.norm(
        perturbed[:, :3, 3] - reference[:, :3, 3], axis=1
    )
    absolute_rotation = []
    for gt, test in zip(reference, perturbed):
        delta = test[:3, :3] @ gt[:3, :3].T
        absolute_rotation.append(np.degrees(Rotation.from_matrix(delta).magnitude()))
    relative_translation = []
    relative_rotation = []
    for index in range(len(reference) - 1):
        error_a_t = perturbed[index, :3, 3] - reference[index, :3, 3]
        error_b_t = perturbed[index + 1, :3, 3] - reference[index + 1, :3, 3]
        relative_translation.append(np.linalg.norm(error_b_t - error_a_t))
        error_a_r = perturbed[index, :3, :3] @ reference[index, :3, :3].T
        error_b_r = perturbed[index + 1, :3, :3] @ reference[index + 1, :3, :3].T
        relative_rotation.append(
            np.degrees(Rotation.from_matrix(error_b_r @ error_a_r.T).magnitude())
        )

    def summarize(values: np.ndarray, scale: float = 1.0) -> dict:
        values = np.asarray(values) * scale
        return {
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "absolute_translation_cm": summarize(absolute_translation, 100.0),
        "absolute_rotation_deg": summarize(absolute_rotation),
        "adjacent_error_change_translation_cm": summarize(relative_translation, 100.0),
        "adjacent_error_change_rotation_deg": summarize(relative_rotation),
    }


def synthetic_arkit_poses(
    poses: np.ndarray,
    diagnostic: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Create stable, time-correlated errors with empirical adjacent magnitudes."""
    records = [
        item
        for item in diagnostic["pairs"]["closed"]
        if item.get("accepted", False)
    ]
    translation_magnitudes = np.array(
        [item["residual_translation_m"] for item in records], dtype=np.float64
    )
    rotation_magnitudes = np.radians(
        [item["residual_rotation_deg"] for item in records]
    )
    rng = np.random.default_rng(seed)
    count = len(poses)
    translation_error = np.zeros((count, 3), dtype=np.float64)
    rotation_error = np.zeros((count, 3), dtype=np.float64)
    rho = 0.82
    for index in range(1, count):
        direction_t = rng.normal(size=3)
        direction_t /= np.linalg.norm(direction_t)
        direction_r = rng.normal(size=3)
        direction_r /= np.linalg.norm(direction_r)
        mag_t = float(rng.choice(translation_magnitudes))
        mag_r = float(rng.choice(rotation_magnitudes))
        translation_error[index] = rho * translation_error[index - 1] + mag_t * direction_t
        rotation_error[index] = rho * rotation_error[index - 1] + mag_r * direction_r

    # Match the measured median adjacent-pair correction.  The empirical
    # residual is an upper bound, but preserving it answers the user's stress
    # test directly.
    target_t = float(np.median(translation_magnitudes))
    target_r = float(np.median(rotation_magnitudes))
    observed_t = np.median(np.linalg.norm(np.diff(translation_error, axis=0), axis=1))
    observed_r = np.median(np.linalg.norm(np.diff(rotation_error, axis=0), axis=1))
    translation_error *= target_t / max(observed_t, 1e-12)
    rotation_error *= target_r / max(observed_r, 1e-12)

    perturbed = poses.copy()
    for index in range(count):
        error_rotation = Rotation.from_rotvec(rotation_error[index]).as_matrix()
        perturbed[index, :3, :3] = error_rotation @ poses[index, :3, :3]
        perturbed[index, :3, 3] = poses[index, :3, 3] + translation_error[index]
    return perturbed, {
        "seed": seed,
        "rho": rho,
        "calibration_source": diagnostic["meaning"],
        "target_adjacent_translation_cm_p50": target_t * 100.0,
        "target_adjacent_rotation_deg_p50": math.degrees(target_r),
        "realized": pose_error_stats(poses, perturbed),
    }


def make_cloud(
    camera_points: list[np.ndarray],
    colors: list[np.ndarray],
    poses: np.ndarray,
    voxel: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    all_points = []
    all_colors = []
    for points, color, pose in zip(camera_points, colors, poses):
        world = points @ pose[:3, :3].T + pose[:3, 3]
        all_points.append(world)
        all_colors.append(color)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(all_points))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))
    raw_count = len(cloud.points)
    cloud = cloud.voxel_down_sample(voxel)
    return cloud, raw_count


def make_tsdf_mesh(
    payload: list[tuple[np.ndarray, np.ndarray]],
    poses: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    voxel: float,
    trunc: float,
    depth_max: float,
) -> o3d.geometry.TriangleMesh:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for (rgb, depth), pose in zip(payload, poses):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def dominant_plane(cloud: o3d.geometry.PointCloud) -> dict:
    model, indices = cloud.segment_plane(
        distance_threshold=0.008, ransac_n=3, num_iterations=4000
    )
    normal = np.asarray(model[:3], dtype=np.float64)
    norm = np.linalg.norm(normal)
    normal /= norm
    offset = float(model[3] / norm)
    points = np.asarray(cloud.points)
    inliers = points[np.asarray(indices)]
    center = np.median(inliers, axis=0)
    if normal @ (center - np.mean(points, axis=0)) < 0:
        normal = -normal
        offset = -offset
    axis_u, axis_v = plane_basis(normal)
    u = (inliers - center) @ axis_u
    v = (inliers - center) @ axis_v
    bounds_u = np.percentile(u, [5, 95])
    bounds_v = np.percentile(v, [5, 95])
    return {
        "normal": normal,
        "offset": offset,
        "center": center,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "bounds_u": bounds_u,
        "bounds_v": bounds_v,
        "inlier_count": len(inliers),
    }


def select_plane_patch(points: np.ndarray, plane: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - plane["center"]
    u = centered @ plane["axis_u"]
    v = centered @ plane["axis_v"]
    distance = points @ plane["normal"] + plane["offset"]
    margin_u = 0.02
    margin_v = 0.02
    selected = (
        (u >= plane["bounds_u"][0] + margin_u)
        & (u <= plane["bounds_u"][1] - margin_u)
        & (v >= plane["bounds_v"][0] + margin_v)
        & (v <= plane["bounds_v"][1] - margin_v)
        & (np.abs(distance) <= 0.08)
    )
    return selected, u, distance


def layer_metrics(cloud: o3d.geometry.PointCloud, plane: dict) -> tuple[dict, dict]:
    points = np.asarray(cloud.points)
    selected, u, distance = select_plane_patch(points, plane)
    d = distance[selected]
    hist, edges = np.histogram(d, bins=120, range=(-0.08, 0.08))
    smooth = np.convolve(hist.astype(float), np.ones(5) / 5.0, mode="same")
    prominence = max(5.0, smooth.max() * 0.08)
    peaks, properties = find_peaks(smooth, prominence=prominence, distance=5)
    order = np.argsort(properties.get("prominences", np.array([])))[::-1]
    peaks = peaks[order[:5]]
    peak_cm = [float((edges[p] + edges[p + 1]) * 50.0) for p in peaks]
    metrics = {
        "patch_points": int(len(d)),
        "signed_distance_cm": {
            "p05": float(np.percentile(d, 5) * 100.0),
            "p50": float(np.percentile(d, 50) * 100.0),
            "p95": float(np.percentile(d, 95) * 100.0),
        },
        "plane_thickness_p05_p95_cm": float(
            (np.percentile(d, 95) - np.percentile(d, 5)) * 100.0
        ),
        "histogram_peak_count": int(len(peaks)),
        "histogram_peak_positions_cm": peak_cm,
    }
    plot_data = {
        "u": u[selected],
        "distance": d,
        "hist": hist,
        "edges": edges,
    }
    return metrics, plot_data


def mesh_metrics(mesh: o3d.geometry.TriangleMesh, plane: dict) -> dict:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return {"triangles": 0}
    tri = vertices[triangles]
    centroids = tri.mean(axis=1)
    selected, _, _ = select_plane_patch(centroids, plane)
    tri = tri[selected]
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge_a, edge_b)
    double_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(double_area[:, None], 1e-12)
    angle = np.degrees(
        np.arccos(np.clip(np.abs(normals @ plane["normal"]), 0.0, 1.0))
    )
    edges = np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ],
        axis=1,
    )
    aspect = np.max(edges, axis=1) ** 2 / np.maximum(double_area, 1e-12)
    return {
        "triangles": int(len(tri)),
        "surface_area_m2": float(np.sum(double_area) * 0.5),
        "normal_deviation_deg": {
            "p50": float(np.percentile(angle, 50)),
            "p90": float(np.percentile(angle, 90)),
            "p95": float(np.percentile(angle, 95)),
            "fraction_over_30deg": float(np.mean(angle > 30.0)),
        },
        "triangle_aspect_proxy": {
            "p50": float(np.percentile(aspect, 50)),
            "p90": float(np.percentile(aspect, 90)),
            "p95": float(np.percentile(aspect, 95)),
        },
    }


def nearest_metrics(reference: o3d.geometry.PointCloud, test: o3d.geometry.PointCloud) -> dict:
    reference_points = np.asarray(reference.points)
    test_points = np.asarray(test.points)
    tree_ref = cKDTree(reference_points)
    tree_test = cKDTree(test_points)
    test_to_ref = tree_ref.query(test_points, workers=-1)[0]
    ref_to_test = tree_test.query(reference_points, workers=-1)[0]

    def summary(values: np.ndarray) -> dict:
        return {
            "p50_cm": float(np.percentile(values, 50) * 100.0),
            "p90_cm": float(np.percentile(values, 90) * 100.0),
            "p95_cm": float(np.percentile(values, 95) * 100.0),
        }

    return {
        "test_to_baseline": summary(test_to_ref),
        "baseline_to_test": summary(ref_to_test),
        "symmetric_mean_cm": float(
            (np.mean(test_to_ref) + np.mean(ref_to_test)) * 50.0
        ),
    }


def save_mask_sheet(records: list[dict], path: Path) -> None:
    indices = np.linspace(0, len(records) - 1, 9).round().astype(int)
    panels = []
    for index in indices:
        record = records[index]
        rgb = record["rgb"].copy()
        overlay = rgb.copy()
        overlay[record["mask"]] = (
            0.35 * overlay[record["mask"]] + 0.65 * np.array([255, 40, 40])
        ).astype(np.uint8)
        panel = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(
            panel,
            str(record["source_index"]),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        panels.append(panel)
    sheet = np.vstack([np.hstack(panels[i : i + 3]) for i in range(0, 9, 3)])
    cv2.imwrite(str(path), sheet)


def save_comparison_plot(plot_data: dict[str, dict], path: Path) -> None:
    names = list(plot_data)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4.2 * len(names)))
    if len(names) == 1:
        axes = np.asarray([axes])
    for row, name in enumerate(names):
        data = plot_data[name]
        u = data["u"]
        distance = data["distance"]
        if len(u) > 30000:
            selection = np.linspace(0, len(u) - 1, 30000).astype(int)
            u = u[selection]
            distance = distance[selection]
        axes[row, 0].scatter(u, distance * 100.0, s=1, alpha=0.28)
        axes[row, 0].set_ylim(-8, 8)
        axes[row, 0].set_title(f"{name}: dominant-plane cross section")
        axes[row, 0].set_xlabel("in-plane coordinate (m)")
        axes[row, 0].set_ylabel("normal distance (cm)")
        centers = (data["edges"][:-1] + data["edges"][1:]) * 50.0
        axes[row, 1].plot(centers, data["hist"])
        axes[row, 1].set_xlim(-8, 8)
        axes[row, 1].set_title(f"{name}: layer histogram")
        axes[row, 1].set_xlabel("normal distance (cm)")
        axes[row, 1].set_ylabel("points")
        axes[row, 1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pinhole = args.source_root / "pinhole_projection"
    all_poses = load_odometry(pinhole / "odometry.log")
    timestamps = load_depth_timestamps(pinhole / "depth.txt")
    rgb_paths = load_rgb_paths(pinhole)
    frame_indices = list(
        range(args.frame_start, args.frame_end + 1, args.frame_stride)
    )
    world_aabb = np.array(
        [float(value) for value in args.world_aabb.split(",")], dtype=np.float64
    )
    if world_aabb.shape != (6,):
        raise ValueError("--world-aabb must contain six comma-separated values")
    selected_poses = all_poses[frame_indices]
    fx, fy, cx, cy = np.loadtxt(pinhole / "calibration.txt").reshape(-1)[:4]
    intrinsic_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    )
    first_rgb = cv2.cvtColor(
        cv2.imread(str(rgb_paths[frame_indices[0]])), cv2.COLOR_BGR2RGB
    )
    height, width = first_rgb.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )

    camera_points = []
    colors = []
    payload = []
    records = []
    for sequence_index, source_index in enumerate(frame_indices):
        ply_path = (
            args.source_root
            / "Depth Long Throw"
            / f"{timestamps[source_index]}.ply"
        )
        world = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
        rgb = cv2.cvtColor(
            cv2.imread(str(rgb_paths[source_index])), cv2.COLOR_BGR2RGB
        )
        depth = project_world_depth(
            world,
            all_poses[source_index],
            intrinsic_matrix,
            height,
            width,
            args.depth_min,
            args.depth_max,
        )
        mask = world_aabb_mask(
            depth, intrinsic_matrix, all_poses[source_index], world_aabb
        )
        masked_depth = np.where(mask, depth, 0.0).astype(np.float32)
        points, point_colors = depth_to_camera_points(
            depth, rgb, mask, intrinsic_matrix
        )
        camera_points.append(points)
        colors.append(point_colors)
        payload.append((rgb, masked_depth))
        records.append(
            {
                "sequence_index": sequence_index,
                "source_index": source_index,
                "rgb": rgb,
                "mask": mask,
                "mask_fraction": float(mask.mean()),
                "valid_points": int(len(points)),
            }
        )
        print(
            f"[prepare {sequence_index + 1:02d}/{len(frame_indices):02d}] "
            f"source={source_index:03d} mask={mask.mean():.3f} points={len(points)}"
        )

    save_mask_sheet(records, args.output_root / "cabinet_mask_validation.jpg")
    diagnostic = json.loads(args.arkit_diagnostic.read_text())
    noisy_poses, noise_report = synthetic_arkit_poses(
        selected_poses, diagnostic, args.seed
    )
    np.save(args.output_root / "hololens_cam2world.npy", selected_poses)
    np.save(args.output_root / "arkit_like_cam2world.npy", noisy_poses)

    scenarios = {
        "hololens_pose_baseline": selected_poses,
        "arkit_like_pose": noisy_poses,
    }
    clouds = {}
    meshes = {}
    raw_counts = {}
    for name, poses in scenarios.items():
        print(f"[fusion] {name}")
        directory = args.output_root / name
        directory.mkdir(exist_ok=True)
        cloud, raw_count = make_cloud(
            camera_points, colors, poses, args.surface_voxel
        )
        mesh = make_tsdf_mesh(
            payload,
            poses,
            intrinsic,
            args.tsdf_voxel,
            args.tsdf_trunc,
            args.depth_max,
        )
        o3d.io.write_point_cloud(str(directory / "cabinet_fused.ply"), cloud)
        o3d.io.write_triangle_mesh(str(directory / "cabinet_tsdf_mesh.ply"), mesh)
        clouds[name] = cloud
        meshes[name] = mesh
        raw_counts[name] = raw_count

    plane = dominant_plane(clouds["hololens_pose_baseline"])
    plane_cloud = clouds["hololens_pose_baseline"].select_by_index(
        np.nonzero(
            select_plane_patch(
                np.asarray(clouds["hololens_pose_baseline"].points), plane
            )[0]
        )[0]
    )
    o3d.io.write_point_cloud(
        str(args.output_root / "dominant_plane_patch_baseline.ply"), plane_cloud
    )

    metrics = {}
    plots = {}
    for name in scenarios:
        layer, plot = layer_metrics(clouds[name], plane)
        mesh = mesh_metrics(meshes[name], plane)
        metrics[name] = {
            "raw_fused_points": raw_counts[name],
            "voxelized_points": len(clouds[name].points),
            "tsdf_vertices": len(meshes[name].vertices),
            "tsdf_triangles": len(meshes[name].triangles),
            "dominant_plane_layering": layer,
            "dominant_plane_mesh": mesh,
        }
        plots[name] = plot
    metrics["arkit_like_pose"]["nearest_to_baseline"] = nearest_metrics(
        clouds["hololens_pose_baseline"], clouds["arkit_like_pose"]
    )
    save_comparison_plot(
        plots, args.output_root / "pose_error_layering_comparison.png"
    )

    report = {
        "experiment_type": "non-official controlled diagnostic; no MONST3R, AutoSeg, or iTACO articulation inference",
        "question": "Can ARKit-scale pose error alone produce point-cloud layering and a spiky TSDF plane?",
        "controlled_variables": [
            "same HoloLens Long Throw world PLY observations",
            "same PLY-to-PV reprojected depth",
            "same deterministic per-frame cabinet masks",
            "same RGB",
            "same frame selection",
            "same TSDF voxel and truncation",
        ],
        "changed_variable": "camera-to-world pose only",
        "frame_indices": frame_indices,
        "frame_interval_approx_s": args.frame_stride / 5.0,
        "frame_records": [
            {
                key: value
                for key, value in record.items()
                if key not in ("rgb", "mask")
            }
            for record in records
        ],
        "parameters": {
            "depth_range_m": [args.depth_min, args.depth_max],
            "surface_voxel_m": args.surface_voxel,
            "tsdf_voxel_m": args.tsdf_voxel,
            "tsdf_trunc_m": args.tsdf_trunc,
            "seed": args.seed,
            "baseline_world_aabb_xyz_m": world_aabb.tolist(),
        },
        "arkit_like_noise": noise_report,
        "dominant_plane": {
            "normal": plane["normal"].tolist(),
            "offset": plane["offset"],
            "center": plane["center"].tolist(),
            "bounds_u_m": plane["bounds_u"].tolist(),
            "bounds_v_m": plane["bounds_v"].tolist(),
            "baseline_ransac_inliers": plane["inlier_count"],
        },
        "metrics": metrics,
        "limitations": [
            "The iPhone ICP residual is an upper bound: it also includes depth noise, calibration error, mask leakage, and ICP ambiguity.",
            "Only residual magnitudes were saved, so synthetic perturbation directions are seeded random directions.",
            "This experiment diagnoses reconstruction sensitivity; it does not estimate the exact unknown iPhone ground-truth trajectory.",
        ],
    }
    (args.output_root / "EXPERIMENT_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    command = (
        "python tools/test_arkit_pose_error_tsdf.py "
        f"--source-root {args.source_root} "
        f"--output-root {args.output_root} "
        f"--arkit-diagnostic {args.arkit_diagnostic} "
        f"--frame-start {args.frame_start} --frame-end {args.frame_end} "
        f"--frame-stride {args.frame_stride} --seed {args.seed} "
        f"--depth-min {args.depth_min} --depth-max {args.depth_max} "
        f"--tsdf-voxel {args.tsdf_voxel} --tsdf-trunc {args.tsdf_trunc} "
        f"--surface-voxel {args.surface_voxel}"
        f" --world-aabb {args.world_aabb}"
    )
    (args.output_root / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
