#!/usr/bin/env python3
"""Build an object-only iTACO surface from masked GT RGB-D and GT poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-min", type=float, default=0.3)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--mask-erode", type=int, default=1)
    parser.add_argument("--mask-fraction-min", type=float, default=0.08)
    parser.add_argument("--mask-fraction-max", type=float, default=0.22)
    parser.add_argument("--registration-voxel", type=float, default=0.02)
    parser.add_argument("--icp-distance", type=float, default=0.06)
    parser.add_argument("--surface-voxel", type=float, default=0.003)
    parser.add_argument("--tsdf-voxel", type=float, default=0.006)
    parser.add_argument("--tsdf-trunc", type=float, default=0.03)
    return parser.parse_args()


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    poses = []
    for start in range(0, len(lines), 5):
        poses.append(np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]]))
    return np.stack(poses)


def load_intrinsics(path: Path, width: int, height: int) -> tuple[np.ndarray, o3d.camera.PinholeCameraIntrinsic]:
    fx, fy, cx, cy = np.loadtxt(path).reshape(-1)[:4]
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return matrix, intrinsic


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(value))


def masked_frame_cloud(
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    depth_min: float,
    depth_max: float,
    erode: int,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = np.load(mask_path).squeeze().astype(np.uint8)
    if bgr is None or depth_mm is None:
        raise FileNotFoundError((rgb_path, depth_path))
    if erode > 0:
        kernel = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m >= depth_min) & (depth_m <= depth_max)
    masked_depth = np.where(valid, depth_m, 0.0).astype(np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(masked_depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return cloud, rgb, masked_depth, valid


def prepared_cloud(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=40)
    )
    return down


def register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_distance,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_distance, result.transformation
    )
    return result.transformation, information, float(result.fitness), float(result.inlier_rmse)


def refine_poses(
    clouds: list[o3d.geometry.PointCloud],
    raw_poses: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, list[dict]]:
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in raw_poses:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose.copy()))

    records = []
    frame_count = len(clouds)
    pairs = {(i, i + 1, False) for i in range(frame_count - 1)}
    for interval in (5, 10):
        for source in range(0, frame_count - interval, interval):
            target = source + interval
            center_distance = np.linalg.norm(raw_poses[source, :3, 3] - raw_poses[target, :3, 3])
            if center_distance < 0.55:
                pairs.add((source, target, True))
    for source in range(frame_count):
        for target in range(source + 15, frame_count, 10):
            center_distance = np.linalg.norm(raw_poses[source, :3, 3] - raw_poses[target, :3, 3])
            if center_distance < 0.22:
                pairs.add((source, target, True))

    for source, target, is_loop in sorted(pairs):
        raw_relative = np.linalg.inv(raw_poses[target]) @ raw_poses[source]
        transformation, information, fitness, rmse = register_pair(
            clouds[source], clouds[target], raw_relative, max_distance
        )
        correction = transformation @ np.linalg.inv(raw_relative)
        correction_translation = float(np.linalg.norm(correction[:3, 3]))
        correction_rotation = rotation_angle_deg(correction[:3, :3])
        accepted = (
            fitness >= (0.20 if is_loop else 0.12)
            and rmse <= max_distance
            and correction_translation <= 0.08
            and correction_rotation <= 8.0
        )
        chosen = transformation if accepted else raw_relative
        chosen_information = information if accepted else np.eye(6) * 20.0
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source,
                target,
                chosen,
                chosen_information,
                uncertain=is_loop,
            )
        )
        records.append(
            {
                "source": source,
                "target": target,
                "loop": is_loop,
                "fitness": fitness,
                "rmse_m": rmse,
                "correction_translation_m": correction_translation,
                "correction_rotation_deg": correction_rotation,
                "accepted": accepted,
            }
        )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    refined = np.stack([node.pose for node in graph.nodes])
    return refined, records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_files = sorted((args.input_dir / "rgb").glob("*.png"))
    if not rgb_files:
        rgb_files = sorted((args.input_dir / "jpg").glob("*.jpg"))
    depth_files = sorted((args.input_dir / "depth").glob("*.png"))
    mask_files = sorted(args.mask_dir.glob("*.npy"))
    raw_poses = load_odometry(args.input_dir / "odometry.log")
    source_count = min(len(rgb_files), len(depth_files), len(mask_files), len(raw_poses))
    if source_count == 0:
        raise RuntimeError("No aligned RGB-D-mask-pose frames")
    rgb_files, depth_files, mask_files, raw_poses = (
        rgb_files[:source_count], depth_files[:source_count], mask_files[:source_count], raw_poses[:source_count]
    )
    mask_fractions = [float(np.load(path).squeeze().astype(bool).mean()) for path in mask_files]
    selected_indices = [
        index for index, fraction in enumerate(mask_fractions)
        if args.mask_fraction_min <= fraction <= args.mask_fraction_max
    ]
    excluded_indices = sorted(set(range(source_count)) - set(selected_indices))
    if len(selected_indices) < 3:
        raise RuntimeError(f"Too few valid cabinet masks: {selected_indices}")
    rgb_files = [rgb_files[index] for index in selected_indices]
    depth_files = [depth_files[index] for index in selected_indices]
    mask_files = [mask_files[index] for index in selected_indices]
    raw_poses = raw_poses[selected_indices]
    count = len(selected_indices)
    first_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    height, width = first_depth.shape
    _, intrinsic = load_intrinsics(args.input_dir / "calibration.txt", width, height)

    frame_clouds = []
    registration_clouds = []
    frame_payload = []
    valid_counts = []
    for index, (rgb_path, depth_path, mask_path) in enumerate(zip(rgb_files, depth_files, mask_files)):
        cloud, rgb, masked_depth, valid = masked_frame_cloud(
            rgb_path,
            depth_path,
            mask_path,
            intrinsic,
            args.depth_min,
            args.depth_max,
            args.mask_erode,
        )
        frame_clouds.append(cloud)
        registration_clouds.append(prepared_cloud(cloud, args.registration_voxel))
        frame_payload.append((rgb, masked_depth))
        valid_counts.append(int(valid.sum()))
        print(f"[load {index + 1:03d}/{count:03d}] valid={valid_counts[-1]} reg_points={len(registration_clouds[-1].points)}")

    refined_poses, registration_records = refine_poses(
        registration_clouds, raw_poses, args.icp_distance
    )
    pose_corrections = []
    for raw, refined in zip(raw_poses, refined_poses):
        delta = np.linalg.inv(raw) @ refined
        pose_corrections.append(
            {
                "translation_m": float(np.linalg.norm(delta[:3, 3])),
                "rotation_deg": rotation_angle_deg(delta[:3, :3]),
            }
        )

    fused = o3d.geometry.PointCloud()
    for cloud, pose in zip(frame_clouds, refined_poses):
        transformed = o3d.geometry.PointCloud(cloud)
        transformed.transform(pose)
        fused += transformed
    raw_point_count = len(fused.points)
    fused = fused.voxel_down_sample(args.surface_voxel)
    if len(fused.points) > 1000:
        fused, _ = fused.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.5)
    fused.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.surface_voxel * 6.0, max_nn=60)
    )
    fused.orient_normals_consistent_tangent_plane(30)

    surface_path = args.output_dir / "surface.ply"
    o3d.io.write_point_cloud(str(surface_path), fused, write_ascii=False, compressed=False)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel,
        sdf_trunc=args.tsdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for (rgb, masked_depth), pose in zip(frame_payload, refined_poses):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0,
            depth_trunc=args.depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.output_dir / "surface_tsdf_mesh.ply"), mesh)

    np.save(args.output_dir / "raw_cam2world.npy", raw_poses)
    np.save(args.output_dir / "refined_cam2world.npy", refined_poses)
    report = {
        "frame_count": count,
        "source_frame_count": source_count,
        "selected_local_indices": selected_indices,
        "excluded_local_indices": excluded_indices,
        "mask_fractions": mask_fractions,
        "mask_fraction_range": [args.mask_fraction_min, args.mask_fraction_max],
        "depth_range_m": [args.depth_min, args.depth_max],
        "mask_erode_pixels": args.mask_erode,
        "registration_voxel_m": args.registration_voxel,
        "icp_distance_m": args.icp_distance,
        "surface_voxel_m": args.surface_voxel,
        "tsdf_voxel_m": args.tsdf_voxel,
        "valid_depth_pixels_per_frame": valid_counts,
        "raw_fused_point_count": raw_point_count,
        "surface_point_count": len(fused.points),
        "tsdf_vertices": len(mesh.vertices),
        "tsdf_triangles": len(mesh.triangles),
        "pose_corrections": pose_corrections,
        "registration_edges": registration_records,
    }
    (args.output_dir / "surface_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("frame_count", "raw_fused_point_count", "surface_point_count", "tsdf_vertices", "tsdf_triangles")}, indent=2))


if __name__ == "__main__":
    main()
