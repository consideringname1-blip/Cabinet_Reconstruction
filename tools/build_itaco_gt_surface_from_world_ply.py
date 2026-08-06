#!/usr/bin/env python3
"""Build the iTACO object surface directly from HoloLens world-space depth PLYs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_odometry(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack(
        [
            np.array([[float(x) for x in row.split()] for row in lines[start + 1 : start + 5]])
            for start in range(0, len(lines), 5)
        ]
    )


def depth_timestamps(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(value))


def masked_world_cloud(
    world: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    depth_min: float,
    depth_max: float,
    visibility_tolerance: float,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, dict]:
    height, width = mask.shape
    camera = (world - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z >= depth_min) & (z <= depth_max)
    ids = np.nonzero(valid)[0]
    camera = camera[valid]
    z = camera[:, 2]
    u = np.rint(intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]).astype(np.int64)
    v = np.rint(intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ids, u, v, z = ids[inside], u[inside], v[inside], z[inside]

    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    pixel = v * width + u
    np.minimum.at(zbuffer, pixel, z)
    visible = z <= zbuffer[pixel] + visibility_tolerance
    selected = visible & mask[v, u]
    ids, u, v = ids[selected], u[selected], v[selected]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(world[ids])
    cloud.colors = o3d.utility.Vector3dVector(rgb[v, u].astype(np.float64) / 255.0)
    dense_depth = zbuffer.reshape(height, width)
    dense_depth[~np.isfinite(dense_depth)] = 0.0
    stats = {
        "input_world_points": int(len(world)),
        "projected_points": int(len(pixel)),
        "selected_visible_masked_points": int(len(ids)),
        "mask_fraction": float(mask.mean()),
    }
    return cloud, dense_depth.astype(np.float32), stats


def prepared_cloud(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=50)
    )
    return down


def register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance, result.transformation
    )
    return result.transformation, information, float(result.fitness), float(result.inlier_rmse)


def refine_world_clouds(
    clouds: list[o3d.geometry.PointCloud],
    source_indices: list[int],
    distance: float,
) -> tuple[np.ndarray, list[dict]]:
    graph = o3d.pipelines.registration.PoseGraph()
    for _ in clouds:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    pairs = {(index, index + 1, False) for index in range(len(clouds) - 1)}
    for interval in (5, 10):
        for source in range(0, len(clouds) - interval, interval):
            pairs.add((source, source + interval, True))

    records = []
    for source, target, loop in sorted(pairs):
        transform, information, fitness, rmse = register_pair(
            clouds[source], clouds[target], distance
        )
        translation = float(np.linalg.norm(transform[:3, 3]))
        rotation = rotation_angle_deg(transform[:3, :3])
        accepted = (
            fitness >= (0.18 if loop else 0.12)
            and rmse <= distance
            and translation <= 0.04
            and rotation <= 4.0
        )
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source,
                target,
                transform if accepted else np.eye(4),
                information if accepted else np.eye(6) * 50.0,
                uncertain=loop,
            )
        )
        records.append(
            {
                "source_node": source,
                "target_node": target,
                "source_local_index": source_indices[source],
                "target_local_index": source_indices[target],
                "loop": loop,
                "fitness": fitness,
                "rmse_m": rmse,
                "translation_m": translation,
                "rotation_deg": rotation,
                "accepted": accepted,
            }
        )

    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=distance,
            edge_prune_threshold=0.25,
            preference_loop_closure=0.1,
            reference_node=0,
        ),
    )
    return np.stack([node.pose for node in graph.nodes]), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--corrected-depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-fraction-min", type=float, default=0.08)
    parser.add_argument("--mask-fraction-max", type=float, default=0.22)
    parser.add_argument("--exclude-local-indices", default="")
    parser.add_argument("--mask-erode", type=int, default=1)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--visibility-tolerance", type=float, default=0.025)
    parser.add_argument("--registration-voxel", type=float, default=0.012)
    parser.add_argument("--icp-distance", type=float, default=0.035)
    parser.add_argument("--surface-voxel", type=float, default=0.0015)
    parser.add_argument("--tsdf-voxel", type=float, default=0.0035)
    parser.add_argument("--tsdf-trunc", type=float, default=0.018)
    parser.add_argument("--skip-registration", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((args.sequence_dir / "frame_mapping.json").read_text())
    source_pinhole = args.source_root / "pinhole_projection"
    all_poses = load_odometry(source_pinhole / "odometry.log")
    timestamps = depth_timestamps(source_pinhole / "depth.txt")
    fx, fy, cx, cy = np.loadtxt(source_pinhole / "calibration.txt").reshape(-1)[:4]
    intrinsic_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    first_rgb = cv2.imread(mapping[0]["rgb"], cv2.IMREAD_COLOR)
    height, width = first_rgb.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    masks = [np.load(args.mask_dir / f"{index:06d}.npy").squeeze().astype(np.uint8) for index in range(len(mapping))]
    if args.mask_erode > 0:
        kernel = np.ones((args.mask_erode * 2 + 1, args.mask_erode * 2 + 1), np.uint8)
        masks = [cv2.erode(mask, kernel, iterations=1) for mask in masks]
    fractions = [float(mask.mean()) for mask in masks]
    selected_indices = [
        index
        for index, fraction in enumerate(fractions)
        if args.mask_fraction_min <= fraction <= args.mask_fraction_max
    ]
    explicit_excluded = {int(value) for value in args.exclude_local_indices.split(",") if value.strip()}
    selected_indices = [index for index in selected_indices if index not in explicit_excluded]
    excluded_indices = sorted(set(range(len(mapping))) - set(selected_indices))

    clouds = []
    registration_clouds = []
    camera_poses = []
    rgbd_payload = []
    frame_records = []
    for node, local_index in enumerate(selected_indices):
        item = mapping[local_index]
        source_index = int(item["source_index"])
        ply_path = args.source_root / "Depth Long Throw" / f"{timestamps[source_index]}.ply"
        world = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
        bgr = cv2.imread(item["rgb"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cloud, _, stats = masked_world_cloud(
            world,
            rgb,
            masks[local_index].astype(bool),
            all_poses[source_index],
            intrinsic_matrix,
            args.depth_min,
            args.depth_max,
            args.visibility_tolerance,
        )
        corrected_depth = np.load(args.corrected_depth_dir / f"{local_index:06d}.npy")
        corrected_depth = np.where(masks[local_index].astype(bool), corrected_depth, 0.0).astype(np.float32)
        clouds.append(cloud)
        registration_clouds.append(prepared_cloud(cloud, args.registration_voxel))
        camera_poses.append(all_poses[source_index])
        rgbd_payload.append((rgb, corrected_depth))
        frame_records.append(
            {
                "node": node,
                "local_index": local_index,
                "source_index": source_index,
                "source_ply": str(ply_path.resolve()),
                **stats,
                "registration_points": len(registration_clouds[-1].points),
            }
        )
        print(
            f"[{node + 1:03d}/{len(selected_indices):03d}] local={local_index:02d} "
            f"points={len(cloud.points)} reg={len(registration_clouds[-1].points)}"
        )

    if args.skip_registration:
        corrections = np.repeat(np.eye(4)[None, ...], len(clouds), axis=0)
        edge_records = []
    else:
        corrections, edge_records = refine_world_clouds(
            registration_clouds, selected_indices, args.icp_distance
        )
    fused = o3d.geometry.PointCloud()
    corrected_camera_poses = []
    for cloud, camera_pose, correction in zip(clouds, camera_poses, corrections):
        transformed = o3d.geometry.PointCloud(cloud)
        transformed.transform(correction)
        fused += transformed
        corrected_camera_poses.append(correction @ camera_pose)
    raw_count = len(fused.points)
    fused = fused.voxel_down_sample(args.surface_voxel)
    if len(fused.points) > 1000:
        fused, _ = fused.remove_statistical_outlier(nb_neighbors=24, std_ratio=3.0)
    fused.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.surface_voxel * 8.0, max_nn=80)
    )
    fused.orient_normals_consistent_tangent_plane(30)
    o3d.io.write_point_cloud(str(args.output_dir / "surface.ply"), fused)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel,
        sdf_trunc=args.tsdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for (rgb, depth), camera_pose in zip(rgbd_payload, corrected_camera_poses):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=args.depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(camera_pose))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.output_dir / "surface_tsdf_mesh.ply"), mesh)

    np.save(args.output_dir / "selected_local_indices.npy", np.array(selected_indices))
    np.save(args.output_dir / "corrections_world.npy", corrections)
    np.save(args.output_dir / "corrected_cam2world.npy", np.stack(corrected_camera_poses))
    report = {
        "method": (
            "masked original Long Throw world PLY fusion with HoloLens PV projection; "
            + ("recorded poses kept fixed" if args.skip_registration else "small ICP pose-graph correction")
        ),
        "skip_registration": args.skip_registration,
        "source_frame_count": len(mapping),
        "selected_local_indices": selected_indices,
        "excluded_local_indices": excluded_indices,
        "explicit_excluded_local_indices": sorted(explicit_excluded),
        "mask_fractions_after_erosion": fractions,
        "surface_point_count_before_voxel": raw_count,
        "surface_point_count": len(fused.points),
        "surface_voxel_m": args.surface_voxel,
        "tsdf_vertices": len(mesh.vertices),
        "tsdf_triangles": len(mesh.triangles),
        "frame_records": frame_records,
        "registration_edges": edge_records,
    }
    (args.output_dir / "surface_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_frames": len(selected_indices),
                "excluded_local_indices": excluded_indices,
        "explicit_excluded_local_indices": sorted(explicit_excluded),
                "surface_point_count": len(fused.points),
                "tsdf_vertices": len(mesh.vertices),
                "tsdf_triangles": len(mesh.triangles),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
