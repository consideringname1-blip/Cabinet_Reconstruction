#!/usr/bin/env python3
"""Build a lightweight movable mesh visualization from the GT+SAM3 iTACO run.

The asset is intended for inspection, not as a watertight reconstruction:

  - static mesh: fused from SAM3 object minus moving/hand masks
  - moving mesh: fused from SAM3 moving masks and canonicalized along the
    predicted prismatic axis using observed mask-centroid travel
  - Blender helper script: imports the PLYs, colors the parts, adds an axis
    arrow, and exports an animated GLB
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


DEFAULT_INPUT = Path("/workspace_whz/data/upload/2026-07-27-175228/pinhole_projection")
DEFAULT_MASK_DIR = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_sam3_formal_masks")
DEFAULT_JOINT = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_itaco_gt_sam3/joint_prediction.npz")
DEFAULT_OUTPUT = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_movable_mesh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--joint-npz", type=Path, default=DEFAULT_JOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--frame-indices", default="", help="comma-separated explicit frame indices; overrides frame-stride")
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--edge-thresh", type=float, default=0.07, help="max triangle edge length in meters")
    parser.add_argument("--min-moving-area", type=int, default=80)
    parser.add_argument("--min-static-area", type=int, default=250)
    parser.add_argument("--static-target-faces", type=int, default=140000)
    parser.add_argument("--moving-target-faces", type=int, default=90000)
    parser.add_argument("--max-preview-travel", type=float, default=0.30)
    parser.add_argument("--static-crop-around-moving-margin", type=int, default=0, help="crop static mask to moving bbox expanded by this many pixels; 0 disables")
    parser.add_argument("--depth-window-around-moving", type=float, default=0.0, help="meters around moving median depth; 0 disables")
    parser.add_argument("--min-preview-travel", type=float, default=0.06)
    return parser.parse_args()


def sorted_files(input_dir: Path) -> tuple[list[Path], list[Path]]:
    rgb_files = sorted((input_dir / "rgb").glob("*.png"))
    if not rgb_files:
        rgb_files = sorted((input_dir / "rgb").glob("*.jpg"))
    depth_files = sorted((input_dir / "depth").glob("*.png"))
    if not rgb_files or not depth_files:
        raise FileNotFoundError(f"Missing RGB/depth under {input_dir}")
    n = min(len(rgb_files), len(depth_files))
    return rgb_files[:n], depth_files[:n]


def load_intrinsics(input_dir: Path) -> np.ndarray:
    vals = np.loadtxt(input_dir / "calibration.txt", dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = vals[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_odometry(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    poses: list[np.ndarray] = []
    i = 0
    while i + 4 < len(lines):
        mat = [[float(x) for x in row.split()] for row in lines[i + 1 : i + 5]]
        poses.append(np.asarray(mat, dtype=np.float64))
        i += 5
    return poses


def depth_png_to_m(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read {depth_path}")
    depth = depth.astype(np.float32)
    if depth.max(initial=0) > 20:
        depth = depth / 1000.0
    return depth


def depth_to_xyz(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    z = depth_m.astype(np.float32)
    x = (xs - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys - float(K[1, 2])) * z / float(K[1, 1])
    return np.stack([x, y, z], axis=-1)


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def read_mask(mask_dir: Path, kind: str, idx: int, shape: tuple[int, int]) -> np.ndarray:
    path = mask_dir / kind / f"{idx:06d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mesh_from_grid(
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_stride: int,
    edge_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    ys = np.arange(0, h, pixel_stride, dtype=np.int64)
    xs = np.arange(0, w, pixel_stride, dtype=np.int64)
    grid_points = points[np.ix_(ys, xs)]
    grid_colors = colors[np.ix_(ys, xs)]
    grid_mask = mask[np.ix_(ys, xs)]
    valid = grid_mask & np.all(np.isfinite(grid_points), axis=-1) & (grid_points[..., 2] > -10)

    vid = np.full(valid.shape, -1, dtype=np.int64)
    rr, cc = np.nonzero(valid)
    if rr.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.uint8)
    vid[rr, cc] = np.arange(rr.size, dtype=np.int64)
    verts = grid_points[rr, cc].astype(np.float32)
    vcols = grid_colors[rr, cc].astype(np.uint8)

    faces: list[list[int]] = []
    rows, cols = valid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            ids = [vid[r, c], vid[r, c + 1], vid[r + 1, c], vid[r + 1, c + 1]]
            if min(ids) < 0:
                continue
            p00, p01, p10, p11 = verts[ids]
            edges = [
                np.linalg.norm(p00 - p01),
                np.linalg.norm(p00 - p10),
                np.linalg.norm(p01 - p11),
                np.linalg.norm(p10 - p11),
                np.linalg.norm(p00 - p11),
                np.linalg.norm(p01 - p10),
            ]
            if max(edges) > edge_thresh:
                continue
            faces.append([ids[0], ids[2], ids[1]])
            faces.append([ids[1], ids[2], ids[3]])
    return verts, np.asarray(faces, dtype=np.int64), vcols


def combine_mesh_chunks(chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    offset = 0
    for verts, faces, colors in chunks:
        if verts.size == 0:
            continue
        verts_all.append(verts)
        colors_all.append(colors)
        if faces.size:
            faces_all.append(faces + offset)
        offset += verts.shape[0]
    if not verts_all:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.uint8),
        )
    verts = np.concatenate(verts_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    faces = np.concatenate(faces_all, axis=0) if faces_all else np.zeros((0, 3), dtype=np.int64)
    return verts, faces, colors


def to_open3d_mesh(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    if colors.size:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def decimate(mesh: o3d.geometry.TriangleMesh, target_faces: int) -> o3d.geometry.TriangleMesh:
    tri_count = np.asarray(mesh.triangles).shape[0]
    if target_faces > 0 and tri_count > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
    return mesh


def write_blender_script(
    script_path: Path,
    *,
    static_ply: Path,
    moving_ply: Path,
    glb_path: Path,
    axis: np.ndarray,
    axis_origin: np.ndarray,
    travel: float,
    metadata_path: Path,
) -> None:
    axis_list = [float(x) for x in axis]
    origin_list = [float(x) for x in axis_origin]
    script = f'''import bpy
import mathutils
from pathlib import Path

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def make_mat(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.7
    return mat

static_mat = make_mat("static_base_soft_green", (0.43, 0.78, 0.55, 1.0))
moving_mat = make_mat("moving_part_orange", (1.0, 0.34, 0.12, 1.0))
axis_mat = make_mat("predicted_prismatic_axis_blue", (0.08, 0.26, 1.0, 1.0))

def import_ply(path, name, mat):
    try:
        bpy.ops.wm.ply_import(filepath=str(path))
    except Exception:
        bpy.ops.import_mesh.ply(filepath=str(path))
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.data.materials.append(mat)
    return obj

static_obj = import_ply(Path(r"{static_ply}"), "Static_GT_SAM3_mesh", static_mat)
moving_obj = import_ply(Path(r"{moving_ply}"), "Moving_GT_SAM3_mesh_animated", moving_mat)

axis = mathutils.Vector({axis_list})
if axis.length == 0:
    axis = mathutils.Vector((1, 0, 0))
axis.normalize()
origin = mathutils.Vector({origin_list})
travel = float({travel})
if travel <= 0:
    travel = 0.10

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 120
bpy.context.scene.render.fps = 24

for frame, scalar in [(1, 0.0), (60, travel), (120, 0.0)]:
    bpy.context.scene.frame_set(frame)
    moving_obj.location = axis * scalar
    moving_obj.keyframe_insert(data_path="location", frame=frame)


arrow_len = max(travel * 1.15, 0.12)
arrow_radius = max(arrow_len * 0.025, 0.004)
center = origin + axis * (arrow_len * 0.5)
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=arrow_radius, depth=arrow_len, location=center)
cylinder = bpy.context.object
cylinder.name = "Predicted_prismatic_axis"
cylinder.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
cylinder.data.materials.append(axis_mat)

bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=arrow_radius * 3.0, radius2=0.0, depth=arrow_radius * 9.0, location=origin + axis * (arrow_len + arrow_radius * 4.5))
cone = bpy.context.object
cone.name = "Axis_arrow_head"
cone.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
cone.data.materials.append(axis_mat)

bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=arrow_radius * 2.4, location=origin)
sphere = bpy.context.object
sphere.name = "Axis_origin_marker"
sphere.data.materials.append(axis_mat)

# Add a simple camera/light for viewers that honor glTF cameras.
all_objs = [static_obj, moving_obj, cylinder, cone, sphere]
coords = []
for obj in all_objs[:2]:
    for corner in obj.bound_box:
        coords.append(obj.matrix_world @ mathutils.Vector(corner))
if coords:
    mins = mathutils.Vector((min(v.x for v in coords), min(v.y for v in coords), min(v.z for v in coords)))
    maxs = mathutils.Vector((max(v.x for v in coords), max(v.y for v in coords), max(v.z for v in coords)))
    target = (mins + maxs) * 0.5
    diag = (maxs - mins).length
else:
    target = origin
    diag = 1.0

bpy.ops.object.light_add(type="AREA", location=target + mathutils.Vector((0.0, -1.2 * diag, 1.4 * diag)))
light = bpy.context.object
light.name = "Preview_area_light"
light.data.energy = 500
light.data.size = max(1.0, diag)

bpy.ops.object.camera_add(location=target + mathutils.Vector((0.0, -1.8 * diag, 0.9 * diag)))
camera = bpy.context.object
direction = target - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 28
bpy.context.scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=str(Path(r"{glb_path}").with_suffix(".blend")))
bpy.ops.export_scene.gltf(
    filepath=str(Path(r"{glb_path}")),
    export_format="GLB",
    export_animations=True,
    export_materials="EXPORT",
    export_cameras=True,
    export_lights=True,
)
print("EXPORTED_GLB", r"{glb_path}")
print("METADATA", r"{metadata_path}")
'''
    script_path.write_text(script)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb_files, depth_files = sorted_files(args.input_dir)
    poses = load_odometry(args.input_dir / "odometry.log")
    n = min(len(rgb_files), len(depth_files), len(poses))
    rgb_files, depth_files, poses = rgb_files[:n], depth_files[:n], poses[:n]
    K = load_intrinsics(args.input_dir)

    joint = np.load(args.joint_npz, allow_pickle=True)
    pred_joint_type = str(joint["pred_joint_type"])
    axis = np.asarray(joint["prismatic_axis"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    average_value = float(joint["prismatic_average_value"])

    if args.frame_indices.strip():
        selected = [int(x) for x in args.frame_indices.split(",") if x.strip()]
        selected = [i for i in selected if 0 <= i < n]
    else:
        selected = list(range(0, n, max(1, args.frame_stride)))
    static_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    moving_frame_data: list[dict[str, Any]] = []

    print(f"[mesh] frames={n} selected={len(selected)} pixel_stride={args.pixel_stride}")
    print(f"[mesh] pred_joint_type={pred_joint_type} axis={axis.tolist()} avg_value={average_value}")

    for idx in selected:
        rgb = rgb_image(rgb_files[idx])
        depth = depth_png_to_m(depth_files[idx])
        xyz_cam = depth_to_xyz(depth, K)
        valid_depth = depth > 0
        world = transform_points(xyz_cam.reshape(-1, 3), poses[idx]).reshape(xyz_cam.shape)

        shape = depth.shape
        obj = read_mask(args.mask_dir, "object", idx, shape)
        moving = read_mask(args.mask_dir, "moving", idx, shape)
        hand = read_mask(args.mask_dir, "hand", idx, shape)
        obj = obj & valid_depth & (~hand)
        moving = moving & valid_depth & (~hand)

        if args.depth_window_around_moving > 0 and int(moving.sum()) > 0:
            moving_depth = depth[moving & (depth > 0)]
            if moving_depth.size:
                depth_med = float(np.median(moving_depth))
                depth_keep = np.abs(depth - depth_med) <= args.depth_window_around_moving
                obj = obj & depth_keep
                moving = moving & depth_keep

        static = obj & (~moving)
        if args.static_crop_around_moving_margin > 0 and int(moving.sum()) > 0:
            ys_m, xs_m = np.nonzero(moving)
            margin = int(args.static_crop_around_moving_margin)
            y0 = max(0, int(ys_m.min()) - margin)
            y1 = min(shape[0], int(ys_m.max()) + margin + 1)
            x0 = max(0, int(xs_m.min()) - margin)
            x1 = min(shape[1], int(xs_m.max()) + margin + 1)
            crop = np.zeros(shape, dtype=bool)
            crop[y0:y1, x0:x1] = True
            static = static & crop

        if int(static.sum()) >= args.min_static_area:
            static_chunks.append(
                mesh_from_grid(world, rgb, static, pixel_stride=args.pixel_stride, edge_thresh=args.edge_thresh)
            )
        if int(moving.sum()) >= args.min_moving_area:
            pts = world[moving]
            proj = pts @ axis
            moving_frame_data.append(
                {
                    "idx": idx,
                    "world": world,
                    "rgb": rgb,
                    "mask": moving,
                    "area": int(moving.sum()),
                    "median_projection": float(np.median(proj)),
                }
            )

    if not static_chunks:
        raise RuntimeError("No static mesh chunks were generated")
    if not moving_frame_data:
        raise RuntimeError("No moving mesh chunks were generated")

    q_values = np.array([d["median_projection"] for d in moving_frame_data], dtype=np.float64)
    q05 = float(np.percentile(q_values, 5))
    q95 = float(np.percentile(q_values, 95))
    observed_travel = abs(q95 - q05)
    preview_travel = float(np.clip(observed_travel, args.min_preview_travel, args.max_preview_travel))
    # Use the lower robust endpoint as canonical pose; this keeps the animation
    # translation positive along the predicted axis.
    canonical_projection = min(q05, q95)

    moving_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for d in moving_frame_data:
        delta = float(d["median_projection"] - canonical_projection)
        canonical_world = d["world"] - delta * axis.reshape(1, 1, 3)
        moving_chunks.append(
            mesh_from_grid(
                canonical_world,
                d["rgb"],
                d["mask"],
                pixel_stride=args.pixel_stride,
                edge_thresh=args.edge_thresh,
            )
        )

    static_verts, static_faces, static_colors = combine_mesh_chunks(static_chunks)
    moving_verts, moving_faces, moving_colors = combine_mesh_chunks(moving_chunks)

    static_mesh = to_open3d_mesh(static_verts, static_faces, static_colors)
    moving_mesh = to_open3d_mesh(moving_verts, moving_faces, moving_colors)
    static_mesh = decimate(static_mesh, args.static_target_faces)
    moving_mesh = decimate(moving_mesh, args.moving_target_faces)

    static_ply = args.output_dir / "static_base_gt_sam3.ply"
    moving_ply = args.output_dir / "moving_part_canonical_gt_sam3.ply"
    o3d.io.write_triangle_mesh(str(static_ply), static_mesh, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(moving_ply), moving_mesh, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True)

    static_bbox = static_mesh.get_axis_aligned_bounding_box()
    moving_bbox = moving_mesh.get_axis_aligned_bounding_box()
    moving_center = np.asarray(moving_bbox.get_center(), dtype=np.float64)
    axis_origin = moving_center - axis * (preview_travel * 0.15)

    metadata = {
        "input_dir": str(args.input_dir),
        "mask_dir": str(args.mask_dir),
        "joint_npz": str(args.joint_npz),
        "pred_joint_type": pred_joint_type,
        "prismatic_axis": axis.tolist(),
        "itaco_average_value_per_frame_m": average_value,
        "itaco_linear_total_over_video_m": abs(average_value) * (n - 1),
        "observed_moving_projection_q05_m": q05,
        "observed_moving_projection_q95_m": q95,
        "observed_travel_m": observed_travel,
        "preview_travel_m": preview_travel,
        "preview_note": "Animation uses robust observed moving-mask centroid travel, clipped for visualization, rather than raw iTACO linear frame-index extrapolation.",
        "frames_total": n,
        "frame_stride": args.frame_stride,
        "pixel_stride": args.pixel_stride,
        "static_crop_around_moving_margin": args.static_crop_around_moving_margin,
        "depth_window_around_moving": args.depth_window_around_moving,
        "selected_frames": selected,
        "moving_frames_used": [int(d["idx"]) for d in moving_frame_data],
        "static_mesh": {
            "path": str(static_ply),
            "vertices": int(np.asarray(static_mesh.vertices).shape[0]),
            "faces": int(np.asarray(static_mesh.triangles).shape[0]),
            "bbox_min": static_bbox.min_bound.tolist(),
            "bbox_max": static_bbox.max_bound.tolist(),
        },
        "moving_mesh": {
            "path": str(moving_ply),
            "vertices": int(np.asarray(moving_mesh.vertices).shape[0]),
            "faces": int(np.asarray(moving_mesh.triangles).shape[0]),
            "bbox_min": moving_bbox.min_bound.tolist(),
            "bbox_max": moving_bbox.max_bound.tolist(),
        },
        "axis_origin": axis_origin.tolist(),
    }
    metadata_path = args.output_dir / "movable_mesh_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    blender_script = args.output_dir / "export_animated_glb.py"
    glb_path = args.output_dir / "itaco_gt_sam3_prismatic_preview.glb"
    write_blender_script(
        blender_script,
        static_ply=static_ply,
        moving_ply=moving_ply,
        glb_path=glb_path,
        axis=axis,
        axis_origin=axis_origin,
        travel=preview_travel,
        metadata_path=metadata_path,
    )

    print(f"[mesh] static vertices={metadata['static_mesh']['vertices']} faces={metadata['static_mesh']['faces']}")
    print(f"[mesh] moving vertices={metadata['moving_mesh']['vertices']} faces={metadata['moving_mesh']['faces']}")
    print(f"[mesh] observed_travel={observed_travel:.4f} preview_travel={preview_travel:.4f}")
    print(f"[mesh] static_ply={static_ply}")
    print(f"[mesh] moving_ply={moving_ply}")
    print(f"[mesh] blender_script={blender_script}")
    print(f"[mesh] metadata={metadata_path}")
    print(f"[mesh] glb_target={glb_path}")


if __name__ == "__main__":
    main()
