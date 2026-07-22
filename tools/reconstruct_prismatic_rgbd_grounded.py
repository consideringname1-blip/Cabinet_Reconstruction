import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


ROOT = Path("/workspace_whz")
QPOS0_CAPTURE = "20260622_080738_525030Z"
QPOS1_CAPTURE = "20260622_081031_636398Z"
POSE_METADATA = ROOT / "data/upload/larm/20260622_joint_0_white_outline_sfm/20260622_joint_0_white_outline_sfm.json"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_rgbd_grounded"

QPOS0_CABINET_MASK = (
    ROOT
    / "data/upload/larm/20260622_joint_0_white_outline_sfm/sam3/20260622_080738_525030Z_sam3_cabinet_mask.png"
)
QPOS1_MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
FITTED_SAM3D = {
    "base": ROOT
    / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_to_rgbd_surface_fit/base/base_sam3d_to_rgbd_surface_rank01.glb",
    "drawer": ROOT
    / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_to_rgbd_surface_fit/drawer/drawer_sam3d_to_rgbd_surface_rank01.glb",
}

BLENDER2OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
PART_COLORS = {
    "base": np.array([80, 220, 90, 255], dtype=np.uint8),
    "drawer": np.array([240, 80, 80, 255], dtype=np.uint8),
    "cabinet": np.array([190, 190, 190, 255], dtype=np.uint8),
}
DRAWER_DISTANCE_THRESHOLD_M = 0.060
DRAWER_CLUSTER_EPS_M = 0.035
DRAWER_CLUSTER_MIN_SAMPLES = 20
RNG = np.random.default_rng(20260702)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def capture_paths(capture):
    return {
        "color": ROOT / f"data/upload/larm_captures/{capture}/color.png",
        "depth": ROOT / f"data/output/hololens2/{capture}_align_depth.png",
        "meta": ROOT / f"data/upload/{capture}_meta.json",
        "capture_json": ROOT / f"data/upload/larm_captures/{capture}/capture.json",
    }


def load_k(capture):
    return np.asarray(read_json(capture_paths(capture)["meta"])["PVCamera"]["k"], dtype=np.float64)


def load_color(capture):
    color = cv2.imread(str(capture_paths(capture)["color"]), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(capture_paths(capture)["color"])
    return color


def load_depth_m(capture):
    depth = cv2.imread(str(capture_paths(capture)["depth"]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(capture_paths(capture)["depth"])
    return depth.astype(np.float32) / 1000.0


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    return trimesh.util.concatenate([geom.copy() for geom in loaded.geometry.values()])


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return (np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=np.float64).T)[:, :3]


def scene_add(scene, mesh, name):
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def make_surface_mesh(name, mask, depth_m, color_bgr, k, stride=2, max_depth_jump=0.045):
    valid = mask & (depth_m > 0.2) & (depth_m < 4.0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    h, w = valid.shape
    ys, xs = np.indices((h, w))
    z = depth_m
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    pts_grid = np.stack([x, y, z], axis=-1)

    valid_small = valid[::stride, ::stride]
    pts_small = pts_grid[::stride, ::stride]
    color_small = color_bgr[::stride, ::stride]
    color_rgba = cv2.cvtColor(color_small, cv2.COLOR_BGR2RGBA)
    coords = np.argwhere(valid_small)
    vertices = pts_small[valid_small].reshape(-1, 3)
    vcolors = color_rgba[valid_small].reshape(-1, 4)
    tint = PART_COLORS.get(name, PART_COLORS["cabinet"]).astype(np.float32)
    vcolors = np.clip(0.84 * vcolors.astype(np.float32) + 0.16 * tint, 0, 255).astype(np.uint8)

    index = -np.ones(valid_small.shape, dtype=np.int64)
    for idx, (yy, xx) in enumerate(coords):
        index[yy, xx] = idx

    faces = []
    hs, ws = valid_small.shape
    for yy in range(hs - 1):
        for xx in range(ws - 1):
            ids = [index[yy, xx], index[yy, xx + 1], index[yy + 1, xx], index[yy + 1, xx + 1]]
            if min(ids) < 0:
                continue
            zz = [vertices[i, 2] for i in ids]
            if max(zz) - min(zz) > max_depth_jump:
                continue
            faces.append([ids[0], ids[2], ids[1]])
            faces.append([ids[1], ids[2], ids[3]])

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=vcolors,
        process=False,
    )
    return mesh, {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_min": vertices.min(axis=0).tolist() if len(vertices) else None,
        "bounds_max": vertices.max(axis=0).tolist() if len(vertices) else None,
        "stride": stride,
        "max_depth_jump_m": max_depth_jump,
    }


def submesh_by_vertex_mask(mesh, vertex_mask, fallback_points=False):
    vertex_mask = np.asarray(vertex_mask, dtype=bool)
    face_mask = np.all(vertex_mask[np.asarray(mesh.faces, dtype=np.int64)], axis=1)
    if np.any(face_mask):
        pieces = mesh.submesh([face_mask], append=True, repair=False)
        if isinstance(pieces, trimesh.Trimesh) and len(pieces.vertices):
            return pieces
    if not fallback_points:
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
    colors = np.asarray(mesh.visual.vertex_colors)[vertex_mask] if mesh.visual.kind == "vertex" else None
    return trimesh.points.PointCloud(np.asarray(mesh.vertices)[vertex_mask], colors=colors)


def voxel_downsample(points, voxel=0.006, max_points=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    if voxel > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(idx)]
    if max_points is not None and len(points) > max_points:
        points = points[RNG.choice(len(points), size=max_points, replace=False)]
    return points


def largest_cluster_mask(points, eps, min_samples):
    if len(points) == 0:
        return np.zeros(0, dtype=bool), {"clusters": 0, "selected_points": 0, "noise_points": 0}
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    valid = labels >= 0
    if not np.any(valid):
        return np.ones(len(points), dtype=bool), {
            "clusters": 0,
            "selected_points": int(len(points)),
            "noise_points": int(len(points)),
            "fallback": "all_candidate_points",
        }
    ids, counts = np.unique(labels[valid], return_counts=True)
    selected = int(ids[np.argmax(counts)])
    return labels == selected, {
        "clusters": int(len(ids)),
        "selected_cluster": selected,
        "selected_points": int(np.max(counts)),
        "noise_points": int(np.sum(~valid)),
    }


def translation_only_icp(closed_points, open_points, initial_translation, iterations=32, trim_quantile=0.65):
    src = voxel_downsample(closed_points, voxel=0.006, max_points=16000)
    dst = voxel_downsample(open_points, voxel=0.006, max_points=18000)
    if len(src) < 64 or len(dst) < 64:
        raise RuntimeError(f"Not enough drawer points for translation ICP: closed={len(src)} open={len(dst)}")
    tree = cKDTree(dst)
    t = np.asarray(initial_translation, dtype=np.float64).reshape(3)
    src0 = src.copy()
    history = []
    for _ in range(iterations):
        moved = src0 + t
        dist, idx = tree.query(moved, k=1, workers=-1)
        keep = dist <= np.percentile(dist, trim_quantile * 100.0)
        if int(np.sum(keep)) < 64:
            keep = dist <= np.percentile(dist, 80.0)
        update = np.median(dst[idx[keep]] - src0[keep], axis=0)
        history.append(float(np.linalg.norm(update - t)))
        t = 0.65 * t + 0.35 * update
        if history[-1] < 1e-5:
            break
    dist, _ = tree.query(src0 + t, k=1, workers=-1)
    return t, {
        "src_points": int(len(src)),
        "dst_points": int(len(dst)),
        "iterations": int(len(history)),
        "last_translation_delta_m": float(history[-1]) if history else None,
        "nearest_distance_percentiles_m": {str(q): float(np.percentile(dist, q)) for q in [25, 50, 75, 90, 95]},
    }


def sfm_camera_to_world(capture):
    meta = read_json(POSE_METADATA)
    for idx, rec in enumerate(meta["grouped_captures"]):
        if capture in str(rec.get("capture", "")):
            token = "0.00" if idx < 3 else "1.00"
            frame_idx = idx if idx < 3 else idx - 3
            transform = np.asarray(meta["inputs"][token][f"input_frame_{frame_idx}"]["transform_matrix"], dtype=np.float64)
            return transform @ BLENDER2OPENCV
    raise KeyError(f"Capture {capture} is not present in {POSE_METADATA}")


def hololens_camera_to_world(capture):
    transform = np.asarray(read_json(capture_paths(capture)["capture_json"])["transform_matrix"], dtype=np.float64)
    return transform @ BLENDER2OPENCV


def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    keep = points[:, 2] > 1e-5
    p = points[keep]
    uv = np.column_stack([k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]])
    return uv


def draw_overlay(color_bgr, k, point_sets, contours, out_path):
    img = color_bgr.copy()
    colors = {
        "base": (70, 220, 70),
        "drawer": (40, 60, 240),
        "closed_drawer": (40, 120, 255),
        "cabinet": (190, 190, 190),
    }
    for name, points in point_sets.items():
        if len(points) == 0:
            continue
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) > 65000:
            pts = pts[RNG.choice(len(pts), size=65000, replace=False)]
        uv = project(pts, k)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        pix = np.round(uv[inb]).astype(np.int32)
        for x, y in pix:
            cv2.circle(img, (int(x), int(y)), 1, colors.get(name, (255, 255, 255)), -1)
    for name, mask in contours.items():
        mask_u8 = (mask.astype(np.uint8) * 255)
        cc, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cc, -1, colors.get(name, (255, 255, 255)), 2)
    cv2.imwrite(str(out_path), img)


def write_urdf(path, base_mesh, drawer_mesh, axis, displacement):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<robot name="cabinet_drawer_rgbd_grounded">
  <link name="base">
    <visual><geometry><mesh filename="{base_mesh}"/></geometry></visual>
    <collision><geometry><mesh filename="{base_mesh}"/></geometry></collision>
  </link>
  <link name="drawer">
    <visual><geometry><mesh filename="{drawer_mesh}"/></geometry></visual>
    <collision><geometry><mesh filename="{drawer_mesh}"/></geometry></collision>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="base"/>
    <child link="drawer"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="{axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g}"/>
    <limit lower="0" upper="{float(displacement):.9g}" effort="0" velocity="0"/>
  </joint>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def mesh_stats(mesh):
    vertices = np.asarray(mesh.vertices)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(mesh.faces)) if hasattr(mesh, "faces") else 0,
        "bounds_min": vertices.min(axis=0).tolist() if len(vertices) else None,
        "bounds_max": vertices.max(axis=0).tolist() if len(vertices) else None,
        "centroid_median": np.median(vertices, axis=0).tolist() if len(vertices) else None,
    }


def export_scene(path, named_meshes):
    scene = trimesh.Scene()
    for name, mesh in named_meshes:
        scene_add(scene, mesh, name)
    scene.export(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    surfaces_dir = OUT / "rgbd_surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    k0 = load_k(QPOS0_CAPTURE)
    k1 = load_k(QPOS1_CAPTURE)
    color0 = load_color(QPOS0_CAPTURE)
    color1 = load_color(QPOS1_CAPTURE)
    depth0 = load_depth_m(QPOS0_CAPTURE)
    depth1 = load_depth_m(QPOS1_CAPTURE)

    qpos0_mask = load_mask(QPOS0_CABINET_MASK)
    qpos1_base_mask = load_mask(QPOS1_MASKS["base"])
    qpos1_drawer_mask = load_mask(QPOS1_MASKS["drawer"])

    qpos0_cam_mesh, qpos0_stats = make_surface_mesh("cabinet", qpos0_mask, depth0, color0, k0)
    qpos1_base_mesh, qpos1_base_stats = make_surface_mesh("base", qpos1_base_mask, depth1, color1, k1)
    qpos1_drawer_mesh, qpos1_drawer_stats = make_surface_mesh("drawer", qpos1_drawer_mask, depth1, color1, k1)

    t0_world = sfm_camera_to_world(QPOS0_CAPTURE)
    t1_world = sfm_camera_to_world(QPOS1_CAPTURE)
    qpos0_to_qpos1 = np.linalg.inv(t1_world) @ t0_world

    qpos0_qpos1_mesh = qpos0_cam_mesh.copy()
    qpos0_qpos1_mesh.vertices = transform_points(qpos0_qpos1_mesh.vertices, qpos0_to_qpos1)

    base_tree = cKDTree(np.asarray(qpos1_base_mesh.vertices, dtype=np.float64))
    dist_to_open_base, _ = base_tree.query(np.asarray(qpos0_qpos1_mesh.vertices), k=1, workers=-1)
    drawer_candidates = dist_to_open_base > DRAWER_DISTANCE_THRESHOLD_M
    cluster_keep, cluster_diag = largest_cluster_mask(
        np.asarray(qpos0_qpos1_mesh.vertices)[drawer_candidates],
        DRAWER_CLUSTER_EPS_M,
        DRAWER_CLUSTER_MIN_SAMPLES,
    )
    qpos0_drawer_vertex_mask = np.zeros(len(qpos0_qpos1_mesh.vertices), dtype=bool)
    qpos0_drawer_vertex_mask[np.where(drawer_candidates)[0][cluster_keep]] = True
    qpos0_base_vertex_mask = ~qpos0_drawer_vertex_mask

    qpos0_closed_drawer_mesh = submesh_by_vertex_mask(qpos0_qpos1_mesh, qpos0_drawer_vertex_mask, fallback_points=True)
    qpos0_closed_base_mesh = submesh_by_vertex_mask(qpos0_qpos1_mesh, qpos0_base_vertex_mask, fallback_points=True)
    qpos0_closed_drawer_cam_mesh = submesh_by_vertex_mask(qpos0_cam_mesh, qpos0_drawer_vertex_mask, fallback_points=True)
    qpos0_closed_base_cam_mesh = submesh_by_vertex_mask(qpos0_cam_mesh, qpos0_base_vertex_mask, fallback_points=True)

    qpos0_cam_mesh.export(surfaces_dir / "qpos0_closed_cabinet_surface_camera.glb")
    qpos0_qpos1_mesh.export(surfaces_dir / "qpos0_closed_cabinet_surface_in_qpos1_camera.glb")
    qpos0_closed_drawer_mesh.export(surfaces_dir / "qpos0_closed_drawer_surface_derived_in_qpos1_camera.glb")
    qpos0_closed_base_mesh.export(surfaces_dir / "qpos0_closed_base_surface_derived_in_qpos1_camera.glb")
    qpos1_base_mesh.export(surfaces_dir / "qpos1_open_base_surface.glb")
    qpos1_drawer_mesh.export(surfaces_dir / "qpos1_open_drawer_surface.glb")
    export_scene(
        surfaces_dir / "qpos0_qpos1_visible_surfaces_for_joint.glb",
        [
            ("qpos0_closed_base_derived", qpos0_closed_base_mesh),
            ("qpos0_closed_drawer_derived", qpos0_closed_drawer_mesh),
            ("qpos1_open_base", qpos1_base_mesh),
            ("qpos1_open_drawer", qpos1_drawer_mesh),
        ],
    )

    closed_drawer_points = np.asarray(qpos0_closed_drawer_mesh.vertices, dtype=np.float64)
    open_drawer_points = np.asarray(qpos1_drawer_mesh.vertices, dtype=np.float64)
    initial_translation = np.median(open_drawer_points, axis=0) - np.median(closed_drawer_points, axis=0)
    translation, icp_diag = translation_only_icp(closed_drawer_points, open_drawer_points, initial_translation)
    displacement = float(np.linalg.norm(translation))
    if displacement <= 1e-8:
        raise RuntimeError("Estimated prismatic displacement is zero")
    axis = translation / displacement
    origin = np.median(closed_drawer_points, axis=0)

    base_mesh = load_mesh(FITTED_SAM3D["base"])
    drawer_open_mesh = load_mesh(FITTED_SAM3D["drawer"])
    drawer_closed_mesh = drawer_open_mesh.copy()
    drawer_closed_mesh.vertices = np.asarray(drawer_closed_mesh.vertices, dtype=np.float64) - translation

    base_out = OUT / "base.glb"
    drawer_out = OUT / "drawer.glb"
    drawer_open_out = OUT / "drawer_open_reference.glb"
    base_mesh.export(base_out)
    drawer_closed_mesh.export(drawer_out)
    drawer_open_mesh.export(drawer_open_out)
    export_scene(OUT / "cabinet_drawer_articulated.glb", [("base", base_mesh), ("drawer_open", drawer_open_mesh)])
    export_scene(OUT / "cabinet_drawer_closed_reference.glb", [("base", base_mesh), ("drawer_closed", drawer_closed_mesh)])

    urdf_path = OUT / "cabinet_drawer_prismatic.urdf"
    write_urdf(urdf_path, "base.glb", "drawer.glb", axis, displacement)

    draw_overlay(
        color0,
        k0,
        {
            "base": np.asarray(qpos0_closed_base_cam_mesh.vertices),
            "closed_drawer": np.asarray(qpos0_closed_drawer_cam_mesh.vertices),
        },
        {"cabinet": qpos0_mask},
        OUT / "qpos0_closed_derived_parts_overlay.png",
    )
    draw_overlay(
        color1,
        k1,
        {
            "base": np.asarray(qpos1_base_mesh.vertices),
            "drawer": np.asarray(qpos1_drawer_mesh.vertices),
        },
        {"base": qpos1_base_mask, "drawer": qpos1_drawer_mask},
        OUT / "qpos1_open_rgbd_parts_overlay.png",
    )
    base_samples, _ = trimesh.sample.sample_surface(base_mesh, min(50000, max(1000, len(base_mesh.faces))))
    drawer_samples, _ = trimesh.sample.sample_surface(drawer_open_mesh, min(50000, max(1000, len(drawer_open_mesh.faces))))
    draw_overlay(
        color1,
        k1,
        {"base": base_samples, "drawer": drawer_samples},
        {"base": qpos1_base_mask, "drawer": qpos1_drawer_mask},
        OUT / "articulated_open_sam3d_overlay.png",
    )

    closed_to_open = np.eye(4, dtype=np.float64)
    closed_to_open[:3, 3] = translation
    open_mesh_to_closed = np.eye(4, dtype=np.float64)
    open_mesh_to_closed[:3, 3] = -translation
    report = {
        "method": (
            "RGB-D grounded prismatic reconstruction. qpos1 uses validated base/drawer SAM masks. "
            "qpos0 has only a whole-cabinet SAM mask, so the closed drawer surface is derived as the "
            "largest qpos0 visible-surface component that does not match the qpos1 static base after "
            "cross-state camera-pose alignment. The joint is translation-only ICP from derived closed "
            "drawer surface to qpos1 drawer surface. SAM3D meshes provide complete shape only."
        ),
        "frame": "qpos1_open_camera",
        "captures": {
            "qpos0_closed": QPOS0_CAPTURE,
            "qpos1_open": QPOS1_CAPTURE,
        },
        "pose_source": {
            "metadata": str(POSE_METADATA),
            "note": "Uses the SFM camera transforms from the LARM preparation metadata only for cross-state camera alignment; LARM part masks/pose outputs are not used as the articulated result.",
            "qpos0_to_qpos1_camera": qpos0_to_qpos1.tolist(),
            "hololens_qpos0_to_qpos1_camera_reference": (
                np.linalg.inv(hololens_camera_to_world(QPOS1_CAPTURE)) @ hololens_camera_to_world(QPOS0_CAPTURE)
            ).tolist(),
        },
        "inputs": {
            "qpos0_cabinet_mask": str(QPOS0_CABINET_MASK),
            "qpos1_base_mask": str(QPOS1_MASKS["base"]),
            "qpos1_drawer_mask": str(QPOS1_MASKS["drawer"]),
            "fitted_sam3d_base": str(FITTED_SAM3D["base"]),
            "fitted_sam3d_drawer": str(FITTED_SAM3D["drawer"]),
        },
        "qpos0_drawer_derivation": {
            "distance_to_qpos1_base_threshold_m": DRAWER_DISTANCE_THRESHOLD_M,
            "cluster_eps_m": DRAWER_CLUSTER_EPS_M,
            "cluster_min_samples": DRAWER_CLUSTER_MIN_SAMPLES,
            "candidate_vertices": int(np.sum(drawer_candidates)),
            "selected_vertices": int(np.sum(qpos0_drawer_vertex_mask)),
            "distance_to_qpos1_base_percentiles_m": {
                str(q): float(np.percentile(dist_to_open_base, q)) for q in [10, 25, 50, 75, 90, 95]
            },
            "cluster": cluster_diag,
        },
        "joint": {
            "type": "prismatic",
            "axis_camera": axis.tolist(),
            "origin_camera_m": origin.tolist(),
            "qpos_closed_m": 0.0,
            "qpos_open_m": displacement,
            "closed_to_open_translation_camera_m": translation.tolist(),
            "drawer_closed_centroid_camera_m": np.median(closed_drawer_points, axis=0).tolist(),
            "drawer_open_centroid_camera_m": np.median(open_drawer_points, axis=0).tolist(),
            "drawer_closed_to_open_transform": closed_to_open.tolist(),
            "drawer_open_mesh_to_closed_mesh_transform": open_mesh_to_closed.tolist(),
            "translation_only_icp": icp_diag,
        },
        "surface_stats": {
            "qpos0_closed_cabinet_camera": qpos0_stats,
            "qpos1_open_base": qpos1_base_stats,
            "qpos1_open_drawer": qpos1_drawer_stats,
            "qpos0_closed_base_derived": mesh_stats(qpos0_closed_base_mesh),
            "qpos0_closed_drawer_derived": mesh_stats(qpos0_closed_drawer_mesh),
        },
        "mesh_stats": {
            "base": mesh_stats(base_mesh),
            "drawer_closed": mesh_stats(drawer_closed_mesh),
            "drawer_open_reference": mesh_stats(drawer_open_mesh),
        },
        "outputs": {
            "base_glb": str(base_out),
            "drawer_glb_closed_qpos0": str(drawer_out),
            "drawer_open_reference_glb": str(drawer_open_out),
            "articulated_open_glb": str(OUT / "cabinet_drawer_articulated.glb"),
            "closed_reference_glb": str(OUT / "cabinet_drawer_closed_reference.glb"),
            "urdf": str(urdf_path),
            "joint_json": str(OUT / "joint.json"),
            "visible_surfaces_glb": str(surfaces_dir / "qpos0_qpos1_visible_surfaces_for_joint.glb"),
            "qpos0_overlay": str(OUT / "qpos0_closed_derived_parts_overlay.png"),
            "qpos1_overlay": str(OUT / "qpos1_open_rgbd_parts_overlay.png"),
            "sam3d_open_overlay": str(OUT / "articulated_open_sam3d_overlay.png"),
        },
    }
    (OUT / "joint.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "joint_json": str(OUT / "joint.json"),
                "axis_camera": report["joint"]["axis_camera"],
                "qpos_open_m": report["joint"]["qpos_open_m"],
                "articulated_open_glb": report["outputs"]["articulated_open_glb"],
                "urdf": report["outputs"]["urdf"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
