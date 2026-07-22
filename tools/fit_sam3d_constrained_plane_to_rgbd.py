import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


ROOT = Path("/workspace_whz")
CAPTURE = "20260622_081031_636398Z"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_constrained_plane_fit"
REF_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"

COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
RAW_MESHES = {
    "base": ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb",
    "drawer": ROOT / "data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb",
}
REF_SURFACES = {
    "base": REF_DIR / "base_rgbd_mask_surface.glb",
    "drawer": REF_DIR / "drawer_rgbd_mask_surface.glb",
}

COLORS = {
    "base": (50, 220, 80),
    "drawer": (40, 70, 245),
    "drawer_closed": (30, 150, 255),
    "axis": (255, 180, 20),
}
RNG = np.random.default_rng(20260708)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    meshes = [geom.copy() for geom in loaded.geometry.values()]
    if not meshes:
        raise ValueError(f"empty mesh scene: {path}")
    return trimesh.util.concatenate(meshes)


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def sample_mesh_points(mesh, count):
    if len(mesh.faces) > 0:
        pts, face_idx = trimesh.sample.sample_surface(mesh, count)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_idx]
        return np.asarray(pts, dtype=np.float64), normals
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) > count:
        vertices = vertices[RNG.choice(len(vertices), size=count, replace=False)]
    return vertices, None


def voxel_downsample(points, voxel=0.0045, max_points=None):
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


def fit_plane_svd(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = unit(vh[-1])
    axes = np.asarray(vh[:2], dtype=np.float64)
    axes[0] = unit(axes[0] - normal * float(axes[0] @ normal))
    axes[1] = unit(np.cross(normal, axes[0]))
    uv = (points - center) @ axes.T
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    if np.linalg.det(np.column_stack([axes[0], axes[1], normal])) < 0:
        axes[1] *= -1.0
    return {
        "center": center,
        "normal": normal,
        "axes": axes,
        "uv_min": uv_min,
        "uv_max": uv_max,
        "extent": uv_max - uv_min,
        "approx_bbox_area": float(np.prod(np.maximum(uv_max - uv_min, 0.0))),
        "singular_values": singular,
    }


def extract_planes(points, normals=None, max_planes=8, threshold=0.012, min_inliers=260, trials=560, normal_sim=0.88):
    points = np.asarray(points, dtype=np.float64)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
    remaining = np.ones(len(points), dtype=bool)
    planes = []
    for plane_index in range(max_planes):
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) < max(min_inliers, 180):
            break
        best = None
        for _ in range(trials):
            if normals is not None:
                i = int(RNG.choice(rem_idx))
                n = normals[i]
                if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
                    continue
                n = unit(n)
                p0 = points[i]
            else:
                ids = RNG.choice(rem_idx, size=3, replace=False)
                p0, p1, p2 = points[ids]
                n = np.cross(p1 - p0, p2 - p0)
                if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
                    continue
                n = unit(n)
            dist = np.abs((points[rem_idx] - p0) @ n)
            inlier_local = dist < threshold
            if normals is not None:
                sim = np.abs(normals[rem_idx] @ n)
                inlier_local &= sim > normal_sim
            count = int(inlier_local.sum())
            if best is None or count > best[0]:
                best = (count, n, rem_idx[inlier_local])
        if best is None or best[0] < min_inliers:
            break
        inliers = best[2]
        plane = fit_plane_svd(points[inliers])
        if normals is not None:
            avg_normal = normals[inliers].mean(axis=0)
            if np.linalg.norm(avg_normal) > 1e-8 and float(plane["normal"] @ avg_normal) < 0:
                plane["normal"] *= -1.0
                plane["axes"][1] *= -1.0
        plane.update(
            {
                "index": int(plane_index),
                "sample_inliers": int(len(inliers)),
                "sample_fraction": float(len(inliers) / max(len(points), 1)),
                "inlier_indices": inliers,
            }
        )
        planes.append(plane)
        remaining[inliers] = False
    return planes


def plane_json(plane):
    return {
        "index": int(plane["index"]),
        "sample_inliers": int(plane["sample_inliers"]),
        "sample_fraction": float(plane["sample_fraction"]),
        "center": np.asarray(plane["center"], dtype=float).tolist(),
        "normal": np.asarray(plane["normal"], dtype=float).tolist(),
        "axes": np.asarray(plane["axes"], dtype=float).tolist(),
        "uv_min": np.asarray(plane["uv_min"], dtype=float).tolist(),
        "uv_max": np.asarray(plane["uv_max"], dtype=float).tolist(),
        "extent": np.asarray(plane["extent"], dtype=float).tolist(),
        "approx_bbox_area": float(plane["approx_bbox_area"]),
        "singular_values": np.asarray(plane["singular_values"], dtype=float).tolist(),
    }


def frame_from_plane(plane, normal_sign=1.0, roll_quarter=0, desired_normal=None):
    n = unit(plane["normal"]) * float(normal_sign)
    if desired_normal is not None and float(n @ desired_normal) < 0:
        n *= -1.0
    u = np.asarray(plane["axes"][0], dtype=np.float64)
    u = unit(u - n * float(u @ n))
    v = unit(np.cross(n, u))
    variants = [
        (u, v),
        (v, -u),
        (-u, -v),
        (-v, u),
    ]
    uu, vv = variants[int(roll_quarter) % 4]
    frame = np.column_stack([uu, vv, n])
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1.0
    return frame


def transform_points(points, rotation, scale, translation):
    return float(scale) * (np.asarray(points, dtype=np.float64) @ rotation.T) + np.asarray(translation, dtype=np.float64)


def solve_scale_translation(src_rotated, dst):
    src_rotated = np.asarray(src_rotated, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mx = src_rotated.mean(axis=0)
    my = dst.mean(axis=0)
    x = src_rotated - mx
    y = dst - my
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        return 1.0, my - mx
    scale = float(np.sum(x * y) / denom)
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    translation = my - scale * mx
    return scale, translation


def robust_extent(points):
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    return np.maximum(hi - lo, 1e-6)


def initial_scale(src_points, tgt_points, rotation):
    src_rot = np.asarray(src_points) @ rotation.T
    src_extent = robust_extent(src_rot)
    tgt_extent = robust_extent(tgt_points)
    ratios = tgt_extent / np.maximum(src_extent, 1e-6)
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-5)]
    if len(ratios) == 0:
        return 1.0
    return float(np.median(ratios))


def refine_fixed_rotation(src_fit, tgt_fit, rotation, scale0, iterations=8):
    src_fit = np.asarray(src_fit, dtype=np.float64)
    tgt_fit = np.asarray(tgt_fit, dtype=np.float64)
    src_rot = src_fit @ rotation.T
    scale = float(scale0)
    translation = np.median(tgt_fit, axis=0) - scale * np.median(src_rot, axis=0)
    min_scale = max(scale0 * 0.45, 1e-7)
    max_scale = max(scale0 * 2.2, min_scale * 1.1)
    history = []
    for _ in range(iterations):
        pred = scale * src_rot + translation
        tree = cKDTree(pred)
        dist, idx = tree.query(tgt_fit, k=1, workers=-1)
        thresh = min(max(float(np.percentile(dist, 78)), 0.012), 0.10)
        keep = dist <= thresh
        if int(keep.sum()) < 180:
            keep = dist <= np.percentile(dist, 90)
        s_new, t_new = solve_scale_translation(src_rot[idx[keep]], tgt_fit[keep])
        s_new = float(np.clip(s_new, min_scale, max_scale))
        update = abs(math.log(max(s_new, 1e-12) / max(scale, 1e-12))) + float(np.linalg.norm(t_new - translation))
        history.append(update)
        scale = 0.70 * scale + 0.30 * s_new
        translation = 0.70 * translation + 0.30 * t_new
        if update < 1e-5:
            break
    return scale, translation, history


def project_all(points, k):
    points = np.asarray(points, dtype=np.float64)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    keep = points[:, 2] > 1e-5
    p = points[keep]
    uv[keep, 0] = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]
    uv[keep, 1] = k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
    return uv, keep


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float64)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def score_pose(src_eval, tgt_eval, rotation, scale, translation, k, mask, depth_m, transformed_plane_normal, desired_axis):
    pred = transform_points(src_eval, rotation, scale, translation)
    tree = cKDTree(pred)
    dist, _ = tree.query(tgt_eval, k=1, workers=-1)
    chamfer_med = float(np.median(dist))
    chamfer_p90 = float(np.percentile(dist, 90))

    uv, z_keep = project_all(pred, k)
    h, w = mask.shape
    inb = z_keep & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if int(inb.sum()) < 80:
        return 1e9, {"reason": "few projected points"}
    px = np.clip(np.round(uv[inb, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(uv[inb, 1]).astype(np.int32), 0, h - 1)
    target = mask > 0
    hit = target[py, px]
    valid_depth = depth_m[py, px] > 0.05
    depth_diff = np.abs(pred[inb, 2][hit & valid_depth] - depth_m[py[hit & valid_depth], px[hit & valid_depth]])
    if len(depth_diff) == 0:
        depth_diff = np.array([1.0], dtype=np.float64)
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    pred_mask[py, px] = 255
    pred_mask = cv2.dilate(pred_mask, np.ones((5, 5), np.uint8), iterations=1) > 0
    iou = float((pred_mask & target).sum() / max(1, (pred_mask | target).sum()))
    coverage = float((pred_mask & target).sum() / max(1, target.sum()))
    leakage = float((pred_mask & ~target).sum() / max(1, pred_mask.sum()))
    uv_in = uv[inb]
    pred_lo = np.percentile(uv_in, 1, axis=0)
    pred_hi = np.percentile(uv_in, 99, axis=0)
    pred_bbox = np.r_[pred_lo, pred_hi]
    tgt_bbox = mask_bbox(mask)
    tgt_size = np.maximum(tgt_bbox[2:] - tgt_bbox[:2], 1.0)
    bbox_center_err = float(
        np.linalg.norm(((pred_lo + pred_hi) * 0.5 - (tgt_bbox[:2] + tgt_bbox[2:]) * 0.5) / tgt_size)
    )
    bbox_size_err = float(np.linalg.norm(((pred_hi - pred_lo) - tgt_size) / tgt_size))
    normal_axis_absdot = float(abs(unit(transformed_plane_normal) @ unit(desired_axis)))
    depth_med = float(np.median(depth_diff))
    depth_p75 = float(np.percentile(depth_diff, 75))
    hit_ratio = float(hit.mean()) if len(hit) else 0.0
    score = (
        chamfer_med / 0.018
        + 0.24 * chamfer_p90 / 0.060
        + 0.80 * depth_med / 0.050
        + 0.25 * depth_p75 / 0.080
        + 1.55 * (1.0 - iou)
        + 0.60 * (1.0 - coverage)
        + 0.85 * leakage
        + 0.75 * bbox_size_err
        + 1.00 * bbox_center_err
        + 2.00 * (1.0 - normal_axis_absdot)
    )
    return float(score), {
        "score": float(score),
        "target_to_source_chamfer_median_m": chamfer_med,
        "target_to_source_chamfer_p90_m": chamfer_p90,
        "depth_abs_median_m": depth_med,
        "depth_abs_p75_m": depth_p75,
        "projected_mask_iou": iou,
        "target_coverage": coverage,
        "leakage": leakage,
        "projected_mask_hit_ratio": hit_ratio,
        "bbox_xyxy_p01_p99": pred_bbox.tolist(),
        "bbox_center_err": bbox_center_err,
        "bbox_size_err": bbox_size_err,
        "projected_points": int(inb.sum()),
        "normal_axis_absdot": normal_axis_absdot,
    }


def draw_overlay(color_bgr, k, samples_by_name, masks_by_name, output):
    img = color_bgr.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for name, points in samples_by_name.items():
        points = np.asarray(points, dtype=np.float64)
        if len(points) > 70000:
            points = points[RNG.choice(len(points), size=70000, replace=False)]
        uv, keep = project_all(points, k)
        h, w = img.shape[:2]
        inb = keep & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        for x, y in pix:
            cv2.circle(img, (int(x), int(y)), 1, COLORS.get(name, (255, 255, 255)), -1)
    for name, mask in masks_by_name.items():
        contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, COLORS.get(name, (255, 255, 255)), 2)
    cv2.imwrite(str(output), img)


def write_urdf(path, base_mesh, drawer_mesh, axis, displacement):
    axis = unit(axis)
    text = f"""<?xml version="1.0"?>
<robot name="cabinet_drawer_sam3d_constrained">
  <link name="base_link">
    <visual name="base_visual"><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{base_mesh}" scale="1 1 1"/></geometry></visual>
  </link>
  <link name="drawer_link">
    <visual name="drawer_visual"><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{drawer_mesh}" scale="1 1 1"/></geometry></visual>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="base_link"/>
    <child link="drawer_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="{axis[0]:.10f} {axis[1]:.10f} {axis[2]:.10f}"/>
    <limit lower="0" upper="{float(displacement):.10g}" effort="1" velocity="0.25"/>
  </joint>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def export_scene(path, named_meshes):
    scene = trimesh.Scene()
    for name, mesh in named_meshes:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def select_target_planes(part_name, target_planes, axis, max_count):
    scored = []
    for plane in target_planes:
        n = unit(plane["normal"])
        axis_absdot = float(abs(n @ axis))
        area = float(plane["approx_bbox_area"])
        inliers = int(plane["sample_inliers"])
        score = 1.8 * axis_absdot + 0.15 * math.log1p(max(inliers, 0)) + 0.25 * math.log1p(max(area, 0.0))
        scored.append((score, axis_absdot, plane))
    scored.sort(key=lambda item: item[0], reverse=True)
    frontish = [item for item in scored if item[1] > (0.70 if part_name == "base" else 0.82)]
    selected = frontish[:max_count]
    if len(selected) < min(2, len(scored)):
        selected = scored[:max_count]
    return [item[2] for item in selected]


def fit_part(part_name, raw_mesh, target_mesh, k, depth_m, mask, axis, color_bgr):
    part_dir = OUT / part_name
    part_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{part_name}] sampling raw mesh", flush=True)
    raw_sample, raw_normals = sample_mesh_points(raw_mesh, 42000 if part_name == "base" else 32000)
    raw_planes = extract_planes(
        raw_sample,
        raw_normals,
        max_planes=6,
        threshold=0.014 if part_name == "base" else 0.012,
        min_inliers=650 if part_name == "base" else 430,
        trials=430,
        normal_sim=0.86,
    )

    target_points_all = voxel_downsample(np.asarray(target_mesh.vertices, dtype=np.float64), voxel=0.0045, max_points=14000)
    target_planes = extract_planes(
        target_points_all,
        None,
        max_planes=5,
        threshold=0.014,
        min_inliers=260,
        trials=460,
    )
    target_selected = select_target_planes(part_name, target_planes, axis, max_count=2)
    print(f"[{part_name}] raw_planes={len(raw_planes)} target_planes={len(target_planes)} selected_targets={len(target_selected)}", flush=True)

    src_fit = raw_sample
    if len(src_fit) > (5200 if part_name == "base" else 4200):
        src_fit = src_fit[RNG.choice(len(src_fit), size=5200 if part_name == "base" else 4200, replace=False)]
    src_eval = raw_sample
    if len(src_eval) > (15000 if part_name == "base" else 12000):
        src_eval = src_eval[RNG.choice(len(src_eval), size=15000 if part_name == "base" else 12000, replace=False)]

    tgt_fit = target_points_all
    if len(tgt_fit) > (3600 if part_name == "base" else 3000):
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), size=3600 if part_name == "base" else 3000, replace=False)]
    tgt_eval = target_points_all
    if len(tgt_eval) > (8000 if part_name == "base" else 6500):
        tgt_eval = tgt_eval[RNG.choice(len(tgt_eval), size=8000 if part_name == "base" else 6500, replace=False)]

    candidates = []
    candidate_id = 0
    for target_plane, source_plane in itertools.product(target_selected, raw_planes[:5]):
        target_normal = unit(target_plane["normal"])
        if float(target_normal @ axis) < 0:
            target_normal *= -1.0
        for source_sign in (-1.0, 1.0):
            for target_roll in range(4):
                target_frame = frame_from_plane(target_plane, normal_sign=1.0, roll_quarter=target_roll, desired_normal=axis)
                source_frame = frame_from_plane(source_plane, normal_sign=source_sign, roll_quarter=0)
                rotation = target_frame @ source_frame.T
                if np.linalg.det(rotation) < 0.0:
                    continue
                scale0 = initial_scale(src_fit, tgt_fit, rotation)
                for scale_factor in (1.0,):
                    scale_init = scale0 * scale_factor
                    scale, translation, history = refine_fixed_rotation(src_fit, tgt_fit, rotation, scale_init)
                    source_normal = unit(source_plane["normal"]) * source_sign
                    transformed_normal = rotation @ source_normal
                    score, metrics = score_pose(
                        src_eval,
                        tgt_eval,
                        rotation,
                        scale,
                        translation,
                        k,
                        mask,
                        depth_m,
                        transformed_normal,
                        axis,
                    )
                    source_center_cam = transform_points(
                        np.asarray(source_plane["center"], dtype=np.float64).reshape(1, 3),
                        rotation,
                        scale,
                        translation,
                    )[0]
                    target_center = np.asarray(target_plane["center"], dtype=np.float64)
                    plane_center_normal_offset = float((source_center_cam - target_center) @ target_normal)
                    score += min(abs(plane_center_normal_offset) / 0.035, 2.5)
                    candidate_id += 1
                    candidates.append(
                        {
                            "id": int(candidate_id),
                            "score": float(score),
                            "rotation": rotation,
                            "scale": float(scale),
                            "translation": translation,
                            "metrics": metrics,
                            "refinement_iterations": int(len(history)),
                            "target_plane_index": int(target_plane["index"]),
                            "source_plane_index": int(source_plane["index"]),
                            "source_normal_sign": float(source_sign),
                            "target_roll_quarter": int(target_roll),
                            "scale_init": float(scale_init),
                            "source_plane_normal_camera": transformed_normal.tolist(),
                            "target_plane_normal_camera": target_normal.tolist(),
                            "plane_center_normal_offset_m": plane_center_normal_offset,
                        }
                    )

    candidates.sort(key=lambda item: item["score"])
    exports = []
    for rank, candidate in enumerate(candidates[:6], start=1):
        mesh = raw_mesh.copy()
        mesh.vertices = transform_points(np.asarray(raw_mesh.vertices), candidate["rotation"], candidate["scale"], candidate["translation"])
        glb = part_dir / f"{part_name}_constrained_plane_rank{rank:02d}.glb"
        mesh.export(glb)
        pts, _ = sample_mesh_points(mesh, 36000 if part_name == "base" else 28000)
        overlay = part_dir / f"{part_name}_constrained_plane_rank{rank:02d}_overlay.png"
        draw_overlay(color_bgr, k, {part_name: pts}, {part_name: mask}, overlay)
        export = {
            "rank": int(rank),
            "glb": str(glb),
            "overlay": str(overlay),
            "score": float(candidate["score"]),
            "scale_uniform": float(candidate["scale"]),
            "rotation_matrix_source_to_camera": candidate["rotation"].tolist(),
            "translation_camera_m": np.asarray(candidate["translation"], dtype=float).tolist(),
            "target_plane_index": int(candidate["target_plane_index"]),
            "source_plane_index": int(candidate["source_plane_index"]),
            "source_normal_sign": float(candidate["source_normal_sign"]),
            "target_roll_quarter": int(candidate["target_roll_quarter"]),
            "source_plane_normal_camera": candidate["source_plane_normal_camera"],
            "target_plane_normal_camera": candidate["target_plane_normal_camera"],
            "plane_center_normal_offset_m": float(candidate["plane_center_normal_offset_m"]),
            "metrics": candidate["metrics"],
        }
        exports.append(export)

    best_mesh = load_mesh(exports[0]["glb"])
    best_samples, _ = sample_mesh_points(best_mesh, 40000 if part_name == "base" else 30000)
    return {
        "mesh": best_mesh,
        "samples": best_samples,
        "exports": exports,
        "raw_planes": [plane_json(p) for p in raw_planes],
        "target_planes": [plane_json(p) for p in target_planes],
        "selected_target_plane_indices": [int(p["index"]) for p in target_selected],
        "candidate_count": int(len(candidates)),
    }


def add_axis_line(origin, axis, length):
    axis = unit(axis)
    origin = np.asarray(origin, dtype=np.float64)
    length = float(max(length, 1e-4))
    mesh = trimesh.creation.cylinder(radius=0.004, height=length, sections=16)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(z_axis, axis)
    dot = float(np.clip(z_axis @ axis, -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-10:
        rot = np.eye(3, dtype=np.float64)
        if dot < 0:
            rot = np.diag([1.0, -1.0, -1.0])
    else:
        cross = unit(cross)
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        angle = math.acos(dot)
        rot = np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = origin
    mesh.apply_transform(transform)
    return mesh


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = read_json(META_PATH)
    k = np.asarray(meta["PVCamera"]["k"], dtype=np.float64)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)
    depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(DEPTH_PATH)
    depth_m = depth.astype(np.float32) / 1000.0
    joint_doc = read_json(JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    axis = unit(joint["axis_camera_closed_to_open"])
    displacement = float(joint["displacement_m"])
    closed_to_open = np.asarray(joint["translation_closed_to_open_camera_m"], dtype=np.float64)
    open_to_closed = np.asarray(joint["translation_open_to_closed_camera_m"], dtype=np.float64)

    masks = {name: load_mask(path) for name, path in MASKS.items()}
    raw_meshes = {name: load_mesh(path) for name, path in RAW_MESHES.items()}
    ref_meshes = {name: load_mesh(path) for name, path in REF_SURFACES.items()}

    results = {}
    for part_name in ("base", "drawer"):
        results[part_name] = fit_part(part_name, raw_meshes[part_name], ref_meshes[part_name], k, depth_m, masks[part_name], axis, color)

    base_mesh = results["base"]["mesh"]
    drawer_open_mesh = results["drawer"]["mesh"]
    drawer_closed_mesh = drawer_open_mesh.copy()
    drawer_closed_mesh.vertices = np.asarray(drawer_closed_mesh.vertices, dtype=np.float64) + open_to_closed

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    base_mesh.export(base_path)
    drawer_open_mesh.export(drawer_open_path)
    drawer_closed_mesh.export(drawer_closed_path)

    export_scene(OUT / "cabinet_drawer_sam3d_constrained_open.glb", [("base", base_mesh), ("drawer_open", drawer_open_mesh)])
    export_scene(OUT / "cabinet_drawer_sam3d_constrained_closed.glb", [("base", base_mesh), ("drawer_closed", drawer_closed_mesh)])
    export_scene(
        OUT / "cabinet_drawer_sam3d_constrained_open_closed_overlay.glb",
        [("base", base_mesh), ("drawer_open", drawer_open_mesh), ("drawer_closed", drawer_closed_mesh)],
    )
    axis_origin = np.median(np.asarray(drawer_closed_mesh.vertices, dtype=np.float64), axis=0)
    axis_line = add_axis_line(axis_origin, axis, max(displacement, 0.18))
    axis_line.export(OUT / "axis_line_closed_to_open.glb")

    urdf_path = OUT / "cabinet_drawer_sam3d_constrained.urdf"
    write_urdf(urdf_path, "base.glb", "drawer_closed_link.glb", axis, displacement)

    draw_overlay(
        color,
        k,
        {"base": results["base"]["samples"], "drawer": results["drawer"]["samples"]},
        masks,
        OUT / "combined_qpos1_constrained_overlay.png",
    )
    closed_samples, _ = sample_mesh_points(drawer_closed_mesh, 28000)
    draw_overlay(
        color,
        k,
        {"base": results["base"]["samples"], "drawer_closed": closed_samples, "drawer": results["drawer"]["samples"]},
        masks,
        OUT / "combined_open_closed_projection_overlay.png",
    )

    base_n = unit(results["base"]["exports"][0]["source_plane_normal_camera"])
    drawer_n = unit(results["drawer"]["exports"][0]["source_plane_normal_camera"])
    report = {
        "method": (
            "Constrained SAM3D completion fit. RGB-D qpos1 mask surfaces and the already-validated "
            "selected prismatic joint define the metric camera frame and joint. Raw SAM3D meshes supply "
            "complete shape only. Candidate rotations are discrete plane-to-plane frame matches; refinement "
            "keeps rotation fixed and optimizes only uniform scale plus camera-frame translation."
        ),
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {
            "+X": "image right",
            "+Y": "image down",
            "+Z": "forward/deeper",
        },
        "joint": {
            "type": "prismatic",
            "axis_camera_closed_to_open": axis.tolist(),
            "displacement_m": displacement,
            "translation_closed_to_open_camera_m": closed_to_open.tolist(),
            "translation_open_to_closed_camera_m": open_to_closed.tolist(),
            "source_json": str(JOINT_JSON),
        },
        "constraints": {
            "uses_closed_partmask": False,
            "updates_rotation_during_icp": False,
            "uses_whole_drawer_pca_axis": False,
            "uses_center_crop_pca_axis": False,
            "rotation_source": "enumerated raw-SAM3D plane frame to RGB-D target plane frame",
            "refined_parameters": ["uniform_scale", "camera_frame_translation"],
        },
        "parts": {},
        "pair_sanity": {
            "base_best_plane_normal_absdot_axis": float(abs(base_n @ axis)),
            "drawer_best_plane_normal_absdot_axis": float(abs(drawer_n @ axis)),
            "base_drawer_best_plane_normal_absdot": float(abs(base_n @ drawer_n)),
        },
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_glb": str(OUT / "cabinet_drawer_sam3d_constrained_open.glb"),
            "closed_glb": str(OUT / "cabinet_drawer_sam3d_constrained_closed.glb"),
            "open_closed_overlay_glb": str(OUT / "cabinet_drawer_sam3d_constrained_open_closed_overlay.glb"),
            "axis_line_glb": str(OUT / "axis_line_closed_to_open.glb"),
            "urdf": str(urdf_path),
            "qpos1_projection_overlay": str(OUT / "combined_qpos1_constrained_overlay.png"),
            "open_closed_projection_overlay": str(OUT / "combined_open_closed_projection_overlay.png"),
        },
    }
    for part_name in ("base", "drawer"):
        report["parts"][part_name] = {
            "raw_mesh": str(RAW_MESHES[part_name]),
            "reference_surface": str(REF_SURFACES[part_name]),
            "candidate_count": results[part_name]["candidate_count"],
            "selected_target_plane_indices": results[part_name]["selected_target_plane_indices"],
            "raw_planes": results[part_name]["raw_planes"],
            "target_rgbd_planes": results[part_name]["target_planes"],
            "top_exports": results[part_name]["exports"],
        }

    report_path = OUT / "sam3d_constrained_plane_fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "open_glb": report["outputs"]["open_glb"],
                "closed_glb": report["outputs"]["closed_glb"],
                "urdf": report["outputs"]["urdf"],
                "overlay": report["outputs"]["qpos1_projection_overlay"],
                "pair_sanity": report["pair_sanity"],
                "base_score": report["parts"]["base"]["top_exports"][0]["score"],
                "drawer_score": report["parts"]["drawer"]["top_exports"][0]["score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
