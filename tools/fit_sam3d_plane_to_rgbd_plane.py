import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


ROOT = Path("/workspace_whz")
sys.path.insert(0, str(ROOT / "tools"))
import build_sam3d_clean_prismatic_qpos1 as clean  # noqa: E402
import fit_sam3d_constrained_plane_to_rgbd as c  # noqa: E402


CAPTURE = "20260622_081031_636398Z"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_plane_to_rgbd_plane_fit"
REF_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth"
OBJECT_POSE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_object_pose_aligned/object_pose_aligned_report.json"
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
RNG = np.random.default_rng(20260719)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    return c.unit(v)


def rotation_about_axis(axis, angle):
    axis = unit(axis)
    x, y, z = axis
    ca = math.cos(float(angle))
    sa = math.sin(float(angle))
    cc = 1.0 - ca
    return np.array(
        [
            [ca + x * x * cc, x * y * cc - z * sa, x * z * cc + y * sa],
            [y * x * cc + z * sa, ca + y * y * cc, y * z * cc - x * sa],
            [z * x * cc - y * sa, z * y * cc + x * sa, ca + z * z * cc],
        ],
        dtype=np.float64,
    )


def frame_from_plane(plane, normal_sign=1.0):
    n = unit(np.asarray(plane["normal"], dtype=np.float64) * float(normal_sign))
    u = np.asarray(plane["axes"][0], dtype=np.float64)
    u = unit(u - n * float(u @ n))
    v = unit(np.cross(n, u))
    frame = np.column_stack([u, v, n])
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1.0
    return frame


def sorted_extent(plane):
    ext = np.asarray(plane["extent"], dtype=np.float64).reshape(2)
    return np.sort(np.maximum(ext, 1e-6))


def plane_scale_seed(source_plane, target_plane):
    src = sorted_extent(source_plane)
    tgt = sorted_extent(target_plane)
    ratios = tgt / src
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
    if len(ratios) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(ratios))))


def transform_points(points, rotation, scale, translation):
    return c.transform_points(points, rotation, scale, translation)


def project_all(points, k):
    return c.project_all(points, k)


def mask_bbox(mask):
    return c.mask_bbox(mask)


def score_pose(
    src_eval,
    tgt_eval,
    rotation,
    scale,
    translation,
    k,
    mask,
    depth_m,
    source_plane,
    source_normal_sign,
    target_plane,
):
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
    target_mask = mask > 0
    hit = target_mask[py, px]
    valid_depth = depth_m[py, px] > 0.05
    depth_diff = np.abs(pred[inb, 2][hit & valid_depth] - depth_m[py[hit & valid_depth], px[hit & valid_depth]])
    if len(depth_diff) == 0:
        depth_diff = np.array([1.0], dtype=np.float64)
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    pred_mask[py, px] = 255
    pred_mask = cv2.dilate(pred_mask, np.ones((5, 5), np.uint8), iterations=1) > 0
    iou = float((pred_mask & target_mask).sum() / max(1, (pred_mask | target_mask).sum()))
    coverage = float((pred_mask & target_mask).sum() / max(1, target_mask.sum()))
    leakage = float((pred_mask & ~target_mask).sum() / max(1, pred_mask.sum()))

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

    src_normal_local = unit(np.asarray(source_plane["normal"], dtype=np.float64) * float(source_normal_sign))
    src_normal_cam = unit(rotation @ src_normal_local)
    tgt_normal = unit(np.asarray(target_plane["normal"], dtype=np.float64))
    plane_normal_absdot = float(abs(src_normal_cam @ tgt_normal))

    src_center_cam = transform_points(
        np.asarray(source_plane["center"], dtype=np.float64).reshape(1, 3),
        rotation,
        scale,
        translation,
    )[0]
    tgt_center = np.asarray(target_plane["center"], dtype=np.float64)
    delta = src_center_cam - tgt_center
    normal_offset = float(abs(delta @ tgt_normal))
    tangent = delta - tgt_normal * float(delta @ tgt_normal)
    tangent_offset = float(np.linalg.norm(tangent))
    extent_ratio = (sorted_extent(target_plane) / np.maximum(sorted_extent(source_plane) * float(scale), 1e-6))
    extent_log_err = float(np.linalg.norm(np.log(np.clip(extent_ratio, 1e-4, 1e4))))

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
        + 2.00 * (1.0 - plane_normal_absdot)
        + 1.25 * min(normal_offset / 0.040, 3.0)
        + 0.35 * min(tangent_offset / 0.140, 3.0)
        + 0.35 * min(extent_log_err, 4.0)
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
        "plane_normal_absdot_after_transform": plane_normal_absdot,
        "plane_center_normal_offset_m": normal_offset,
        "plane_center_tangent_offset_m": tangent_offset,
        "plane_extent_log_err": extent_log_err,
    }


def solve_scale_translation(src_rotated, dst):
    return c.solve_scale_translation(src_rotated, dst)


def refine_fixed_rotation_from_init(src_fit, tgt_fit, rotation, scale0, translation0, iterations=7):
    src_fit = np.asarray(src_fit, dtype=np.float64)
    tgt_fit = np.asarray(tgt_fit, dtype=np.float64)
    src_rot = src_fit @ rotation.T
    scale = float(scale0)
    translation = np.asarray(translation0, dtype=np.float64).reshape(3)
    min_scale = max(scale0 * 0.55, 1e-7)
    max_scale = max(scale0 * 1.9, min_scale * 1.1)
    history = []
    for _ in range(iterations):
        pred = scale * src_rot + translation.reshape(1, 3)
        tree = cKDTree(pred)
        dist, idx = tree.query(tgt_fit, k=1, workers=-1)
        thresh = min(max(float(np.percentile(dist, 78)), 0.012), 0.10)
        keep = dist <= thresh
        if int(keep.sum()) < 180:
            keep = dist <= np.percentile(dist, 90)
        s_new, t_new = solve_scale_translation(src_rot[idx[keep]], tgt_fit[keep])
        s_new = float(np.clip(s_new, min_scale, max_scale))
        update = abs(math.log(max(s_new, 1e-12) / max(scale, 1e-12))) + float(np.linalg.norm(t_new - translation))
        residual = np.linalg.norm(scale * src_rot[idx[keep]] + translation.reshape(1, 3) - tgt_fit[keep], axis=1)
        history.append(
            {
                "update": float(update),
                "pairs": int(keep.sum()),
                "median_m": float(np.median(residual)),
                "p90_m": float(np.percentile(residual, 90)),
                "scale": float(scale),
            }
        )
        scale = 0.68 * scale + 0.32 * s_new
        translation = 0.68 * translation + 0.32 * t_new
        if update < 1e-5:
            break
    return float(scale), translation, history


def coarse_candidate_score(candidate):
    metrics = candidate.get("coarse_metrics", {})
    if "score" not in metrics:
        return 1e9
    return float(metrics["score"])


def make_candidate_rotation(source_plane, target_plane, source_sign, roll_angle):
    source_frame = frame_from_plane(source_plane, normal_sign=source_sign)
    target_frame = frame_from_plane(target_plane, normal_sign=1.0)
    base_rotation = target_frame @ source_frame.T
    roll_rotation = rotation_about_axis(target_frame[:, 2], roll_angle)
    rotation = roll_rotation @ base_rotation
    if np.linalg.det(rotation) < 0:
        return None
    return rotation


def fit_part(part_name, raw_mesh, target_mesh, k, depth_m, mask, color_bgr):
    part_dir = OUT / part_name
    part_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{part_name}] sampling raw mesh", flush=True)
    raw_sample, raw_normals = c.sample_mesh_points(raw_mesh, 52000 if part_name == "base" else 40000)
    raw_planes = c.extract_planes(
        raw_sample,
        raw_normals,
        max_planes=7,
        threshold=0.014 if part_name == "base" else 0.012,
        min_inliers=620 if part_name == "base" else 430,
        trials=520,
        normal_sim=0.86,
    )
    target_points_all = c.voxel_downsample(
        np.asarray(target_mesh.vertices, dtype=np.float64),
        voxel=0.0045,
        max_points=15000 if part_name == "base" else 12000,
    )
    target_planes = c.extract_planes(
        target_points_all,
        None,
        max_planes=6,
        threshold=0.014,
        min_inliers=250,
        trials=520,
    )
    print(f"[{part_name}] raw_planes={len(raw_planes)} target_planes={len(target_planes)}", flush=True)
    if not raw_planes or not target_planes:
        raise RuntimeError(f"{part_name}: no usable plane candidates")

    src_fit = raw_sample
    if len(src_fit) > (5600 if part_name == "base" else 4400):
        src_fit = src_fit[RNG.choice(len(src_fit), size=5600 if part_name == "base" else 4400, replace=False)]
    src_eval = raw_sample
    if len(src_eval) > (16000 if part_name == "base" else 12500):
        src_eval = src_eval[RNG.choice(len(src_eval), size=16000 if part_name == "base" else 12500, replace=False)]
    src_coarse = src_eval
    if len(src_coarse) > 3800:
        src_coarse = src_coarse[RNG.choice(len(src_coarse), size=3800, replace=False)]

    tgt_fit = target_points_all
    if len(tgt_fit) > (4200 if part_name == "base" else 3300):
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), size=4200 if part_name == "base" else 3300, replace=False)]
    tgt_eval = target_points_all
    if len(tgt_eval) > (8500 if part_name == "base" else 6800):
        tgt_eval = tgt_eval[RNG.choice(len(tgt_eval), size=8500 if part_name == "base" else 6800, replace=False)]
    tgt_coarse = tgt_eval
    if len(tgt_coarse) > 2600:
        tgt_coarse = tgt_coarse[RNG.choice(len(tgt_coarse), size=2600, replace=False)]

    source_planes = raw_planes[:6]
    target_plane_candidates = target_planes[:6]
    roll_angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    coarse_candidates = []
    candidate_id = 0
    for target_plane, source_plane in itertools.product(target_plane_candidates, source_planes):
        base_scale = plane_scale_seed(source_plane, target_plane)
        full_scale = c.initial_scale(src_fit, tgt_fit, np.eye(3))
        scale_seeds = sorted({round(base_scale, 8), round(math.sqrt(max(base_scale * full_scale, 1e-12)), 8)})
        for source_sign in (-1.0, 1.0):
            for roll_angle in roll_angles:
                rotation = make_candidate_rotation(source_plane, target_plane, source_sign, roll_angle)
                if rotation is None:
                    continue
                source_center = np.asarray(source_plane["center"], dtype=np.float64)
                target_center = np.asarray(target_plane["center"], dtype=np.float64)
                for scale0 in scale_seeds:
                    translation0 = target_center - float(scale0) * (source_center @ rotation.T)
                    score, metrics = score_pose(
                        src_coarse,
                        tgt_coarse,
                        rotation,
                        float(scale0),
                        translation0,
                        k,
                        mask,
                        depth_m,
                        source_plane,
                        source_sign,
                        target_plane,
                    )
                    candidate_id += 1
                    coarse_candidates.append(
                        {
                            "id": int(candidate_id),
                            "coarse_score": float(score),
                            "coarse_metrics": metrics,
                            "rotation": rotation,
                            "scale_init": float(scale0),
                            "translation_init": translation0,
                            "target_plane_index": int(target_plane["index"]),
                            "source_plane_index": int(source_plane["index"]),
                            "source_normal_sign": float(source_sign),
                            "roll_angle_rad": float(roll_angle),
                            "source_plane": source_plane,
                            "target_plane": target_plane,
                        }
                    )
    coarse_candidates.sort(key=coarse_candidate_score)
    keep_n = min(96 if part_name == "base" else 80, len(coarse_candidates))
    print(f"[{part_name}] coarse_candidates={len(coarse_candidates)} refining={keep_n}", flush=True)

    full_candidates = []
    for item in coarse_candidates[:keep_n]:
        scale, translation, history = refine_fixed_rotation_from_init(
            src_fit,
            tgt_fit,
            item["rotation"],
            item["scale_init"],
            item["translation_init"],
            iterations=7,
        )
        score, metrics = score_pose(
            src_eval,
            tgt_eval,
            item["rotation"],
            scale,
            translation,
            k,
            mask,
            depth_m,
            item["source_plane"],
            item["source_normal_sign"],
            item["target_plane"],
        )
        full_candidates.append(
            {
                **{k2: v for k2, v in item.items() if k2 not in {"source_plane", "target_plane", "coarse_metrics"}},
                "score": float(score),
                "metrics": metrics,
                "scale": float(scale),
                "translation": translation,
                "refinement_history": history,
                "source_plane_normal_camera": (
                    item["rotation"] @ (unit(item["source_plane"]["normal"]) * float(item["source_normal_sign"]))
                ).tolist(),
                "target_plane_normal_camera": unit(item["target_plane"]["normal"]).tolist(),
            }
        )
    full_candidates.sort(key=lambda item: item["score"])
    if not full_candidates:
        raise RuntimeError(f"{part_name}: no full candidates")

    exports = []
    for rank, candidate in enumerate(full_candidates[:8], start=1):
        mesh = raw_mesh.copy()
        mesh.vertices = transform_points(
            np.asarray(raw_mesh.vertices, dtype=np.float64),
            candidate["rotation"],
            candidate["scale"],
            candidate["translation"],
        )
        glb = part_dir / f"{part_name}_plane_to_rgbd_plane_rank{rank:02d}.glb"
        mesh.export(glb)
        pts, _ = c.sample_mesh_points(mesh, 36000 if part_name == "base" else 28000)
        overlay = part_dir / f"{part_name}_plane_to_rgbd_plane_rank{rank:02d}_overlay.png"
        c.draw_overlay(color_bgr, k, {part_name: pts}, {part_name: mask}, overlay)
        exports.append(
            {
                "rank": int(rank),
                "glb": str(glb),
                "overlay": str(overlay),
                "score": float(candidate["score"]),
                "coarse_score": float(candidate["coarse_score"]),
                "scale_uniform": float(candidate["scale"]),
                "rotation_matrix_source_to_camera": candidate["rotation"].tolist(),
                "translation_camera_m": np.asarray(candidate["translation"], dtype=float).tolist(),
                "target_plane_index": int(candidate["target_plane_index"]),
                "source_plane_index": int(candidate["source_plane_index"]),
                "source_normal_sign": float(candidate["source_normal_sign"]),
                "roll_angle_rad": float(candidate["roll_angle_rad"]),
                "source_plane_normal_camera": candidate["source_plane_normal_camera"],
                "target_plane_normal_camera": candidate["target_plane_normal_camera"],
                "metrics": candidate["metrics"],
                "refinement_last": candidate["refinement_history"][-1] if candidate["refinement_history"] else None,
            }
        )
    best_mesh = c.load_mesh(exports[0]["glb"])
    samples, _ = c.sample_mesh_points(best_mesh, 40000 if part_name == "base" else 30000)
    return {
        "mesh": best_mesh,
        "samples": samples,
        "exports": exports,
        "raw_planes": [c.plane_json(p) for p in raw_planes],
        "target_planes": [c.plane_json(p) for p in target_planes],
        "candidate_count": int(len(coarse_candidates)),
        "refined_candidate_count": int(len(full_candidates)),
    }


def load_joint():
    report = read_json(OBJECT_POSE_REPORT)
    joint = report["joint_axis_kept_fixed"]
    axis = unit(joint["axis_camera_closed_to_open"])
    q = float(joint["best_residual_q_m"])
    return report, axis, q


def export_scene(path, named_meshes):
    scene = trimesh.Scene()
    for name, mesh in named_meshes:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def select_pair(base_exports, drawer_exports):
    best = None
    for base_item in base_exports[:6]:
        bn = unit(base_item["source_plane_normal_camera"])
        for drawer_item in drawer_exports[:6]:
            dn = unit(drawer_item["source_plane_normal_camera"])
            normal_parallel_penalty = 1.50 * (1.0 - abs(float(bn @ dn)))
            pair_score = float(base_item["score"]) + float(drawer_item["score"]) + normal_parallel_penalty
            item = {
                "pair_score": float(pair_score),
                "base_rank": int(base_item["rank"]),
                "drawer_rank": int(drawer_item["rank"]),
                "base_drawer_selected_plane_normal_absdot": float(abs(bn @ dn)),
                "normal_parallel_penalty": float(normal_parallel_penalty),
            }
            if best is None or pair_score < best["pair_score"]:
                best = item
    return best


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

    object_pose_report, axis, q = load_joint()
    translation_closed_to_open = axis * q
    translation_open_to_closed = -translation_closed_to_open

    masks = {name: c.load_mask(path) for name, path in MASKS.items()}
    raw_meshes = {name: c.load_mesh(path) for name, path in RAW_MESHES.items()}
    ref_meshes = {name: c.load_mesh(path) for name, path in REF_SURFACES.items()}

    results = {}
    for part_name in ("base", "drawer"):
        results[part_name] = fit_part(part_name, raw_meshes[part_name], ref_meshes[part_name], k, depth_m, masks[part_name], color)

    pair = select_pair(results["base"]["exports"], results["drawer"]["exports"])
    base_export = results["base"]["exports"][pair["base_rank"] - 1]
    drawer_export = results["drawer"]["exports"][pair["drawer_rank"] - 1]
    base_mesh = c.load_mesh(base_export["glb"])
    drawer_open_mesh = c.load_mesh(drawer_export["glb"])
    drawer_closed_mesh = drawer_open_mesh.copy()
    drawer_closed_mesh.vertices = np.asarray(drawer_closed_mesh.vertices, dtype=np.float64) + translation_open_to_closed

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    open_scene = OUT / "cabinet_drawer_sam3d_plane_to_rgbd_plane_open.glb"
    closed_scene = OUT / "cabinet_drawer_sam3d_plane_to_rgbd_plane_closed.glb"
    open_closed_scene = OUT / "cabinet_drawer_sam3d_plane_to_rgbd_plane_open_closed_overlay.glb"
    animated_glb = OUT / "cabinet_drawer_sam3d_plane_to_rgbd_plane_prismatic_animated.glb"
    urdf = OUT / "cabinet_drawer_sam3d_plane_to_rgbd_plane.urdf"

    base_mesh.export(base_path)
    drawer_open_mesh.export(drawer_open_path)
    drawer_closed_mesh.export(drawer_closed_path)
    export_scene(open_scene, [("base", base_mesh), ("drawer_open", drawer_open_mesh)])
    export_scene(closed_scene, [("base", base_mesh), ("drawer_closed", drawer_closed_mesh)])
    export_scene(open_closed_scene, [("base", base_mesh), ("drawer_open", drawer_open_mesh), ("drawer_closed", drawer_closed_mesh)])
    clean.build_animated_glb(animated_glb, base_mesh, drawer_closed_mesh, axis, q)
    clean.write_urdf(urdf, axis, q)

    base_samples, _ = c.sample_mesh_points(base_mesh, 42000)
    drawer_samples, _ = c.sample_mesh_points(drawer_open_mesh, 32000)
    drawer_closed_samples, _ = c.sample_mesh_points(drawer_closed_mesh, 32000)
    c.draw_overlay(
        color,
        k,
        {"base": base_samples, "drawer": drawer_samples},
        masks,
        OUT / "combined_qpos1_plane_to_rgbd_plane_overlay.png",
    )
    c.draw_overlay(
        color,
        k,
        {"base": base_samples, "drawer": drawer_samples, "drawer_closed": drawer_closed_samples},
        masks,
        OUT / "combined_open_closed_plane_to_rgbd_plane_overlay.png",
    )

    joint = {
        "type": "prismatic",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "axis_camera_closed_to_open": axis.astype(float).tolist(),
        "qpos_closed_m": 0.0,
        "qpos_open_m": float(q),
        "displacement_m": float(q),
        "translation_open_to_closed_camera_m": translation_open_to_closed.astype(float).tolist(),
        "translation_closed_to_open_camera_m": translation_closed_to_open.astype(float).tolist(),
        "source": str(OBJECT_POSE_REPORT),
    }
    report = {
        "method": (
            "Plane-to-plane SAM3D to RGB-D fit. Raw SAM3D plane normals are never compared to the motion axis. "
            "Each raw SAM3D plane is only compared to an RGB-D target plane after a candidate source-to-camera "
            "transform has been constructed. The prismatic axis is used only for final articulated export."
        ),
        "frame": joint["frame"],
        "camera_axes": joint["camera_axes"],
        "constraints": {
            "uses_sfm": False,
            "uses_raw_camera_pose": False,
            "uses_closed_partmask": False,
            "uses_foundationpose": False,
            "uses_whole_mesh_pca_for_world_frame": False,
            "uses_raw_sam3d_normal_vs_motion_axis": False,
            "uses_target_plane_axis_preselection": False,
            "updates_rotation_during_icp": False,
            "rotation_source": "enumerated source SAM3D plane frame to target RGB-D plane frame plus in-plane roll",
            "refined_parameters": ["uniform_scale", "camera_frame_translation"],
        },
        "joint": joint,
        "q_evidence": {
            "object_pose_aligned_report": str(OBJECT_POSE_REPORT),
            "best_residual_q_m": float(q),
            "q_sweep_top3": object_pose_report.get("q_sweep_top15", [])[:3],
        },
        "pair_selection": pair,
        "parts": {},
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_glb": str(open_scene),
            "closed_glb": str(closed_scene),
            "open_closed_overlay_glb": str(open_closed_scene),
            "animated_prismatic_glb": str(animated_glb),
            "urdf": str(urdf),
            "joint_json": str(OUT / "joint.json"),
            "qpos1_projection_overlay": str(OUT / "combined_qpos1_plane_to_rgbd_plane_overlay.png"),
            "open_closed_projection_overlay": str(OUT / "combined_open_closed_plane_to_rgbd_plane_overlay.png"),
            "report": str(OUT / "sam3d_plane_to_rgbd_plane_fit_report.json"),
        },
    }
    for part_name in ("base", "drawer"):
        report["parts"][part_name] = {
            "raw_mesh": str(RAW_MESHES[part_name]),
            "reference_surface": str(REF_SURFACES[part_name]),
            "candidate_count": results[part_name]["candidate_count"],
            "refined_candidate_count": results[part_name]["refined_candidate_count"],
            "selected_rank": int(pair[f"{part_name}_rank"]),
            "raw_planes": results[part_name]["raw_planes"],
            "target_rgbd_planes": results[part_name]["target_planes"],
            "top_exports": results[part_name]["exports"],
        }

    (OUT / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    report_path = OUT / "sam3d_plane_to_rgbd_plane_fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "open_glb": str(open_scene),
                "animated_prismatic_glb": str(animated_glb),
                "overlay": str(OUT / "combined_qpos1_plane_to_rgbd_plane_overlay.png"),
                "pair_selection": pair,
                "base_score": base_export["score"],
                "drawer_score": drawer_export["score"],
                "base_metrics": base_export["metrics"],
                "drawer_metrics": drawer_export["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
