import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import build_rgbd_rectified_panel_proxy as rect
import build_sam3d_clean_prismatic_qpos1 as clean


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_rectified_frame_prismatic_qpos1"
OBJECT_POSE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_object_pose_aligned/object_pose_aligned_report.json"
SURFACE_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth"
CAPTURE = "20260622_081031_636398Z"
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
    "base": SURFACE_DIR / "base_rgbd_mask_surface.glb",
    "drawer": SURFACE_DIR / "drawer_rgbd_mask_surface.glb",
}
COLORS = {"base": (60, 220, 80), "drawer": (40, 70, 245), "drawer_closed": (20, 165, 255)}
RNG = np.random.default_rng(20260718)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        mesh = trimesh.util.concatenate([g.copy() for g in loaded.geometry.values()])
    return mesh


def sample_mesh(mesh, count):
    if len(mesh.faces):
        pts, _ = trimesh.sample.sample_surface(mesh, count)
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(pts) > count:
        pts = pts[RNG.choice(len(pts), size=count, replace=False)]
    return np.asarray(pts, dtype=np.float64)


def pca_axes(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return center, axes


def candidate_maps():
    out = []
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            m = p @ np.diag(signs)
            if np.linalg.det(m) > 0:
                out.append((perm, signs, m))
    return out


def local_coords(points, axes):
    frame = np.column_stack(axes)
    return np.asarray(points, dtype=np.float64) @ frame


def world_from_local(local, axes):
    frame = np.column_stack(axes)
    return np.asarray(local, dtype=np.float64) @ frame.T


def robust_extent(points, lo=5, hi=95):
    points = np.asarray(points, dtype=np.float64)
    return np.maximum(np.percentile(points, hi, axis=0) - np.percentile(points, lo, axis=0), 1e-6)


def initial_scale(src_local, tgt_local, weights):
    src_e = robust_extent(src_local)
    tgt_e = robust_extent(tgt_local)
    ratios = tgt_e / src_e
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(ratios) & (ratios > 1e-7) & (weights > 0)
    if not np.any(keep):
        return 1.0
    return float(np.average(ratios[keep], weights=weights[keep]))


def solve_scale_translation(src, dst, weights):
    weights = np.asarray(weights, dtype=np.float64).reshape(1, 3)
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    ms = src.mean(axis=0)
    md = dst.mean(axis=0)
    x = (src - ms) * weights
    y = (dst - md) * weights
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(x * y) / denom)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    trans = md - scale * ms
    return scale, trans


def refine_fixed_frame(src_local_fit, tgt_local_fit, scale0, weights, iterations=18):
    weights = np.asarray(weights, dtype=np.float64).reshape(1, 3)
    scale = float(scale0)
    trans = np.median(tgt_local_fit, axis=0) - scale * np.median(src_local_fit, axis=0)
    min_scale = max(scale0 * 0.55, 1e-8)
    max_scale = max(scale0 * 1.8, min_scale * 1.1)
    history = []
    for _ in range(iterations):
        moved = scale * src_local_fit + trans.reshape(1, 3)
        tree = cKDTree(moved * weights)
        dist, idx = tree.query(tgt_local_fit * weights, k=1, workers=-1)
        cutoff = min(float(np.percentile(dist, 82)), 0.10)
        keep = dist <= cutoff
        if int(keep.sum()) < 350:
            keep = dist <= float(np.percentile(dist, 94))
        s_new, t_new = solve_scale_translation(src_local_fit[idx[keep]], tgt_local_fit[keep], weights.reshape(3))
        scale = float(np.clip(0.60 * scale + 0.40 * s_new, min_scale, max_scale))
        trans = 0.60 * trans + 0.40 * t_new
        residual = np.linalg.norm((scale * src_local_fit[idx[keep]] + trans.reshape(1, 3) - tgt_local_fit[keep]) * weights, axis=1)
        history.append(
            {
                "weighted_median": float(np.median(residual)),
                "weighted_p90": float(np.percentile(residual, 90)),
                "pairs": int(keep.sum()),
                "scale": float(scale),
            }
        )
    return scale, trans, history


def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    keep = points[:, 2] > 1e-5
    p = points[keep]
    uv[keep, 0] = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]
    uv[keep, 1] = k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
    return uv, keep


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float64)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def score_world_points(src_world_eval, tgt_world_eval, k, mask, depth_obs):
    tree = cKDTree(np.asarray(src_world_eval, dtype=np.float64))
    dist, _ = tree.query(tgt_world_eval, k=1, workers=-1)
    chamfer_med = float(np.median(dist))
    chamfer_p90 = float(np.percentile(dist, 90))
    uv, keep = project(src_world_eval, k)
    h, w = mask.shape
    inb = keep & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if int(inb.sum()) < 100:
        return 1e9, {"reason": "few_projected"}
    pix = np.round(uv[inb]).astype(np.int32)
    px = np.clip(pix[:, 0], 0, w - 1)
    py = np.clip(pix[:, 1], 0, h - 1)
    hit = mask[py, px] > 0
    valid = depth_obs[py, px] > 0.05
    diff = np.abs(src_world_eval[inb, 2][hit & valid] - depth_obs[py[hit & valid], px[hit & valid]]) if np.any(hit & valid) else np.array([1.0])
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[py, px] = 255
    pred = cv2.dilate(pred, np.ones((5, 5), np.uint8), iterations=1) > 0
    target = mask > 0
    iou = float((pred & target).sum() / max(1, (pred | target).sum()))
    coverage = float((pred & target).sum() / max(1, target.sum()))
    leakage = float((pred & ~target).sum() / max(1, pred.sum()))
    uvi = uv[inb]
    lo = np.percentile(uvi, 1, axis=0)
    hi = np.percentile(uvi, 99, axis=0)
    bbox = np.r_[lo, hi]
    tb = mask_bbox(mask)
    ts = np.maximum(tb[2:] - tb[:2], 1)
    bbox_size_err = float(np.linalg.norm(((hi - lo) - ts) / ts))
    bbox_center_err = float(np.linalg.norm((((hi + lo) * 0.5) - ((tb[:2] + tb[2:]) * 0.5)) / ts))
    depth_med = float(np.median(diff))
    score = (
        chamfer_med / 0.018
        + 0.22 * chamfer_p90 / 0.055
        + 0.75 * depth_med / 0.05
        + 1.4 * (1.0 - iou)
        + 0.5 * (1.0 - coverage)
        + 0.65 * leakage
        + 0.8 * bbox_size_err
        + 1.2 * bbox_center_err
    )
    return float(score), {
        "score": float(score),
        "target_to_source_chamfer_median_m": chamfer_med,
        "target_to_source_chamfer_p90_m": chamfer_p90,
        "projected_mask_iou": iou,
        "target_coverage": coverage,
        "leakage": leakage,
        "projected_mask_hit_ratio": float(hit.mean()) if len(hit) else 0.0,
        "depth_abs_median_m": depth_med,
        "bbox_xyxy_p01_p99": bbox.tolist(),
        "bbox_size_err": bbox_size_err,
        "bbox_center_err": bbox_center_err,
        "projected_points": int(inb.sum()),
    }


def fit_part(name, axes, k, depth_obs, color):
    raw = load_mesh(RAW_MESHES[name])
    ref = load_mesh(REF_SURFACES[name])
    mask = cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(MASKS[name])
    mask = mask > 127
    src_surface = sample_mesh(raw, 80000 if name == "base" else 65000)
    src_fit = src_surface[RNG.choice(len(src_surface), min(len(src_surface), 16000), replace=False)]
    src_eval = src_surface[RNG.choice(len(src_surface), min(len(src_surface), 30000), replace=False)]
    tgt_all = np.asarray(ref.vertices, dtype=np.float64)
    tgt_fit = tgt_all
    if len(tgt_fit) > 10000:
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), size=10000, replace=False)]
    tgt_eval = tgt_all

    src_center, src_axes = pca_axes(src_fit)
    tgt_local_fit = local_coords(tgt_fit, axes)
    weights = np.array([1.0, 1.0, 0.45 if name == "drawer" else 0.70], dtype=np.float64)
    candidates = []
    frame = np.column_stack(axes)
    for perm, signs, m in candidate_maps():
        src_local_fit = ((src_fit - src_center.reshape(1, 3)) @ src_axes) @ m.T
        src_local_eval = ((src_eval - src_center.reshape(1, 3)) @ src_axes) @ m.T
        scale0 = initial_scale(src_local_fit, tgt_local_fit, weights)
        scale, trans_local, history = refine_fixed_frame(src_local_fit, tgt_local_fit, scale0, weights)
        moved_eval = world_from_local(scale * src_local_eval + trans_local.reshape(1, 3), axes)
        score, metrics = score_world_points(moved_eval, tgt_eval, k, mask, depth_obs)
        rotation_source_to_camera = frame @ m @ src_axes.T
        candidates.append(
            {
                "score": score,
                "metrics": metrics,
                "source_center": src_center,
                "source_axes": src_axes,
                "map_matrix": m,
                "perm": perm,
                "signs": signs,
                "scale_uniform": scale,
                "translation_local": trans_local,
                "rotation_source_to_camera": rotation_source_to_camera,
                "history": history,
            }
        )
    candidates.sort(key=lambda item: item["score"])

    part_dir = OUT / name
    part_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for rank, cand in enumerate(candidates[:4], start=1):
        mesh = raw.copy()
        raw_local = ((np.asarray(raw.vertices, dtype=np.float64) - cand["source_center"].reshape(1, 3)) @ cand["source_axes"]) @ cand["map_matrix"].T
        mesh.vertices = world_from_local(cand["scale_uniform"] * raw_local + cand["translation_local"].reshape(1, 3), axes)
        glb = part_dir / f"{name}_sam3d_rectified_frame_rank{rank:02d}.glb"
        mesh.export(glb)
        pts = sample_mesh(mesh, 45000 if name == "base" else 32000)
        overlay = part_dir / f"{name}_rectified_rank{rank:02d}_overlay.png"
        draw_overlay(color, k, {name: pts}, overlay)
        exports.append(
            {
                "rank": rank,
                "glb": str(glb),
                "overlay": str(overlay),
                "score": cand["score"],
                "scale_uniform": cand["scale_uniform"],
                "translation_local_uvq": cand["translation_local"].astype(float).tolist(),
                "rotation_matrix_source_to_camera_fixed_frame": cand["rotation_source_to_camera"].astype(float).tolist(),
                "source_center": cand["source_center"].astype(float).tolist(),
                "map_perm": list(cand["perm"]),
                "map_signs": list(cand["signs"]),
                "metrics": cand["metrics"],
                "last_refine": cand["history"][-1] if cand["history"] else {},
            }
        )
    return exports


def draw_overlay(color, k, samples, path):
    img = color.copy()
    for name, points in samples.items():
        uv, keep = project(points, k)
        h, w = img.shape[:2]
        inb = keep & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 70000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, COLORS.get(name, (255, 255, 255)), -1, cv2.LINE_AA)
        mask_name = "drawer" if name == "drawer_closed" else name
        mask = cv2.imread(str(MASKS[mask_name]), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, COLORS.get(name, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    object_report = read_json(OBJECT_POSE_REPORT)
    axis = unit(object_report["joint_axis_kept_fixed"]["axis_camera_closed_to_open"])
    q = float(object_report["joint_axis_kept_fixed"]["best_residual_q_m"])
    open_to_closed = -axis * q
    closed_to_open = axis * q
    k = np.asarray(read_json(META_PATH)["PVCamera"]["k"], dtype=np.float64)
    depth_obs = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)

    drawer_ref = load_mesh(REF_SURFACES["drawer"])
    axes, rect_info = rect.rectified_frame(np.asarray(drawer_ref.vertices, dtype=np.float64), axis, k)
    frame = np.column_stack(axes)
    if abs(np.linalg.det(frame) - 1.0) > 1e-5:
        raise RuntimeError("rectified frame is not right-handed")

    report = {
        "method": "SAM3D fit constrained to a qpos1 RGB-D rectified orthogonal frame. The target frame is fixed from the prismatic axis plus qpos1 drawer mask rectangle; optimization only chooses raw SAM3D local-axis permutation/sign, one uniform scale, and translation. It does not allow arbitrary 3D rotation/ICP drift.",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "sfm_used": False,
        "raw_camera_pose_used": False,
        "closed_partmask_used": False,
        "foundationpose_used": False,
        "joint": {
            "axis_camera_closed_to_open": axis.astype(float).tolist(),
            "qpos_open_m": q,
            "qpos_closed_m": 0.0,
            "translation_open_to_closed_camera_m": open_to_closed.astype(float).tolist(),
            "translation_closed_to_open_camera_m": closed_to_open.astype(float).tolist(),
            "source": str(OBJECT_POSE_REPORT),
        },
        "rectified_frame": {
            "u": axes[0].astype(float).tolist(),
            "v": axes[1].astype(float).tolist(),
            "axis_n": axes[2].astype(float).tolist(),
            "determinant": float(np.linalg.det(frame)),
            "mask_center_px": rect_info["center_px"].astype(float).tolist(),
            "mask_horizontal_px_dir": rect_info["horizontal_px_dir"].astype(float).tolist(),
            "mask_vertical_px_dir": rect_info["vertical_px_dir"].astype(float).tolist(),
        },
        "parts": {},
    }

    for name in ("base", "drawer"):
        report["parts"][name] = {
            "raw_mesh": str(RAW_MESHES[name]),
            "reference_surface": str(REF_SURFACES[name]),
            "top_exports": fit_part(name, axes, k, depth_obs, color),
        }

    base = load_mesh(report["parts"]["base"]["top_exports"][0]["glb"])
    drawer_open = load_mesh(report["parts"]["drawer"]["top_exports"][0]["glb"])
    drawer_closed = clean.translate_mesh(drawer_open, open_to_closed)
    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    open_scene = OUT / "cabinet_drawer_sam3d_rectified_frame_open.glb"
    closed_scene = OUT / "cabinet_drawer_sam3d_rectified_frame_closed.glb"
    overlay_scene = OUT / "cabinet_drawer_sam3d_rectified_frame_open_closed_overlay.glb"
    animated_glb = OUT / "cabinet_drawer_sam3d_rectified_frame_prismatic_animated.glb"
    urdf = OUT / "cabinet_drawer_sam3d_rectified_frame_prismatic.urdf"
    base.export(base_path)
    drawer_open.export(drawer_open_path)
    drawer_closed.export(drawer_closed_path)
    clean.export_scene(open_scene, {"base": base, "drawer_open": drawer_open})
    clean.export_scene(closed_scene, {"base": base, "drawer_closed": drawer_closed})
    clean.export_scene(overlay_scene, {"base": base, "drawer_open": drawer_open, "drawer_closed": drawer_closed})
    clean.build_animated_glb(animated_glb, base, drawer_closed, axis, q)
    clean.write_urdf(urdf, axis, q)

    base_pts = sample_mesh(base, 70000)
    drawer_pts = sample_mesh(drawer_open, 45000)
    drawer_closed_pts = sample_mesh(drawer_closed, 45000)
    open_overlay = OUT / "qpos1_sam3d_rectified_frame_open_overlay.png"
    open_closed_overlay = OUT / "qpos1_sam3d_rectified_frame_open_closed_overlay.png"
    draw_overlay(color, k, {"base": base_pts, "drawer": drawer_pts}, open_overlay)
    draw_overlay(color, k, {"base": base_pts, "drawer": drawer_pts, "drawer_closed": drawer_closed_pts}, open_closed_overlay)

    report["outputs"] = {
        "base_glb": str(base_path),
        "drawer_open_reference_glb": str(drawer_open_path),
        "drawer_closed_link_glb": str(drawer_closed_path),
        "open_scene_glb": str(open_scene),
        "closed_scene_glb": str(closed_scene),
        "open_closed_overlay_glb": str(overlay_scene),
        "animated_prismatic_glb": str(animated_glb),
        "urdf": str(urdf),
        "joint_json": str(OUT / "joint.json"),
        "qpos1_open_overlay": str(open_overlay),
        "qpos1_open_closed_overlay": str(open_closed_overlay),
        "report": str(OUT / "sam3d_rectified_frame_prismatic_report.json"),
    }
    joint = {
        "type": "prismatic",
        "frame": report["frame"],
        "camera_axes": report["camera_axes"],
        "axis_camera_closed_to_open": axis.astype(float).tolist(),
        "qpos_closed_m": 0.0,
        "qpos_open_m": q,
        "displacement_m": q,
        "translation_open_to_closed_camera_m": open_to_closed.astype(float).tolist(),
        "translation_closed_to_open_camera_m": closed_to_open.astype(float).tolist(),
        "sfm_used": False,
        "raw_camera_pose_used": False,
        "closed_partmask_used": False,
        "sam3d_fit_constraint": "fixed_qpos1_rgbd_rectified_frame",
    }
    (OUT / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    (OUT / "sam3d_rectified_frame_prismatic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), **report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
