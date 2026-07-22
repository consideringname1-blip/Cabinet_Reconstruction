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
OUT_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit"

COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"

PARTS = {
    "base": {
        "mesh": ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb",
        "mask": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
        "color": (0, 220, 0),
    },
    "drawer": {
        "mesh": ROOT / "data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb",
        "mask": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
        "color": (0, 0, 240),
    },
}

RNG = np.random.default_rng(20260624)


def load_k() -> np.ndarray:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return np.asarray(meta["PVCamera"]["k"], dtype=np.float64)


def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    meshes = []
    for geom in scene.geometry.values():
        meshes.append(geom.copy())
    mesh = trimesh.util.concatenate(meshes)
    mesh.process(validate=False)
    return mesh


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def depth_mask_points(mask: np.ndarray, depth_mm: np.ndarray, k: np.ndarray, max_points: int = 12000) -> tuple[np.ndarray, dict]:
    valid = (mask > 0) & (depth_mm > 0)
    ys, xs = np.where(valid)
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    good = (z > 0.2) & (z < 4.0)
    xs = xs[good].astype(np.float64)
    ys = ys[good].astype(np.float64)
    z = z[good]
    points = np.column_stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z))
    if len(points) > max_points:
        idx = RNG.choice(len(points), max_points, replace=False)
        points = points[idx]
    stats = {
        "points": int(len(points)),
        "median_depth_m": float(np.median(points[:, 2])),
        "center_median_m": np.median(points, axis=0).tolist(),
        "bbox_xyxy": mask_bbox(mask),
    }
    return points, stats


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    x = points - center
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1
    return center, vecs, vals


def rotation_candidates(src_axes: np.ndarray, tgt_axes: np.ndarray) -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            s = np.diag(signs)
            m = p @ s
            r = tgt_axes @ m @ src_axes.T
            if np.linalg.det(r) > 0.0:
                out.append(r)
    return out


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    mu_x = src.mean(axis=0)
    mu_y = dst.mean(axis=0)
    x = src - mu_x
    y = dst - mu_y
    cov = (y.T @ x) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    var_x = np.mean(np.sum(x * x, axis=1))
    scale = float(np.trace(np.diag(singular) @ d) / max(var_x, 1e-12))
    t = mu_y - scale * (mu_x @ r.T)
    return r, scale, t


def transform(points: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return scale * (points @ r.T) + t


def project(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    good = z > 1e-5
    p = points[good]
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    return uv, good


def raster_iou(uv: np.ndarray, mask: np.ndarray) -> float:
    h, w = mask.shape
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = np.round(uv[inb]).astype(np.int32)
    if len(uv) == 0:
        return 0.0
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[uv[:, 1], uv[:, 0]] = 255
    pred = cv2.dilate(pred, np.ones((7, 7), np.uint8), iterations=1)
    tgt = mask > 0
    pr = pred > 0
    inter = int(np.logical_and(tgt, pr).sum())
    union = int(np.logical_or(tgt, pr).sum())
    return float(inter / union) if union else 0.0


def score_pose(
    src_eval: np.ndarray,
    tgt_eval: np.ndarray,
    r: np.ndarray,
    scale: float,
    t: np.ndarray,
    k: np.ndarray,
    mask: np.ndarray,
    target_stats: dict,
) -> tuple[float, dict]:
    p = transform(src_eval, r, scale, t)
    tree = cKDTree(p)
    dist, _ = tree.query(tgt_eval, k=1, workers=-1)
    chamfer_med = float(np.median(dist))
    chamfer_p90 = float(np.percentile(dist, 90))

    uv, good = project(p, k)
    h, w = mask.shape
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if inb.sum() < 100:
        return 1e9, {"reason": "few projected points"}
    uv_in = uv[inb]
    lo = np.percentile(uv_in, 1.0, axis=0)
    hi = np.percentile(uv_in, 99.0, axis=0)
    bbox = np.r_[lo, hi]
    target_bbox = np.asarray(target_stats["bbox_xyxy"], dtype=np.float64)
    target_size = np.maximum(target_bbox[2:] - target_bbox[:2], 1.0)
    bbox_size = hi - lo
    bbox_center = (hi + lo) * 0.5
    target_center = (target_bbox[:2] + target_bbox[2:]) * 0.5
    bbox_err = float(np.linalg.norm((bbox_size - target_size) / target_size))
    center_err = float(np.linalg.norm((bbox_center - target_center) / target_size))
    depth_err = float(abs(np.median(p[:, 2]) - target_stats["median_depth_m"]))
    iou = raster_iou(uv, mask)
    score = chamfer_med / 0.025 + 0.35 * chamfer_p90 / 0.06 + bbox_err + 1.5 * center_err + depth_err / 0.08 + 2.0 * (1.0 - iou)
    details = {
        "score": score,
        "chamfer_median_m": chamfer_med,
        "chamfer_p90_m": chamfer_p90,
        "bbox_xyxy_p01_p99": bbox.tolist(),
        "bbox_size_px": bbox_size.tolist(),
        "bbox_center_px": bbox_center.tolist(),
        "bbox_err": bbox_err,
        "center_err": center_err,
        "depth_median_m": float(np.median(p[:, 2])),
        "depth_err_m": depth_err,
        "iou_point_raster": iou,
        "projected_in_frame": int(inb.sum()),
    }
    return score, details


def refine_icp(src_fit: np.ndarray, tgt_fit: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray, iterations: int = 18):
    for i in range(iterations):
        p = transform(src_fit, r, scale, t)
        tree = cKDTree(p)
        dist, idx = tree.query(tgt_fit, k=1, workers=-1)
        cutoff = min(float(np.percentile(dist, 82)), 0.10)
        keep = dist <= max(cutoff, 0.015)
        if keep.sum() < 200:
            keep = dist <= np.percentile(dist, 92)
        matched_src = src_fit[idx[keep]]
        matched_tgt = tgt_fit[keep]
        r_new, s_new, t_new = umeyama_similarity(matched_src, matched_tgt)
        # Damp scale updates to avoid a subset of the visible surface collapsing the full mesh.
        scale = 0.65 * scale + 0.35 * s_new
        r = r_new
        t = 0.65 * t + 0.35 * t_new
    return r, float(scale), t


def fit_part(name: str, cfg: dict, k: np.ndarray, depth_mm: np.ndarray, color: np.ndarray) -> dict:
    mask = cv2.imread(str(cfg["mask"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(cfg["mask"])
    mask = (mask > 127).astype(np.uint8) * 255
    target_points, target_stats = depth_mask_points(mask, depth_mm, k)
    mesh = load_mesh(cfg["mesh"])
    src_surface, _ = trimesh.sample.sample_surface(mesh, 42000)
    src_fit = src_surface
    if len(src_fit) > 18000:
        src_fit = src_fit[RNG.choice(len(src_fit), 18000, replace=False)]
    tgt_fit = target_points
    if len(tgt_fit) > 9000:
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), 9000, replace=False)]
    src_eval = src_surface
    if len(src_eval) > 30000:
        src_eval = src_eval[RNG.choice(len(src_eval), 30000, replace=False)]
    tgt_eval = target_points

    src_center, src_axes, src_vals = pca_axes(src_fit)
    tgt_center, tgt_axes, tgt_vals = pca_axes(tgt_fit)
    src_rms = math.sqrt(float(src_vals.sum()))
    tgt_rms = math.sqrt(float(tgt_vals.sum()))
    initial_scale = tgt_rms / max(src_rms, 1e-9)

    candidates = []
    for r0 in rotation_candidates(src_axes, tgt_axes):
        t0 = tgt_center - initial_scale * (src_center @ r0.T)
        r1, s1, t1 = refine_icp(src_fit, tgt_fit, r0, initial_scale, t0)
        score, details = score_pose(src_eval, tgt_eval, r1, s1, t1, k, mask, target_stats)
        candidates.append((score, r1, s1, t1, details))
    candidates.sort(key=lambda item: item[0])
    best_score, best_r, best_s, best_t, best_details = candidates[0]

    fitted = mesh.copy()
    fitted.vertices = transform(np.asarray(fitted.vertices, dtype=np.float64), best_r, best_s, best_t)
    glb_path = OUT_DIR / f"{name}_rgbd_similarity_fit.glb"
    ply_path = OUT_DIR / f"{name}_rgbd_similarity_fit.ply"
    fitted.export(glb_path)
    fitted.export(ply_path)

    uv, _ = project(transform(src_eval, best_r, best_s, best_t), k)
    overlay = color.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    if overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    draw = overlay.copy()
    h, w = mask.shape
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    pts = np.round(uv[inb]).astype(np.int32)
    step = max(1, len(pts) // 35000)
    for x, y in pts[::step]:
        cv2.circle(draw, (int(x), int(y)), 1, cfg["color"], -1)
    b = target_stats["bbox_xyxy"]
    cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), cfg["color"], 2)
    overlay_path = OUT_DIR / f"{name}_rgbd_similarity_projection_overlay.png"
    cv2.imwrite(str(overlay_path), draw)

    return {
        "name": name,
        "source_mesh": str(cfg["mesh"]),
        "mask": str(cfg["mask"]),
        "target": target_stats,
        "fit": {
            "scale_uniform": best_s,
            "rotation_matrix_source_to_camera": best_r.tolist(),
            "translation_camera_m": best_t.tolist(),
            "score": best_score,
            **best_details,
        },
        "top_candidates": [
            {
                "score": float(c[0]),
                "scale_uniform": float(c[2]),
                "translation_camera_m": c[3].tolist(),
                "fit_details": c[4],
            }
            for c in candidates[:5]
        ],
        "outputs": {
            "glb": str(glb_path),
            "ply": str(ply_path),
            "projection_overlay": str(overlay_path),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k = load_k()
    depth_mm = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(DEPTH_PATH)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)

    report = {
        "method": "Fit each raw SAM3DObject mesh to its qpos1 SAM mask + aligned depth using a 7DoF similarity transform only: 3D rotation, uniform scale, and 3D translation. No anisotropic scaling, no topology changes.",
        "capture": CAPTURE,
        "camera_k": k.tolist(),
        "depth": str(DEPTH_PATH),
        "parts": {},
    }
    combined = color.copy()
    if combined.shape[2] == 4:
        combined = cv2.cvtColor(combined, cv2.COLOR_BGRA2BGR)
    for name, cfg in PARTS.items():
        result = fit_part(name, cfg, k, depth_mm, color)
        report["parts"][name] = result
        overlay = cv2.imread(result["outputs"]["projection_overlay"], cv2.IMREAD_COLOR)
        if overlay is not None:
            # Re-draw projected points on a shared overlay by extracting the non-background color is unnecessary;
            # fit_part writes individual overlays for inspection.
            pass
    # Build a combined overlay from fitted GLBs using fresh samples.
    for name, cfg in PARTS.items():
        part = report["parts"][name]
        mesh = trimesh.load(part["outputs"]["glb"], force="mesh", process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, 30000)
        uv, _ = project(pts, k)
        h, w = depth_mm.shape
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        for x, y in pix:
            cv2.circle(combined, (int(x), int(y)), 1, cfg["color"], -1)
        b = report["parts"][name]["target"]["bbox_xyxy"]
        cv2.rectangle(combined, (b[0], b[1]), (b[2], b[3]), cfg["color"], 2)
    combined_path = OUT_DIR / "combined_rgbd_similarity_projection_overlay.png"
    cv2.imwrite(str(combined_path), combined)
    report["outputs"] = {"combined_projection_overlay": str(combined_path)}
    report_path = OUT_DIR / "rgbd_similarity_fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_overlay": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
