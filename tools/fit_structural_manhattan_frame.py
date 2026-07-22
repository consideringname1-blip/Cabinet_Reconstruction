import itertools
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
CAPTURE = "20260622_081031_636398Z"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/structural_manhattan_fit"
RGBD_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json"
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
    return trimesh.util.concatenate([g.copy() for g in scene.geometry.values()])


def mask_points(mask_path: Path, depth_mm: np.ndarray, k: np.ndarray, max_points: int = 30000) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    ys, xs = np.where((mask > 0) & (depth_mm > 0))
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    good = (z > 0.2) & (z < 4.0)
    xs = xs[good].astype(np.float64)
    ys = ys[good].astype(np.float64)
    z = z[good]
    pts = np.column_stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z))
    if len(pts) > max_points:
        idx = RNG.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return pts, mask, bbox


def fit_plane_ransac(points: np.ndarray, trials: int = 2500, threshold: float = 0.014) -> dict:
    best_inliers = None
    best_n = None
    best_d = None
    n_points = len(points)
    for _ in range(trials):
        idx = RNG.choice(n_points, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(n @ p0)
        dist = np.abs(points @ n + d)
        inliers = dist < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_n = n
            best_d = d
    inlier_pts = points[best_inliers]
    center = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - center, full_matrices=False)
    n = vh[-1]
    if n @ center > 0:
        n = -n
    d = -float(n @ center)
    return {
        "normal": n,
        "d": d,
        "center": center,
        "inliers": int(best_inliers.sum()),
        "inlier_ratio": float(best_inliers.mean()),
    }


def cabinet_frame_from_drawer_plane(drawer_points: np.ndarray) -> dict:
    plane = fit_plane_ransac(drawer_points)
    n = plane["normal"]
    n = n / np.linalg.norm(n)
    # Camera image-y/down direction, projected onto the front plane.
    image_y = np.array([0.0, 1.0, 0.0])
    v = image_y - n * float(image_y @ n)
    if np.linalg.norm(v) < 1e-6:
        v = np.array([0.0, 0.0, 1.0]) - n * float(np.array([0.0, 0.0, 1.0]) @ n)
    v = v / np.linalg.norm(v)
    h = np.cross(v, n)
    h = h / np.linalg.norm(h)
    axes = np.column_stack([h, v, n])
    if np.linalg.det(axes) < 0:
        h = -h
        axes = np.column_stack([h, v, n])
    return {
        "front_normal_camera": n,
        "vertical_camera": v,
        "horizontal_camera": h,
        "axes_camera_columns_h_v_n": axes,
        "drawer_plane": {k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in plane.items()},
    }


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    x = points - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def signed_permutations() -> list[tuple[str, np.ndarray]]:
    out = []
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            m = p @ np.diag(signs)
            if np.linalg.det(m) > 0:
                out.append((f"perm{perm}_sign{tuple(int(s) for s in signs)}", m))
    return out


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = points[points[:, 2] > 1e-5]
    return np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))


def snap_xy(points: np.ndarray, k: np.ndarray, bbox: list[int], width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict]:
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[inb]
    if len(uv) < 100:
        return points, np.zeros(3), {}
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    center = (lo + hi) * 0.5
    target = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])
    z = float(np.median(points[:, 2]))
    delta_uv = target - center
    delta = np.array([delta_uv[0] * z / k[0, 0], delta_uv[1] * z / k[1, 1], 0.0])
    shifted = points + delta
    uv2 = project(shifted, k)
    inb2 = (uv2[:, 0] >= 0) & (uv2[:, 0] < width) & (uv2[:, 1] >= 0) & (uv2[:, 1] < height)
    uv2 = uv2[inb2]
    lo2 = np.percentile(uv2, 1, axis=0)
    hi2 = np.percentile(uv2, 99, axis=0)
    return shifted, delta, {
        "bbox_before_xyxy": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "bbox_after_xyxy": [float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1])],
        "center_before_px": center.tolist(),
        "center_after_px": ((lo2 + hi2) * 0.5).tolist(),
        "target_center_px": target.tolist(),
        "xy_delta_m": delta.tolist(),
    }


def zbuffer(points: np.ndarray, k: np.ndarray, width: int, height: int) -> np.ndarray:
    uv = project(points, k)
    p = points[points[:, 2] > 1e-5]
    z = p[:, 2]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if ok.any():
        np.minimum.at(depth, (py[ok], px[ok]), z[ok])
    return depth


def score(points: np.ndarray, mask: np.ndarray, depth_obs: np.ndarray, k: np.ndarray, bbox: list[int]) -> tuple[float, dict]:
    h, w = mask.shape
    depth_render = zbuffer(points, k, w, h)
    pred = np.isfinite(depth_render)
    target = mask > 0
    valid = depth_obs > 0.05
    inter = pred & target & valid
    if inter.sum() < 100:
        return 1e9, {"reason": "empty"}
    diff = np.abs(depth_render[inter] - depth_obs[inter])
    iou = float((pred & target).sum() / max(1, (pred | target).sum()))
    cov = float((pred & target).sum() / max(1, target.sum()))
    leak = float((pred & ~target).sum() / max(1, pred.sum()))
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inb]
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    target_bbox = np.asarray(bbox, dtype=np.float64)
    target_size = np.maximum(target_bbox[2:] - target_bbox[:2], 1)
    size_err = float(np.linalg.norm(((hi - lo) - target_size) / target_size))
    center_err = float(np.linalg.norm((((lo + hi) * 0.5) - ((target_bbox[:2] + target_bbox[2:]) * 0.5)) / target_size))
    med = float(np.median(diff))
    p80 = float(np.percentile(diff, 80))
    score_value = med / 0.025 + 0.35 * p80 / 0.055 + 1.1 * (1 - iou) + 0.5 * (1 - cov) + 0.8 * leak + 0.7 * size_err + 1.0 * center_err
    return score_value, {
        "score": float(score_value),
        "depth_abs_median_m": med,
        "depth_abs_p80_m": p80,
        "mask_iou": iou,
        "target_coverage": cov,
        "leakage": leak,
        "bbox_xyxy_p01_p99": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "bbox_size_err": size_err,
        "bbox_center_err": center_err,
        "intersection_pixels": int(inter.sum()),
    }


def draw_overlay(color: np.ndarray, part_points: dict[str, np.ndarray], k: np.ndarray, masks: dict[str, np.ndarray], bboxes: dict[str, list[int]], output: Path) -> None:
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for name, points in part_points.items():
        uv = project(points, k)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 50000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, PARTS[name]["color"], -1)
        b = bboxes[name]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), PARTS[name]["color"], 2)
    cv2.imwrite(str(output), img)


def fit_part_to_frame(name: str, mesh: trimesh.Trimesh, target_points: np.ndarray, target_mask: np.ndarray, bbox: list[int], target_axes: np.ndarray, k: np.ndarray, depth_obs: np.ndarray, scale_hint: float) -> dict:
    sample, _ = trimesh.sample.sample_surface(mesh, 45000 if name == "base" else 30000)
    source_center, source_axes = pca_axes(sample)
    target_center = np.median(target_points, axis=0)
    target_center[2] = float(np.median(target_points[:, 2]))
    h, w = target_mask.shape
    candidates = []
    for label, signed_perm in signed_permutations():
        r = target_axes @ signed_perm @ source_axes.T
        for scale_mult in (0.90, 0.97, 1.03, 1.10):
            s = scale_hint * scale_mult
            for dz in (-0.03, 0.0, 0.03):
                p = target_center + s * ((sample - source_center) @ r.T)
                p[:, 2] += dz
                p, delta, snap = snap_xy(p, k, bbox, w, h)
                sc, metrics = score(p, target_mask, depth_obs, k, bbox)
                metrics["snap"] = snap
                candidates.append((sc, label, r, s, target_center + np.array([delta[0], delta[1], dz]), metrics))
    candidates.sort(key=lambda item: item[0])
    sc, label, r, s, t_center, metrics = candidates[0]
    full_vertices = t_center + s * ((np.asarray(mesh.vertices) - source_center) @ r.T)
    fitted = mesh.copy()
    fitted.vertices = full_vertices
    return {
        "mesh": fitted,
        "selected_label": label,
        "score": float(sc),
        "scale_uniform": float(s),
        "rotation_matrix_source_to_structural_camera": r.tolist(),
        "source_center": source_center.tolist(),
        "target_center_after_snap": t_center.tolist(),
        "metrics": metrics,
        "top_candidates": [
            {
                "rank": i + 1,
                "label": c[1],
                "score": float(c[0]),
                "scale_uniform": float(c[3]),
                "target_center_after_snap": c[4].tolist(),
                "metrics": c[5],
            }
            for i, c in enumerate(candidates[:8])
        ],
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    k = load_k()
    depth_mm = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    depth_obs = depth_mm.astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    rgbd = json.loads(RGBD_REPORT.read_text(encoding="utf-8"))

    points = {}
    masks = {}
    bboxes = {}
    for name, cfg in PARTS.items():
        points[name], masks[name], bboxes[name] = mask_points(cfg["mask"], depth_mm, k)

    frame = cabinet_frame_from_drawer_plane(points["drawer"])
    target_axes = frame["axes_camera_columns_h_v_n"]

    report = {
        "method": "Structural Manhattan-frame fit. A shared cabinet coordinate frame is estimated from the drawer front depth plane, then both base and drawer meshes are constrained to map their PCA axes onto this same orthonormal frame. Only uniform scale and translation are searched afterward.",
        "frame": {
            "horizontal_camera": frame["horizontal_camera"].tolist(),
            "vertical_camera": frame["vertical_camera"].tolist(),
            "front_normal_camera": frame["front_normal_camera"].tolist(),
            "drawer_plane": frame["drawer_plane"],
        },
        "parts": {},
        "orthogonality": {},
    }

    scene = trimesh.Scene()
    overlay_samples = {}
    for name, cfg in PARTS.items():
        mesh = load_mesh(cfg["mesh"])
        scale_hint = float(rgbd["parts"][name]["scale_uniform"])
        fitted = fit_part_to_frame(name, mesh, points[name], masks[name], bboxes[name], target_axes, k, depth_obs, scale_hint)
        glb = OUT / f"{name}_structural_manhattan.glb"
        ply = OUT / f"{name}_structural_manhattan.ply"
        fitted["mesh"].export(glb)
        fitted["mesh"].export(ply)
        scene.add_geometry(fitted["mesh"], geom_name=name, node_name=name)
        sample, _ = trimesh.sample.sample_surface(fitted["mesh"], 70000 if name == "base" else 50000)
        overlay_samples[name] = sample
        report["parts"][name] = {k2: v for k2, v in fitted.items() if k2 != "mesh"}
        report["parts"][name]["outputs"] = {"glb": str(glb), "ply": str(ply)}

    combined = OUT / "cabinet_drawer_structural_manhattan_open.glb"
    scene.export(combined)
    overlay = OUT / "structural_manhattan_projection_overlay.png"
    draw_overlay(color, overlay_samples, k, masks, bboxes, overlay)
    report["outputs"] = {"combined_open_glb": str(combined), "projection_overlay": str(overlay)}
    # Both parts are constrained to this same frame by construction; store dot products for clarity.
    axes = target_axes
    report["orthogonality"] = {
        "horizontal_dot_vertical": float(axes[:, 0] @ axes[:, 1]),
        "horizontal_dot_front_normal": float(axes[:, 0] @ axes[:, 2]),
        "vertical_dot_front_normal": float(axes[:, 1] @ axes[:, 2]),
        "determinant": float(np.linalg.det(axes)),
    }
    report_path = OUT / "structural_manhattan_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_open_glb": str(combined), "overlay": str(overlay)}, indent=2))


if __name__ == "__main__":
    main()
