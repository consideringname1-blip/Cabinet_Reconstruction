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
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/plane_component_frame_fit"
PLANE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/mesh_plane_components/mesh_plane_components_report.json"
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


def mask_points(mask_path: Path, depth_mm: np.ndarray, k: np.ndarray, max_points: int = 40000):
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
        pts = pts[RNG.choice(len(pts), max_points, replace=False)]
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return pts, mask, bbox


def fit_plane(points: np.ndarray) -> dict:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    # Make the normal point roughly toward camera.
    if n @ center > 0:
        n = -n
    d = -float(n @ center)
    return {"center": center, "normal": n, "d": d}


def ransac_plane(points: np.ndarray, trials: int = 2200, threshold: float = 0.014) -> dict:
    best = None
    for _ in range(trials):
        idx = RNG.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(n @ p0)
        dist = np.abs(points @ n + d)
        inliers = dist < threshold
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), inliers)
    plane = fit_plane(points[best[1]])
    plane["inliers"] = best[0]
    plane["inlier_ratio"] = float(best[0] / len(points))
    return plane


def cabinet_frame(drawer_points: np.ndarray) -> dict:
    plane = ransac_plane(drawer_points)
    n = plane["normal"]
    image_y = np.array([0.0, 1.0, 0.0])
    v = image_y - n * float(image_y @ n)
    v = v / np.linalg.norm(v)
    h = np.cross(v, n)
    h = h / np.linalg.norm(h)
    axes = np.column_stack([h, v, n])
    if np.linalg.det(axes) < 0:
        h = -h
        axes = np.column_stack([h, v, n])
    return {
        "horizontal": h,
        "vertical": v,
        "front_normal": n,
        "axes": axes,
        "drawer_plane": plane,
    }


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = points[points[:, 2] > 1e-5]
    return np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))


def zbuffer(points: np.ndarray, k: np.ndarray, width: int, height: int) -> np.ndarray:
    good = points[:, 2] > 1e-5
    p = points[good]
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if ok.any():
        np.minimum.at(depth, (py[ok], px[ok]), p[:, 2][ok])
    return depth


def snap_xy(points: np.ndarray, k: np.ndarray, bbox: list[int], width: int, height: int):
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
    duv = target - center
    delta = np.array([duv[0] * z / k[0, 0], duv[1] * z / k[1, 1], 0.0])
    shifted = points + delta
    uv2 = project(shifted, k)
    inb2 = (uv2[:, 0] >= 0) & (uv2[:, 0] < width) & (uv2[:, 1] >= 0) & (uv2[:, 1] < height)
    uv2 = uv2[inb2]
    lo2 = np.percentile(uv2, 1, axis=0)
    hi2 = np.percentile(uv2, 99, axis=0)
    return shifted, delta, {
        "bbox_before": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "bbox_after": [float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1])],
        "center_after": ((lo2 + hi2) * 0.5).tolist(),
        "target_center": target.tolist(),
        "xy_delta_m": delta.tolist(),
    }


def score(points: np.ndarray, mask: np.ndarray, depth_obs: np.ndarray, k: np.ndarray, bbox: list[int]):
    h, w = mask.shape
    dr = zbuffer(points, k, w, h)
    pred = np.isfinite(dr)
    target = mask > 0
    valid = depth_obs > 0.05
    inter = pred & target & valid
    if inter.sum() < 100:
        return 1e9, {"reason": "empty"}
    diff = np.abs(dr[inter] - depth_obs[inter])
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
    value = med / 0.025 + 0.3 * p80 / 0.055 + 1.4 * (1 - iou) + 0.6 * (1 - cov) + 0.9 * leak + 0.7 * size_err + 0.9 * center_err
    return value, {
        "score": float(value),
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


def source_frame_from_planes(front_n: np.ndarray, aux_n: np.ndarray, aux_target: str):
    n = front_n / np.linalg.norm(front_n)
    aux = aux_n - n * float(aux_n @ n)
    if np.linalg.norm(aux) < 1e-5:
        return None
    aux = aux / np.linalg.norm(aux)
    if aux_target == "horizontal":
        h = aux
        v = np.cross(n, h)
    else:
        v = aux
        h = np.cross(v, n)
    h = h / np.linalg.norm(h)
    v = v / np.linalg.norm(v)
    frame = np.column_stack([h, v, n])
    if np.linalg.det(frame) < 0:
        h = -h
        frame = np.column_stack([h, v, n])
    return frame


def candidate_rotations_from_planes(planes: list[dict], target_axes: np.ndarray):
    top = planes[:5]
    for i, front in enumerate(top):
        fn = np.asarray(front["normal"], dtype=np.float64)
        for j, aux in enumerate(top):
            if i == j:
                continue
            an = np.asarray(aux["normal"], dtype=np.float64)
            if abs(float(fn @ an)) > 0.38:
                continue
            for sf, sa in itertools.product([-1.0, 1.0], repeat=2):
                for aux_target in ("horizontal", "vertical"):
                    src_frame = source_frame_from_planes(sf * fn, sa * an, aux_target)
                    if src_frame is None:
                        continue
                    r = target_axes @ src_frame.T
                    if np.linalg.det(r) < 0:
                        continue
                    yield {
                        "front_plane": front["index"],
                        "aux_plane": aux["index"],
                        "front_sign": sf,
                        "aux_sign": sa,
                        "aux_target": aux_target,
                        "rotation": r,
                    }


def fit_part(name, mesh, planes, target_points, mask, bbox, frame, k, depth_obs, scale_hint):
    sample, _ = trimesh.sample.sample_surface(mesh, 18000 if name == "base" else 14000)
    source_center = sample.mean(axis=0)
    target_center = np.median(target_points, axis=0)
    h, w = mask.shape
    candidates = []
    rotations = list(candidate_rotations_from_planes(planes, frame["axes"]))
    for cand in rotations:
        r = cand["rotation"]
        for scale_mult in (0.95, 1.0, 1.05):
            s = float(scale_hint * scale_mult)
            base = target_center + s * ((sample - source_center) @ r.T)
            for dz in (-0.025, 0.0, 0.025):
                p = base.copy()
                p[:, 2] += dz
                p, delta, snap = snap_xy(p, k, bbox, w, h)
                sc, metrics = score(p, mask, depth_obs, k, bbox)
                metrics["snap"] = snap
                t_center = target_center + np.array([delta[0], delta[1], dz])
                candidates.append((sc, cand, r, s, t_center, metrics))
    candidates.sort(key=lambda x: x[0])
    sc, cand, r, s, t_center, metrics = candidates[0]
    verts = t_center + s * ((np.asarray(mesh.vertices) - source_center) @ r.T)
    fitted = mesh.copy()
    fitted.vertices = verts
    return {
        "mesh": fitted,
        "source_center": source_center.tolist(),
        "target_center": target_center.tolist(),
        "selected": {
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cand.items() if k != "rotation"},
            "score": float(sc),
            "scale_uniform": float(s),
            "target_center_after_snap": t_center.tolist(),
            "rotation_matrix_source_to_camera": r.tolist(),
            "metrics": metrics,
        },
        "top_candidates": [
            {
                "rank": idx + 1,
                "score": float(c[0]),
                "front_plane": c[1]["front_plane"],
                "aux_plane": c[1]["aux_plane"],
                "front_sign": c[1]["front_sign"],
                "aux_sign": c[1]["aux_sign"],
                "aux_target": c[1]["aux_target"],
                "scale_uniform": float(c[3]),
                "target_center_after_snap": c[4].tolist(),
                "metrics": c[5],
            }
            for idx, c in enumerate(candidates[:12])
        ],
    }


def draw_overlay(color, samples, k, bboxes, output):
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for name, pts in samples.items():
        uv = project(pts, k)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 50000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, PARTS[name]["color"], -1)
        b = bboxes[name]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), PARTS[name]["color"], 2)
    cv2.imwrite(str(output), img)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    k = load_k()
    depth_mm = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    depth_obs = depth_mm.astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    plane_report = json.loads(PLANE_REPORT.read_text(encoding="utf-8"))
    rgbd_report = json.loads(RGBD_REPORT.read_text(encoding="utf-8"))

    points, masks, bboxes = {}, {}, {}
    for name, cfg in PARTS.items():
        points[name], masks[name], bboxes[name] = mask_points(cfg["mask"], depth_mm, k)
    frame = cabinet_frame(points["drawer"])
    report = {
        "method": "Plane-component frame fit. Mesh coordinate axes and global PCA are ignored. Candidate rotations are built by matching detected raw mesh planar component normals to a shared RGB-D cabinet frame derived from the drawer front plane.",
        "frame": {
            "horizontal_camera": frame["horizontal"].tolist(),
            "vertical_camera": frame["vertical"].tolist(),
            "front_normal_camera": frame["front_normal"].tolist(),
            "drawer_plane_center": frame["drawer_plane"]["center"].tolist(),
            "drawer_plane_inliers": frame["drawer_plane"]["inliers"],
            "drawer_plane_inlier_ratio": frame["drawer_plane"]["inlier_ratio"],
        },
        "parts": {},
        "outputs": {},
    }
    scene = trimesh.Scene()
    overlay_samples = {}
    for name, cfg in PARTS.items():
        mesh = load_mesh(cfg["mesh"])
        planes = plane_report["parts"][name]["planes"]
        scale_hint = float(rgbd_report["parts"][name]["scale_uniform"])
        result = fit_part(name, mesh, planes, points[name], masks[name], bboxes[name], frame, k, depth_obs, scale_hint)
        glb = OUT / f"{name}_plane_component_frame.glb"
        ply = OUT / f"{name}_plane_component_frame.ply"
        result["mesh"].export(glb)
        result["mesh"].export(ply)
        scene.add_geometry(result["mesh"], geom_name=name, node_name=name)
        sample, _ = trimesh.sample.sample_surface(result["mesh"], 45000 if name == "base" else 35000)
        overlay_samples[name] = sample
        report["parts"][name] = {
            "source_mesh": str(cfg["mesh"]),
            "selected": result["selected"],
            "source_center": result["source_center"],
            "target_center": result["target_center"],
            "top_candidates": result["top_candidates"],
            "outputs": {"glb": str(glb), "ply": str(ply)},
        }
    combined = OUT / "cabinet_drawer_plane_component_frame_open.glb"
    scene.export(combined)
    overlay = OUT / "plane_component_frame_projection_overlay.png"
    draw_overlay(color, overlay_samples, k, bboxes, overlay)
    report["outputs"] = {"combined_open_glb": str(combined), "projection_overlay": str(overlay)}
    report_path = OUT / "plane_component_frame_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_open_glb": str(combined), "overlay": str(overlay)}, indent=2))


if __name__ == "__main__":
    main()
