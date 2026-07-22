import csv
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate"
POSE_METADATA = ROOT / "data/upload/larm/20260622_joint_0_white_outline_sfm/20260622_joint_0_white_outline_sfm.json"
QPOS1_CAPTURE = "20260622_081031_636398Z"
QPOS1_BASE_MASK = ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png"
QPOS1_DRAWER_MASK = ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png"
RNG = np.random.default_rng(20260702)
BLENDER2OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def capture_paths(capture):
    return {
        "color": ROOT / f"data/upload/larm_captures/{capture}/color.png",
        "depth": ROOT / f"data/output/hololens2/{capture}_align_depth.png",
        "meta": ROOT / f"data/upload/{capture}_meta.json",
    }


def load_k(capture):
    return np.asarray(read_json(capture_paths(capture)["meta"])["PVCamera"]["k"], dtype=np.float64)


def load_depth_m(capture):
    depth = cv2.imread(str(capture_paths(capture)["depth"]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(capture_paths(capture)["depth"])
    return depth.astype(np.float32) / 1000.0


def load_color(capture):
    color = cv2.imread(str(capture_paths(capture)["color"]), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(capture_paths(capture)["color"])
    return color


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def unproject_mask(capture, mask_path, stride=1, erode=True, max_points=None):
    k = load_k(capture)
    depth = load_depth_m(capture)
    mask = load_mask(mask_path)
    valid = mask & (depth > 0.2) & (depth < 4.0)
    if erode:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    ys, xs = np.nonzero(valid)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]
    if max_points is not None and len(xs) > max_points:
        idx = RNG.choice(len(xs), size=max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.column_stack([x, y, z]), np.column_stack([xs, ys])


def sfm_camera_to_world(capture):
    meta = read_json(POSE_METADATA)
    for idx, rec in enumerate(meta["grouped_captures"]):
        if capture in str(rec.get("capture", "")):
            token = "0.00" if idx < 3 else "1.00"
            frame_idx = idx if idx < 3 else idx - 3
            transform = np.asarray(meta["inputs"][token][f"input_frame_{frame_idx}"]["transform_matrix"], dtype=np.float64)
            return transform @ BLENDER2OPENCV
    raise KeyError(capture)


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return (np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=np.float64).T)[:, :3]


def voxel_downsample(points, voxel=0.008, max_points=None):
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


def robust_percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {str(q): float(np.percentile(values, q)) for q in [10, 25, 50, 75, 90, 95]}


def load_closed_views():
    meta = read_json(POSE_METADATA)
    views = []
    for idx, rec in enumerate(meta["grouped_captures"]):
        if float(rec["qpos"]) != 0.0:
            continue
        capture = Path(rec["capture"]).name
        mask_path = Path(rec["mask_path"])
        pts_cam, pix = unproject_mask(capture, mask_path, stride=1, max_points=45000)
        t_view_world = sfm_camera_to_world(capture)
        t_world_qpos1 = np.linalg.inv(sfm_camera_to_world(QPOS1_CAPTURE))
        pts_qpos1 = transform_points(pts_cam, t_world_qpos1 @ t_view_world)
        views.append(
            {
                "capture": capture,
                "mask_path": str(mask_path),
                "points_cam": pts_cam,
                "points_qpos1": pts_qpos1,
                "pixels": pix,
                "k": load_k(capture),
                "depth": load_depth_m(capture),
                "mask": load_mask(mask_path),
                "color": load_color(capture),
                "qpos1_to_view": np.linalg.inv(t_view_world) @ sfm_camera_to_world(QPOS1_CAPTURE),
            }
        )
    return views



def orient_axis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    # Deterministic sign for reporting only; q search still covers both signs.
    if axis[1] > 0:
        axis = -axis
    return axis


def pca_plane_axis(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    return orient_axis(vh[-1]), center


def estimate_drawer_axis_candidates(drawer_points, iterations=1600, threshold=0.014):
    pts = voxel_downsample(drawer_points, voxel=0.004, max_points=10000)
    if len(pts) < 100:
        raise RuntimeError("Not enough drawer points for plane estimation")

    candidates = []
    remaining = pts.copy()
    for plane_idx in range(5):
        best = None
        if len(remaining) < 160:
            break
        for _ in range(iterations):
            idx = RNG.choice(len(remaining), size=3, replace=False)
            a, b, c = remaining[idx]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-8:
                continue
            n = n / norm
            d = -float(np.dot(n, a))
            dist = np.abs(remaining @ n + d)
            inliers = dist < threshold
            count = int(np.sum(inliers))
            if best is None or count > best["count"]:
                best = {"normal": n, "d": d, "count": count, "inliers": inliers}
        if best is None or best["count"] < 100:
            break
        inlier_pts = remaining[best["inliers"]]
        axis, center = pca_plane_axis(inlier_pts)
        duplicate = False
        for cand in candidates:
            old = np.asarray(cand["axis_camera"], dtype=np.float64)
            if abs(float(np.dot(old, axis))) > 0.965:
                duplicate = True
                break
        if not duplicate:
            candidates.append({
                "label": f"ransac_plane_{plane_idx + 1}",
                "axis_camera": axis.tolist(),
                "inliers": int(len(inlier_pts)),
                "total_points": int(len(pts)),
                "inlier_ratio": float(len(inlier_pts) / max(1, len(pts))),
                "threshold_m": float(threshold),
                "center_camera": center.tolist(),
            })
        remaining = remaining[~best["inliers"]]

    return candidates

def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    valid = points[:, 2] > 1e-5
    p = points[valid]
    uv = np.column_stack([k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]])
    return uv, p


def projection_penalty(points_qpos1, views):
    outside_ratios = []
    front_ratios = []
    near_ratios = []
    depth_abs = []
    sample = points_qpos1
    if len(sample) > 9000:
        sample = sample[RNG.choice(len(sample), size=9000, replace=False)]
    for view in views:
        pts = transform_points(sample, view["qpos1_to_view"])
        uv, valid_pts = project(pts, view["k"])
        h, w = view["mask"].shape
        in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if len(in_img) == 0:
            outside_ratios.append(1.0)
            front_ratios.append(1.0)
            near_ratios.append(0.0)
            depth_abs.append(0.2)
            continue
        uv_i = np.round(uv[in_img]).astype(np.int32)
        pts_i = valid_pts[in_img]
        xs = np.clip(uv_i[:, 0], 0, w - 1)
        ys = np.clip(uv_i[:, 1], 0, h - 1)
        in_mask = view["mask"][ys, xs]
        outside_ratios.append(float(1.0 - np.mean(in_mask)) if len(in_mask) else 1.0)
        obs_z = view["depth"][ys, xs].astype(np.float64)
        depth_valid = in_mask & (obs_z > 0.2) & (obs_z < 4.0)
        if np.any(depth_valid):
            diff = pts_i[depth_valid, 2] - obs_z[depth_valid]
            # In front of the observed closed surface is especially bad for a synthesized surface.
            front_ratios.append(float(np.mean(diff < -0.035)))
            near_ratios.append(float(np.mean(np.abs(diff) < 0.055)))
            depth_abs.append(float(np.median(np.minimum(np.abs(diff), 0.16))))
        else:
            front_ratios.append(1.0)
            near_ratios.append(0.0)
            depth_abs.append(0.16)
    return {
        "outside_mask_ratio": float(np.mean(outside_ratios)),
        "front_violation_ratio": float(np.mean(front_ratios)),
        "near_depth_ratio": float(np.mean(near_ratios)),
        "depth_abs_median_m": float(np.mean(depth_abs)),
    }


def score_translation(t, base_points, drawer_points, closed_points, residual_points, closed_tree, views):
    shifted_drawer = drawer_points + np.asarray(t, dtype=np.float64)
    drawer_tree = cKDTree(shifted_drawer)
    residual_dist, _ = drawer_tree.query(residual_points, k=1, workers=-1)
    drawer_to_closed_dist, _ = closed_tree.query(shifted_drawer, k=1, workers=-1)

    # The base distance is precomputed for all closed points by the caller and stored globally
    # through closed_points[:, 3] would be ugly; pass explicit residuals instead and keep the
    # full closed->drawer term light.
    closed_sample = closed_points
    if len(closed_sample) > 16000:
        closed_sample = closed_sample[RNG.choice(len(closed_sample), size=16000, replace=False)]
    closed_to_drawer, _ = drawer_tree.query(closed_sample, k=1, workers=-1)

    proj = projection_penalty(shifted_drawer, views)
    residual_med = float(np.percentile(residual_dist, 50))
    residual_p80 = float(np.percentile(residual_dist, 80))
    drawer_med = float(np.percentile(drawer_to_closed_dist, 50))
    drawer_p80 = float(np.percentile(drawer_to_closed_dist, 80))
    closed_drawer_p25 = float(np.percentile(closed_to_drawer, 25))

    score = (
        residual_med / 0.025
        + 0.55 * residual_p80 / 0.060
        + 0.35 * drawer_med / 0.035
        + 0.25 * drawer_p80 / 0.080
        + 0.25 * closed_drawer_p25 / 0.050
        + 1.8 * proj["outside_mask_ratio"]
        + 2.2 * proj["front_violation_ratio"]
        + 0.8 * proj["depth_abs_median_m"] / 0.055
        - 0.55 * proj["near_depth_ratio"]
    )
    return float(score), {
        "translation_open_to_closed_camera": [float(v) for v in np.asarray(t).tolist()],
        "score": float(score),
        "residual_to_drawer_percentiles_m": robust_percentiles(residual_dist),
        "drawer_to_closed_percentiles_m": robust_percentiles(drawer_to_closed_dist),
        "closed_to_drawer_p25_m": closed_drawer_p25,
        **proj,
    }


def translation_icp_to_residual(drawer_points, residual_points, initial_t, iterations=32):
    src = voxel_downsample(drawer_points, voxel=0.006, max_points=12000)
    dst = voxel_downsample(residual_points, voxel=0.006, max_points=18000)
    tree = cKDTree(dst)
    t = np.asarray(initial_t, dtype=np.float64)
    for _ in range(iterations):
        dist, idx = tree.query(src + t, k=1, workers=-1)
        keep = dist <= np.percentile(dist, 65)
        if int(np.sum(keep)) < 64:
            keep = dist <= np.percentile(dist, 85)
        update = np.median(dst[idx[keep]] - src[keep], axis=0)
        if np.linalg.norm(update - t) < 1e-5:
            t = update
            break
        t = 0.65 * t + 0.35 * update
    return t


def make_residual_points(closed_points, base_points):
    base_tree = cKDTree(base_points)
    dist, _ = base_tree.query(closed_points, k=1, workers=-1)
    # Data-driven latent residual: points not explained by static base. This is not an input mask.
    thresh = max(0.042, float(np.percentile(dist, 78)))
    residual = closed_points[dist > thresh]
    if len(residual) > 30000:
        residual = residual[RNG.choice(len(residual), size=30000, replace=False)]
    return residual, {"threshold_m": thresh, "base_distance_percentiles_m": robust_percentiles(dist), "points": int(len(residual))}


def cluster_initial_translations(drawer_points, residual_points):
    labels = DBSCAN(eps=0.045, min_samples=28).fit_predict(residual_points)
    initials = []
    drawer_med = np.median(drawer_points, axis=0)
    if np.any(labels >= 0):
        ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        order = ids[np.argsort(counts)[::-1]]
        for cluster_id in order[:8]:
            cluster = residual_points[labels == cluster_id]
            initials.append(np.median(cluster, axis=0) - drawer_med)
    initials.append(np.median(residual_points, axis=0) - drawer_med)
    return initials


def draw_candidate_overlay(path, view, base_points, drawer_points, t):
    img = view["color"].copy()
    contours, _ = cv2.findContours((view["mask"].astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
    for name, pts_q1, color in [
        ("base", base_points, (60, 220, 60)),
        ("drawer_closed", drawer_points + t, (40, 80, 240)),
    ]:
        sample = pts_q1
        if len(sample) > 45000:
            sample = sample[RNG.choice(len(sample), size=45000, replace=False)]
        pts_view = transform_points(sample, view["qpos1_to_view"])
        uv, valid_pts = project(pts_view, view["k"])
        h, w = img.shape[:2]
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        for x, y in pix:
            cv2.circle(img, (int(x), int(y)), 1, color, -1)
        if name == "drawer_closed" and len(pix):
            c = pix.mean(axis=0).astype(int)
            cv2.putText(img, name, (int(c[0]) + 8, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def draw_axis_overlay(path, axis, t, drawer_points):
    img = load_color(QPOS1_CAPTURE)
    k = load_k(QPOS1_CAPTURE)
    center = np.median(drawer_points, axis=0)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-9)

    def to_uv(p):
        return np.array([k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]], dtype=np.float64)

    def arrow(a, b, color, label):
        ua = tuple(np.round(to_uv(a)).astype(int))
        ub = tuple(np.round(to_uv(b)).astype(int))
        cv2.arrowedLine(img, ua, ub, color, 3, tipLength=0.12)
        cv2.putText(img, label, (ub[0] + 6, ub[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    arrow(center, center + axis * 0.24, (255, 0, 255), "axis")
    arrow(center, center + np.asarray(t) * 0.55, (0, 255, 255), "open->closed")
    cv2.imwrite(str(path), img)


def export_surface_scene(path, base_points, drawer_points, t):
    scene = trimesh.Scene()
    base = trimesh.points.PointCloud(base_points, colors=np.tile(np.array([70, 220, 70, 255], dtype=np.uint8), (len(base_points), 1)))
    open_drawer = trimesh.points.PointCloud(drawer_points, colors=np.tile(np.array([240, 80, 80, 255], dtype=np.uint8), (len(drawer_points), 1)))
    closed_drawer = trimesh.points.PointCloud(drawer_points + t, colors=np.tile(np.array([255, 190, 30, 255], dtype=np.uint8), (len(drawer_points), 1)))
    scene.add_geometry(base, geom_name="qpos1_base_surface")
    scene.add_geometry(open_drawer, geom_name="qpos1_drawer_open_surface")
    scene.add_geometry(closed_drawer, geom_name="synthesized_drawer_closed_surface")
    scene.export(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_points, _ = unproject_mask(QPOS1_CAPTURE, QPOS1_BASE_MASK, stride=1, max_points=70000)
    drawer_points, _ = unproject_mask(QPOS1_CAPTURE, QPOS1_DRAWER_MASK, stride=1, max_points=50000)
    base_points = voxel_downsample(base_points, voxel=0.006, max_points=28000)
    drawer_points = voxel_downsample(drawer_points, voxel=0.006, max_points=18000)

    closed_views = load_closed_views()
    closed_points = np.concatenate([view["points_qpos1"] for view in closed_views], axis=0)
    closed_points = voxel_downsample(closed_points, voxel=0.007, max_points=80000)
    closed_tree = cKDTree(closed_points)
    residual_points, residual_diag = make_residual_points(closed_points, base_points)

    axis_candidates = estimate_drawer_axis_candidates(drawer_points)

    candidates = []

    # Candidate family 1: axes inferred from qpos1 drawer RANSAC plane normals; displacement is synthesized.
    q_values = np.linspace(-0.70, 0.70, 141)
    for axis_info in axis_candidates:
        axis_vec = np.asarray(axis_info["axis_camera"], dtype=np.float64)
        label = axis_info["label"]
        coarse = []
        for q in q_values:
            t = axis_vec * q
            score, diag = score_translation(t, base_points, drawer_points, closed_points, residual_points, closed_tree, closed_views)
            diag.update({"family": "drawer_surface_axis_1d", "axis_label": label, "q_open_to_closed_m": float(q), "axis_camera": axis_vec.tolist()})
            candidates.append(diag)
            coarse.append(diag)
        best_axis = min(coarse, key=lambda c: c["score"])
        q0 = float(best_axis["q_open_to_closed_m"])
        for q in np.linspace(q0 - 0.06, q0 + 0.06, 121):
            t = axis_vec * q
            score, diag = score_translation(t, base_points, drawer_points, closed_points, residual_points, closed_tree, closed_views)
            diag.update({"family": "drawer_surface_axis_1d_refined", "axis_label": label, "q_open_to_closed_m": float(q), "axis_camera": axis_vec.tolist()})
            candidates.append(diag)

    # Candidate family 2: free translation initialized from latent residual clusters, then locally jittered.
    for init in cluster_initial_translations(drawer_points, residual_points):
        refined = translation_icp_to_residual(drawer_points, residual_points, init)
        for radius in [0.0, 0.025, 0.05]:
            jitters = [np.zeros(3)] if radius == 0.0 else RNG.normal(size=(24, 3))
            for jitter in jitters:
                if radius > 0.0:
                    jitter = jitter / max(np.linalg.norm(jitter), 1e-9) * radius
                t = refined + jitter
                score, diag = score_translation(t, base_points, drawer_points, closed_points, residual_points, closed_tree, closed_views)
                axis = np.asarray(t, dtype=np.float64)
                axis = axis / max(np.linalg.norm(axis), 1e-9)
                diag.update({"family": "free_3d_latent_residual", "axis_label": "latent_residual_icp", "axis_camera": axis.tolist()})
                candidates.append(diag)

    candidates.sort(key=lambda c: c["score"])
    best = candidates[0]
    best_t = np.asarray(best["translation_open_to_closed_camera"], dtype=np.float64)
    best_axis_closed_to_open = -best_t / max(float(np.linalg.norm(best_t)), 1e-9)
    best_displacement = float(np.linalg.norm(best_t))
    family_best = {}
    for cand in candidates:
        key = cand.get("family", "unknown") + "/" + cand.get("axis_label", "")
        if key not in family_best or cand["score"] < family_best[key]["score"]:
            family_best[key] = cand
    family_best = dict(sorted(family_best.items(), key=lambda item: item[1]["score"]))

    with (OUT / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "rank",
            "family",
            "score",
            "translation_open_to_closed_camera",
            "axis_camera",
            "axis_label",
            "q_open_to_closed_m",
            "outside_mask_ratio",
            "front_violation_ratio",
            "near_depth_ratio",
            "depth_abs_median_m",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, cand in enumerate(candidates, start=1):
            writer.writerow({field: cand.get(field, "") if field != "rank" else rank for field in fields})

    draw_candidate_overlay(OUT / "best_qpos0_synthesis_overlay.png", closed_views[0], base_points, drawer_points, best_t)
    draw_axis_overlay(OUT / "best_axis_qpos1_overlay.png", -best_t, best_t, drawer_points)
    export_surface_scene(OUT / "best_synthesized_surfaces_qpos1_frame.glb", base_points, drawer_points, best_t)
    best_axis1d = next((cand for cand in candidates if cand["family"].startswith("drawer_surface_axis")), None)
    if best_axis1d is not None:
        t_axis1d = np.asarray(best_axis1d["translation_open_to_closed_camera"], dtype=np.float64)
        draw_candidate_overlay(OUT / "best_axis1d_qpos0_synthesis_overlay.png", closed_views[0], base_points, drawer_points, t_axis1d)
        draw_axis_overlay(OUT / "best_axis1d_qpos1_overlay.png", -t_axis1d, t_axis1d, drawer_points)
        export_surface_scene(OUT / "best_axis1d_synthesized_surfaces_qpos1_frame.glb", base_points, drawer_points, t_axis1d)

    report = {
        "method": (
            "Analysis-by-synthesis prismatic estimate without closed-state part masks. "
            "The qpos1 base/drawer RGB-D surfaces are trusted. qpos0 uses only whole-object SAM masks and aligned depth. "
            "The closed drawer surface is a latent explanation scored by depth, whole-mask projection, static-base residuals, and multi-view consistency."
        ),
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "inputs": {
            "pose_metadata": str(POSE_METADATA),
            "qpos1_capture": QPOS1_CAPTURE,
            "qpos1_base_mask": str(QPOS1_BASE_MASK),
            "qpos1_drawer_mask": str(QPOS1_DRAWER_MASK),
            "qpos0_closed_whole_masks": [view["mask_path"] for view in closed_views],
        },
        "qpos1_drawer_axis_candidates": axis_candidates,
        "latent_residual": residual_diag,
        "best": {
            **best,
            "axis_closed_to_open_camera": best_axis_closed_to_open.tolist(),
            "displacement_m": best_displacement,
        },
        "family_best": family_best,
        "top_candidates": candidates[:30],
        "outputs": {
            "report": str(OUT / "synthesis_axis_estimate_report.json"),
            "candidate_scores_csv": str(OUT / "candidate_scores.csv"),
            "best_qpos0_overlay": str(OUT / "best_qpos0_synthesis_overlay.png"),
            "best_axis_qpos1_overlay": str(OUT / "best_axis_qpos1_overlay.png"),
            "best_surfaces_glb": str(OUT / "best_synthesized_surfaces_qpos1_frame.glb"),
            "best_axis1d_qpos0_overlay": str(OUT / "best_axis1d_qpos0_synthesis_overlay.png"),
            "best_axis1d_qpos1_overlay": str(OUT / "best_axis1d_qpos1_overlay.png"),
        },
    }
    (OUT / "synthesis_axis_estimate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": report["outputs"]["report"],
                "best_family": best["family"],
                "best_score": best["score"],
                "open_to_closed_translation_camera": best["translation_open_to_closed_camera"],
                "axis_closed_to_open_camera": report["best"]["axis_closed_to_open_camera"],
                "displacement_m": report["best"]["displacement_m"],
                "overlay": report["outputs"]["best_qpos0_overlay"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
