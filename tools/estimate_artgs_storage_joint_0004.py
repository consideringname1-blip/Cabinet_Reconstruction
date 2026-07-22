import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_real_pointcloud_animation as glb


ROOT = Path("/workspace_whz")
DATASET = ROOT / "datasets/artgs_data/ArtGS_raw_data/paris/sapien/storage_45135"
OUT = ROOT / "data/output/artgs_joint_stability/storage_45135_train_end0004_start0031"
SPLIT = "train"
END_VIEW = "0004"
START_VIEW = "0031"
RNG = np.random.default_rng(20260718)

BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in [5, 10, 25, 50, 75, 90, 95]}


def voxel_downsample(points, colors=None, voxel=0.006, max_points=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
    if len(points) == 0:
        return (points, colors) if colors is not None else points
    if voxel > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    if max_points is not None and len(points) > max_points:
        idx = RNG.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    return (points.astype(np.float32), colors.astype(np.uint8)) if colors is not None else points.astype(np.float32)


def sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points
    return points[RNG.choice(len(points), size=max_points, replace=False)]


def tint(colors, rgb, alpha=0.55):
    colors = np.asarray(colors, dtype=np.uint8)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(1, 3)
    out = (1.0 - alpha) * colors[:, :3].astype(np.float32) + alpha * rgb
    return np.column_stack([np.clip(out, 0, 255).astype(np.uint8), colors[:, 3]]).astype(np.uint8)


def state_paths(state):
    view = START_VIEW if state == "start" else END_VIEW
    return {
        "rgba": DATASET / state / SPLIT / "rgba" / f"{view}.png",
        "depth": DATASET / state / SPLIT / "depth" / f"{view}.png",
        "camera": DATASET / state / "camera_train.json",
    }


def load_camera(state):
    view = START_VIEW if state == "start" else END_VIEW
    doc = read_json(state_paths(state)["camera"])
    return np.asarray(doc["K"], dtype=np.float64), np.asarray(doc[view], dtype=np.float64)


def camera_to_world_cv(state):
    _, pose_blender = load_camera(state)
    # ArtGS/NeRF-style camera matrices use Blender camera coordinates. Depth
    # unprojection below is OpenCV-style, so insert the standard axis flip.
    return pose_blender @ BLENDER_TO_OPENCV


def world_to_camera_cv(state):
    return np.linalg.inv(camera_to_world_cv(state))


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=np.float64).T)[:, :3]


def unproject_state(state, max_points=None):
    paths = state_paths(state)
    rgba = cv2.imread(str(paths["rgba"]), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(paths["rgba"])
    if depth is None:
        raise FileNotFoundError(paths["depth"])
    if rgba.shape[2] != 4:
        raise RuntimeError(f"Expected RGBA image: {paths['rgba']}")
    depth_m = depth.astype(np.float64) / 1000.0
    k, _ = load_camera(state)
    valid = (rgba[:, :, 3] > 0) & (depth_m > 0.05) & (depth_m < 10.0)
    ys, xs = np.nonzero(valid)
    if max_points is not None and len(xs) > max_points:
        idx = RNG.choice(len(xs), size=max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    z = depth_m[ys, xs]
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    cam = np.column_stack([x, y, z])
    world = transform_points(cam, camera_to_world_cv(state))
    rgb = rgba[ys, xs][:, [2, 1, 0]]
    colors = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)]).astype(np.uint8)
    return {
        "state": state,
        "points_world": world.astype(np.float32),
        "colors": colors,
        "pixels": np.column_stack([xs, ys]).astype(np.int32),
        "valid_mask": valid,
        "rgba": rgba,
        "depth_m": depth_m,
        "k": k,
        "stats": {
            "view": START_VIEW if state == "start" else END_VIEW,
            "rgba": str(paths["rgba"]),
            "depth": str(paths["depth"]),
            "camera": str(paths["camera"]),
            "valid_depth_alpha_pixels": int(valid.sum()),
            "exported_points": int(len(world)),
        },
    }


def read_ply_mesh(path):
    data = Path(path).read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("latin1").splitlines()
    vertex_count = 0
    face_count = 0
    vertex_props = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
            face_count = int(parts[2])
            in_vertex = False
            continue
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = False
            continue
        if in_vertex and len(parts) >= 3 and parts[0] == "property":
            vertex_props.append(parts[1])
    if vertex_props[:3] != ["double", "double", "double"]:
        raise RuntimeError(f"Unexpected PLY vertex properties in {path}: {vertex_props}")
    rec = struct.Struct("<" + "".join({"double": "d", "float": "f", "uchar": "B", "uint": "I", "int": "i"}[p] for p in vertex_props))
    offset = header_end
    vertices = np.empty((vertex_count, 3), dtype=np.float64)
    for i in range(vertex_count):
        vertices[i] = rec.unpack_from(data, offset)[:3]
        offset += rec.size
    faces = []
    for _ in range(face_count):
        n = data[offset]
        offset += 1
        idx = struct.unpack_from("<" + "I" * n, data, offset)
        offset += 4 * n
        if n == 3:
            faces.append(idx)
        elif n > 3:
            for j in range(1, n - 1):
                faces.append((idx[0], idx[j], idx[j + 1]))
    return vertices, np.asarray(faces, dtype=np.int32)


def sample_mesh_surface(vertices, faces, count):
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    probs = areas / max(float(areas.sum()), 1e-12)
    idx = RNG.choice(len(faces), size=count, replace=True, p=probs)
    chosen = tri[idx]
    r1 = np.sqrt(RNG.random(count))
    r2 = RNG.random(count)
    pts = (1.0 - r1)[:, None] * chosen[:, 0] + (r1 * (1.0 - r2))[:, None] * chosen[:, 1] + (r1 * r2)[:, None] * chosen[:, 2]
    return pts.astype(np.float32)


def load_gt_mesh_samples():
    gt = DATASET / "gt"
    meshes = {
        "start_static": read_ply_mesh(gt / "start/start_static_rotate.ply"),
        "start_dynamic": read_ply_mesh(gt / "start/start_dynamic_rotate.ply"),
        "end_static": read_ply_mesh(gt / "end/end_static_rotate.ply"),
        "end_dynamic": read_ply_mesh(gt / "end/end_dynamic_rotate.ply"),
    }
    samples = {}
    for name, (vertices, faces) in meshes.items():
        samples[name] = sample_mesh_surface(vertices, faces, 120000 if "static" in name else 80000)
    return meshes, samples


def classify_by_mesh(points, static_samples, dynamic_samples):
    points = np.asarray(points, dtype=np.float64)
    st = cKDTree(static_samples)
    dy = cKDTree(dynamic_samples)
    ds, _ = st.query(points, k=1, workers=-1)
    dd, _ = dy.query(points, k=1, workers=-1)
    label_dynamic = dd < ds
    return label_dynamic, ds, dd


def residual_from_base(closed_points, base_points, label):
    base_tree = cKDTree(sample_points(base_points, 70000))
    dist, _ = base_tree.query(closed_points, k=1, workers=-1)
    threshold = max(0.035, float(np.percentile(dist, 76)))
    residual = closed_points[dist > threshold]
    if len(residual) < 800:
        threshold = float(np.percentile(dist, 65))
        residual = closed_points[dist > threshold]
    return residual.astype(np.float32), {
        "label": label,
        "threshold_m": float(threshold),
        "closed_points": int(len(closed_points)),
        "residual_points": int(len(residual)),
        "base_distance_percentiles_m": percentiles(dist),
    }


def plane_axis_candidates(points, iterations=650, threshold=0.012):
    pts = voxel_downsample(points, voxel=0.010, max_points=6500).astype(np.float64)
    candidates = []
    remaining = pts.copy()
    for plane_idx in range(5):
        if len(remaining) < 180:
            break
        best = None
        for _ in range(iterations):
            ids = RNG.choice(len(remaining), size=3, replace=False)
            a, b, c = remaining[ids]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n = n / norm
            d = -float(np.dot(n, a))
            dist = np.abs(remaining @ n + d)
            inliers = dist < threshold
            count = int(np.sum(inliers))
            if best is None or count > best["count"]:
                best = {"normal": n, "d": d, "inliers": inliers, "count": count}
        if best is None or best["count"] < 120:
            break
        inlier_pts = remaining[best["inliers"]]
        _, _, vh = np.linalg.svd(inlier_pts - inlier_pts.mean(axis=0), full_matrices=False)
        axis = unit(vh[-1])
        duplicate = False
        for cand in candidates:
            if abs(float(np.dot(axis, cand["axis_world"]))) > 0.965:
                duplicate = True
                break
        if not duplicate:
            candidates.append(
                {
                    "label": f"ransac_plane_{plane_idx + 1}",
                    "axis_world": axis,
                    "inliers": int(len(inlier_pts)),
                    "inlier_ratio_remaining": float(len(inlier_pts) / max(1, len(remaining))),
                    "threshold_m": float(threshold),
                    "center_world": inlier_pts.mean(axis=0).astype(float).tolist(),
                }
            )
        remaining = remaining[~best["inliers"]]
    return candidates


def score_translation(drawer_open, residual, translation_open_to_closed):
    shifted = np.asarray(drawer_open, dtype=np.float64) + np.asarray(translation_open_to_closed, dtype=np.float64).reshape(1, 3)
    residual_s = sample_points(residual, 35000)
    shifted_s = sample_points(shifted, 26000)
    drawer_tree = cKDTree(shifted_s)
    residual_to_drawer, _ = drawer_tree.query(residual_s, k=1, workers=1)
    residual_tree = cKDTree(residual_s)
    drawer_to_residual, _ = residual_tree.query(shifted_s, k=1, workers=1)
    score = (
        float(np.median(residual_to_drawer))
        + 0.42 * float(np.percentile(residual_to_drawer, 85))
        + 0.32 * float(np.median(drawer_to_residual))
        + 0.16 * float(np.percentile(drawer_to_residual, 85))
    )
    t = np.asarray(translation_open_to_closed, dtype=np.float64)
    q = float(np.linalg.norm(t))
    axis_closed_to_open = unit(-t) if q > 1e-9 else np.array([np.nan, np.nan, np.nan])
    return {
        "score": float(score),
        "translation_open_to_closed_world": t.astype(float).tolist(),
        "translation_closed_to_open_world": (-t).astype(float).tolist(),
        "axis_closed_to_open_world": axis_closed_to_open.astype(float).tolist(),
        "displacement_m": q,
        "residual_to_closed_drawer_percentiles_m": percentiles(residual_to_drawer),
        "closed_drawer_to_residual_percentiles_m": percentiles(drawer_to_residual),
    }


def refine_translation_icp(drawer_open, residual, initial_t, iterations=36):
    src = voxel_downsample(drawer_open, voxel=0.012, max_points=6000).astype(np.float64)
    dst = voxel_downsample(residual, voxel=0.012, max_points=9000).astype(np.float64)
    tree = cKDTree(dst)
    t = np.asarray(initial_t, dtype=np.float64).reshape(3)
    for _ in range(iterations):
        dist, idx = tree.query(src + t.reshape(1, 3), k=1, workers=1)
        keep = dist <= np.percentile(dist, 62)
        if int(keep.sum()) < 200:
            keep = dist <= np.percentile(dist, 82)
        new_t = np.median(dst[idx[keep]] - src[keep], axis=0)
        if np.linalg.norm(new_t - t) < 1e-5:
            t = new_t
            break
        t = 0.58 * t + 0.42 * new_t
    return t


def residual_cluster_initials(residual, drawer_open):
    residual = np.asarray(residual, dtype=np.float64)
    drawer_med = np.median(drawer_open, axis=0)
    initials = [np.median(residual, axis=0) - drawer_med]
    keys = np.floor(residual / 0.08).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1][:18]
    tree = cKDTree(residual)
    for key_idx in order:
        seed = (uniq[key_idx].astype(np.float64) + 0.5) * 0.08
        ids = tree.query_ball_point(seed, r=0.18)
        if len(ids) < 120:
            ids = np.nonzero(inv == key_idx)[0].tolist()
        cluster = residual[ids]
        if len(cluster) >= 80:
            initials.append(np.median(cluster, axis=0) - drawer_med)
    dedup = []
    for t in initials:
        if not any(np.linalg.norm(t - old) < 0.025 for old in dedup):
            dedup.append(t)
    return dedup


def estimate_from_residual(label, drawer_open, residual):
    candidates = []
    drawer_metric = voxel_downsample(drawer_open, voxel=0.012, max_points=6000).astype(np.float64)
    residual_metric = voxel_downsample(residual, voxel=0.012, max_points=9000).astype(np.float64)
    plane_candidates = plane_axis_candidates(drawer_metric)
    q_grid = np.linspace(-0.75, 0.75, 81)
    for axis_info in plane_candidates:
        axis = axis_info["axis_world"]
        best_for_axis = []
        for q in q_grid:
            diag = score_translation(drawer_metric, residual_metric, axis * float(q))
            diag.update(
                {
                    "mode": label,
                    "family": "drawer_surface_plane_axis_1d",
                    "axis_label": axis_info["label"],
                    "axis_candidate_world": axis.astype(float).tolist(),
                    "q_open_to_closed_signed_m": float(q),
                }
            )
            best_for_axis.append(diag)
            candidates.append(diag)
        best_q = min(best_for_axis, key=lambda x: x["score"])["q_open_to_closed_signed_m"]
        for q in np.linspace(best_q - 0.055, best_q + 0.055, 41):
            diag = score_translation(drawer_metric, residual_metric, axis * float(q))
            diag.update(
                {
                    "mode": label,
                    "family": "drawer_surface_plane_axis_1d_refined",
                    "axis_label": axis_info["label"],
                    "axis_candidate_world": axis.astype(float).tolist(),
                    "q_open_to_closed_signed_m": float(q),
                }
            )
            candidates.append(diag)

    for init in residual_cluster_initials(residual_metric, drawer_metric):
        refined = refine_translation_icp(drawer_metric, residual_metric, init)
        for t in [init, refined]:
            diag = score_translation(drawer_metric, residual_metric, t)
            diag.update({"mode": label, "family": "free_translation_residual_icp", "axis_label": "residual_cluster"})
            candidates.append(diag)

    candidates.sort(key=lambda x: x["score"])
    family_best = {}
    for cand in candidates:
        key = f"{cand['family']}/{cand.get('axis_label', '')}"
        if key not in family_best or cand["score"] < family_best[key]["score"]:
            family_best[key] = cand
    return {
        "label": label,
        "best": candidates[0],
        "best_axis1d": next((c for c in candidates if c["family"].startswith("drawer_surface_plane_axis")), None),
        "plane_axis_candidates": [
            {**c, "axis_world": c["axis_world"].astype(float).tolist()} for c in plane_candidates
        ],
        "family_best": dict(sorted(family_best.items(), key=lambda item: item[1]["score"])),
        "top_candidates": candidates[:25],
    }


def angle_to_gt(axis, gt_axis):
    axis = unit(axis)
    gt_axis = unit(gt_axis)
    cos_abs = float(np.clip(abs(np.dot(axis, gt_axis)), -1.0, 1.0))
    cos_signed = float(np.clip(np.dot(axis, gt_axis), -1.0, 1.0))
    return {
        "unsigned_deg": float(np.degrees(np.arccos(cos_abs))),
        "signed_deg": float(np.degrees(np.arccos(cos_signed))),
        "dot_signed": cos_signed,
        "dot_abs": cos_abs,
    }


def evaluate_candidate(name, cand, gt_axis, gt_displacement, start_dynamic_points=None):
    axis = np.asarray(cand["axis_closed_to_open_world"], dtype=np.float64)
    q = float(cand["displacement_m"])
    out = {
        "name": name,
        "score": float(cand["score"]),
        "axis_closed_to_open_world": axis.astype(float).tolist(),
        "translation_closed_to_open_world": cand["translation_closed_to_open_world"],
        "translation_open_to_closed_world": cand["translation_open_to_closed_world"],
        "displacement_m": q,
        "axis_vs_gt": angle_to_gt(axis, gt_axis),
        "displacement_error_m": float(q - gt_displacement),
        "abs_displacement_error_m": float(abs(q - gt_displacement)),
    }
    if start_dynamic_points is not None and len(start_dynamic_points):
        # How well does the estimated closed drawer explain true dynamic pixels in the start render?
        # This metric is for validation only and is not used during estimation.
        out["validation_note"] = "Uses gt start dynamic classification only for validation."
    return out


def project_world(points_world, state):
    k, _ = load_camera(state)
    cam = transform_points(points_world, world_to_camera_cv(state))
    valid = cam[:, 2] > 1e-6
    cam = cam[valid]
    uv = np.column_stack([k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2], k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]])
    return uv, cam


def draw_points(img, points_world, state, color, limit=45000, radius=1):
    pts = sample_points(points_world, limit)
    uv, _ = project_world(pts, state)
    h, w = img.shape[:2]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    pix = np.round(uv[inb]).astype(np.int32)
    for x, y in pix:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def draw_overlay(path, state, base_points, drawer_open, residual, translation_open_to_closed, title):
    rgba = cv2.imread(str(state_paths(state)["rgba"]), cv2.IMREAD_UNCHANGED)
    img = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3] > 0
    contours, _ = cv2.findContours((alpha.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    drawer_closed = drawer_open + np.asarray(translation_open_to_closed, dtype=np.float64).reshape(1, 3)
    draw_points(img, base_points, state, (50, 220, 70), limit=45000, radius=1)
    draw_points(img, residual, state, (255, 210, 40), limit=45000, radius=1)
    draw_points(img, drawer_closed, state, (40, 70, 245), limit=45000, radius=1)
    cv2.putText(img, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def draw_axis_overlay(path, state, drawer_open, translation_open_to_closed):
    rgba = cv2.imread(str(state_paths(state)["rgba"]), cv2.IMREAD_UNCHANGED)
    img = rgba[:, :, :3].copy()
    t = np.asarray(translation_open_to_closed, dtype=np.float64)
    axis = unit(-t)
    q = float(np.linalg.norm(t))
    center = np.median(drawer_open + t.reshape(1, 3), axis=0)
    p0 = center
    p1 = center + axis * max(q, 0.2)
    uv, _ = project_world(np.stack([p0, p1], axis=0), state)
    if len(uv) == 2:
        a = tuple(np.round(uv[0]).astype(int))
        b = tuple(np.round(uv[1]).astype(int))
        cv2.arrowedLine(img, a, b, (255, 0, 255), 4, tipLength=0.14)
        cv2.putText(img, "closed->open axis", (b[0] + 6, b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def build_animation(path, base_points, base_colors, drawer_open, drawer_colors, residual, translation_open_to_closed):
    t = np.asarray(translation_open_to_closed, dtype=np.float32)
    drawer_closed = (np.asarray(drawer_open, dtype=np.float32) + t.reshape(1, 3)).astype(np.float32)
    base_points, base_colors = voxel_downsample(base_points, base_colors, voxel=0.008, max_points=26000)
    drawer_closed, drawer_colors = voxel_downsample(drawer_closed, drawer_colors, voxel=0.008, max_points=18000)
    residual = voxel_downsample(residual, voxel=0.010, max_points=26000)
    residual_colors = np.tile(np.array([255, 210, 40, 255], dtype=np.uint8), (len(residual), 1))

    builder = glb.GlbBuilder("artgs-storage-45135-joint-stability")
    bv, bc, bi = glb.make_splats(base_points, base_colors, half_size=0.006)
    dv, dc, di = glb.make_splats(drawer_closed, drawer_colors, half_size=0.006)
    rv, rc, ri = glb.make_splats(residual, residual_colors, half_size=0.007)
    base_mesh = builder.add_mesh("end_visible_base_splats", bv, bc, glb.TRIANGLES, bi)
    drawer_mesh = builder.add_mesh("synth_closed_drawer_splats", dv, dc, glb.TRIANGLES, di)
    residual_mesh = builder.add_mesh("start_residual_reference_splats", rv, rc, glb.TRIANGLES, ri)
    axis_pos, axis_col = glb.make_axis_line(drawer_closed, -t)
    axis_mesh = builder.add_mesh("axis_line_closed_to_open", axis_pos, axis_col, glb.LINES)
    builder.add_node("end_visible_base_static", base_mesh)
    drawer_node = builder.add_node("drawer_closed_animated_to_end_open", drawer_mesh)
    builder.add_node("start_residual_reference_static", residual_mesh)
    builder.add_node("axis_line_closed_to_open", axis_mesh)
    builder.add_translation_animation("drawer_closed_to_open", drawer_node, -t)
    builder.write(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = unproject_state("start")
    end = unproject_state("end")
    meshes, mesh_samples = load_gt_mesh_samples()

    end_dyn, end_ds, end_dd = classify_by_mesh(end["points_world"], mesh_samples["end_static"], mesh_samples["end_dynamic"])
    start_dyn, start_ds, start_dd = classify_by_mesh(start["points_world"], mesh_samples["start_static"], mesh_samples["start_dynamic"])

    end_base_points = end["points_world"][~end_dyn]
    end_base_colors = tint(end["colors"][~end_dyn], [60, 220, 70], alpha=0.45)
    end_drawer_points = end["points_world"][end_dyn]
    end_drawer_colors = tint(end["colors"][end_dyn], [240, 80, 80], alpha=0.30)
    start_whole_points = start["points_world"]
    start_whole_colors = tint(start["colors"], [80, 190, 255], alpha=0.40)
    start_dynamic_points = start["points_world"][start_dyn]

    end_base_points, end_base_colors = voxel_downsample(end_base_points, end_base_colors, voxel=0.005, max_points=80000)
    end_drawer_points, end_drawer_colors = voxel_downsample(end_drawer_points, end_drawer_colors, voxel=0.005, max_points=50000)
    start_whole_points, start_whole_colors = voxel_downsample(start_whole_points, start_whole_colors, voxel=0.005, max_points=90000)

    residual_visible, residual_visible_diag = residual_from_base(start_whole_points, end_base_points, "visible_end_base_only")
    residual_gt_static, residual_gt_static_diag = residual_from_base(start_whole_points, mesh_samples["end_static"], "gt_static_mesh_reference")

    visible_est = estimate_from_residual("visible_end_base_only", end_drawer_points, residual_visible)
    gt_static_est = estimate_from_residual("gt_static_mesh_reference", end_drawer_points, residual_gt_static)

    trans_doc = read_json(DATASET / "gt/trans.json")
    gt_axis = np.asarray(trans_doc["trans_info"]["axis"]["d"], dtype=np.float64)
    gt_q = float(trans_doc["trans_info"]["translate"]["r"] - trans_doc["trans_info"]["translate"]["l"])

    outputs = {}
    for est_name, est_doc, residual in [
        ("visible_end_base_only", visible_est, residual_visible),
        ("gt_static_mesh_reference", gt_static_est, residual_gt_static),
    ]:
        best = est_doc["best"]
        t = np.asarray(best["translation_open_to_closed_world"], dtype=np.float64)
        overlay_end = OUT / f"{est_name}_end_overlay.png"
        overlay_start = OUT / f"{est_name}_start_overlay.png"
        axis_overlay = OUT / f"{est_name}_axis_overlay.png"
        animated = OUT / f"{est_name}_animated_splats.glb"
        draw_overlay(
            overlay_end,
            "end",
            end_base_points,
            end_drawer_points,
            residual,
            t,
            f"{est_name}: green=end base yellow=start residual red=synth closed drawer",
        )
        draw_overlay(
            overlay_start,
            "start",
            end_base_points,
            end_drawer_points,
            residual,
            t,
            f"{est_name}: red=synth closed drawer projected into start view",
        )
        draw_axis_overlay(axis_overlay, "end", end_drawer_points, t)
        build_animation(animated, end_base_points, end_base_colors, end_drawer_points, end_drawer_colors, residual, t)
        outputs[est_name] = {
            "end_overlay": str(overlay_end),
            "start_overlay": str(overlay_start),
            "axis_overlay": str(axis_overlay),
            "animated_splats_glb": str(animated),
        }

    evals = {
        "visible_end_base_only_best": evaluate_candidate("visible_end_base_only_best", visible_est["best"], gt_axis, gt_q, start_dynamic_points),
        "visible_end_base_only_best_axis1d": evaluate_candidate("visible_end_base_only_best_axis1d", visible_est["best_axis1d"], gt_axis, gt_q, start_dynamic_points)
        if visible_est["best_axis1d"]
        else None,
        "gt_static_mesh_reference_best": evaluate_candidate("gt_static_mesh_reference_best", gt_static_est["best"], gt_axis, gt_q, start_dynamic_points),
        "gt_static_mesh_reference_best_axis1d": evaluate_candidate("gt_static_mesh_reference_best_axis1d", gt_static_est["best_axis1d"], gt_axis, gt_q, start_dynamic_points)
        if gt_static_est["best_axis1d"]
        else None,
    }

    report = {
        "method": (
            "ArtGS/SAPIEN stability probe for the current RGB-D geometric prismatic estimator. "
            f"End/train/{END_VIEW} supplies qpos1 visible base/drawer surfaces. Start/train/{START_VIEW} supplies whole-object closed-state RGB-D. "
            "No closed-state part mask is used. Camera matrices are used only to put both depth maps into the common SAPIEN world frame."
        ),
        "coordinate_frame": "ArtGS/SAPIEN world frame",
        "camera_convention": "depth unprojects as OpenCV camera coordinates; world = camera_train[VIEW] @ diag(1,-1,-1,1) @ camera_opencv",
        "inputs": {
            "dataset": str(DATASET),
            "split": SPLIT,
            "end_view": END_VIEW,
            "start_view": START_VIEW,
            "start": start["stats"],
            "end": end["stats"],
            "gt_trans": str(DATASET / "gt/trans.json"),
        },
        "gt_joint": {
            "axis_closed_to_open_world": gt_axis.astype(float).tolist(),
            "displacement_start_to_end_m": gt_q,
            "translate_l": float(trans_doc["trans_info"]["translate"]["l"]),
            "translate_r": float(trans_doc["trans_info"]["translate"]["r"]),
            "type": trans_doc["trans_info"]["type"],
        },
        "segmentation_from_gt_mesh_projection": {
            "note": "GT meshes are used to classify end visible depth into base/drawer for this stability probe, matching the current pipeline assumption that qpos1 masks are trusted.",
            "end_visible_base_points": int(len(end_base_points)),
            "end_visible_drawer_points": int(len(end_drawer_points)),
            "start_gt_dynamic_visible_points_validation_only": int(np.sum(start_dyn)),
            "end_mesh_distance_static_percentiles_m": percentiles(end_ds),
            "end_mesh_distance_dynamic_percentiles_m": percentiles(end_dd),
            "start_mesh_distance_static_percentiles_m": percentiles(start_ds),
            "start_mesh_distance_dynamic_percentiles_m": percentiles(start_dd),
        },
        "residuals": {
            "visible_end_base_only": residual_visible_diag,
            "gt_static_mesh_reference": residual_gt_static_diag,
        },
        "estimates": {
            "visible_end_base_only": visible_est,
            "gt_static_mesh_reference": gt_static_est,
        },
        "evaluation_against_gt": evals,
        "outputs": outputs,
    }
    report_path = OUT / "artgs_storage_45135_train_end0004_start0031_joint_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joint_paths = {}
    for joint_label, joint_est in [
        ("visible_end_base_only", visible_est),
        ("gt_static_mesh_reference", gt_static_est),
    ]:
        joint_path = OUT / f"joint_{joint_label}.json"
        joint_path.write_text(
            json.dumps(
                {
                    "joint": {
                        "type": "prismatic",
                        "axis_closed_to_open_world": joint_est["best"]["axis_closed_to_open_world"],
                        "origin_world": np.median(end_drawer_points, axis=0).astype(float).tolist(),
                        "displacement_m": joint_est["best"]["displacement_m"],
                        "translation_open_to_closed_world": joint_est["best"]["translation_open_to_closed_world"],
                        "translation_closed_to_open_world": joint_est["best"]["translation_closed_to_open_world"],
                    },
                    "source_report": str(report_path),
                    "estimate_label": joint_label,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        joint_paths[joint_label] = str(joint_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "joint_jsons": joint_paths,
                "visible_best": evals["visible_end_base_only_best"],
                "gt_static_best": evals["gt_static_mesh_reference_best"],
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
