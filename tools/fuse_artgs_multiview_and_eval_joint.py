import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_real_pointcloud_animation as glb
import estimate_artgs_storage_joint_0004 as est


ROOT = Path("/workspace_whz")
DATASET = est.DATASET
OUT = ROOT / "data/output/artgs_joint_stability/storage_45135_multiview_train_reconstruction_eval"
SPLIT = "train"
RNG = np.random.default_rng(20260722)

PER_VIEW_MAX_PIXELS = 26000
FUSED_VOXEL_M = 0.006
FUSED_MAX_POINTS = 260000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    return est.unit(v)


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}


def voxel_downsample_with_colors(points, colors, voxel=0.006, max_points=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
    if len(points) == 0:
        return points.astype(np.float32), colors
    if voxel > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx)
        points = points[idx]
        colors = colors[idx]
    if max_points is not None and len(points) > max_points:
        idx = RNG.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float32), colors.astype(np.uint8)


def sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= max_points:
        return points
    return points[RNG.choice(len(points), size=max_points, replace=False)]


def sample_pair(points, colors, max_points):
    if len(points) <= max_points:
        return points, colors
    idx = RNG.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def tint(colors, rgb, alpha=0.55):
    colors = np.asarray(colors, dtype=np.uint8)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(1, 3)
    out = (1.0 - alpha) * colors[:, :3].astype(np.float32) + alpha * rgb
    return np.column_stack([np.clip(out, 0, 255).astype(np.uint8), colors[:, 3]]).astype(np.uint8)


def camera_doc(state):
    return read_json(DATASET / state / "camera_train.json")


def view_ids(state):
    doc = camera_doc(state)
    ids = sorted(k for k in doc.keys() if k != "K")
    valid = []
    for vid in ids:
        rgba = DATASET / state / SPLIT / "rgba" / f"{vid}.png"
        depth = DATASET / state / SPLIT / "depth" / f"{vid}.png"
        if rgba.exists() and depth.exists():
            valid.append(vid)
    return valid


def camera_to_world_cv(state, view_id):
    pose_blender = np.asarray(camera_doc(state)[view_id], dtype=np.float64)
    return pose_blender @ est.BLENDER_TO_OPENCV


def unproject_view(state, view_id, max_pixels=PER_VIEW_MAX_PIXELS):
    rgba_path = DATASET / state / SPLIT / "rgba" / f"{view_id}.png"
    depth_path = DATASET / state / SPLIT / "depth" / f"{view_id}.png"
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(rgba_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    k = np.asarray(camera_doc(state)["K"], dtype=np.float64)
    depth_m = depth.astype(np.float64) / 1000.0
    valid = (rgba[:, :, 3] > 0) & (depth_m > 0.05) & (depth_m < 10.0)
    ys, xs = np.nonzero(valid)
    valid_pixels = int(len(xs))
    if max_pixels is not None and len(xs) > max_pixels:
        idx = RNG.choice(len(xs), size=max_pixels, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    z = depth_m[ys, xs]
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    cam = np.column_stack([x, y, z])
    world = est.transform_points(cam, camera_to_world_cv(state, view_id))
    rgb = rgba[ys, xs][:, [2, 1, 0]]
    colors = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)]).astype(np.uint8)
    return world.astype(np.float32), colors, {"view": view_id, "valid_pixels": valid_pixels, "sampled_pixels": int(len(xs))}


def fuse_state(state):
    ids = view_ids(state)
    pts_all = []
    col_all = []
    per_view = []
    for vid in ids:
        pts, colors, stats = unproject_view(state, vid)
        pts_all.append(pts)
        col_all.append(colors)
        per_view.append(stats)
    points = np.vstack(pts_all)
    colors = np.vstack(col_all)
    before = int(len(points))
    points, colors = voxel_downsample_with_colors(points, colors, voxel=FUSED_VOXEL_M, max_points=FUSED_MAX_POINTS)
    return {
        "state": state,
        "points": points,
        "colors": colors,
        "stats": {
            "state": state,
            "split": SPLIT,
            "views": len(ids),
            "view_ids": ids,
            "points_before_voxel": before,
            "points_after_voxel": int(len(points)),
            "voxel_m": FUSED_VOXEL_M,
            "per_view_max_pixels": PER_VIEW_MAX_PIXELS,
            "per_view": per_view,
        },
    }


def load_gt_samples():
    gt = DATASET / "gt"
    mesh_paths = {
        "start_static": gt / "start/start_static_rotate.ply",
        "start_dynamic": gt / "start/start_dynamic_rotate.ply",
        "end_static": gt / "end/end_static_rotate.ply",
        "end_dynamic": gt / "end/end_dynamic_rotate.ply",
    }
    samples = {}
    for key, path in mesh_paths.items():
        vertices, faces = est.read_ply_mesh(path)
        count = 260000 if "static" in key else 180000
        samples[key] = est.sample_mesh_surface(vertices, faces, count)
    samples["start_whole"] = np.vstack([samples["start_static"], samples["start_dynamic"]]).astype(np.float32)
    samples["end_whole"] = np.vstack([samples["end_static"], samples["end_dynamic"]]).astype(np.float32)
    return samples


def classify_static_dynamic(points, static_samples, dynamic_samples):
    pts = np.asarray(points, dtype=np.float64)
    static_tree = cKDTree(static_samples)
    dynamic_tree = cKDTree(dynamic_samples)
    d_static, _ = static_tree.query(pts, k=1, workers=1)
    d_dynamic, _ = dynamic_tree.query(pts, k=1, workers=1)
    dynamic = d_dynamic < d_static
    return dynamic, d_static, d_dynamic


def nn_stats(source, target, max_source=90000, max_target=220000):
    src = sample_points(source, max_source)
    tgt = sample_points(target, max_target)
    tree = cKDTree(tgt)
    dist, _ = tree.query(src, k=1, workers=1)
    return {"source_points": int(len(src)), "target_points": int(len(tgt)), "percentiles_m": percentiles(dist)}


def chamfer_stats(a, b):
    ab = nn_stats(a, b)
    ba = nn_stats(b, a)
    med = 0.5 * (ab["percentiles_m"].get("50", np.nan) + ba["percentiles_m"].get("50", np.nan))
    p90 = 0.5 * (ab["percentiles_m"].get("90", np.nan) + ba["percentiles_m"].get("90", np.nan))
    return {"a_to_b": ab, "b_to_a": ba, "symmetric_median_m": float(med), "symmetric_p90_m": float(p90)}


def residual_from_base(closed_points, base_points):
    base_tree = cKDTree(sample_points(base_points, 180000))
    dist, _ = base_tree.query(closed_points, k=1, workers=1)
    threshold = max(0.025, float(np.percentile(dist, 82)))
    residual = closed_points[dist > threshold]
    if len(residual) < 1000:
        threshold = float(np.percentile(dist, 70))
        residual = closed_points[dist > threshold]
    return residual.astype(np.float32), {
        "threshold_m": float(threshold),
        "closed_points": int(len(closed_points)),
        "base_points": int(len(base_points)),
        "residual_points": int(len(residual)),
        "base_distance_percentiles_m": percentiles(dist),
    }


def rigid_transform(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = cb - r @ ca
    return r, t


def transform_points_rt(points, r, t):
    return np.asarray(points, dtype=np.float64) @ r.T + np.asarray(t, dtype=np.float64).reshape(1, 3)


def icp_alignment_check(source, target, max_corr=0.035, iterations=16):
    src = sample_points(source, 45000)
    tgt = sample_points(target, 90000)
    tree = cKDTree(tgt)
    r = np.eye(3)
    t = np.zeros(3)
    history = []
    for _ in range(iterations):
        moved = transform_points_rt(src, r, t)
        dist, idx = tree.query(moved, k=1, workers=1)
        keep = dist <= min(float(np.percentile(dist, 65)), max_corr)
        if int(keep.sum()) < 800:
            keep = dist <= float(np.percentile(dist, 55))
        dr, dt = rigid_transform(moved[keep], tgt[idx[keep]])
        r = dr @ r
        t = dr @ t + dt
        moved_new = transform_points_rt(src, r, t)
        dist_new, _ = tree.query(moved_new, k=1, workers=1)
        history.append({"median_m": float(np.median(dist_new)), "p90_m": float(np.percentile(dist_new, 90)), "pairs": int(keep.sum())})
    rot_angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))
    moved = transform_points_rt(src, r, t)
    final_dist, _ = tree.query(moved, k=1, workers=1)
    return {
        "source_points": int(len(src)),
        "target_points": int(len(tgt)),
        "rotation_matrix_source_to_target": r.astype(float).tolist(),
        "translation_source_to_target_m": t.astype(float).tolist(),
        "rotation_angle_deg": rot_angle,
        "translation_norm_m": float(np.linalg.norm(t)),
        "final_source_to_target_percentiles_m": percentiles(final_dist),
        "history": history,
    }


def axis_angle(pred_axis, gt_axis):
    pred = unit(pred_axis)
    gt = unit(gt_axis)
    dot = float(np.clip(np.dot(pred, gt), -1.0, 1.0))
    return {
        "dot_signed": dot,
        "dot_abs": float(abs(dot)),
        "signed_angle_deg": float(np.degrees(np.arccos(dot))),
        "unsigned_angle_deg": float(np.degrees(np.arccos(abs(dot)))),
    }


def point_to_line_distance(point, origin, direction):
    return float(np.linalg.norm(np.cross(np.asarray(point) - np.asarray(origin), unit(direction))))


def make_joint_doc(joint_est, drawer_points):
    best = joint_est["best"]
    return {
        "type": "prismatic",
        "axis_closed_to_open_world": best["axis_closed_to_open_world"],
        "origin_world": np.median(drawer_points, axis=0).astype(float).tolist(),
        "displacement_m": float(best["displacement_m"]),
        "translation_open_to_closed_world": best["translation_open_to_closed_world"],
        "translation_closed_to_open_world": best["translation_closed_to_open_world"],
        "score": float(best["score"]),
    }


def build_fused_cloud_glb(path, start_static, start_dynamic, end_static, end_dynamic):
    builder = glb.GlbBuilder("artgs-multiview-fused-pointcloud")
    parts = [
        ("start_static_blue", start_static, [70, 170, 255, 255], 26000),
        ("start_dynamic_orange", start_dynamic, [255, 170, 40, 255], 16000),
        ("end_static_green", end_static, [70, 225, 90, 255], 26000),
        ("end_dynamic_red", end_dynamic, [245, 70, 70, 255], 16000),
    ]
    for name, points, color, limit in parts:
        pts = sample_points(points, limit).astype(np.float32)
        colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 4), (len(pts), 1))
        vertices, vertex_colors, indices = glb.make_splats(pts, colors, half_size=0.005)
        mesh = builder.add_mesh(name, vertices, vertex_colors, glb.TRIANGLES, indices)
        builder.add_node(name, mesh)
    builder.write(path)


def build_reconstruction_glb(path, base_points, drawer_open, residual, joint, gt_axis, gt_origin):
    t_open_to_closed = np.asarray(joint["translation_open_to_closed_world"], dtype=np.float32)
    t_closed_to_open = np.asarray(joint["translation_closed_to_open_world"], dtype=np.float32)
    drawer_closed = (np.asarray(drawer_open, dtype=np.float32) + t_open_to_closed.reshape(1, 3)).astype(np.float32)
    builder = glb.GlbBuilder("artgs-multiview-prismatic-reconstruction")
    parts = [
        ("end_base_static_green", base_points, [70, 225, 90, 255], 30000, 0.005),
        ("synth_closed_drawer_red", drawer_closed, [245, 70, 70, 255], 22000, 0.005),
        ("start_residual_yellow", residual, [255, 215, 40, 255], 22000, 0.006),
    ]
    drawer_node = None
    for name, points, color, limit, half in parts:
        pts = sample_points(points, limit).astype(np.float32)
        colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 4), (len(pts), 1))
        vertices, vertex_colors, indices = glb.make_splats(pts, colors, half_size=half)
        mesh = builder.add_mesh(name, vertices, vertex_colors, glb.TRIANGLES, indices)
        node = builder.add_node(name, mesh)
        if name == "synth_closed_drawer_red":
            drawer_node = node
    pred_axis_pos, pred_axis_colors = glb.make_axis_line(drawer_closed, t_closed_to_open)
    pred_axis_mesh = builder.add_mesh("pred_axis_yellow_closed_to_open", pred_axis_pos, pred_axis_colors, glb.LINES)
    builder.add_node("pred_axis_yellow_closed_to_open", pred_axis_mesh)
    gt_origin = np.asarray(gt_origin, dtype=np.float32)
    gt_axis = unit(gt_axis).astype(np.float32)
    gt_line = np.stack([gt_origin - gt_axis * 0.28, gt_origin + gt_axis * 0.55], axis=0)
    gt_colors = np.tile(np.asarray([20, 230, 255, 255], dtype=np.uint8).reshape(1, 4), (2, 1))
    gt_mesh = builder.add_mesh("gt_axis_cyan_closed_to_open", gt_line, gt_colors, glb.LINES)
    builder.add_node("gt_axis_cyan_closed_to_open", gt_mesh)
    builder.add_translation_animation("drawer_closed_to_open", drawer_node, t_closed_to_open)
    builder.write(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = fuse_state("start")
    end = fuse_state("end")
    gt_samples = load_gt_samples()

    start_dynamic_mask, start_d_static, start_d_dynamic = classify_static_dynamic(start["points"], gt_samples["start_static"], gt_samples["start_dynamic"])
    end_dynamic_mask, end_d_static, end_d_dynamic = classify_static_dynamic(end["points"], gt_samples["end_static"], gt_samples["end_dynamic"])
    start_static = start["points"][~start_dynamic_mask]
    start_dynamic = start["points"][start_dynamic_mask]
    end_static = end["points"][~end_dynamic_mask]
    end_dynamic = end["points"][end_dynamic_mask]

    start_static_colors = start["colors"][~start_dynamic_mask]
    start_dynamic_colors = start["colors"][start_dynamic_mask]
    end_static_colors = end["colors"][~end_dynamic_mask]
    end_dynamic_colors = end["colors"][end_dynamic_mask]

    residual, residual_diag = residual_from_base(start["points"], end_static)
    joint_est = est.estimate_from_residual("multiview_fused_visible_end_base", end_dynamic, residual)
    joint = make_joint_doc(joint_est, end_dynamic)

    trans = read_json(DATASET / "gt/trans.json")
    gt_axis = unit(trans["trans_info"]["axis"]["d"])
    gt_origin = np.asarray(trans["trans_info"]["axis"]["o"], dtype=np.float64)
    gt_q = float(trans["trans_info"]["translate"]["r"] - trans["trans_info"]["translate"]["l"])
    gt_translation = gt_axis * gt_q
    pred_axis = unit(joint["axis_closed_to_open_world"])
    pred_translation = np.asarray(joint["translation_closed_to_open_world"], dtype=np.float64)

    drawer_closed = end_dynamic + np.asarray(joint["translation_open_to_closed_world"], dtype=np.float64).reshape(1, 3)
    eval_report = {
        "fused_to_gt_surface": {
            "start_whole_to_gt_start": nn_stats(start["points"], gt_samples["start_whole"]),
            "gt_start_to_start_whole": nn_stats(gt_samples["start_whole"], start["points"]),
            "end_whole_to_gt_end": nn_stats(end["points"], gt_samples["end_whole"]),
            "gt_end_to_end_whole": nn_stats(gt_samples["end_whole"], end["points"]),
        },
        "cross_state_static_alignment_by_camera_poses": {
            "start_static_to_end_static": nn_stats(start_static, end_static),
            "end_static_to_start_static": nn_stats(end_static, start_static),
            "icp_start_static_to_end_static_correction": icp_alignment_check(start_static, end_static),
        },
        "joint_reconstruction_alignment": {
            "synth_closed_drawer_to_start_dynamic": chamfer_stats(drawer_closed, start_dynamic),
            "synth_closed_drawer_to_start_residual": chamfer_stats(drawer_closed, residual),
            "open_drawer_to_end_dynamic_self_check": chamfer_stats(end_dynamic, end_dynamic),
        },
        "joint_vs_sapien_reference": {
            "pred_axis_closed_to_open_world": pred_axis.astype(float).tolist(),
            "gt_axis_closed_to_open_world": gt_axis.astype(float).tolist(),
            "pred_origin_world": joint["origin_world"],
            "gt_origin_world": gt_origin.astype(float).tolist(),
            "pred_displacement_m": float(joint["displacement_m"]),
            "gt_displacement_m": gt_q,
            "pred_translation_closed_to_open_world": pred_translation.astype(float).tolist(),
            "gt_translation_closed_to_open_world": gt_translation.astype(float).tolist(),
            "axis_error": axis_angle(pred_axis, gt_axis),
            "displacement_error_m": float(joint["displacement_m"] - gt_q),
            "abs_displacement_error_m": float(abs(joint["displacement_m"] - gt_q)),
            "translation_vector_error_m": float(np.linalg.norm(pred_translation - gt_translation)),
            "pred_origin_to_gt_axis_perpendicular_m": point_to_line_distance(joint["origin_world"], gt_origin, gt_axis),
            "origin_note": "Prismatic axis origin is convention-dependent; this pipeline uses the median open drawer point as origin and evaluates it by perpendicular distance to the SAPIEN reference axis.",
        },
    }

    fused_glb = OUT / "multiview_fused_start_end_parts_splats.glb"
    recon_glb = OUT / "multiview_prismatic_reconstruction_animated.glb"
    build_fused_cloud_glb(fused_glb, start_static, start_dynamic, end_static, end_dynamic)
    build_reconstruction_glb(recon_glb, end_static, end_dynamic, residual, joint, gt_axis, gt_origin)

    joint_path = OUT / "joint_multiview_fused_visible_end_base.json"
    joint_path.write_text(json.dumps({"joint": joint, "source": "multiview fused RGB-D + residual analysis-by-synthesis"}, indent=2), encoding="utf-8")
    report = {
        "method": (
            "Fuse all available train RGB-D views into SAPIEN world coordinates using camera_train.json intrinsics/poses. "
            "Use fused end-state base/drawer surfaces as qpos1 geometry, use fused start-state whole-object surface as closed observation, "
            "estimate a prismatic joint from closed residuals, and evaluate against fused point clouds and SAPIEN gt/trans.json."
        ),
        "dataset": str(DATASET),
        "coordinate_frame": "SAPIEN/ArtGS world frame",
        "camera_convention": "world = camera_train[view] @ diag(1,-1,-1,1) @ point_camera_opencv",
        "inputs": {
            "start_fusion": start["stats"],
            "end_fusion": end["stats"],
        },
        "classification": {
            "note": "For this dataset probe, GT static/dynamic meshes classify fused visible points into base/drawer, simulating correct qpos1 masks and providing validation-only start dynamic labels.",
            "start_static_points": int(len(start_static)),
            "start_dynamic_points": int(len(start_dynamic)),
            "end_static_points": int(len(end_static)),
            "end_dynamic_points": int(len(end_dynamic)),
            "start_to_gt_static_dist_percentiles_m": percentiles(start_d_static),
            "start_to_gt_dynamic_dist_percentiles_m": percentiles(start_d_dynamic),
            "end_to_gt_static_dist_percentiles_m": percentiles(end_d_static),
            "end_to_gt_dynamic_dist_percentiles_m": percentiles(end_d_dynamic),
        },
        "residual": residual_diag,
        "joint_estimation": {
            "joint": joint,
            "best_candidate": joint_est["best"],
            "best_axis1d": joint_est["best_axis1d"],
            "plane_axis_candidates": joint_est["plane_axis_candidates"],
            "top_candidates": joint_est["top_candidates"][:15],
        },
        "evaluation": eval_report,
        "outputs": {
            "report": str(OUT / "multiview_reconstruction_alignment_report.json"),
            "joint_json": str(joint_path),
            "fused_parts_glb": str(fused_glb),
            "animated_reconstruction_glb": str(recon_glb),
        },
    }
    report_path = OUT / "multiview_reconstruction_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "joint_json": str(joint_path),
                "fused_parts_glb": str(fused_glb),
                "animated_reconstruction_glb": str(recon_glb),
                "joint_vs_sapien_reference": eval_report["joint_vs_sapien_reference"],
                "static_alignment": eval_report["cross_state_static_alignment_by_camera_poses"],
                "drawer_alignment": eval_report["joint_reconstruction_alignment"]["synth_closed_drawer_to_start_dynamic"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
