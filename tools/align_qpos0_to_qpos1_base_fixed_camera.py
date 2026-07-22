import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_fixed_camera_qpos0_compare as fixed
import build_rgbd_real_pointcloud_animation as anim


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_object_pose_aligned"
RNG = np.random.default_rng(20260716)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def rotation_about(axis, angle):
    axis = unit(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def rigid_transform(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = cb - r @ ca
    return r, t


def transform_points(points, r, t):
    return np.asarray(points, dtype=np.float64) @ r.T + np.asarray(t, dtype=np.float64).reshape(1, 3)


def sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points
    return points[RNG.choice(len(points), size=max_points, replace=False)]


def partial_icp_source_to_target_base(source_whole, target_base, init_r, init_t, iterations=48, trim=0.70, max_corr=0.11):
    # Target-driven partial ICP: each qpos1 base point picks its nearest transformed
    # qpos0 whole point. This lets the qpos0 drawer/front act as an outlier.
    source = sample_points(source_whole, 36000)
    target = sample_points(target_base, 30000)
    r = np.asarray(init_r, dtype=np.float64).copy()
    t = np.asarray(init_t, dtype=np.float64).reshape(3).copy()
    last_score = np.inf
    history = []
    for _ in range(iterations):
        moved = transform_points(source, r, t)
        tree = cKDTree(moved)
        dist, src_idx = tree.query(target, k=1, workers=-1)
        cutoff = min(float(np.percentile(dist, trim * 100.0)), max_corr)
        keep = dist <= cutoff
        if int(keep.sum()) < 500:
            keep = dist <= float(np.percentile(dist, 55.0))
        a = source[src_idx[keep]]
        b = target[keep]
        new_r, new_t = rigid_transform(a, b)
        moved_keep = transform_points(a, new_r, new_t)
        residual = np.linalg.norm(moved_keep - b, axis=1)
        score = float(np.median(residual) + 0.35 * np.percentile(residual, 90))
        history.append({"score": score, "median_m": float(np.median(residual)), "p90_m": float(np.percentile(residual, 90)), "pairs": int(len(a))})
        r, t = new_r, new_t
        if abs(last_score - score) < 1e-5:
            break
        last_score = score

    moved = transform_points(source, r, t)
    tree = cKDTree(moved)
    dist, _ = tree.query(target, k=1, workers=-1)
    metric = {
        "target_base_to_aligned_qpos0_percentiles_m": fixed.percentile_dict(dist),
        "target_base_points_scored": int(len(target)),
        "icp_iterations": int(len(history)),
        "last_iteration": history[-1] if history else {},
    }
    score = float(np.median(dist) + 0.35 * np.percentile(dist, 90))
    return {"r": r, "t": t, "score": score, "metric": metric, "history": history}


def estimate_object_pose(qpos0_whole, qpos1_base):
    source_center = np.median(qpos0_whole, axis=0)
    target_center = np.median(qpos1_base, axis=0)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    starts = []
    for deg in np.linspace(-50.0, 50.0, 11):
        r = rotation_about(up, np.deg2rad(deg))
        t = target_center - r @ source_center
        starts.append((f"camera_up_yaw_{deg:+.0f}", r, t))
    starts.append(("identity_centroid", np.eye(3), target_center - source_center))
    results = []
    for label, r, t in starts:
        result = partial_icp_source_to_target_base(qpos0_whole, qpos1_base, r, t)
        result["init_label"] = label
        results.append(result)
    return sorted(results, key=lambda x: x["score"])


def residual_after_base_alignment(qpos0_aligned, qpos1_base, threshold=0.045):
    base_tree = cKDTree(sample_points(qpos1_base, 50000))
    dist, _ = base_tree.query(qpos0_aligned, k=1, workers=-1)
    residual = qpos0_aligned[dist > threshold]
    if len(residual) < 400:
        residual = qpos0_aligned[dist > np.percentile(dist, 70)]
    return residual, dist


def score_q(residual, drawer_open, axis, q):
    drawer_closed = drawer_open - axis.reshape(1, 3) * float(q)
    residual_s = sample_points(residual, 30000)
    drawer_s = sample_points(drawer_closed, 22000)
    drawer_tree = cKDTree(drawer_s)
    res_to_drawer, _ = drawer_tree.query(residual_s, k=1, workers=-1)
    residual_tree = cKDTree(residual_s)
    drawer_to_res, _ = residual_tree.query(drawer_s, k=1, workers=-1)
    score = float(np.median(res_to_drawer) + 0.45 * np.percentile(res_to_drawer, 90) + 0.30 * np.median(drawer_to_res))
    origin = np.median(drawer_open, axis=0)
    axis_delta = float(np.median((drawer_closed - origin) @ axis) - np.median((residual_s - origin) @ axis))
    return {
        "q_open_m": float(q),
        "score": score,
        "qpos0_residual_to_closed_drawer_percentiles_m": fixed.percentile_dict(res_to_drawer),
        "closed_drawer_to_qpos0_residual_percentiles_m": fixed.percentile_dict(drawer_to_res),
        "signed_axis_delta_closed_drawer_minus_qpos0_residual_m": axis_delta,
    }


def estimate_q(residual, drawer_open, axis, old_q):
    qs = sorted(set([float(old_q)] + [float(x) for x in np.linspace(0.12, 0.46, 69)]))
    scored = [score_q(residual, drawer_open, axis, q) for q in qs]
    return sorted(scored, key=lambda x: x["score"])


def project_overlay_qpos1(path, qpos1_color, k, base_points, drawer_closed, qpos0_aligned):
    img = qpos1_color.copy()
    for points, bgr, limit in [
        (qpos0_aligned, (255, 210, 40), 50000),
        (base_points, (60, 220, 60), 45000),
        (drawer_closed, (40, 70, 245), 30000),
    ]:
        pts = sample_points(points, limit)
        uv, _ = fixed.est.project(pts, k)
        h, w = img.shape[:2]
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        for x, y in pix:
            cv2.circle(img, (int(x), int(y)), 1, bgr, -1, cv2.LINE_AA)
    cv2.putText(img, "qpos1 view: yellow=aligned qpos0 whole, green=base, red=synth closed drawer", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def project_overlay_qpos0(path, qpos0_color, k, qpos0_mask, target_closed_in_qpos0):
    img = qpos0_color.copy()
    contours, _ = cv2.findContours(qpos0_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    pts = sample_points(target_closed_in_qpos0, 65000)
    uv, _ = fixed.est.project(pts, k)
    h, w = img.shape[:2]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    pix = np.round(uv[inb]).astype(np.int32)
    for x, y in pix:
        cv2.circle(img, (int(x), int(y)), 1, (40, 70, 245), -1, cv2.LINE_AA)
    cv2.putText(img, "qpos0 view: white=qpos0 whole mask, red=qpos1 closed model after object-pose inverse", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def invert_transform(r, t):
    rinv = r.T
    tinv = -rinv @ t
    return rinv, tinv


def build_glb(path, mode, base_points, base_colors, drawer_closed, drawer_colors, qpos0_aligned, qpos0_colors, translation_closed_to_open):
    builder = anim.GlbBuilder(f"fixed-camera-object-pose-aligned-{mode}")
    if mode == "points":
        base_mesh = builder.add_mesh("qpos1_base_points", base_points, base_colors, anim.POINTS)
        drawer_mesh = builder.add_mesh("drawer_closed_points", drawer_closed, drawer_colors, anim.POINTS)
        q0_mesh = builder.add_mesh("aligned_qpos0_closed_whole_points", qpos0_aligned, qpos0_colors, anim.POINTS)
    else:
        bv, bc, bi = anim.make_splats(base_points, base_colors, half_size=0.002)
        dv, dc, di = anim.make_splats(drawer_closed, drawer_colors, half_size=0.002)
        qv, qc, qi = anim.make_splats(qpos0_aligned, qpos0_colors, half_size=0.0024)
        base_mesh = builder.add_mesh("qpos1_base_splats", bv, bc, anim.TRIANGLES, bi)
        drawer_mesh = builder.add_mesh("drawer_closed_splats", dv, dc, anim.TRIANGLES, di)
        q0_mesh = builder.add_mesh("aligned_qpos0_closed_whole_splats", qv, qc, anim.TRIANGLES, qi)
    axis_pos, axis_col = anim.make_axis_line(drawer_closed, translation_closed_to_open)
    axis_mesh = builder.add_mesh("axis_line_closed_to_open", axis_pos, axis_col, anim.LINES)
    builder.add_node("qpos1_base_static", base_mesh)
    drawer_node = builder.add_node("drawer_closed_animated_to_qpos1_open", drawer_mesh)
    builder.add_node("aligned_qpos0_closed_whole_static", q0_mesh)
    builder.add_node("axis_line_closed_to_open", axis_mesh)
    builder.add_translation_animation("drawer_closed_to_open_after_object_pose_alignment", drawer_node, translation_closed_to_open.astype(np.float32))
    builder.write(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint = read_json(fixed.JOINT_JSON)["joint"]
    axis = unit(joint["axis_camera_closed_to_open"])
    old_q = float(joint["displacement_m"])

    base_points, base_colors, base_stats = fixed.unproject_color_mask(fixed.QPOS1_CAPTURE, fixed.QPOS1_BASE_MASK, max_points=90000)
    drawer_points, drawer_colors, drawer_stats = fixed.unproject_color_mask(fixed.QPOS1_CAPTURE, fixed.QPOS1_DRAWER_MASK, max_points=70000)
    qpos0_points, qpos0_colors, qpos0_stats = fixed.unproject_color_mask(fixed.QPOS0_CAPTURE, fixed.QPOS0_WHOLE_MASK, max_points=90000)
    qpos0_colors = fixed.tint_rgba(qpos0_colors, [80, 200, 255], alpha=0.55)

    base_points, base_colors = fixed.voxel_downsample_with_colors(base_points, base_colors, voxel=0.0035, max_points=65000)
    drawer_points, drawer_colors = fixed.voxel_downsample_with_colors(drawer_points, drawer_colors, voxel=0.0035, max_points=45000)
    qpos0_points, qpos0_colors = fixed.voxel_downsample_with_colors(qpos0_points, qpos0_colors, voxel=0.0035, max_points=65000)

    pose_candidates = estimate_object_pose(qpos0_points, base_points)
    pose = pose_candidates[0]
    r = pose["r"]
    t = pose["t"]
    qpos0_aligned = transform_points(qpos0_points, r, t).astype(np.float32)
    qpos0_residual, qpos0_to_base_dist = residual_after_base_alignment(qpos0_aligned, base_points)
    q_scores = estimate_q(qpos0_residual, drawer_points.astype(np.float64), axis, old_q)
    best_q = float(q_scores[0]["q_open_m"])

    outputs = []
    qpos1_color = fixed.load_color(fixed.QPOS1_CAPTURE)
    qpos1_k = fixed.est.load_k(fixed.QPOS1_CAPTURE)
    qpos0_color = fixed.load_color(fixed.QPOS0_CAPTURE)
    qpos0_k = fixed.est.load_k(fixed.QPOS0_CAPTURE)
    qpos0_mask = fixed.est.load_mask(fixed.QPOS0_WHOLE_MASK)
    rinv, tinv = invert_transform(r, t)
    for label, q in [("oldq", old_q), ("base_icp_residual_bestq", best_q)]:
        translation_open_to_closed = -axis.astype(np.float32) * np.float32(q)
        translation_closed_to_open = axis.astype(np.float32) * np.float32(q)
        drawer_closed = drawer_points + translation_open_to_closed.reshape(1, 3)
        target_closed = np.vstack([base_points, drawer_closed]).astype(np.float32)
        target_closed_qpos0 = transform_points(target_closed, rinv, tinv)
        qpos1_overlay = OUT / f"qpos1_overlay_aligned_qpos0_{label}.png"
        qpos0_overlay = OUT / f"qpos0_overlay_inverse_projected_closed_model_{label}.png"
        project_overlay_qpos1(qpos1_overlay, qpos1_color, qpos1_k, base_points, drawer_closed, qpos0_aligned)
        project_overlay_qpos0(qpos0_overlay, qpos0_color, qpos0_k, qpos0_mask, target_closed_qpos0)
        bp, bc = fixed.sample_pair(base_points, base_colors, 22000)
        dp, dc = fixed.sample_pair(drawer_closed, drawer_colors, 16000)
        qp, qc = fixed.sample_pair(qpos0_aligned, qpos0_colors, 28000)
        points_path = OUT / f"cabinet_drawer_object_pose_aligned_{label}_points_animated.glb"
        splats_path = OUT / f"cabinet_drawer_object_pose_aligned_{label}_splats_animated.glb"
        build_glb(points_path, "points", base_points, base_colors, drawer_closed, drawer_colors, qpos0_aligned, qpos0_colors, translation_closed_to_open)
        build_glb(splats_path, "splats", bp, bc, dp, dc, qp, qc, translation_closed_to_open)
        outputs.append(
            {
                "label": label,
                "q_open_m": float(q),
                "q_score": score_q(qpos0_residual, drawer_points.astype(np.float64), axis, q),
                "animated_points_glb": str(points_path),
                "animated_splats_glb": str(splats_path),
                "qpos1_overlay": str(qpos1_overlay),
                "qpos0_overlay": str(qpos0_overlay),
            }
        )

    report = {
        "method": "Fixed-camera object-pose alignment. Camera poses and SfM are ignored. A rigid object-pose transform from qpos0 closed whole-cabinet RGB-D points to qpos1 base RGB-D points is estimated with target-driven partial ICP, treating qpos0 drawer/front points as outliers. Then q is scored from qpos0 residual points not explained by qpos1 base.",
        "frame": "qpos1 OpenCV/PV camera frame after applying object pose transform to qpos0 points",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "inputs": {
            "qpos0_whole": qpos0_stats,
            "qpos1_base": base_stats,
            "qpos1_drawer": drawer_stats,
            "closed_partmask_used": False,
            "sfm_used": False,
            "raw_camera_pose_used": False,
        },
        "object_pose_qpos0_to_qpos1": {
            "rotation_matrix": r.astype(float).tolist(),
            "translation_m": t.astype(float).tolist(),
            "best_init_label": pose["init_label"],
            "score": float(pose["score"]),
            "metric": pose["metric"],
            "top_pose_candidates": [
                {
                    "rank": i + 1,
                    "init_label": item["init_label"],
                    "score": float(item["score"]),
                    "metric": item["metric"],
                }
                for i, item in enumerate(pose_candidates[:6])
            ],
        },
        "base_alignment_residual": {
            "qpos0_to_qpos1_base_percentiles_m": fixed.percentile_dict(qpos0_to_base_dist),
            "qpos0_residual_points_after_base_removal": int(len(qpos0_residual)),
            "base_removal_threshold_m": 0.045,
        },
        "joint_axis_kept_fixed": {
            "axis_camera_closed_to_open": axis.astype(float).tolist(),
            "old_q_m": old_q,
            "best_residual_q_m": best_q,
            "q_identifiability_note": "Once object pose is free, global translation along the drawer axis and drawer displacement are partially coupled. The residual q score is diagnostic unless base alignment is visually validated.",
        },
        "q_sweep_top15": q_scores[:15],
        "outputs": outputs,
    }
    report_path = OUT / "object_pose_aligned_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
