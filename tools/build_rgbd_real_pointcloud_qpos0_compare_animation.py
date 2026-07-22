import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_real_pointcloud_animation as anim
import estimate_prismatic_rgbd_synthesis as est


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_real_pointcloud_qpos0_compare"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
REFINE_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/fixed_axis_q_zbuffer_refine/fixed_axis_q_zbuffer_refine_report.json"
RNG = np.random.default_rng(20260713)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgba_from_bgr(color_bgr, ys, xs):
    rgb = color_bgr[ys, xs][:, ::-1]
    return np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)]).astype(np.uint8)


def tint_rgba(colors, tint_rgb, alpha=0.45):
    colors = np.asarray(colors, dtype=np.uint8)
    tint = np.asarray(tint_rgb, dtype=np.float32).reshape(1, 3)
    rgb = colors[:, :3].astype(np.float32)
    out = np.clip((1.0 - alpha) * rgb + alpha * tint, 0, 255).astype(np.uint8)
    return np.column_stack([out, colors[:, 3]]).astype(np.uint8)


def unproject_color_mask(capture, mask_path, max_points=None, erode=True):
    k = est.load_k(capture)
    depth = est.load_depth_m(capture)
    color_bgr = est.load_color(capture)
    mask = est.load_mask(mask_path)
    valid = mask & (depth > 0.2) & (depth < 4.0)
    if erode:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    ys, xs = np.nonzero(valid)
    if max_points is not None and len(xs) > max_points:
        idx = RNG.choice(len(xs), size=max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = rgba_from_bgr(color_bgr, ys, xs)
    return points, colors, {"capture": capture, "mask": str(mask_path), "valid_pixels": int(valid.sum()), "exported_points": int(len(points))}


def voxel_downsample_with_colors(points, colors, voxel=0.006, max_points=None):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
    if len(points) == 0:
        return points, colors
    if voxel > 0:
        keys = np.floor(points.astype(np.float64) / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx)
        points = points[idx]
        colors = colors[idx]
    if max_points is not None and len(points) > max_points:
        idx = RNG.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float32), colors.astype(np.uint8)


def sample_pair(points, colors, max_points):
    if len(points) <= max_points:
        return points, colors
    idx = RNG.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def load_qpos1_parts():
    base_points, base_colors, base_stats = unproject_color_mask(
        est.QPOS1_CAPTURE,
        est.QPOS1_BASE_MASK,
        max_points=90000,
    )
    drawer_points, drawer_colors, drawer_stats = unproject_color_mask(
        est.QPOS1_CAPTURE,
        est.QPOS1_DRAWER_MASK,
        max_points=70000,
    )
    base_points, base_colors = voxel_downsample_with_colors(base_points, base_colors, voxel=0.0035, max_points=70000)
    drawer_points, drawer_colors = voxel_downsample_with_colors(drawer_points, drawer_colors, voxel=0.0035, max_points=50000)
    return base_points, base_colors, base_stats, drawer_points, drawer_colors, drawer_stats


def load_qpos0_closed_reference():
    meta = est.read_json(est.POSE_METADATA)
    qpos1_cam_to_world = est.sfm_camera_to_world(est.QPOS1_CAPTURE)
    world_to_qpos1 = np.linalg.inv(qpos1_cam_to_world)
    points_all = []
    colors_all = []
    stats = []
    view_tints = [
        np.array([60, 210, 255], dtype=np.uint8),
        np.array([120, 150, 255], dtype=np.uint8),
        np.array([190, 120, 255], dtype=np.uint8),
    ]
    view_idx = 0
    for rec in meta["grouped_captures"]:
        if float(rec["qpos"]) != 0.0:
            continue
        capture = Path(rec["capture"]).name
        mask_path = Path(rec["mask_path"])
        pts_cam, colors, item = unproject_color_mask(capture, mask_path, max_points=65000)
        pts_qpos1 = est.transform_points(pts_cam, world_to_qpos1 @ est.sfm_camera_to_world(capture)).astype(np.float32)
        colors = tint_rgba(colors, view_tints[view_idx % len(view_tints)], alpha=0.50)
        points_all.append(pts_qpos1)
        colors_all.append(colors)
        item["transformed_to"] = "qpos1 OpenCV/PV camera frame"
        item["tint_rgb"] = view_tints[view_idx % len(view_tints)].astype(int).tolist()
        stats.append(item)
        view_idx += 1
    if not points_all:
        raise RuntimeError("No qpos0 closed whole-cabinet views found in SfM metadata")
    points = np.vstack(points_all).astype(np.float32)
    colors = np.vstack(colors_all).astype(np.uint8)
    points, colors = voxel_downsample_with_colors(points, colors, voxel=0.0045, max_points=90000)
    return points, colors, stats


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in [5, 10, 25, 50, 75, 90, 95]}


def metric_points(points, voxel=0.006, max_points=50000):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = est.voxel_downsample(points, voxel=voxel, max_points=max_points)
    return points.astype(np.float64)


def compare_closed_to_qpos0(base_points, drawer_open_points, qpos0_points, axis, q, residual_threshold=0.045):
    base_metric = metric_points(base_points, voxel=0.006, max_points=50000)
    drawer_open_metric = metric_points(drawer_open_points, voxel=0.006, max_points=40000)
    qpos0_metric = metric_points(qpos0_points, voxel=0.006, max_points=70000)
    drawer_closed = drawer_open_metric - axis.reshape(1, 3) * float(q)

    base_tree = cKDTree(base_metric)
    dist_qpos0_to_base, _ = base_tree.query(qpos0_metric, k=1, workers=-1)
    residual_mask = dist_qpos0_to_base > residual_threshold
    qpos0_residual = qpos0_metric[residual_mask]
    if len(qpos0_residual) < 200:
        fallback_threshold = float(np.percentile(dist_qpos0_to_base, 70))
        residual_mask = dist_qpos0_to_base > fallback_threshold
        qpos0_residual = qpos0_metric[residual_mask]
    else:
        fallback_threshold = residual_threshold

    combined_tree = cKDTree(np.vstack([base_metric, drawer_closed]))
    drawer_tree = cKDTree(drawer_closed)
    qpos0_tree = cKDTree(qpos0_metric)
    dist_qpos0_to_combined, _ = combined_tree.query(qpos0_metric, k=1, workers=-1)
    dist_residual_to_drawer, _ = drawer_tree.query(qpos0_residual, k=1, workers=-1)
    dist_drawer_to_qpos0, _ = qpos0_tree.query(drawer_closed, k=1, workers=-1)

    if len(qpos0_residual):
        origin = np.median(drawer_open_metric, axis=0)
        qpos0_axis = (qpos0_residual - origin.reshape(1, 3)) @ axis
        closed_axis = (drawer_closed - origin.reshape(1, 3)) @ axis
        axis_delta = float(np.median(closed_axis) - np.median(qpos0_axis))
    else:
        axis_delta = float("nan")

    return {
        "q_open_m": float(q),
        "translation_open_to_closed_camera_m": (-axis * float(q)).astype(float).tolist(),
        "residual_threshold_m": float(residual_threshold),
        "effective_residual_threshold_m": float(fallback_threshold),
        "base_explained_qpos0_ratio": float(np.mean(dist_qpos0_to_base <= residual_threshold)),
        "qpos0_metric_points": int(len(qpos0_metric)),
        "qpos0_residual_points": int(len(qpos0_residual)),
        "qpos0_to_base_percentiles_m": percentiles(dist_qpos0_to_base),
        "qpos0_to_synth_base_plus_drawer_percentiles_m": percentiles(dist_qpos0_to_combined),
        "qpos0_residual_to_closed_drawer_percentiles_m": percentiles(dist_residual_to_drawer),
        "closed_drawer_to_qpos0_percentiles_m": percentiles(dist_drawer_to_qpos0),
        "signed_axis_delta_closed_drawer_minus_qpos0_residual_m": axis_delta,
        "signed_axis_delta_note": "axis is closed->open. Negative means synthesized closed drawer median is farther in the closed/deeper direction than qpos0 residual.",
    }


def sweep_q(base_points, drawer_open_points, qpos0_points, axis):
    qs = np.linspace(0.18, 0.42, 49)
    rows = []
    for q in qs:
        metrics = compare_closed_to_qpos0(base_points, drawer_open_points, qpos0_points, axis, float(q))
        res = metrics["qpos0_residual_to_closed_drawer_percentiles_m"]
        drw = metrics["closed_drawer_to_qpos0_percentiles_m"]
        score = res.get("50", 9.0) + 0.4 * res.get("90", 9.0) + 0.25 * drw.get("50", 9.0)
        rows.append(
            {
                "q_open_m": float(q),
                "score": float(score),
                "residual_to_drawer_median_m": float(res.get("50", np.nan)),
                "residual_to_drawer_p90_m": float(res.get("90", np.nan)),
                "drawer_to_qpos0_median_m": float(drw.get("50", np.nan)),
                "signed_axis_delta_m": metrics["signed_axis_delta_closed_drawer_minus_qpos0_residual_m"],
                "qpos0_residual_points": metrics["qpos0_residual_points"],
            }
        )
    return sorted(rows, key=lambda x: x["score"])


def make_axis_line_for_q(drawer_closed_points, translation_closed_to_open):
    return anim.make_axis_line(drawer_closed_points, translation_closed_to_open)


def build_compare_glb(
    path,
    mode_name,
    base_points,
    base_colors,
    drawer_closed_points,
    drawer_colors,
    qpos0_points,
    qpos0_colors,
    translation_closed_to_open,
):
    builder = anim.GlbBuilder(f"rgbd-real-pointcloud-qpos0-compare-{mode_name}")
    if mode_name == "points":
        base_mesh = builder.add_mesh("qpos1_base_real_points", base_points, base_colors, anim.POINTS)
        drawer_mesh = builder.add_mesh("synth_drawer_closed_real_points", drawer_closed_points, drawer_colors, anim.POINTS)
        qpos0_mesh = builder.add_mesh("qpos0_closed_whole_reference_points", qpos0_points, qpos0_colors, anim.POINTS)
    elif mode_name == "splats":
        b_v, b_c, b_i = anim.make_splats(base_points, base_colors, half_size=0.0020)
        d_v, d_c, d_i = anim.make_splats(drawer_closed_points, drawer_colors, half_size=0.0020)
        q_v, q_c, q_i = anim.make_splats(qpos0_points, qpos0_colors, half_size=0.0024)
        base_mesh = builder.add_mesh("qpos1_base_real_splats", b_v, b_c, anim.TRIANGLES, b_i)
        drawer_mesh = builder.add_mesh("synth_drawer_closed_real_splats", d_v, d_c, anim.TRIANGLES, d_i)
        qpos0_mesh = builder.add_mesh("qpos0_closed_whole_reference_splats", q_v, q_c, anim.TRIANGLES, q_i)
    else:
        raise ValueError(mode_name)

    axis_pos, axis_colors = make_axis_line_for_q(drawer_closed_points, translation_closed_to_open)
    axis_mesh = builder.add_mesh("axis_line_closed_to_open", axis_pos, axis_colors, anim.LINES)
    builder.add_node("qpos1_base_static", base_mesh)
    drawer_node = builder.add_node("synth_drawer_closed_animated_to_qpos1_open", drawer_mesh)
    builder.add_node("qpos0_closed_whole_reference_static", qpos0_mesh)
    builder.add_node("axis_line_closed_to_open", axis_mesh)
    builder.add_translation_animation(
        "drawer_closed_to_open_camera_frame_direct",
        drawer_node,
        np.asarray(translation_closed_to_open, dtype=np.float32),
    )
    builder.write(path)


def build_outputs_for_q(label, q, axis, base_points, base_colors, drawer_open_points, drawer_colors, qpos0_points, qpos0_colors):
    translation_open_to_closed = -axis.astype(np.float32) * np.float32(q)
    translation_closed_to_open = axis.astype(np.float32) * np.float32(q)
    drawer_closed_points = drawer_open_points + translation_open_to_closed.reshape(1, 3)

    points_path = OUT / f"cabinet_drawer_rgbd_points_qpos0_compare_{label}_animated.glb"
    splats_path = OUT / f"cabinet_drawer_rgbd_splats_qpos0_compare_{label}_animated.glb"
    build_compare_glb(
        points_path,
        "points",
        base_points,
        base_colors,
        drawer_closed_points,
        drawer_colors,
        qpos0_points,
        qpos0_colors,
        translation_closed_to_open,
    )

    base_splat_points, base_splat_colors = sample_pair(base_points, base_colors, 22000)
    drawer_splat_points, drawer_splat_colors = sample_pair(drawer_closed_points, drawer_colors, 16000)
    qpos0_splat_points, qpos0_splat_colors = sample_pair(qpos0_points, qpos0_colors, 32000)
    build_compare_glb(
        splats_path,
        "splats",
        base_splat_points,
        base_splat_colors,
        drawer_splat_points,
        drawer_splat_colors,
        qpos0_splat_points,
        qpos0_splat_colors,
        translation_closed_to_open,
    )

    metrics = compare_closed_to_qpos0(base_points, drawer_open_points, qpos0_points, axis, q)
    return {
        "label": label,
        "q_open_m": float(q),
        "translation_open_to_closed_camera_m": translation_open_to_closed.astype(float).tolist(),
        "translation_closed_to_open_camera_m": translation_closed_to_open.astype(float).tolist(),
        "outputs": {
            "animated_points_glb": str(points_path),
            "animated_splats_glb": str(splats_path),
        },
        "metrics": metrics,
    }


def load_q_values(joint):
    out = {"oldq": float(joint["displacement_m"])}
    if REFINE_JSON.exists():
        refine = read_json(REFINE_JSON)
        out["zbuffer_bestq"] = float(refine["best_q_m"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint_doc = read_json(JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    axis = anim.unit(joint["axis_camera_closed_to_open"]).astype(np.float64)
    q_values = load_q_values(joint)

    base_points, base_colors, base_stats, drawer_points, drawer_colors, drawer_stats = load_qpos1_parts()
    qpos0_points, qpos0_colors, qpos0_stats = load_qpos0_closed_reference()

    q_sweep = sweep_q(base_points, drawer_points, qpos0_points, axis)
    outputs = []
    for label, q in q_values.items():
        outputs.append(build_outputs_for_q(label, q, axis, base_points, base_colors, drawer_points, drawer_colors, qpos0_points, qpos0_colors))
    nn_best = q_sweep[0]["q_open_m"]
    if all(abs(nn_best - item["q_open_m"]) > 0.006 for item in outputs):
        outputs.append(build_outputs_for_q("residual_nn_bestq", nn_best, axis, base_points, base_colors, drawer_points, drawer_colors, qpos0_points, qpos0_colors))

    report = {
        "method": "Animated real RGB-D point cloud with qpos0 whole-cabinet point cloud reference. No qpos0 part mask is used. qpos0 closed RGB-D points are transformed into the qpos1 OpenCV/PV camera frame using the SfM camera poses, then shown as a static tinted reference while the synthesized drawer moves from closed to open.",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "joint_axis_kept_fixed": {
            "axis_camera_closed_to_open": axis.astype(float).tolist(),
            "source_json": str(JOINT_JSON),
        },
        "qpos0_reference": {
            "source": "qpos0 whole-cabinet SAM3 mask + aligned depth + SfM qpos0->qpos1 camera transforms",
            "closed_partmask_used": False,
            "merged_points": int(len(qpos0_points)),
            "views": qpos0_stats,
        },
        "qpos1_parts": {
            "base": {**base_stats, "points_after_downsample": int(len(base_points))},
            "drawer": {**drawer_stats, "points_after_downsample": int(len(drawer_points))},
        },
        "distance_metric_note": "qpos0 residual is computed by removing qpos0 points already explained by qpos1 base within 4.5 cm. This is only a diagnostic because the visible surface changes between open and closed.",
        "fixed_q_outputs": outputs,
        "residual_nn_q_sweep_top10": q_sweep[:10],
    }
    report_path = OUT / "rgbd_real_pointcloud_qpos0_compare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "outputs": [item["outputs"] for item in outputs],
                "q_sweep_best": q_sweep[0],
                "qpos0_points": int(len(qpos0_points)),
                "base_points": int(len(base_points)),
                "drawer_points": int(len(drawer_points)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
