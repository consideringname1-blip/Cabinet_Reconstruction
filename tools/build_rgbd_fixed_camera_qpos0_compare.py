import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_real_pointcloud_animation as anim
import estimate_prismatic_rgbd_synthesis as est


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_qpos0_compare"
QPOS0_CAPTURE = "20260622_080738_525030Z"
QPOS1_CAPTURE = "20260622_081031_636398Z"
QPOS0_WHOLE_MASK = ROOT / "data/upload/larm/20260622_joint_0_white_outline_sfm/sam3/20260622_080738_525030Z_sam3_cabinet_mask.png"
QPOS1_BASE_MASK = ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png"
QPOS1_DRAWER_MASK = ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
RNG = np.random.default_rng(20260715)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_color(capture):
    color = cv2.imread(str(ROOT / f"data/upload/larm_captures/{capture}/color.png"), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(capture)
    return color


def rgba_from_bgr(color_bgr, ys, xs):
    rgb = color_bgr[ys, xs][:, ::-1]
    return np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)]).astype(np.uint8)


def tint_rgba(colors, tint_rgb, alpha=0.45):
    tint = np.asarray(tint_rgb, dtype=np.float32).reshape(1, 3)
    rgb = colors[:, :3].astype(np.float32)
    out = np.clip((1.0 - alpha) * rgb + alpha * tint, 0, 255).astype(np.uint8)
    return np.column_stack([out, colors[:, 3]]).astype(np.uint8)


def unproject_color_mask(capture, mask_path, max_points=None, erode=True):
    k = est.load_k(capture)
    depth = est.load_depth_m(capture)
    color = load_color(capture)
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
    colors = rgba_from_bgr(color, ys, xs)
    return points, colors, {"capture": capture, "mask": str(mask_path), "valid_pixels": int(valid.sum()), "exported_points": int(len(points))}


def voxel_downsample_with_colors(points, colors, voxel=0.004, max_points=None):
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


def percentile_dict(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in [5, 10, 25, 50, 75, 90, 95]}


def metric_points(points, voxel=0.006, max_points=60000):
    return est.voxel_downsample(np.asarray(points, dtype=np.float64), voxel=voxel, max_points=max_points)


def compare(base_points, drawer_points, qpos0_points, axis, q):
    base_m = metric_points(base_points, voxel=0.006, max_points=50000)
    drawer_open_m = metric_points(drawer_points, voxel=0.006, max_points=40000)
    qpos0_m = metric_points(qpos0_points, voxel=0.006, max_points=60000)
    drawer_closed_m = drawer_open_m - axis.reshape(1, 3) * float(q)

    base_tree = cKDTree(base_m)
    q0_to_base, _ = base_tree.query(qpos0_m, k=1, workers=-1)
    residual = qpos0_m[q0_to_base > 0.045]
    if len(residual) < 200:
        residual = qpos0_m[q0_to_base > np.percentile(q0_to_base, 70)]
    drawer_tree = cKDTree(drawer_closed_m)
    residual_to_drawer, _ = drawer_tree.query(residual, k=1, workers=-1)
    q0_tree = cKDTree(qpos0_m)
    drawer_to_q0, _ = q0_tree.query(drawer_closed_m, k=1, workers=-1)

    origin = np.median(drawer_open_m, axis=0)
    residual_axis = (residual - origin.reshape(1, 3)) @ axis
    drawer_axis = (drawer_closed_m - origin.reshape(1, 3)) @ axis
    axis_delta = float(np.median(drawer_axis) - np.median(residual_axis)) if len(residual) else float("nan")

    return {
        "q_open_m": float(q),
        "qpos0_to_qpos1_camera_transform": "identity_fixed_camera_assumption",
        "qpos0_to_base_percentiles_m": percentile_dict(q0_to_base),
        "qpos0_residual_points": int(len(residual)),
        "qpos0_residual_to_closed_drawer_percentiles_m": percentile_dict(residual_to_drawer),
        "closed_drawer_to_qpos0_percentiles_m": percentile_dict(drawer_to_q0),
        "signed_axis_delta_closed_drawer_minus_qpos0_residual_m": axis_delta,
        "signed_axis_delta_note": "axis is closed->open. Negative means synthesized closed drawer median is farther in the closed direction than qpos0 residual.",
    }


def render_depth_zbuffer(points, k, width, height, radius=2):
    depth = np.full((height, width), np.inf, dtype=np.float32)
    uv, valid_points = est.project(points, k)
    if len(uv) == 0:
        return depth
    z = valid_points[:, 2].astype(np.float32)
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    for dy in range(-radius, radius + 1):
        yy = py + dy
        ok_y = (yy >= 0) & (yy < height)
        for dx in range(-radius, radius + 1):
            xx = px + dx
            ok = ok_y & (xx >= 0) & (xx < width)
            if not np.any(ok):
                continue
            yy_ok = yy[ok]
            xx_ok = xx[ok]
            z_ok = z[ok]
            old = depth[yy_ok, xx_ok]
            closer = z_ok < old
            if np.any(closer):
                depth[yy_ok[closer], xx_ok[closer]] = z_ok[closer]
    return depth


def depth_metrics(drawer_points, axis, q):
    depth = est.load_depth_m(QPOS0_CAPTURE)
    color = load_color(QPOS0_CAPTURE)
    mask = est.load_mask(QPOS0_WHOLE_MASK)
    k = est.load_k(QPOS0_CAPTURE)
    h, w = mask.shape
    drawer_closed = drawer_points - axis.reshape(1, 3) * float(q)
    pred_depth = render_depth_zbuffer(drawer_closed, k, w, h, radius=2)
    pred = np.isfinite(pred_depth)
    target = mask & (depth > 0.2) & (depth < 4.0)
    overlap = pred & target
    diff = pred_depth[overlap] - depth[overlap]

    img = color.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    uv, _ = est.project(drawer_closed, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    pix = np.round(uv[inb]).astype(np.int32)
    if len(pix) > 40000:
        pix = pix[RNG.choice(len(pix), size=40000, replace=False)]
    for x, y in pix:
        cv2.circle(img, (int(x), int(y)), 1, (40, 60, 245), -1, cv2.LINE_AA)
    cv2.putText(img, f"fixed-camera q={q:.3f}m white=qpos0 whole mask red=synth closed drawer", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    path = OUT / f"fixed_camera_qpos0_projection_q{q:.3f}.png"
    cv2.imwrite(str(path), img)

    return {
        "q_open_m": float(q),
        "overlap_pixels": int(overlap.sum()),
        "target_mask_pixels": int(target.sum()),
        "drawer_projected_pixels": int(pred.sum()),
        "drawer_overlap_target_ratio": float(overlap.sum() / max(1, target.sum())),
        "drawer_depth_minus_qpos0_depth_percentiles_m": percentile_dict(diff),
        "signed_median_m": float(np.median(diff)) if len(diff) else float("nan"),
        "abs_median_m": float(np.median(np.abs(diff))) if len(diff) else float("nan"),
        "behind_ratio_gt_3cm": float(np.mean(diff > 0.03)) if len(diff) else float("nan"),
        "front_ratio_gt_3cm": float(np.mean(diff < -0.03)) if len(diff) else float("nan"),
        "projection_overlay": str(path),
    }


def make_axis_line(points, translation):
    return anim.make_axis_line(points, translation)


def build_glb(path, mode, base_points, base_colors, drawer_closed, drawer_colors, qpos0_points, qpos0_colors, translation_closed_to_open):
    builder = anim.GlbBuilder(f"fixed-camera-qpos0-compare-{mode}")
    if mode == "points":
        base_mesh = builder.add_mesh("qpos1_base_points", base_points, base_colors, anim.POINTS)
        drawer_mesh = builder.add_mesh("synth_closed_drawer_points", drawer_closed, drawer_colors, anim.POINTS)
        qpos0_mesh = builder.add_mesh("qpos0_closed_whole_reference_points_identity", qpos0_points, qpos0_colors, anim.POINTS)
    else:
        bv, bc, bi = anim.make_splats(base_points, base_colors, half_size=0.002)
        dv, dc, di = anim.make_splats(drawer_closed, drawer_colors, half_size=0.002)
        qv, qc, qi = anim.make_splats(qpos0_points, qpos0_colors, half_size=0.0024)
        base_mesh = builder.add_mesh("qpos1_base_splats", bv, bc, anim.TRIANGLES, bi)
        drawer_mesh = builder.add_mesh("synth_closed_drawer_splats", dv, dc, anim.TRIANGLES, di)
        qpos0_mesh = builder.add_mesh("qpos0_closed_whole_reference_splats_identity", qv, qc, anim.TRIANGLES, qi)
    axis_pos, axis_col = make_axis_line(drawer_closed, translation_closed_to_open)
    axis_mesh = builder.add_mesh("axis_line_closed_to_open", axis_pos, axis_col, anim.LINES)
    builder.add_node("qpos1_base_static", base_mesh)
    drawer_node = builder.add_node("synth_drawer_closed_animated_to_qpos1_open", drawer_mesh)
    builder.add_node("qpos0_closed_whole_reference_identity_static", qpos0_mesh)
    builder.add_node("axis_line_closed_to_open", axis_mesh)
    builder.add_translation_animation("drawer_closed_to_open_fixed_camera_direct", drawer_node, translation_closed_to_open.astype(np.float32))
    builder.write(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint = read_json(JOINT_JSON)["joint"]
    axis = anim.unit(joint["axis_camera_closed_to_open"]).astype(np.float64)
    q = float(joint["displacement_m"])
    t_open_to_closed = -axis.astype(np.float32) * np.float32(q)
    t_closed_to_open = axis.astype(np.float32) * np.float32(q)

    base_points, base_colors, base_stats = unproject_color_mask(QPOS1_CAPTURE, QPOS1_BASE_MASK, max_points=80000)
    drawer_points, drawer_colors, drawer_stats = unproject_color_mask(QPOS1_CAPTURE, QPOS1_DRAWER_MASK, max_points=60000)
    qpos0_points, qpos0_colors, qpos0_stats = unproject_color_mask(QPOS0_CAPTURE, QPOS0_WHOLE_MASK, max_points=80000)
    qpos0_colors = tint_rgba(qpos0_colors, [80, 200, 255], alpha=0.55)

    base_points, base_colors = voxel_downsample_with_colors(base_points, base_colors, voxel=0.0035, max_points=60000)
    drawer_points, drawer_colors = voxel_downsample_with_colors(drawer_points, drawer_colors, voxel=0.0035, max_points=40000)
    qpos0_points, qpos0_colors = voxel_downsample_with_colors(qpos0_points, qpos0_colors, voxel=0.0035, max_points=60000)
    drawer_closed = drawer_points + t_open_to_closed.reshape(1, 3)

    points_path = OUT / "cabinet_drawer_fixed_camera_qpos0_identity_points_animated.glb"
    splats_path = OUT / "cabinet_drawer_fixed_camera_qpos0_identity_splats_animated.glb"
    build_glb(points_path, "points", base_points, base_colors, drawer_closed, drawer_colors, qpos0_points, qpos0_colors, t_closed_to_open)
    bs, bcs = sample_pair(base_points, base_colors, 22000)
    ds, dcs = sample_pair(drawer_closed, drawer_colors, 16000)
    qs, qcs = sample_pair(qpos0_points, qpos0_colors, 28000)
    build_glb(splats_path, "splats", bs, bcs, ds, dcs, qs, qcs, t_closed_to_open)

    report = {
        "method": "Fixed-camera qpos0/qpos1 comparison. No SfM, no raw camera pose, no qpos0 part mask. qpos0 closed whole-cabinet RGB-D points and qpos1 open base/drawer RGB-D points are interpreted in the same OpenCV/PV camera frame.",
        "invalidated_assumption": "Previous SfM assumed a static cabinet and moving cameras. User clarified the camera was fixed and the cabinet/object was moved instead.",
        "frame": "shared OpenCV/PV camera frame under fixed-camera assumption",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "qpos0_reference": {**qpos0_stats, "points_after_downsample": int(len(qpos0_points)), "closed_partmask_used": False},
        "qpos1_parts": {
            "base": {**base_stats, "points_after_downsample": int(len(base_points))},
            "drawer": {**drawer_stats, "points_after_downsample": int(len(drawer_points))},
        },
        "joint_kept_fixed": {
            "axis_camera_closed_to_open": axis.astype(float).tolist(),
            "q_open_m": q,
            "translation_open_to_closed_camera_m": t_open_to_closed.astype(float).tolist(),
            "translation_closed_to_open_camera_m": t_closed_to_open.astype(float).tolist(),
        },
        "nearest_neighbor_metrics": compare(base_points, drawer_points, qpos0_points, axis, q),
        "qpos0_projection_depth_metrics": depth_metrics(drawer_points, axis, q),
        "outputs": {
            "animated_points_glb": str(points_path),
            "animated_splats_glb": str(splats_path),
        },
    }
    report_path = OUT / "fixed_camera_qpos0_compare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
