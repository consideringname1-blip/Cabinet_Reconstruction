import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_real_pointcloud_qpos0_compare_animation as cmp
import estimate_prismatic_rgbd_synthesis as est


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_real_pointcloud_qpos0_compare"
REPORT = OUT / "rgbd_real_pointcloud_qpos0_compare_report.json"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
RNG = np.random.default_rng(20260714)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def draw_points(img, points_view, k, bgr, radius=1, max_points=45000):
    points = np.asarray(points_view, dtype=np.float64).reshape(-1, 3)
    if len(points) > max_points:
        points = points[RNG.choice(len(points), size=max_points, replace=False)]
    uv, _ = est.project(points, k)
    h, w = img.shape[:2]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    pix = np.round(uv[inb]).astype(np.int32)
    for x, y in pix:
        cv2.circle(img, (int(x), int(y)), radius, bgr, -1, lineType=cv2.LINE_AA)
    return int(len(pix))


def draw_mask_outline(img, mask, bgr=(255, 255, 255)):
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr, 2, cv2.LINE_AA)


def label(img, text, y, bgr):
    cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, bgr, 2, cv2.LINE_AA)


def q_values_from_report():
    report = read_json(REPORT)
    out = []
    for item in report["fixed_q_outputs"]:
        out.append((item["label"], float(item["q_open_m"])))
    return out


def render_closed_view_overlays(axis, base_points, drawer_points, q_values):
    views = est.load_closed_views()
    outputs = []
    for view_idx, view in enumerate(views):
        base_view = est.transform_points(base_points, view["qpos1_to_view"])
        for q_label, q in q_values:
            drawer_closed = drawer_points - axis.reshape(1, 3) * q
            drawer_view = est.transform_points(drawer_closed, view["qpos1_to_view"])
            img = view["color"].copy()
            draw_mask_outline(img, view["mask"], (245, 245, 245))
            nb = draw_points(img, base_view, view["k"], (65, 220, 65), radius=1, max_points=42000)
            nd = draw_points(img, drawer_view, view["k"], (40, 60, 245), radius=1, max_points=25000)
            label(img, f"{view['capture']}  {q_label} q={q:.3f}m", 28, (0, 255, 255))
            label(img, "white=qpos0 whole mask  green=qpos1 base  red=synth closed drawer", 56, (245, 245, 245))
            label(img, f"projected points base={nb} drawer={nd}", 84, (245, 245, 245))
            path = OUT / f"diagnostic_qpos0_view{view_idx}_{q_label}_projection.png"
            cv2.imwrite(str(path), img)
            outputs.append(str(path))
    return outputs


def draw_scatter(canvas, coords, bgr, bounds, radius=1, max_points=18000):
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    if len(coords) > max_points:
        coords = coords[RNG.choice(len(coords), size=max_points, replace=False)]
    xmin, xmax, ymin, ymax = bounds
    w = canvas.shape[1]
    h = canvas.shape[0]
    x = (coords[:, 0] - xmin) / max(xmax - xmin, 1e-6)
    y = (coords[:, 1] - ymin) / max(ymax - ymin, 1e-6)
    px = np.round(60 + x * (w - 100)).astype(np.int32)
    py = np.round(h - 50 - y * (h - 100)).astype(np.int32)
    inb = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    for xx, yy in zip(px[inb], py[inb]):
        cv2.circle(canvas, (int(xx), int(yy)), radius, bgr, -1, lineType=cv2.LINE_AA)


def residual_points(base_points, qpos0_points, threshold=0.045):
    base_metric = est.voxel_downsample(base_points, voxel=0.006, max_points=50000)
    qpos0_metric = est.voxel_downsample(qpos0_points, voxel=0.006, max_points=70000)
    dist, _ = cKDTree(base_metric).query(qpos0_metric, k=1, workers=-1)
    return qpos0_metric[dist > threshold], qpos0_metric, dist


def render_axis_profile(axis, base_points, drawer_points, qpos0_points, q_values):
    qpos0_residual, qpos0_metric, dist_base = residual_points(base_points, qpos0_points)
    origin = np.median(drawer_points, axis=0)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = up - axis * float(np.dot(up, axis))
    if np.linalg.norm(up) < 1e-6:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = up - axis * float(np.dot(up, axis))
    up = up / np.linalg.norm(up)

    def coords(points):
        rel = np.asarray(points, dtype=np.float64) - origin.reshape(1, 3)
        return np.column_stack([rel @ axis, rel @ up])

    all_coords = [coords(qpos0_residual), coords(drawer_points)]
    for _, q in q_values:
        all_coords.append(coords(drawer_points - axis.reshape(1, 3) * q))
    stacked = np.vstack([c for c in all_coords if len(c)])
    xmin, xmax = np.percentile(stacked[:, 0], [1, 99])
    ymin, ymax = np.percentile(stacked[:, 1], [1, 99])
    pad_x = max(0.04, 0.08 * (xmax - xmin))
    pad_y = max(0.04, 0.08 * (ymax - ymin))
    bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    outputs = []
    for q_label, q in q_values:
        drawer_closed = drawer_points - axis.reshape(1, 3) * q
        canvas = np.full((900, 1300, 3), 20, dtype=np.uint8)
        draw_scatter(canvas, coords(qpos0_residual), (255, 230, 70), bounds, radius=1, max_points=22000)
        draw_scatter(canvas, coords(drawer_points), (70, 170, 255), bounds, radius=1, max_points=12000)
        draw_scatter(canvas, coords(drawer_closed), (40, 60, 245), bounds, radius=1, max_points=12000)
        x0 = int(60 + (0.0 - bounds[0]) / max(bounds[1] - bounds[0], 1e-6) * (canvas.shape[1] - 100))
        cv2.line(canvas, (x0, 40), (x0, canvas.shape[0] - 40), (120, 120, 120), 1, cv2.LINE_AA)
        label(canvas, f"{q_label} q={q:.3f}m   horizontal=joint axis closed->open", 34, (0, 255, 255))
        label(canvas, "cyan/yellow=qpos0 residual after removing qpos1 base   orange=qpos1 open drawer   red=synth closed drawer", 64, (235, 235, 235))
        median_delta = float(np.median(coords(drawer_closed)[:, 0]) - np.median(coords(qpos0_residual)[:, 0]))
        label(canvas, f"median signed axis delta red - qpos0 residual = {median_delta:+.3f}m", 94, (235, 235, 235))
        path = OUT / f"diagnostic_axis_profile_{q_label}.png"
        cv2.imwrite(str(path), canvas)
        outputs.append(str(path))

    hist = np.full((720, 1200, 3), 255, dtype=np.uint8)
    label(hist, "qpos0-to-base distance histogram: residual is points > 4.5cm from qpos1 base", 34, (40, 40, 40))
    vals = np.minimum(dist_base, 0.25)
    bins = np.linspace(0.0, 0.25, 80)
    counts, _ = np.histogram(vals, bins=bins)
    counts = counts.astype(np.float64) / max(float(counts.max()), 1.0)
    for i, c in enumerate(counts):
        x1 = 60 + int(i * 1080 / len(counts))
        x2 = 60 + int((i + 1) * 1080 / len(counts))
        y = 660 - int(c * 560)
        cv2.rectangle(hist, (x1, y), (x2, 660), (90, 150, 220), -1)
    threshold_x = 60 + int(0.045 / 0.25 * 1080)
    cv2.line(hist, (threshold_x, 80), (threshold_x, 670), (20, 20, 230), 2, cv2.LINE_AA)
    label(hist, "red line = 4.5cm residual threshold", 68, (20, 20, 230))
    path = OUT / "diagnostic_qpos0_base_distance_histogram.png"
    cv2.imwrite(str(path), hist)
    outputs.append(str(path))
    return outputs


def main():
    joint_doc = read_json(JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    axis = cmp.anim.unit(joint["axis_camera_closed_to_open"]).astype(np.float64)
    base_points, base_colors, _, drawer_points, drawer_colors, _ = cmp.load_qpos1_parts()
    qpos0_points, qpos0_colors, _ = cmp.load_qpos0_closed_reference()
    q_values = q_values_from_report()
    closed_view_outputs = render_closed_view_overlays(axis, base_points, drawer_points, q_values)
    profile_outputs = render_axis_profile(axis, base_points, drawer_points, qpos0_points, q_values)
    out = {
        "closed_view_projection_overlays": closed_view_outputs,
        "axis_profile_overlays": profile_outputs,
    }
    path = OUT / "diagnostic_image_manifest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
