import csv
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

import estimate_prismatic_rgbd_synthesis as est


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/fixed_axis_q_zbuffer_refine"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
RNG = np.random.default_rng(20260710)


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def render_zbuffer(points_view, k, width, height, radius=1):
    uv, valid_points = est.project(points_view, k)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int16)
    if len(uv) == 0:
        return depth, owner
    z = valid_points[:, 2]
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
    return depth, owner


def zbuffer_named(base_view, drawer_view, k, width, height, radius=1):
    depth = np.full((height, width), np.inf, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int16)
    for owner_id, points in [(0, base_view), (1, drawer_view)]:
        uv, valid_points = est.project(points, k)
        if len(uv) == 0:
            continue
        z = valid_points[:, 2]
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
                    owner[yy_ok[closer], xx_ok[closer]] = owner_id
    return depth, owner


def bbox_err(pred, target):
    ys, xs = np.where(pred)
    yt, xt = np.where(target)
    if len(xs) < 10 or len(xt) < 10:
        return 99.0, 99.0, [np.nan] * 4
    rb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    tb = np.array([xt.min(), yt.min(), xt.max(), yt.max()], dtype=np.float64)
    size = np.maximum(tb[2:] - tb[:2], 1.0)
    size_err = float(np.linalg.norm(((rb[2:] - rb[:2]) - size) / size))
    center_err = float(np.linalg.norm((((rb[:2] + rb[2:]) - (tb[:2] + tb[2:])) * 0.5) / size))
    return size_err, center_err, rb.tolist()


def score_q(q, axis_closed_to_open, base_points, drawer_points, views):
    t_open_to_closed = -axis_closed_to_open * float(q)
    per_view = []
    values = []
    for view in views:
        h, w = view["mask"].shape
        base_v = est.transform_points(base_points, view["qpos1_to_view"])
        drawer_v = est.transform_points(drawer_points + t_open_to_closed.reshape(1, 3), view["qpos1_to_view"])
        depth_render, owner = zbuffer_named(base_v, drawer_v, view["k"], w, h, radius=2)
        pred = np.isfinite(depth_render)
        target = view["mask"] > 0
        valid = view["depth"] > 0.20
        inter = pred & target & valid
        drawer_pred = owner == 1
        drawer_inter = drawer_pred & target & valid
        if int(inter.sum()) < 120:
            metrics = {"reason": "small_intersection", "score": 1e9}
            per_view.append(metrics)
            values.append(1e9)
            continue
        diff = depth_render[inter] - view["depth"][inter]
        abs_diff = np.abs(diff)
        drawer_abs = np.abs(depth_render[drawer_inter] - view["depth"][drawer_inter]) if np.any(drawer_inter) else np.array([0.25])
        iou = float((pred & target).sum() / max(1, (pred | target).sum()))
        coverage = float((pred & target).sum() / max(1, target.sum()))
        leakage = float((pred & ~target).sum() / max(1, pred.sum()))
        drawer_coverage = float((drawer_pred & target).sum() / max(1, target.sum()))
        behind_ratio = float(np.mean(diff > 0.055))
        front_ratio = float(np.mean(diff < -0.055))
        size_err, center_err, rb = bbox_err(pred, target)
        med = float(np.median(abs_diff))
        p80 = float(np.percentile(abs_diff, 80))
        drawer_med = float(np.median(drawer_abs))
        value = (
            med / 0.035
            + 0.30 * p80 / 0.080
            + 0.70 * drawer_med / 0.045
            + 1.25 * (1.0 - iou)
            + 0.55 * (1.0 - coverage)
            + 0.85 * leakage
            + 0.85 * behind_ratio
            + 0.55 * front_ratio
            + 0.28 * size_err
            + 0.35 * center_err
        )
        metrics = {
            "capture": view["capture"],
            "score": float(value),
            "depth_abs_median_m": med,
            "depth_abs_p80_m": p80,
            "drawer_depth_abs_median_m": drawer_med,
            "mask_iou": iou,
            "target_coverage": coverage,
            "leakage": leakage,
            "drawer_target_coverage": drawer_coverage,
            "behind_ratio": behind_ratio,
            "front_ratio": front_ratio,
            "bbox_size_err": size_err,
            "bbox_center_err": center_err,
            "render_bbox_xyxy": rb,
            "intersection_pixels": int(inter.sum()),
            "drawer_intersection_pixels": int(drawer_inter.sum()),
        }
        per_view.append(metrics)
        values.append(value)
    return {
        "q_open_m": float(q),
        "translation_open_to_closed_camera_m": t_open_to_closed.tolist(),
        "score": float(np.mean(values)),
        "per_view": per_view,
        "mean_metrics": {
            key: float(np.mean([m[key] for m in per_view if key in m]))
            for key in [
                "depth_abs_median_m",
                "depth_abs_p80_m",
                "drawer_depth_abs_median_m",
                "mask_iou",
                "target_coverage",
                "leakage",
                "drawer_target_coverage",
                "behind_ratio",
                "front_ratio",
                "bbox_size_err",
                "bbox_center_err",
            ]
        },
    }


def draw_overlay(path, view, axis, q, base_points, drawer_points):
    t = -axis * float(q)
    img = view["color"].copy()
    contours, _ = cv2.findContours((view["mask"].astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
    for name, points, color in [
        ("base", base_points, (70, 220, 70)),
        ("drawer_closed", drawer_points + t.reshape(1, 3), (40, 70, 245)),
    ]:
        pts = points
        if len(pts) > 55000:
            pts = pts[RNG.choice(len(pts), size=55000, replace=False)]
        pv = est.transform_points(pts, view["qpos1_to_view"])
        uv, _ = est.project(pv, view["k"])
        h, w = img.shape[:2]
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        for x, y in pix:
            cv2.circle(img, (int(x), int(y)), 1, color, -1)
        if len(pix):
            c = pix.mean(axis=0).astype(int)
            cv2.putText(img, name, (int(c[0]) + 8, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"q={q:.3f}m", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def export_scene(path, axis, q, base_points, drawer_points):
    t = -axis * float(q)
    scene = trimesh.Scene()
    base = trimesh.points.PointCloud(base_points, colors=np.tile(np.array([70, 220, 70, 255], dtype=np.uint8), (len(base_points), 1)))
    drawer_open = trimesh.points.PointCloud(drawer_points, colors=np.tile(np.array([240, 80, 80, 255], dtype=np.uint8), (len(drawer_points), 1)))
    drawer_closed = trimesh.points.PointCloud(drawer_points + t.reshape(1, 3), colors=np.tile(np.array([255, 190, 30, 255], dtype=np.uint8), (len(drawer_points), 1)))
    scene.add_geometry(base, geom_name="qpos1_base_surface")
    scene.add_geometry(drawer_open, geom_name="qpos1_drawer_open_surface")
    scene.add_geometry(drawer_closed, geom_name="drawer_closed_fixed_axis_zbuffer_refined")
    scene.export(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint_doc = est.read_json(JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    axis = unit(joint["axis_camera_closed_to_open"])
    old_q = float(joint["displacement_m"])

    base_points, _ = est.unproject_mask(est.QPOS1_CAPTURE, est.QPOS1_BASE_MASK, stride=1, max_points=80000)
    drawer_points, _ = est.unproject_mask(est.QPOS1_CAPTURE, est.QPOS1_DRAWER_MASK, stride=1, max_points=60000)
    base_points = est.voxel_downsample(base_points, voxel=0.0055, max_points=36000)
    drawer_points = est.voxel_downsample(drawer_points, voxel=0.0055, max_points=26000)
    views = est.load_closed_views()

    coarse_qs = np.linspace(0.08, 0.46, 77)
    coarse = [score_q(q, axis, base_points, drawer_points, views) for q in coarse_qs]
    coarse_best = min(coarse, key=lambda x: x["score"])
    q0 = float(coarse_best["q_open_m"])
    fine_qs = np.linspace(max(0.04, q0 - 0.045), min(0.52, q0 + 0.045), 91)
    fine = [score_q(q, axis, base_points, drawer_points, views) for q in fine_qs]
    all_scores = sorted(coarse + fine, key=lambda x: x["score"])
    best = all_scores[0]
    old = score_q(old_q, axis, base_points, drawer_points, views)

    with (OUT / "fixed_axis_q_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "q_open_m",
                "score",
                "depth_abs_median_m",
                "drawer_depth_abs_median_m",
                "mask_iou",
                "target_coverage",
                "leakage",
                "behind_ratio",
                "front_ratio",
            ],
        )
        writer.writeheader()
        seen = set()
        rank = 0
        for row in all_scores:
            key = round(row["q_open_m"], 6)
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            m = row["mean_metrics"]
            writer.writerow(
                {
                    "rank": rank,
                    "q_open_m": row["q_open_m"],
                    "score": row["score"],
                    "depth_abs_median_m": m["depth_abs_median_m"],
                    "drawer_depth_abs_median_m": m["drawer_depth_abs_median_m"],
                    "mask_iou": m["mask_iou"],
                    "target_coverage": m["target_coverage"],
                    "leakage": m["leakage"],
                    "behind_ratio": m["behind_ratio"],
                    "front_ratio": m["front_ratio"],
                }
            )

    draw_overlay(OUT / "best_qpos0_zbuffer_overlay.png", views[0], axis, best["q_open_m"], base_points, drawer_points)
    draw_overlay(OUT / "old_qpos0_zbuffer_overlay.png", views[0], axis, old_q, base_points, drawer_points)
    export_scene(OUT / "best_fixed_axis_surfaces_qpos1_frame.glb", axis, best["q_open_m"], base_points, drawer_points)
    export_scene(OUT / "old_fixed_axis_surfaces_qpos1_frame.glb", axis, old_q, base_points, drawer_points)

    report = {
        "method": "Fixed-axis q refinement with multi-view qpos0 whole-mask/depth z-buffer scoring. Axis direction is kept from selected_axis1d_joint; only displacement length is rescored.",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "axis_camera_closed_to_open": axis.tolist(),
        "old_selected_q_m": old_q,
        "best_q_m": best["q_open_m"],
        "best_translation_open_to_closed_camera_m": best["translation_open_to_closed_camera_m"],
        "delta_q_m": float(best["q_open_m"] - old_q),
        "old_score": old,
        "best_score": best,
        "top_scores": all_scores[:20],
        "outputs": {
            "scores_csv": str(OUT / "fixed_axis_q_scores.csv"),
            "best_qpos0_overlay": str(OUT / "best_qpos0_zbuffer_overlay.png"),
            "old_qpos0_overlay": str(OUT / "old_qpos0_zbuffer_overlay.png"),
            "best_surfaces_glb": str(OUT / "best_fixed_axis_surfaces_qpos1_frame.glb"),
            "old_surfaces_glb": str(OUT / "old_fixed_axis_surfaces_qpos1_frame.glb"),
        },
    }
    report_path = OUT / "fixed_axis_q_zbuffer_refine_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "old_q_m": old_q,
                "best_q_m": best["q_open_m"],
                "delta_q_m": report["delta_q_m"],
                "old_score": old["score"],
                "best_score": best["score"],
                "best_overlay": report["outputs"]["best_qpos0_overlay"],
                "old_overlay": report["outputs"]["old_qpos0_overlay"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
