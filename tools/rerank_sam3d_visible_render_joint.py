import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
sys.path.insert(0, str(ROOT / "tools"))
import build_sam3d_clean_prismatic_qpos1 as clean  # noqa: E402
import fit_sam3d_constrained_plane_to_rgbd as c  # noqa: E402


CAPTURE = "20260622_081031_636398Z"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_visible_render_joint_rerank"
PLANE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_plane_to_rgbd_plane_fit/sam3d_plane_to_rgbd_plane_fit_report.json"
TWOPLANE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_base_twoplane_to_rgbd_plane_fit/sam3d_base_twoplane_to_rgbd_plane_fit_report.json"
OBJECT_POSE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_object_pose_aligned/object_pose_aligned_report.json"
COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
COLORS = {
    "base": np.array([60, 220, 80], dtype=np.float32),
    "drawer": np.array([40, 70, 245], dtype=np.float32),
    "drawer_closed": np.array([20, 165, 255], dtype=np.float32),
}
RNG = np.random.default_rng(20260721)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    return c.unit(v)


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    meshes = [g.copy() for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices)]
    if not meshes:
        raise ValueError(f"empty mesh scene: {path}")
    return trimesh.util.concatenate(meshes)


def sample_mesh(mesh, count):
    if len(mesh.faces):
        pts, _ = trimesh.sample.sample_surface(mesh, count)
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(pts) > count:
        pts = pts[RNG.choice(len(pts), size=count, replace=False)]
    return np.asarray(pts, dtype=np.float64)


def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    keep = points[:, 2] > 1e-5
    p = points[keep]
    uv[keep, 0] = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]
    uv[keep, 1] = k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
    return uv, keep


def render_zbuffer(points, k, height, width, splat_radius=1):
    uv, keep = project(points, k)
    inb = keep & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if int(inb.sum()) == 0:
        zbuf = np.full((height, width), np.inf, dtype=np.float32)
        return zbuf, np.zeros((height, width), dtype=bool), 0
    px0 = np.round(uv[inb, 0]).astype(np.int32)
    py0 = np.round(uv[inb, 1]).astype(np.int32)
    z = np.asarray(points[inb, 2], dtype=np.float32)
    flat = np.full(height * width, np.inf, dtype=np.float32)
    for dy in range(-splat_radius, splat_radius + 1):
        py = py0 + dy
        ok_y = (py >= 0) & (py < height)
        if not np.any(ok_y):
            continue
        for dx in range(-splat_radius, splat_radius + 1):
            px = px0 + dx
            ok = ok_y & (px >= 0) & (px < width)
            if not np.any(ok):
                continue
            idx = py[ok] * width + px[ok]
            np.minimum.at(flat, idx, z[ok])
    zbuf = flat.reshape(height, width)
    raw_mask = np.isfinite(zbuf)
    return zbuf, raw_mask, int(inb.sum())


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float64)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def score_render(zbuf, pred_mask_raw, projected_points, target_mask, depth_m):
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    pred_mask = cv2.dilate(pred_mask.astype(np.uint8), kernel, iterations=1) > 0
    target = target_mask > 0
    valid_depth = depth_m > 0.05
    hit = pred_mask_raw & target & valid_depth
    if int(hit.sum()) == 0:
        depth_med = 1.0
        depth_p75 = 1.0
        depth_p90 = 1.0
        signed_depth_med = 1.0
    else:
        diff = zbuf[hit].astype(np.float64) - depth_m[hit].astype(np.float64)
        depth_abs = np.abs(diff)
        depth_med = float(np.median(depth_abs))
        depth_p75 = float(np.percentile(depth_abs, 75))
        depth_p90 = float(np.percentile(depth_abs, 90))
        signed_depth_med = float(np.median(diff))
    union = pred_mask | target
    iou = float((pred_mask & target).sum() / max(1, union.sum()))
    coverage = float((pred_mask & target).sum() / max(1, target.sum()))
    leakage = float((pred_mask & ~target).sum() / max(1, pred_mask.sum()))
    pred_hit_ratio = float((pred_mask_raw & target).sum() / max(1, pred_mask_raw.sum()))

    if pred_mask.any():
        ys, xs = np.where(pred_mask)
        pb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    else:
        pb = np.array([0, 0, 1, 1], dtype=np.float64)
    tb = mask_bbox(target)
    ts = np.maximum(tb[2:] - tb[:2], 1.0)
    bbox_center_err = float(np.linalg.norm(((pb[:2] + pb[2:]) * 0.5 - (tb[:2] + tb[2:]) * 0.5) / ts))
    bbox_size_err = float(np.linalg.norm(((pb[2:] - pb[:2]) - ts) / ts))

    score = (
        2.20 * (1.0 - iou)
        + 0.85 * (1.0 - coverage)
        + 1.25 * leakage
        + 1.10 * min(depth_med / 0.050, 5.0)
        + 0.32 * min(depth_p75 / 0.100, 5.0)
        + 0.16 * min(depth_p90 / 0.160, 5.0)
        + 0.65 * bbox_size_err
        + 0.90 * bbox_center_err
    )
    return float(score), {
        "visible_render_score": float(score),
        "render_mask_iou": iou,
        "render_target_coverage": coverage,
        "render_leakage": leakage,
        "render_projected_hit_ratio": pred_hit_ratio,
        "visible_depth_abs_median_m": depth_med,
        "visible_depth_abs_p75_m": depth_p75,
        "visible_depth_abs_p90_m": depth_p90,
        "visible_depth_signed_median_m": signed_depth_med,
        "render_bbox_xyxy": pb.tolist(),
        "target_bbox_xyxy": tb.tolist(),
        "render_bbox_center_err": bbox_center_err,
        "render_bbox_size_err": bbox_size_err,
        "render_visible_pixels_raw": int(pred_mask_raw.sum()),
        "render_visible_pixels_scored": int(pred_mask.sum()),
        "render_depth_overlap_pixels": int(hit.sum()),
        "projected_sample_points": int(projected_points),
    }, pred_mask


def depth_vis(zbuf, pred_mask, depth_m, target_mask):
    img = np.zeros((*zbuf.shape, 3), dtype=np.uint8)
    use = pred_mask & np.isfinite(zbuf)
    if np.any(use):
        z = zbuf[use]
        lo, hi = np.percentile(z, [2, 98])
        scaled = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        img[use] = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    contours, _ = cv2.findContours((target_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
    return img


def draw_visible_overlay(color_bgr, render_items, masks, output):
    img = color_bgr.copy()
    for name, pred_mask in render_items.items():
        color = COLORS[name]
        use = pred_mask > 0
        img[use] = np.clip(0.56 * img[use].astype(np.float32) + 0.44 * color.reshape(1, 3), 0, 255).astype(np.uint8)
    for name, mask in masks.items():
        contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, tuple(int(x) for x in COLORS[name]), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), img)


def candidate_normal(export):
    for key in ("primary_normal_camera", "source_plane_normal_camera"):
        if key in export:
            return unit(export[key])
    return None


def candidate_secondary_normal(export):
    if "secondary_normal_camera" in export:
        return unit(export["secondary_normal_camera"])
    metrics = export.get("metrics", {})
    if "secondary_plane_normal_camera" in metrics:
        return unit(metrics["secondary_plane_normal_camera"])
    return None


def candidate_id(source_name, part, export):
    return f"{source_name}_{part}_rank{int(export['rank']):02d}"


def collect_candidates(plane_report, twoplane_report):
    out = {"base": [], "drawer": []}
    for export in plane_report["parts"]["base"]["top_exports"]:
        item = dict(export)
        item["candidate_source"] = "single_plane"
        item["candidate_id"] = candidate_id("single", "base", item)
        out["base"].append(item)
    for export in twoplane_report["parts"]["base"]["top_exports"]:
        item = dict(export)
        item["candidate_source"] = "base_two_plane"
        item["candidate_id"] = candidate_id("twoplane", "base", item)
        out["base"].append(item)
    for export in plane_report["parts"]["drawer"]["top_exports"]:
        item = dict(export)
        item["candidate_source"] = "single_plane"
        item["candidate_id"] = candidate_id("single", "drawer", item)
        out["drawer"].append(item)
    return out


def score_candidate(part, export, k, height, width, target_mask, depth_m, output_dir):
    mesh = load_mesh(export["glb"])
    count = 120000 if part == "base" else 90000
    points = sample_mesh(mesh, count)
    zbuf, pred_raw, projected = render_zbuffer(points, k, height, width, splat_radius=1)
    score, metrics, pred_mask = score_render(zbuf, pred_raw, projected, target_mask, depth_m)
    candidate_dir = output_dir / part
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prefix = export["candidate_id"]
    cv2.imwrite(str(candidate_dir / f"{prefix}_visible_depth.png"), depth_vis(zbuf, pred_raw, depth_m, target_mask))
    metrics.update(
        {
            "candidate_id": export["candidate_id"],
            "candidate_source": export["candidate_source"],
            "rank": int(export["rank"]),
            "glb": export["glb"],
            "visible_depth_png": str(candidate_dir / f"{prefix}_visible_depth.png"),
        }
    )
    return {**export, "visible_render": metrics, "mesh": mesh, "pred_mask": pred_mask, "zbuf": zbuf}


def load_joint():
    report = read_json(OBJECT_POSE_REPORT)
    joint = report["joint_axis_kept_fixed"]
    axis = unit(joint["axis_camera_closed_to_open"])
    q = float(joint["best_residual_q_m"])
    return report, axis, q


def export_scene(path, items):
    scene = trimesh.Scene()
    for name, mesh in items:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def translate_mesh(mesh, translation):
    out = mesh.copy()
    out.vertices = np.asarray(out.vertices, dtype=np.float64) + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return out


def choose_pair(base_scored, drawer_scored):
    best = None
    all_pairs = []
    for b in base_scored:
        bn = candidate_normal(b)
        bs = candidate_secondary_normal(b)
        for d in drawer_scored:
            dn = candidate_normal(d)
            pair_penalty = 0.0
            pair_metrics = {}
            if bn is not None and dn is not None:
                front_parallel = float(abs(bn @ dn))
                pair_penalty += 1.25 * (1.0 - front_parallel)
                pair_metrics["base_drawer_front_normal_absdot"] = front_parallel
            if bn is not None and bs is not None:
                base_orth = float(abs(bn @ bs))
                pair_penalty += 0.65 * base_orth
                pair_metrics["base_front_secondary_absdot"] = base_orth
            total = (
                float(b["visible_render"]["visible_render_score"])
                + float(d["visible_render"]["visible_render_score"])
                + pair_penalty
            )
            item = {
                "selection_score": float(total),
                "base_candidate_id": b["candidate_id"],
                "drawer_candidate_id": d["candidate_id"],
                "base_visible_score": float(b["visible_render"]["visible_render_score"]),
                "drawer_visible_score": float(d["visible_render"]["visible_render_score"]),
                "pair_penalty": float(pair_penalty),
                **pair_metrics,
            }
            all_pairs.append(item)
            if best is None or total < best["selection_score"]:
                best = item
    all_pairs.sort(key=lambda x: x["selection_score"])
    return best, all_pairs[:20]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plane_report = read_json(PLANE_REPORT)
    twoplane_report = read_json(TWOPLANE_REPORT)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)
    depth_raw = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(DEPTH_PATH)
    depth_m = depth_raw.astype(np.float32) / 1000.0
    k = np.asarray(read_json(META_PATH)["PVCamera"]["k"], dtype=np.float64)
    height, width = depth_m.shape
    masks = {}
    for part, path in MASKS.items():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        masks[part] = mask > 127

    object_pose_report, axis, q = load_joint()
    open_to_closed = -axis * q
    closed_to_open = axis * q
    proposals = collect_candidates(plane_report, twoplane_report)

    scored = {"base": [], "drawer": []}
    for part in ("base", "drawer"):
        print(f"[{part}] visible-render scoring {len(proposals[part])} candidates", flush=True)
        for export in proposals[part]:
            scored[part].append(score_candidate(part, export, k, height, width, masks[part], depth_m, OUT))
        scored[part].sort(key=lambda item: item["visible_render"]["visible_render_score"])
        print(
            f"[{part}] best={scored[part][0]['candidate_id']} "
            f"score={scored[part][0]['visible_render']['visible_render_score']:.4f}",
            flush=True,
        )

    pair, top_pairs = choose_pair(scored["base"], scored["drawer"])
    base = next(item for item in scored["base"] if item["candidate_id"] == pair["base_candidate_id"])
    drawer = next(item for item in scored["drawer"] if item["candidate_id"] == pair["drawer_candidate_id"])
    base_mesh = base["mesh"]
    drawer_open = drawer["mesh"]
    drawer_closed = translate_mesh(drawer_open, open_to_closed)

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    open_scene = OUT / "cabinet_drawer_sam3d_visible_render_open.glb"
    closed_scene = OUT / "cabinet_drawer_sam3d_visible_render_closed.glb"
    overlay_scene = OUT / "cabinet_drawer_sam3d_visible_render_open_closed_overlay.glb"
    animated_glb = OUT / "cabinet_drawer_sam3d_visible_render_prismatic_animated.glb"
    urdf = OUT / "cabinet_drawer_sam3d_visible_render.urdf"
    base_mesh.export(base_path)
    drawer_open.export(drawer_open_path)
    drawer_closed.export(drawer_closed_path)
    export_scene(open_scene, [("base", base_mesh), ("drawer_open", drawer_open)])
    export_scene(closed_scene, [("base", base_mesh), ("drawer_closed", drawer_closed)])
    export_scene(overlay_scene, [("base", base_mesh), ("drawer_open", drawer_open), ("drawer_closed", drawer_closed)])
    clean.build_animated_glb(animated_glb, base_mesh, drawer_closed, axis, q)
    clean.write_urdf(urdf, axis, q)

    closed_points = sample_mesh(drawer_closed, 90000)
    _, closed_pred_raw, _ = render_zbuffer(closed_points, k, height, width, splat_radius=1)
    draw_visible_overlay(
        color,
        {"base": base["pred_mask"], "drawer": drawer["pred_mask"]},
        masks,
        OUT / "combined_qpos1_visible_render_overlay.png",
    )
    draw_visible_overlay(
        color,
        {"base": base["pred_mask"], "drawer": drawer["pred_mask"], "drawer_closed": closed_pred_raw},
        masks,
        OUT / "combined_open_closed_visible_render_overlay.png",
    )

    joint = {
        "type": "prismatic",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "axis_camera_closed_to_open": axis.astype(float).tolist(),
        "qpos_closed_m": 0.0,
        "qpos_open_m": float(q),
        "displacement_m": float(q),
        "translation_open_to_closed_camera_m": open_to_closed.astype(float).tolist(),
        "translation_closed_to_open_camera_m": closed_to_open.astype(float).tolist(),
        "source": str(OBJECT_POSE_REPORT),
    }
    report = {
        "method": (
            "Visible-render/z-buffer reranking of SAM3D plane-pose candidates. Complete SAM3D meshes are not "
            "fit to the single-view RGB-D point cloud with full-surface Chamfer. For each candidate pose, only "
            "the camera-visible rendered surface is scored against qpos1 mask/depth. Base and drawer are selected "
            "jointly with a weak shared-frame normal consistency term."
        ),
        "frame": joint["frame"],
        "camera_axes": joint["camera_axes"],
        "constraints": {
            "uses_sfm": False,
            "uses_raw_camera_pose": False,
            "uses_closed_partmask": False,
            "uses_complete_mesh_to_visible_pointcloud_chamfer": False,
            "uses_visible_zbuffer_depth_score": True,
            "uses_raw_sam3d_normal_vs_motion_axis": False,
            "axis_used_for_scoring": False,
            "axis_used_for_animation": True,
            "pair_selection": "base_visible_score + drawer_visible_score + shared transformed normal consistency",
        },
        "joint": joint,
        "q_evidence": {
            "object_pose_aligned_report": str(OBJECT_POSE_REPORT),
            "best_residual_q_m": float(q),
            "q_sweep_top3": object_pose_report.get("q_sweep_top15", [])[:3],
        },
        "selected_pair": pair,
        "top_pairs": top_pairs,
        "parts": {
            "base": {
                "selected_candidate_id": base["candidate_id"],
                "selected_source": base["candidate_source"],
                "selected_glb": base["glb"],
                "visible_render": base["visible_render"],
                "top_visible_candidates": [
                    {
                        "candidate_id": item["candidate_id"],
                        "candidate_source": item["candidate_source"],
                        "rank": int(item["rank"]),
                        "glb": item["glb"],
                        "visible_render": item["visible_render"],
                    }
                    for item in scored["base"][:10]
                ],
            },
            "drawer": {
                "selected_candidate_id": drawer["candidate_id"],
                "selected_source": drawer["candidate_source"],
                "selected_glb": drawer["glb"],
                "visible_render": drawer["visible_render"],
                "top_visible_candidates": [
                    {
                        "candidate_id": item["candidate_id"],
                        "candidate_source": item["candidate_source"],
                        "rank": int(item["rank"]),
                        "glb": item["glb"],
                        "visible_render": item["visible_render"],
                    }
                    for item in scored["drawer"][:10]
                ],
            },
        },
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_glb": str(open_scene),
            "closed_glb": str(closed_scene),
            "open_closed_overlay_glb": str(overlay_scene),
            "animated_prismatic_glb": str(animated_glb),
            "urdf": str(urdf),
            "joint_json": str(OUT / "joint.json"),
            "qpos1_visible_overlay": str(OUT / "combined_qpos1_visible_render_overlay.png"),
            "open_closed_visible_overlay": str(OUT / "combined_open_closed_visible_render_overlay.png"),
            "report": str(OUT / "sam3d_visible_render_joint_rerank_report.json"),
        },
    }
    (OUT / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    report_path = OUT / "sam3d_visible_render_joint_rerank_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "open_glb": str(open_scene),
                "animated_prismatic_glb": str(animated_glb),
                "qpos1_visible_overlay": report["outputs"]["qpos1_visible_overlay"],
                "open_closed_visible_overlay": report["outputs"]["open_closed_visible_overlay"],
                "selected_pair": pair,
                "base_visible": base["visible_render"],
                "drawer_visible": drawer["visible_render"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
