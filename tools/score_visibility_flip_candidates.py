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
REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json"
COLOR = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/visibility_flip_score"

MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
COLORS = {"base": (0, 220, 0), "drawer": (0, 0, 240)}
RNG = np.random.default_rng(20260624)


def signed_permutation_rotations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            m = p @ np.diag(signs)
            if np.linalg.det(m) > 0:
                mats.append(m.astype(np.float64))
    return mats


def rot_axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    axis /= np.linalg.norm(axis)
    a = math.radians(deg)
    x, y, z = axis
    c = math.cos(a)
    s = math.sin(a)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    return trimesh.util.concatenate([g.copy() for g in scene.geometry.values()])


def project(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    good = z > 1e-5
    p = points[good]
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    return uv, good


def transform_similarity(points: np.ndarray, r: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    return s * (points @ r.T) + t


def apply_source_rotation(points: np.ndarray, center: np.ndarray, f: np.ndarray) -> np.ndarray:
    return center + (points - center) @ f.T


def apply_camera_rotation(points: np.ndarray, center: np.ndarray, q: np.ndarray) -> np.ndarray:
    return center + (points - center) @ q.T


def bbox_center(uv: np.ndarray, w: int, h: int) -> tuple[np.ndarray, list[float], int]:
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inb]
    if len(uv) < 50:
        return np.array([np.nan, np.nan]), [np.nan] * 4, int(len(uv))
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    return (lo + hi) * 0.5, [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])], int(len(uv))


def snap_xy(points: np.ndarray, k: np.ndarray, target_bbox: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, dict]:
    uv, _ = project(points, k)
    ctr, bbox, count = bbox_center(uv, w, h)
    if not np.isfinite(ctr).all():
        return points, np.zeros(3), {"before_center_px": ctr.tolist(), "before_bbox": bbox, "projected_count": count}
    target_ctr = (target_bbox[:2] + target_bbox[2:]) * 0.5
    z = float(np.nanmedian(points[:, 2]))
    duv = target_ctr - ctr
    delta = np.array([duv[0] * z / k[0, 0], duv[1] * z / k[1, 1], 0.0], dtype=np.float64)
    snapped = points + delta.reshape(1, 3)
    uv2, _ = project(snapped, k)
    ctr2, bbox2, _ = bbox_center(uv2, w, h)
    return snapped, delta, {
        "before_center_px": ctr.tolist(),
        "before_bbox": bbox,
        "after_center_px": ctr2.tolist(),
        "after_bbox": bbox2,
        "target_center_px": target_ctr.tolist(),
        "xy_delta_m": delta.tolist(),
        "projected_count": count,
    }


def zbuffer(points: np.ndarray, k: np.ndarray, width: int, height: int, radius: int = 1) -> np.ndarray:
    uv, good = project(points, k)
    z = points[good, 2]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if len(uv) == 0:
        return depth
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    for dy in range(-radius, radius + 1):
        yy = py + dy
        ok_y = (yy >= 0) & (yy < height)
        for dx in range(-radius, radius + 1):
            xx = px + dx
            ok = ok_y & (xx >= 0) & (xx < width)
            if ok.any():
                np.minimum.at(depth, (yy[ok], xx[ok]), z[ok])
    return depth


def score_depth(depth_render: np.ndarray, depth_obs_m: np.ndarray, mask: np.ndarray, bbox: list[float]) -> tuple[float, dict]:
    pred = np.isfinite(depth_render)
    target = mask > 0
    valid_obs = depth_obs_m > 0.05
    inter = pred & target & valid_obs
    union = pred | target
    if union.sum() == 0 or inter.sum() < 50:
        return 1e9, {"reason": "empty render or intersection"}
    diff = np.abs(depth_render[inter] - depth_obs_m[inter])
    med = float(np.median(diff))
    p80 = float(np.percentile(diff, 80))
    p95 = float(np.percentile(diff, 95))
    iou = float((pred & target).sum() / max(1, union.sum()))
    coverage = float((pred & target).sum() / max(1, target.sum()))
    leakage = float((pred & ~target).sum() / max(1, pred.sum()))
    ys, xs = np.where(pred)
    if len(xs):
        rb = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        target_bbox = np.asarray(bbox, dtype=np.float64)
        rb_arr = np.asarray(rb, dtype=np.float64)
        target_size = np.maximum(target_bbox[2:] - target_bbox[:2], 1.0)
        size_err = float(np.linalg.norm(((rb_arr[2:] - rb_arr[:2]) - target_size) / target_size))
        center_err = float(np.linalg.norm((((rb_arr[:2] + rb_arr[2:]) - (target_bbox[:2] + target_bbox[2:])) * 0.5) / target_size))
    else:
        rb = [np.nan] * 4
        size_err = center_err = 99.0
    # Median depth and coverage dominate; leakage catches wrong/back-side candidates that spill out.
    score = med / 0.025 + 0.35 * p80 / 0.055 + 0.15 * p95 / 0.10 + 1.4 * (1 - iou) + 0.7 * (1 - coverage) + 0.9 * leakage + 0.35 * size_err + 0.45 * center_err
    return score, {
        "score": float(score),
        "depth_abs_median_m": med,
        "depth_abs_p80_m": p80,
        "depth_abs_p95_m": p95,
        "mask_iou": iou,
        "target_coverage": coverage,
        "leakage": leakage,
        "render_bbox_xyxy": rb,
        "render_bbox_size_err": size_err,
        "render_bbox_center_err": center_err,
        "intersection_pixels": int(inter.sum()),
        "render_pixels": int(pred.sum()),
        "target_pixels": int(target.sum()),
    }


def draw_overlay(color: np.ndarray, depth_render: np.ndarray, mask: np.ndarray, bgr: tuple[int, int, int], path: Path) -> None:
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    pred = np.isfinite(depth_render)
    tint = np.zeros_like(img)
    tint[pred] = bgr
    img = cv2.addWeighted(img, 0.72, tint, 0.28, 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def make_contact_sheet(paths: list[Path], labels: list[str], output: Path) -> None:
    imgs = []
    for path, label in zip(paths, labels):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (640, 36), (255, 255, 255), -1)
        cv2.putText(img, label[:70], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        imgs.append(img)
    while len(imgs) % 2:
        imgs.append(np.zeros_like(imgs[0]) + 255)
    rows = [np.hstack(imgs[i : i + 2]) for i in range(0, len(imgs), 2)]
    cv2.imwrite(str(output), np.vstack(rows))


def part_candidates(name: str, mesh: trimesh.Trimesh, part: dict, k: np.ndarray, depth_obs_m: np.ndarray, color: np.ndarray) -> dict:
    mask = cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    h, w = mask.shape
    target_bbox = np.asarray(part["target_bbox_xyxy"], dtype=np.float64)
    r = np.asarray(part["rotation_matrix_source_to_camera"], dtype=np.float64)
    s = float(part["scale_uniform"])
    t = np.asarray(part["translation_camera_m_after"], dtype=np.float64)

    # Samples for scoring; full vertices are only transformed for the selected output.
    sample, _ = trimesh.sample.sample_surface(mesh, 260000 if name == "base" else 180000)
    src_center = (np.asarray(mesh.bounds[0]) + np.asarray(mesh.bounds[1])) * 0.5
    current = transform_similarity(sample, r, s, t)
    cam_center = np.median(current, axis=0)

    raw_candidates = []
    raw_candidates.append(("current", "identity", sample, current))
    for idx, f in enumerate(signed_permutation_rotations()):
        if np.allclose(f, np.eye(3)):
            continue
        p_src = apply_source_rotation(sample, src_center, f)
        p_cam = transform_similarity(p_src, r, s, t)
        raw_candidates.append((f"source_perm_{idx:02d}", "source_frame_signed_permutation", p_src, p_cam))
    axes = {
        "cam_x": np.array([1.0, 0.0, 0.0]),
        "cam_y": np.array([0.0, 1.0, 0.0]),
        "cam_z": np.array([0.0, 0.0, 1.0]),
    }
    for axis_name, axis in axes.items():
        for deg in (90, 180, 270):
            q = rot_axis_angle(axis, deg)
            p_cam = apply_camera_rotation(current, cam_center, q)
            raw_candidates.append((f"{axis_name}_{deg}", "camera_axis_rotation", sample, p_cam))

    results = []
    cand_dir = OUT / name
    cand_dir.mkdir(parents=True, exist_ok=True)
    for label, kind, src_points_after_local, p_cam0 in raw_candidates:
        p_cam, delta, snap = snap_xy(p_cam0, k, target_bbox, w, h)
        depth_render = zbuffer(p_cam, k, w, h, radius=1)
        score, metrics = score_depth(depth_render, depth_obs_m, mask, target_bbox.tolist())
        overlay_path = cand_dir / f"{label}_overlay.png"
        draw_overlay(color, depth_render, mask, COLORS[name], overlay_path)
        results.append(
            {
                "label": label,
                "kind": kind,
                "score": float(score),
                "metrics": metrics,
                "snap": snap,
                "overlay": str(overlay_path),
                # Store only the information needed to reproduce the selected transform.
                "source_rotation_index": label if kind == "source_frame_signed_permutation" else None,
            }
        )
    results.sort(key=lambda x: x["score"])

    top_paths = [Path(r["overlay"]) for r in results[:8]]
    top_labels = [f"{i+1}. {r['label']} score={r['score']:.2f}" for i, r in enumerate(results[:8])]
    contact = cand_dir / f"{name}_visibility_candidate_contact_sheet.png"
    make_contact_sheet(top_paths, top_labels, contact)
    return {"top": results[:12], "contact_sheet": str(contact)}


def export_selected(name: str, mesh: trimesh.Trimesh, part: dict, selection_label: str, k: np.ndarray) -> trimesh.Trimesh:
    # Recompute the selected candidate on full vertices.
    r = np.asarray(part["rotation_matrix_source_to_camera"], dtype=np.float64)
    s = float(part["scale_uniform"])
    t = np.asarray(part["translation_camera_m_after"], dtype=np.float64)
    mask = cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    target_bbox = np.asarray(part["target_bbox_xyxy"], dtype=np.float64)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    src_center = (np.asarray(mesh.bounds[0]) + np.asarray(mesh.bounds[1])) * 0.5

    if selection_label == "current":
        p = transform_similarity(v, r, s, t)
    elif selection_label.startswith("source_perm_"):
        f = None
        for idx, candidate in enumerate(signed_permutation_rotations()):
            if np.allclose(candidate, np.eye(3)):
                continue
            if selection_label == f"source_perm_{idx:02d}":
                f = candidate
                break
        if f is None:
            raise ValueError(selection_label)
        p = transform_similarity(apply_source_rotation(v, src_center, f), r, s, t)
    elif selection_label.startswith("cam_"):
        current = transform_similarity(v, r, s, t)
        sample_current = current[RNG.choice(len(current), min(len(current), 50000), replace=False)]
        cam_center = np.median(sample_current, axis=0)
        _, axis_name, deg_text = selection_label.split("_")
        axis = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}[axis_name]
        q = rot_axis_angle(axis, float(deg_text))
        p = apply_camera_rotation(current, cam_center, q)
    else:
        raise ValueError(selection_label)
    p, _, _ = snap_xy(p, k, target_bbox, w, h)
    out = mesh.copy()
    out.vertices = p
    return out


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    k = np.asarray(report["camera_k"], dtype=np.float64)
    color = cv2.imread(str(COLOR), cv2.IMREAD_UNCHANGED)
    depth_obs = cv2.imread(str(DEPTH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    summary = {
        "method": "Visibility-aware flip candidate scoring. Each candidate is rendered as a point-splat z-buffer, then scored against SAM mask IoU/coverage/leakage and aligned-depth visible-surface error. This is meant to resolve 180-degree orientation ambiguity left by ICP/bbox fitting.",
        "source_report": str(REPORT),
        "parts": {},
        "outputs": {},
    }
    combined = trimesh.Scene()
    for name, part in report["parts"].items():
        mesh = load_mesh(part["source_mesh"])
        res = part_candidates(name, mesh, part, k, depth_obs, color)
        best = res["top"][0]
        selected = export_selected(name, mesh, part, best["label"], k)
        glb = OUT / f"{name}_visibility_best.glb"
        ply = OUT / f"{name}_visibility_best.ply"
        selected.export(glb)
        selected.export(ply)
        combined.add_geometry(selected, geom_name=name, node_name=name)
        summary["parts"][name] = {
            "selected_label": best["label"],
            "selected_kind": best["kind"],
            "selected_score": best["score"],
            "selected_metrics": best["metrics"],
            "candidate_contact_sheet": res["contact_sheet"],
            "top_candidates": res["top"],
            "outputs": {"glb": str(glb), "ply": str(ply)},
        }
    combined_path = OUT / "cabinet_drawer_visibility_best_open.glb"
    combined.export(combined_path)
    summary["outputs"]["combined_open_glb"] = str(combined_path)
    report_path = OUT / "visibility_flip_score_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_open_glb": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
