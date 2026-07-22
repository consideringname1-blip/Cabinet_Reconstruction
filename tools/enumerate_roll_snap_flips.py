import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
SRC_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/plane_component_roll_snap"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/plane_flip_enumeration"
REPORT = SRC_DIR / "plane_roll_snap_report.json"
COLOR_PATH = ROOT / "data/upload/larm_captures/20260622_081031_636398Z/color.png"
DEPTH_PATH = ROOT / "data/output/hololens2/20260622_081031_636398Z_align_depth.png"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
MESHES = {
    "base": SRC_DIR / "base_plane_roll_snap.glb",
    "drawer": SRC_DIR / "drawer_plane_roll_snap.glb",
}
COLORS = {"base": (0, 220, 0), "drawer": (0, 0, 240)}
RNG = np.random.default_rng(20260625)


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def rot_axis_angle(axis, deg):
    axis = normalize(axis)
    a = math.radians(deg)
    x, y, z = axis
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


def load_mesh(path):
    scene = trimesh.load(path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    return trimesh.util.concatenate([g.copy() for g in scene.geometry.values()])


def project(points, k):
    p = points[points[:, 2] > 1e-5]
    return np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))


def zbuffer(points, k, w, h):
    p = points[points[:, 2] > 1e-5]
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    px = np.round(uv[:, 0]).astype(int)
    py = np.round(uv[:, 1]).astype(int)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if ok.any():
        np.minimum.at(depth, (py[ok], px[ok]), p[:, 2][ok])
    return depth


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def snap_xy(points, k, bbox, w, h):
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inb]
    if len(uv) < 100:
        return points, np.zeros(3), {}
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    ctr = (lo + hi) * 0.5
    tgt = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])
    z = float(np.median(points[:, 2]))
    delta_uv = tgt - ctr
    delta = np.array([delta_uv[0] * z / k[0, 0], delta_uv[1] * z / k[1, 1], 0.0])
    shifted = points + delta
    uv2 = project(shifted, k)
    inb2 = (uv2[:, 0] >= 0) & (uv2[:, 0] < w) & (uv2[:, 1] >= 0) & (uv2[:, 1] < h)
    uv2 = uv2[inb2]
    if len(uv2) < 100:
        return shifted, delta, {"xy_delta_m": delta.tolist()}
    lo2 = np.percentile(uv2, 1, axis=0)
    hi2 = np.percentile(uv2, 99, axis=0)
    return shifted, delta, {
        "xy_delta_m": delta.tolist(),
        "bbox_after": [float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1])],
        "center_after": ((lo2 + hi2) * 0.5).tolist(),
    }


def score(points, mask, depth_obs, k, bbox):
    h, w = mask.shape
    dr = zbuffer(points, k, w, h)
    pred = np.isfinite(dr)
    target = mask > 0
    valid = depth_obs > 0.05
    inter = pred & target & valid
    if inter.sum() < 100:
        return 1e9, {"reason": "empty"}
    diff = np.abs(dr[inter] - depth_obs[inter])
    iou = float((pred & target).sum() / max(1, (pred | target).sum()))
    cov = float((pred & target).sum() / max(1, target.sum()))
    leak = float((pred & ~target).sum() / max(1, pred.sum()))
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inb]
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    target_bbox = np.asarray(bbox, dtype=float)
    target_size = np.maximum(target_bbox[2:] - target_bbox[:2], 1)
    size_err = float(np.linalg.norm(((hi - lo) - target_size) / target_size))
    center_err = float(np.linalg.norm((((lo + hi) * 0.5) - ((target_bbox[:2] + target_bbox[2:]) * 0.5)) / target_size))
    med = float(np.median(diff))
    p80 = float(np.percentile(diff, 80))
    value = med / 0.025 + 0.30 * p80 / 0.055 + 1.45 * (1 - iou) + 0.55 * (1 - cov) + 0.85 * leak + 0.55 * size_err + 0.9 * center_err
    return value, {
        "score": float(value),
        "depth_abs_median_m": med,
        "depth_abs_p80_m": p80,
        "mask_iou": iou,
        "target_coverage": cov,
        "leakage": leak,
        "bbox_xyxy_p01_p99": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "bbox_size_err": size_err,
        "bbox_center_err": center_err,
        "intersection_pixels": int(inter.sum()),
    }


def draw_overlay(color, points, k, mask, bgr, path):
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
    pix = np.round(uv[inb]).astype(int)
    step = max(1, len(pix) // 35000)
    for x, y in pix[::step]:
        cv2.circle(img, (int(x), int(y)), 1, bgr, -1)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr, 2)
    cv2.imwrite(str(path), img)


def contact_sheet(paths, labels, output):
    imgs = []
    for path, label in zip(paths, labels):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (640, 34), (255, 255, 255), -1)
        cv2.putText(img, label[:70], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        imgs.append(img)
    while len(imgs) % 2:
        imgs.append(np.zeros_like(imgs[0]) + 255)
    rows = [np.hstack(imgs[i : i + 2]) for i in range(0, len(imgs), 2)]
    cv2.imwrite(str(output), np.vstack(rows))


def candidate_rotations(frame):
    axes = {"front": np.array(frame["n"]), "vertical": np.array(frame["v"]), "horizontal": np.array(frame["h"])}
    candidates = [("identity", np.eye(3))]
    for axis_name, axis in axes.items():
        for deg in (90, 180, 270):
            candidates.append((f"{axis_name}_{deg}", rot_axis_angle(axis, deg)))
    # Combined 180-degree flips.
    for a, b in [("front", "vertical"), ("front", "horizontal"), ("vertical", "horizontal")]:
        candidates.append((f"{a}_180__{b}_180", rot_axis_angle(axes[b], 180) @ rot_axis_angle(axes[a], 180)))
    return candidates


def eval_part(name, mesh, frame, k, depth_obs, color):
    mask = (cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    bbox = bbox_from_mask(mask)
    sample, _ = trimesh.sample.sample_surface(mesh, 90000 if name == "base" else 70000)
    center = np.median(sample, axis=0)
    part_dir = OUT / name
    part_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, q in candidate_rotations(frame):
        p = center + ((sample - center) @ q.T)
        for dz in (-0.035, -0.0175, 0.0, 0.0175, 0.035):
            pp = p.copy()
            pp[:, 2] += dz
            pp, delta, snap = snap_xy(pp, k, bbox, mask.shape[1], mask.shape[0])
            sc, metrics = score(pp, mask, depth_obs, k, bbox)
            metrics["snap"] = snap
            overlay = part_dir / f"{label}_dz_{dz:+.4f}.png"
            draw_overlay(color, pp, k, mask, COLORS[name], overlay)
            results.append({"label": label, "dz_m": dz, "score": float(sc), "metrics": metrics, "overlay": str(overlay)})
    results.sort(key=lambda r: r["score"])
    contact = part_dir / f"{name}_flip_candidates_contact_sheet.png"
    contact_sheet([Path(r["overlay"]) for r in results[:10]], [f"{i+1}. {r['label']} dz={r['dz_m']:+.3f} s={r['score']:.2f}" for i, r in enumerate(results[:10])], contact)
    best = results[0]
    # Apply selected transform to full mesh.
    q = dict(candidate_rotations(frame))[best["label"]]
    vertices = np.asarray(mesh.vertices, dtype=float)
    full_center = np.median(vertices, axis=0)
    transformed = full_center + ((vertices - full_center) @ q.T)
    transformed[:, 2] += best["dz_m"]
    transformed, delta, snap = snap_xy(transformed, k, bbox, mask.shape[1], mask.shape[0])
    out_mesh = mesh.copy()
    out_mesh.vertices = transformed
    return out_mesh, {"selected": best, "contact_sheet": str(contact), "top_candidates": results[:12]}


def draw_combined(color, samples, k, output):
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for name, pts in samples.items():
        uv = project(pts, k)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        pix = np.round(uv[inb]).astype(int)
        step = max(1, len(pix) // 45000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, COLORS[name], -1)
        mask = (cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        bbox = bbox_from_mask(mask)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[name], 2)
    cv2.imwrite(str(output), img)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    frame = report["frame"]
    k = np.asarray(report.get("camera_k") or json.loads((ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json").read_text())["camera_k"], dtype=float)
    depth_obs = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    scene = trimesh.Scene()
    summary = {
        "method": "Discrete flip enumeration around plane_component_roll_snap result. Candidate rotations are 90/180/270 degrees around the shared front, vertical, and horizontal cabinet-frame axes, followed by depth offset and xy center snap.",
        "source": str(SRC_DIR),
        "parts": {},
        "outputs": {},
    }
    overlay_samples = {}
    for name in ("base", "drawer"):
        mesh = load_mesh(MESHES[name])
        fitted, res = eval_part(name, mesh, frame, k, depth_obs, color)
        glb = OUT / f"{name}_flip_best.glb"
        ply = OUT / f"{name}_flip_best.ply"
        fitted.export(glb)
        fitted.export(ply)
        scene.add_geometry(fitted, geom_name=name, node_name=name)
        sample, _ = trimesh.sample.sample_surface(fitted, 70000 if name == "base" else 50000)
        overlay_samples[name] = sample
        summary["parts"][name] = {**res, "outputs": {"glb": str(glb), "ply": str(ply)}}
    combined = OUT / "cabinet_drawer_flip_best_open.glb"
    scene.export(combined)
    overlay = OUT / "flip_best_projection_overlay.png"
    draw_combined(color, overlay_samples, k, overlay)
    summary["outputs"] = {"combined_open_glb": str(combined), "projection_overlay": str(overlay)}
    report_path = OUT / "flip_enumeration_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_open_glb": str(combined), "overlay": str(overlay)}, indent=2))


if __name__ == "__main__":
    main()
