import itertools
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/plane_component_roll_snap"
PLANE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/mesh_plane_components/mesh_plane_components_report.json"
RGBD_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json"
COLOR_PATH = ROOT / "data/upload/larm_captures/20260622_081031_636398Z/color.png"
DEPTH_PATH = ROOT / "data/output/hololens2/20260622_081031_636398Z_align_depth.png"

PARTS = {
    "base": {
        "mesh": ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb",
        "mask": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
        "color": (0, 220, 0),
    },
    "drawer": {
        "mesh": ROOT / "data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb",
        "mask": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
        "color": (0, 0, 240),
    },
}

RNG = np.random.default_rng(20260624)


def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    return trimesh.util.concatenate([g.copy() for g in scene.geometry.values()])


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def rot_axis_angle(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


def signed_angle_around(a, b, axis):
    a = normalize(a - axis * float(a @ axis))
    b = normalize(b - axis * float(b @ axis))
    return math.atan2(float(axis @ np.cross(a, b)), float(a @ b))


def transform(v, r, s, t):
    return s * (v @ r.T) + t


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
    duv = tgt - ctr
    delta = np.array([duv[0] * z / k[0, 0], duv[1] * z / k[1, 1], 0.0])
    shifted = points + delta
    uv2 = project(shifted, k)
    inb2 = (uv2[:, 0] >= 0) & (uv2[:, 0] < w) & (uv2[:, 1] >= 0) & (uv2[:, 1] < h)
    uv2 = uv2[inb2]
    lo2 = np.percentile(uv2, 1, axis=0)
    hi2 = np.percentile(uv2, 99, axis=0)
    return shifted, delta, {"bbox_after": [float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1])], "center_after": ((lo2 + hi2) * 0.5).tolist(), "xy_delta_m": delta.tolist()}


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
    value = med / 0.025 + 0.3 * p80 / 0.055 + 1.2 * (1 - iou) + 0.5 * (1 - cov) + 0.8 * leak + 0.6 * size_err + center_err
    return value, {"score": float(value), "depth_abs_median_m": med, "depth_abs_p80_m": p80, "mask_iou": iou, "target_coverage": cov, "leakage": leak, "bbox_xyxy_p01_p99": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])], "bbox_size_err": size_err, "bbox_center_err": center_err}


def fit_drawer_front_frame(depth_mm, k):
    mask = cv2.imread(str(PARTS["drawer"]["mask"]), cv2.IMREAD_GRAYSCALE) > 127
    ys, xs = np.where(mask & (depth_mm > 0))
    z = depth_mm[ys, xs].astype(float) / 1000.0
    pts = np.column_stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z))
    pts = pts[(pts[:, 2] > 0.2) & (pts[:, 2] < 4.0)]
    if len(pts) > 25000:
        pts = pts[RNG.choice(len(pts), 25000, replace=False)]
    best = None
    for _ in range(1600):
        p0, p1, p2 = pts[RNG.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-8:
            continue
        n = normalize(n)
        d = -float(n @ p0)
        inl = np.abs(pts @ n + d) < 0.014
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    center = pts[best[1]].mean(axis=0)
    _, _, vh = np.linalg.svd(pts[best[1]] - center, full_matrices=False)
    n = vh[-1]
    if n @ center > 0:
        n = -n
    n = normalize(n)
    image_y = np.array([0.0, 1.0, 0.0])
    v = normalize(image_y - n * float(image_y @ n))
    h = normalize(np.cross(v, n))
    return {"n": n, "v": v, "h": h, "center": center, "inliers": best[0], "inlier_ratio": float(best[0] / len(pts))}


def mask_bbox(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], m.astype(np.uint8)


def fit_part(name, mesh, planes, part_report, frame, k, depth_obs):
    r0 = np.asarray(part_report["rotation_matrix_source_to_camera"], dtype=float)
    s = float(part_report["scale_uniform"])
    t = np.asarray(part_report["translation_camera_m_after"], dtype=float)
    bbox, mask = mask_bbox(PARTS[name]["mask"])
    sample, _ = trimesh.sample.sample_surface(mesh, 60000 if name == "base" else 45000)
    current = transform(sample, r0, s, t)
    center_cam = np.median(current, axis=0)
    front = frame["n"]
    target_auxes = {"horizontal": frame["h"], "vertical": frame["v"]}
    candidates = []

    # Candidate front planes are actual mesh planes already nearly parallel to observed front in current fit.
    front_planes = []
    for pl in planes[:8]:
        n_cam = normalize(r0 @ np.asarray(pl["normal"], dtype=float))
        if abs(float(n_cam @ front)) > 0.86:
            front_planes.append(pl)
    if not front_planes:
        front_planes = planes[:4]

    for fp in front_planes:
        fn_src = np.asarray(fp["normal"], dtype=float)
        for ap in planes[:8]:
            if ap["index"] == fp["index"]:
                continue
            an_src = np.asarray(ap["normal"], dtype=float)
            if abs(float(fn_src @ an_src)) > 0.42:
                continue
            for sign_f, sign_a, aux_name in itertools.product([-1.0, 1.0], [-1.0, 1.0], ["horizontal", "vertical"]):
                fn_cam = normalize(sign_f * (r0 @ fn_src))
                # First align the chosen front plane to the observed front normal.
                axis1 = np.cross(fn_cam, front)
                if np.linalg.norm(axis1) < 1e-6:
                    q1 = np.eye(3)
                else:
                    q1 = rot_axis_angle(axis1, math.atan2(np.linalg.norm(axis1), float(fn_cam @ front)))
                an_cam = normalize(q1 @ (sign_a * (r0 @ an_src)))
                target_aux = target_auxes[aux_name]
                angle = signed_angle_around(an_cam, target_aux, front)
                q2 = rot_axis_angle(front, angle)
                q = q2 @ q1
                for dz in (-0.025, 0.0, 0.025):
                    p = center_cam + ((current - center_cam) @ q.T)
                    p[:, 2] += dz
                    p, delta, snap = snap_xy(p, k, bbox, mask.shape[1], mask.shape[0])
                    sc, metrics = score(p, mask, depth_obs, k, bbox)
                    metrics["snap"] = snap
                    candidates.append((sc, q, dz, delta, fp, ap, sign_f, sign_a, aux_name, metrics))
    candidates.sort(key=lambda c: c[0])
    sc, q, dz, delta, fp, ap, sign_f, sign_a, aux_name, metrics = candidates[0]
    verts_current = transform(np.asarray(mesh.vertices, dtype=float), r0, s, t)
    verts = center_cam + ((verts_current - center_cam) @ q.T)
    verts[:, 2] += dz
    verts += delta
    fitted = mesh.copy()
    fitted.vertices = verts
    return {
        "mesh": fitted,
        "selected": {
            "score": float(sc),
            "front_plane": fp["index"],
            "aux_plane": ap["index"],
            "front_sign": sign_f,
            "aux_sign": sign_a,
            "aux_target": aux_name,
            "roll_snap_rotation_camera": q.tolist(),
            "dz_m": float(dz),
            "xy_delta_m": delta.tolist(),
            "metrics": metrics,
        },
        "top_candidates": [
            {"rank": i + 1, "score": float(c[0]), "front_plane": c[4]["index"], "aux_plane": c[5]["index"], "front_sign": c[6], "aux_sign": c[7], "aux_target": c[8], "dz_m": float(c[2]), "xy_delta_m": c[3].tolist(), "metrics": c[9]}
            for i, c in enumerate(candidates[:12])
        ],
    }


def draw_overlay(color, samples, k, out):
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for name, pts in samples.items():
        uv = project(pts, k)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        pix = np.round(uv[inb]).astype(int)
        step = max(1, len(pix) // 50000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, PARTS[name]["color"], -1)
        bbox, _ = mask_bbox(PARTS[name]["mask"])
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), PARTS[name]["color"], 2)
    cv2.imwrite(str(out), img)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rgbd = json.loads(RGBD_REPORT.read_text(encoding="utf-8"))
    planes = json.loads(PLANE_REPORT.read_text(encoding="utf-8"))
    k = np.asarray(rgbd["camera_k"], dtype=float)
    depth_obs = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    frame = fit_drawer_front_frame((depth_obs * 1000).astype(np.uint16), k)
    report = {
        "method": "Local plane-component roll snap. Start from the good RGB-D similarity fit, then rotate each part only enough to align actual detected mesh plane normals to the shared RGB-D front/horizontal/vertical frame. This constrains mesh geometry planes, not model coordinate axes.",
        "frame": {k2: (v.tolist() if isinstance(v, np.ndarray) else v) for k2, v in frame.items()},
        "parts": {},
        "outputs": {},
    }
    scene = trimesh.Scene()
    overlay_samples = {}
    for name in ("base", "drawer"):
        mesh = load_mesh(PARTS[name]["mesh"])
        res = fit_part(name, mesh, planes["parts"][name]["planes"], rgbd["parts"][name], frame, k, depth_obs)
        glb = OUT / f"{name}_plane_roll_snap.glb"
        ply = OUT / f"{name}_plane_roll_snap.ply"
        res["mesh"].export(glb)
        res["mesh"].export(ply)
        scene.add_geometry(res["mesh"], geom_name=name, node_name=name)
        sample, _ = trimesh.sample.sample_surface(res["mesh"], 70000 if name == "base" else 50000)
        overlay_samples[name] = sample
        report["parts"][name] = {"selected": res["selected"], "top_candidates": res["top_candidates"], "outputs": {"glb": str(glb), "ply": str(ply)}}
    combined = OUT / "cabinet_drawer_plane_roll_snap_open.glb"
    scene.export(combined)
    overlay = OUT / "plane_roll_snap_projection_overlay.png"
    draw_overlay(color, overlay_samples, k, overlay)
    report["outputs"] = {"combined_open_glb": str(combined), "projection_overlay": str(overlay)}
    report_path = OUT / "plane_roll_snap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_open_glb": str(combined), "overlay": str(overlay)}, indent=2))


if __name__ == "__main__":
    main()
