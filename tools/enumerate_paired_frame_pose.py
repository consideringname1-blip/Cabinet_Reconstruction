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
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/paired_frame_enumeration"
PLANE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/mesh_plane_components/mesh_plane_components_report.json"
RGBD_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json"
COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
MESHES = {
    "base": ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb",
    "drawer": ROOT / "data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb",
}
COLORS = {"base": (0, 220, 0), "drawer": (0, 0, 240)}
RNG = np.random.default_rng(20260625)


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return trimesh.util.concatenate([g.copy() for g in loaded.geometry.values()])


def rot_axis_angle(axis, radians):
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(radians)
    s = math.sin(radians)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def project(points, k):
    p = points[points[:, 2] > 1e-5]
    return np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))


def zbuffer(points, k, w, h):
    p = points[points[:, 2] > 1e-5]
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if ok.any():
        np.minimum.at(depth, (py[ok], px[ok]), p[:, 2][ok])
    return depth


def mask_bbox(mask_path):
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    return mask, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def snap_xy(points, k, bbox, w, h):
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inb]
    if len(uv) < 100:
        return points, np.zeros(3), {}
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    center = (lo + hi) * 0.5
    target = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])
    z = float(np.median(points[:, 2]))
    delta_uv = target - center
    delta = np.array([delta_uv[0] * z / k[0, 0], delta_uv[1] * z / k[1, 1], 0.0])
    shifted = points + delta
    uv2 = project(shifted, k)
    inb2 = (uv2[:, 0] >= 0) & (uv2[:, 0] < w) & (uv2[:, 1] >= 0) & (uv2[:, 1] < h)
    uv2 = uv2[inb2]
    lo2 = np.percentile(uv2, 1, axis=0)
    hi2 = np.percentile(uv2, 99, axis=0)
    return shifted, delta, {
        "xy_delta_m": delta.tolist(),
        "bbox_after": [float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1])],
        "center_after": ((lo2 + hi2) * 0.5).tolist(),
    }


def score_points(points, mask, depth_obs, k, bbox):
    h, w = mask.shape
    good = points[:, 2] > 1e-5
    p = points[good]
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    inb = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if inb.sum() < 100:
        return 1e9, {"reason": "empty"}
    px = px[inb]
    py = py[inb]
    p = p[inb]
    uv = uv[inb]
    target_hit = mask[py, px] > 0
    valid = depth_obs[py, px] > 0.05
    inter = target_hit & valid
    if inter.sum() < 80:
        return 1e9, {"reason": "no_mask_depth_intersection"}
    diff = np.abs(p[:, 2][inter] - depth_obs[py[inter], px[inter]])
    pred = np.zeros((h, w), dtype=bool)
    pred[py, px] = True
    target = mask > 0
    iou = float((pred & target).sum() / max(1, (pred | target).sum()))
    coverage = float((pred & target).sum() / max(1, target.sum()))
    leakage = float((pred & ~target).sum() / max(1, pred.sum()))
    lo = np.percentile(uv, 1, axis=0)
    hi = np.percentile(uv, 99, axis=0)
    target_bbox = np.asarray(bbox, dtype=np.float64)
    target_size = np.maximum(target_bbox[2:] - target_bbox[:2], 1)
    size_err = float(np.linalg.norm(((hi - lo) - target_size) / target_size))
    center_err = float(np.linalg.norm((((lo + hi) * 0.5) - ((target_bbox[:2] + target_bbox[2:]) * 0.5)) / target_size))
    med = float(np.median(diff))
    p80 = float(np.percentile(diff, 80))
    hit_ratio = float(target_hit.mean())
    value = med / 0.025 + 0.25 * p80 / 0.055 + 0.85 * (1 - hit_ratio) + 0.55 * (1 - coverage) + 0.55 * leakage + 0.65 * size_err + 0.9 * center_err
    return value, {
        "score": float(value),
        "depth_abs_median_m": med,
        "depth_abs_p80_m": p80,
        "mask_iou": iou,
        "target_coverage": coverage,
        "leakage": leakage,
        "point_mask_hit_ratio": hit_ratio,
        "bbox_xyxy_p01_p99": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "bbox_size_err": size_err,
        "bbox_center_err": center_err,
        "intersection_points": int(inter.sum()),
    }

def signed_permutation_frames():
    # Columns are target horizontal, vertical, front axes expressed in source frame.
    frames = []
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            f = p @ np.diag(signs)
            if np.linalg.det(f) > 0:
                frames.append(f)
    return frames


def source_frame_from_planes(front_n, aux_n, aux_kind):
    n = normalize(front_n)
    aux = normalize(aux_n - n * float(aux_n @ n))
    if aux_kind == "horizontal":
        h = aux
        v = normalize(np.cross(n, h))
    else:
        v = aux
        h = normalize(np.cross(v, n))
    frame = np.column_stack([h, v, n])
    if np.linalg.det(frame) < 0:
        h = -h
        frame = np.column_stack([h, v, n])
    return frame


def make_mesh_frames(planes, max_planes):
    frames = []
    top = planes[:max_planes]
    for fp in top:
        fn = np.asarray(fp["normal"], dtype=np.float64)
        for ap in top:
            if ap["index"] == fp["index"]:
                continue
            an = np.asarray(ap["normal"], dtype=np.float64)
            if abs(float(fn @ an)) > 0.42:
                continue
            for sf, sa, aux_kind in itertools.product([-1.0, 1.0], [-1.0, 1.0], ["horizontal", "vertical"]):
                frame = source_frame_from_planes(sf * fn, sa * an, aux_kind)
                frames.append(
                    {
                        "frame": frame,
                        "front_plane": fp["index"],
                        "aux_plane": ap["index"],
                        "front_sign": sf,
                        "aux_sign": sa,
                        "aux_kind": aux_kind,
                    }
                )
    return frames


def drawer_rgbd_frame(depth_obs, k):
    mask, _ = mask_bbox(MASKS["drawer"])
    ys, xs = np.where((mask > 0) & (depth_obs > 0.05))
    z = depth_obs[ys, xs].astype(np.float64)
    pts = np.column_stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z))
    if len(pts) > 30000:
        pts = pts[RNG.choice(len(pts), 30000, replace=False)]
    best = None
    for _ in range(1800):
        p0, p1, p2 = pts[RNG.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-8:
            continue
        n = normalize(n)
        d = -float(n @ p0)
        inliers = np.abs(pts @ n + d) < 0.014
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), inliers)
    inlier_pts = pts[best[1]]
    center = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - center, full_matrices=False)
    n = normalize(vh[-1])
    if n @ center > 0:
        n = -n
    image_y = np.array([0.0, 1.0, 0.0])
    v = normalize(image_y - n * float(image_y @ n))
    h = normalize(np.cross(v, n))
    return {"frame": np.column_stack([h, v, n]), "h": h, "v": v, "n": n, "center": center, "inliers": best[0], "inlier_ratio": float(best[0] / len(pts))}


def initial_part_state(name, mesh, rgbd_report):
    part = rgbd_report["parts"][name]
    r = np.asarray(part["rotation_matrix_source_to_camera"], dtype=np.float64)
    s = float(part["scale_uniform"])
    t = np.asarray(part["translation_camera_m_after"], dtype=np.float64)
    vertices = s * (np.asarray(mesh.vertices, dtype=np.float64) @ r.T) + t
    sample, _ = trimesh.sample.sample_surface(mesh, 5000 if name == "base" else 4000)
    sample = s * (sample @ r.T) + t
    return vertices, sample, r, s, t


def eval_part_candidates(name, mesh, mesh_frames, target_frame, k, depth_obs, rgbd_report, top_keep=24):
    mask, bbox = mask_bbox(MASKS[name])
    vertices0, sample0, r0, _, _ = initial_part_state(name, mesh, rgbd_report)
    src_current_to_source = r0.T
    source_to_current = r0
    center_sample = np.median(sample0, axis=0)
    candidates = []
    cube_frames = signed_permutation_frames()
    for mf in mesh_frames:
        # Current axes of this semantic source frame.
        current_frame = source_to_current @ mf["frame"]
        for perm_frame in cube_frames:
            desired = target_frame @ perm_frame
            q = desired @ current_frame.T
            if np.linalg.det(q) < 0.0:
                continue
            p0 = center_sample + ((sample0 - center_sample) @ q.T)
            for dz in (-0.025, 0.0, 0.025):
                p = p0.copy()
                p[:, 2] += dz
                p, delta, snap = snap_xy(p, k, bbox, mask.shape[1], mask.shape[0])
                sc, metrics = score_points(p, mask, depth_obs, k, bbox)
                metrics["snap"] = snap
                # Small bias against mappings whose front axis is not close to target front.
                front_align = abs(float((q @ current_frame[:, 2]) @ target_frame[:, 2]))
                structural_penalty = 0.35 * (1.0 - front_align)
                total = float(sc + structural_penalty)
                candidates.append(
                    {
                        "score": total,
                        "image_score": float(sc),
                        "metrics": metrics,
                        "q": q,
                        "dz_m": float(dz),
                        "xy_delta_m": delta,
                        "mesh_frame": {k: v for k, v in mf.items() if k != "frame"},
                        "perm_frame": perm_frame,
                        "front_align_abs": front_align,
                    }
                )
    candidates.sort(key=lambda x: x["score"])
    deduped = []
    for cand in candidates:
        key = (
            cand["mesh_frame"]["front_plane"],
            cand["mesh_frame"]["aux_plane"],
            cand["mesh_frame"]["front_sign"],
            cand["mesh_frame"]["aux_sign"],
            cand["mesh_frame"]["aux_kind"],
            tuple(np.round(cand["perm_frame"].reshape(-1), 3)),
            cand["dz_m"],
        )
        if key in {d["key"] for d in deduped}:
            continue
        deduped.append({"key": key, **cand})
        if len(deduped) >= top_keep:
            break
    for d in deduped:
        d.pop("key", None)
    return deduped, vertices0, sample0


def apply_candidate(mesh, vertices0, cand):
    center = np.median(vertices0, axis=0)
    vertices = center + ((vertices0 - center) @ cand["q"].T)
    vertices[:, 2] += cand["dz_m"]
    vertices += cand["xy_delta_m"]
    out = mesh.copy()
    out.vertices = vertices
    return out


def draw_part_overlay(color, points, k, mask, bgr, path):
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    uv = project(points, k)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
    pix = np.round(uv[inb]).astype(np.int32)
    step = max(1, len(pix) // 45000)
    for x, y in pix[::step]:
        cv2.circle(img, (int(x), int(y)), 1, bgr, -1)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr, 2)
    cv2.imwrite(str(path), img)


def draw_combined(color, samples, k, output):
    img = color.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for name, pts in samples.items():
        uv = project(pts, k)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 45000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, COLORS[name], -1)
        mask, bbox = mask_bbox(MASKS[name])
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[name], 2)
    cv2.imwrite(str(output), img)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rgbd_report = json.loads(RGBD_REPORT.read_text(encoding="utf-8"))
    plane_report = json.loads(PLANE_REPORT.read_text(encoding="utf-8"))
    k = np.asarray(json.loads(META_PATH.read_text(encoding="utf-8"))["PVCamera"]["k"], dtype=np.float64)
    depth_obs = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    target = drawer_rgbd_frame(depth_obs, k)["frame"]
    meshes = {name: load_mesh(path) for name, path in MESHES.items()}
    mesh_frames = {
        "base": make_mesh_frames(plane_report["parts"]["base"]["planes"], 7),
        "drawer": make_mesh_frames(plane_report["parts"]["drawer"]["planes"], 5),
    }
    part_candidates = {}
    initial_vertices = {}
    initial_samples = {}
    for name in ("base", "drawer"):
        cands, verts0, sample0 = eval_part_candidates(name, meshes[name], mesh_frames[name], target, k, depth_obs, rgbd_report)
        part_candidates[name] = cands
        initial_vertices[name] = verts0
        initial_samples[name] = sample0

    paired = []
    for bi, bc in enumerate(part_candidates["base"][:16]):
        for di, dc in enumerate(part_candidates["drawer"][:16]):
            # Since both candidates map their selected geometric frames into the same target frame
            # up to signed permutations, this penalty guards against selecting unrelated semantic faces.
            frame_delta = bc["perm_frame"].T @ dc["perm_frame"]
            off_diag = frame_delta - np.diag(np.diag(frame_delta))
            relation_penalty = 0.55 * float(np.linalg.norm(off_diag))
            total = bc["score"] + dc["score"] + relation_penalty
            paired.append({"score": total, "base_index": bi, "drawer_index": di, "relation_penalty": relation_penalty})
    paired.sort(key=lambda x: x["score"])

    exports = []
    for rank, pair in enumerate(paired[:8], start=1):
        base_cand = part_candidates["base"][pair["base_index"]]
        drawer_cand = part_candidates["drawer"][pair["drawer_index"]]
        base_mesh = apply_candidate(meshes["base"], initial_vertices["base"], base_cand)
        drawer_mesh = apply_candidate(meshes["drawer"], initial_vertices["drawer"], drawer_cand)
        scene = trimesh.Scene()
        scene.add_geometry(base_mesh, geom_name="base", node_name="base")
        scene.add_geometry(drawer_mesh, geom_name="drawer", node_name="drawer")
        glb = OUT / f"cabinet_drawer_paired_frame_rank{rank:02d}.glb"
        scene.export(glb)
        base_sample, _ = trimesh.sample.sample_surface(base_mesh, 70000)
        drawer_sample, _ = trimesh.sample.sample_surface(drawer_mesh, 50000)
        overlay = OUT / f"paired_frame_rank{rank:02d}_projection_overlay.png"
        draw_combined(color, {"base": base_sample, "drawer": drawer_sample}, k, overlay)
        exports.append(
            {
                "rank": rank,
                "combined_open_glb": str(glb),
                "projection_overlay": str(overlay),
                "pair_score": pair["score"],
                "relation_penalty": pair["relation_penalty"],
                "base": {
                    "candidate_index": pair["base_index"],
                    "score": base_cand["score"],
                    "image_score": base_cand["image_score"],
                    "metrics": base_cand["metrics"],
                    "mesh_frame": base_cand["mesh_frame"],
                    "perm_frame": base_cand["perm_frame"].tolist(),
                    "front_align_abs": base_cand["front_align_abs"],
                },
                "drawer": {
                    "candidate_index": pair["drawer_index"],
                    "score": drawer_cand["score"],
                    "image_score": drawer_cand["image_score"],
                    "metrics": drawer_cand["metrics"],
                    "mesh_frame": drawer_cand["mesh_frame"],
                    "perm_frame": drawer_cand["perm_frame"].tolist(),
                    "front_align_abs": drawer_cand["front_align_abs"],
                },
            }
        )

    report = {
        "method": "Paired enumeration of true mesh plane frames. For each part, actual detected SAM3D mesh planes form source front/horizontal/vertical frames. All 24 proper signed axis permutations are enumerated into one shared drawer RGB-D frame, then base/drawer pairs are scored together. This enumerates relative orthogonal poses rather than rotating one part independently.",
        "target_frame_columns_h_v_n": target.tolist(),
        "candidate_counts": {name: len(part_candidates[name]) for name in ("base", "drawer")},
        "top_part_candidates": {
            name: [
                {
                    "rank": i + 1,
                    "score": c["score"],
                    "image_score": c["image_score"],
                    "metrics": c["metrics"],
                    "mesh_frame": c["mesh_frame"],
                    "perm_frame": c["perm_frame"].tolist(),
                    "front_align_abs": c["front_align_abs"],
                }
                for i, c in enumerate(part_candidates[name][:16])
            ]
            for name in ("base", "drawer")
        },
        "exports": exports,
    }
    report_path = OUT / "paired_frame_enumeration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "top_glb": exports[0]["combined_open_glb"] if exports else None, "top_overlay": exports[0]["projection_overlay"] if exports else None, "exports": len(exports)}, indent=2))


if __name__ == "__main__":
    main()
