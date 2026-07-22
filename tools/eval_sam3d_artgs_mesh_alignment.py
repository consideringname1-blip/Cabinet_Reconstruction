import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import build_rgbd_real_pointcloud_animation as glb
import estimate_artgs_storage_joint_0004 as est
import fuse_artgs_multiview_and_eval_joint as mv


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/artgs_sam3d_eval/storage_45135_end0004"
SAM3D_OBJ = ROOT / "data/output/sam3d-objects/meshes/artgs_storage_45135_end0004_whole_rgb_sam3d_processed.obj"
SAM3D_RAW_GLB = ROOT / "data/output/sam3d-objects/meshes/artgs_storage_45135_end0004_whole_rgb_sam3d_raw.glb"
RNG = np.random.default_rng(20260722)


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}


def sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= max_points:
        return points
    return points[RNG.choice(len(points), size=max_points, replace=False)]


def load_obj(path):
    vertices = []
    faces = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            ids = []
            for token in line.split()[1:]:
                idx = int(token.split("/")[0])
                if idx < 0:
                    idx = len(vertices) + idx + 1
                ids.append(idx - 1)
            if len(ids) == 3:
                faces.append(ids)
            elif len(ids) > 3:
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# fitted SAM3DObject mesh in SAPIEN world coordinates\n")
        for v in np.asarray(vertices, dtype=np.float64):
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in np.asarray(faces, dtype=np.int32):
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def sample_mesh_surface(vertices, faces, count):
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int32)]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    probs = areas / max(float(areas.sum()), 1e-12)
    idx = RNG.choice(len(faces), size=count, replace=True, p=probs)
    chosen = tri[idx]
    r1 = np.sqrt(RNG.random(count))
    r2 = RNG.random(count)
    pts = (1.0 - r1)[:, None] * chosen[:, 0] + (r1 * (1.0 - r2))[:, None] * chosen[:, 1] + (r1 * r2)[:, None] * chosen[:, 2]
    return pts.astype(np.float32)


def pca_frame(points):
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    x = points - center
    cov = np.cov(x.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    frame = vecs[:, order]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1.0
    scale = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    return center, frame, max(scale, 1e-9)


def signed_permutation_matrices():
    mats = []
    for perm in permutations(range(3)):
        p = np.zeros((3, 3), dtype=np.float64)
        for i, j in enumerate(perm):
            p[i, j] = 1.0
        for signs in product([-1.0, 1.0], repeat=3):
            m = np.diag(signs) @ p
            if np.linalg.det(m) > 0.0:
                mats.append(m)
    return mats


def transform_points(points, scale, rot, trans):
    return float(scale) * (np.asarray(points, dtype=np.float64) @ np.asarray(rot, dtype=np.float64).T) + np.asarray(trans, dtype=np.float64).reshape(1, 3)


def estimate_similarity_umeyama(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    cov = (xd.T @ xs) / max(len(src), 1)
    u, _, vt = np.linalg.svd(cov)
    sfix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sfix[-1, -1] = -1.0
    rot = u @ sfix @ vt
    var = float(np.mean(np.sum(xs * xs, axis=1)))
    scale = float(np.trace(np.diag(np.linalg.svd(cov, compute_uv=False)) @ sfix) / max(var, 1e-12))
    # The trace expression above is intentionally simple, but singular values
    # need the reflection sign. Recompute explicitly for clarity.
    _, singular, _ = np.linalg.svd(cov)
    scale = float(np.sum(singular * np.diag(sfix)) / max(var, 1e-12))
    trans = cd - scale * (rot @ cs)
    return scale, rot, trans


def fit_similarity_icp(source_points, target_points):
    source_eval = sample_points(source_points, 50000)
    source_icp = sample_points(source_points, 14000)
    target_eval = sample_points(target_points, 90000)
    target_icp = sample_points(target_points, 120000)
    tree_eval = cKDTree(target_eval)
    tree_icp = cKDTree(target_icp)
    cs, fs, ss = pca_frame(source_eval)
    ct, ft, st = pca_frame(target_eval)
    candidates = []
    for m in signed_permutation_matrices():
        rot = ft @ m @ fs.T
        scale = st / ss
        trans = ct - scale * (rot @ cs)
        moved = transform_points(source_eval, scale, rot, trans)
        dist, _ = tree_eval.query(moved, k=1, workers=1)
        score = float(np.median(dist) + 0.35 * np.percentile(dist, 90))
        candidates.append({"scale": scale, "rot": rot, "trans": trans, "score": score, "init": m.astype(float).tolist()})
    candidates.sort(key=lambda x: x["score"])

    results = []
    for cand in candidates[:12]:
        scale = float(cand["scale"])
        rot = np.asarray(cand["rot"], dtype=np.float64)
        trans = np.asarray(cand["trans"], dtype=np.float64)
        history = []
        last_score = None
        for _ in range(32):
            moved = transform_points(source_icp, scale, rot, trans)
            dist, idx = tree_icp.query(moved, k=1, workers=1)
            cutoff = float(np.percentile(dist, 68))
            keep = dist <= cutoff
            if int(np.sum(keep)) < 500:
                keep = dist <= float(np.percentile(dist, 80))
            new_scale, new_rot, new_trans = estimate_similarity_umeyama(source_icp[keep], target_icp[idx[keep]])
            moved_new = transform_points(source_icp, new_scale, new_rot, new_trans)
            dist_new, _ = tree_icp.query(moved_new, k=1, workers=1)
            score = float(np.median(dist_new) + 0.35 * np.percentile(dist_new, 90))
            history.append({"score": score, "median_m": float(np.median(dist_new)), "p90_m": float(np.percentile(dist_new, 90)), "pairs": int(np.sum(keep))})
            scale, rot, trans = new_scale, new_rot, new_trans
            if last_score is not None and abs(score - last_score) < 1e-8:
                break
            last_score = score
        moved_eval = transform_points(source_eval, scale, rot, trans)
        dist_eval, _ = tree_eval.query(moved_eval, k=1, workers=1)
        final_score = float(np.median(dist_eval) + 0.35 * np.percentile(dist_eval, 90))
        results.append(
            {
                "scale": float(scale),
                "rot": rot,
                "trans": trans,
                "score": final_score,
                "source_to_target_percentiles_m": percentiles(dist_eval),
                "history": history,
                "init_score": float(cand["score"]),
            }
        )
    results.sort(key=lambda x: x["score"])
    return results[0], results[:6]


def nn_stats(source, target, max_source=100000, max_target=220000):
    src = sample_points(source, max_source)
    tgt = sample_points(target, max_target)
    tree = cKDTree(tgt)
    dist, _ = tree.query(src, k=1, workers=1)
    return {"source_points": int(len(src)), "target_points": int(len(tgt)), "percentiles_m": percentiles(dist)}


def chamfer_stats(a, b):
    ab = nn_stats(a, b)
    ba = nn_stats(b, a)
    return {
        "a_to_b": ab,
        "b_to_a": ba,
        "symmetric_median_m": float(0.5 * (ab["percentiles_m"]["50"] + ba["percentiles_m"]["50"])),
        "symmetric_p90_m": float(0.5 * (ab["percentiles_m"]["90"] + ba["percentiles_m"]["90"])),
    }


def build_eval_glb(path, fused_points, gt_points, sam3d_points):
    builder = glb.GlbBuilder("artgs-sam3d-fit-to-fused-rgbd-eval")
    parts = [
        ("fused_end_rgbd_green", sample_points(fused_points, 36000), [70, 225, 90, 255], 0.005),
        ("gt_end_mesh_cyan", sample_points(gt_points, 36000), [20, 230, 255, 255], 0.005),
        ("sam3d_fitted_yellow", sample_points(sam3d_points, 36000), [255, 210, 35, 255], 0.0055),
    ]
    for name, pts, color, half in parts:
        colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 4), (len(pts), 1))
        vertices, vertex_colors, indices = glb.make_splats(pts.astype(np.float32), colors, half_size=half)
        mesh = builder.add_mesh(name, vertices, vertex_colors, glb.TRIANGLES, indices)
        builder.add_node(name, mesh)
    builder.write(path)


def rotation_angle(rot):
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_vertices, source_faces = load_obj(SAM3D_OBJ)
    source_samples = sample_mesh_surface(source_vertices, source_faces, 90000)

    fused_end = mv.fuse_state("end")
    fused_points = fused_end["points"]

    gt_vertices, gt_faces = est.read_ply_mesh(est.DATASET / "gt/end/end_rotate.ply")
    gt_samples = est.sample_mesh_surface(gt_vertices, gt_faces, 240000)

    fit, top_fits = fit_similarity_icp(source_samples, fused_points)
    fitted_vertices = transform_points(source_vertices, fit["scale"], fit["rot"], fit["trans"]).astype(np.float32)
    fitted_samples = transform_points(source_samples, fit["scale"], fit["rot"], fit["trans"]).astype(np.float32)

    fitted_obj = OUT / "artgs_storage_45135_end0004_sam3d_fitted_to_fused_rgbd.obj"
    eval_glb = OUT / "artgs_storage_45135_end0004_sam3d_fitted_vs_fused_vs_gt.glb"
    write_obj(fitted_obj, fitted_vertices, source_faces)
    build_eval_glb(eval_glb, fused_points, gt_samples, fitted_samples)

    report = {
        "method": (
            "Generate SAM3DObject mesh from single end/train/rgba/0004 whole-object mask, "
            "fit it to multiview fused end-state RGB-D point cloud with best-case 7DoF similarity + trimmed ICP, "
            "then compare the fitted SAM3D surface against SAPIEN gt/end/end_rotate.ply."
        ),
        "note": "The 7DoF fitting is an evaluation upper bound for shape quality; it should not be interpreted as a reliable deployment pose estimator.",
        "inputs": {
            "sam3d_processed_obj": str(SAM3D_OBJ),
            "sam3d_raw_glb": str(SAM3D_RAW_GLB),
            "single_image": str(est.DATASET / "end/train/rgba/0004.png"),
            "single_image_mask": str(ROOT / "data/output/sam3/artgs_storage_45135_end0004_whole_mask.png"),
            "fused_end_views": fused_end["stats"],
            "gt_mesh": str(est.DATASET / "gt/end/end_rotate.ply"),
        },
        "sam3d_mesh_stats": {
            "vertices": int(len(source_vertices)),
            "faces": int(len(source_faces)),
            "source_surface_samples": int(len(source_samples)),
        },
        "fit_to_fused_rgbd": {
            "scale": float(fit["scale"]),
            "rotation_matrix": np.asarray(fit["rot"], dtype=float).tolist(),
            "rotation_angle_deg": rotation_angle(fit["rot"]),
            "translation_m": np.asarray(fit["trans"], dtype=float).tolist(),
            "score": float(fit["score"]),
            "source_to_fused_percentiles_m": fit["source_to_target_percentiles_m"],
            "top_fit_candidates": [
                {
                    "rank": i + 1,
                    "scale": float(item["scale"]),
                    "rotation_angle_deg": rotation_angle(item["rot"]),
                    "translation_m": np.asarray(item["trans"], dtype=float).tolist(),
                    "score": float(item["score"]),
                    "source_to_fused_percentiles_m": item["source_to_target_percentiles_m"],
                    "init_score": float(item["init_score"]),
                    "last_history": item["history"][-1] if item["history"] else {},
                }
                for i, item in enumerate(top_fits)
            ],
        },
        "metrics": {
            "sam3d_fitted_to_fused_rgbd": chamfer_stats(fitted_samples, fused_points),
            "sam3d_fitted_to_gt_mesh": chamfer_stats(fitted_samples, gt_samples),
            "fused_rgbd_to_gt_mesh_reference": chamfer_stats(fused_points, gt_samples),
        },
        "outputs": {
            "report": str(OUT / "sam3d_fitted_to_fused_rgbd_vs_gt_report.json"),
            "fitted_obj": str(fitted_obj),
            "eval_glb": str(eval_glb),
        },
    }
    report_path = OUT / "sam3d_fitted_to_fused_rgbd_vs_gt_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "fitted_obj": str(fitted_obj),
                "eval_glb": str(eval_glb),
                "metrics": report["metrics"],
                "fit": report["fit_to_fused_rgbd"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
