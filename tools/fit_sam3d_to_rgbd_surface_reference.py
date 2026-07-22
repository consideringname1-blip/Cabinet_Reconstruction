
import itertools
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path('/workspace_whz')
CAPTURE = '20260622_081031_636398Z'
OUT = ROOT / 'data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_to_rgbd_surface_fit'
REF_DIR = ROOT / 'data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth'
COLOR_PATH = ROOT / f'data/upload/larm_captures/{CAPTURE}/color.png'
DEPTH_PATH = ROOT / f'data/output/hololens2/{CAPTURE}_align_depth.png'
META_PATH = ROOT / f'data/upload/{CAPTURE}_meta.json'
MASKS = {
    'base': ROOT / 'data/output/geometric_joint_estimate_sam3door/base_mask.png',
    'drawer': ROOT / 'data/output/geometric_joint_estimate_sam3door/door_mask.png',
}
RAW_MESHES = {
    'base': ROOT / 'data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb',
    'drawer': ROOT / 'data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb',
}
REF_SURFACES = {
    'base': REF_DIR / 'base_rgbd_mask_surface.glb',
    'drawer': REF_DIR / 'drawer_rgbd_mask_surface.glb',
}
COLORS = {'base': (0, 220, 0), 'drawer': (0, 0, 240)}
RNG = np.random.default_rng(20260625)


def load_mesh(path):
    loaded = trimesh.load(path, force='scene', process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return trimesh.util.concatenate([g.copy() for g in loaded.geometry.values()])


def pca_axes(points):
    c = points.mean(axis=0)
    x = points - c
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    vals = np.var(x @ axes, axis=0)
    return c, axes, vals


def rotation_candidates(src_axes, tgt_axes):
    out = []
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            m = p @ np.diag(signs)
            r = tgt_axes @ m @ src_axes.T
            if np.linalg.det(r) > 0:
                out.append({'r': r, 'perm': perm, 'signs': signs})
    return out


def umeyama(src, dst):
    mx = src.mean(axis=0)
    my = dst.mean(axis=0)
    x = src - mx
    y = dst - my
    cov = (y.T @ x) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    r = u @ d @ vt
    var = np.mean(np.sum(x * x, axis=1))
    s = float(np.trace(np.diag(singular) @ d) / max(var, 1e-12))
    t = my - s * (mx @ r.T)
    return r, s, t


def transform(points, r, s, t):
    return s * (points @ r.T) + t


def refine_target_to_source(src_fit, tgt_fit, r, s, t, iterations=24):
    # Align full source to the visible RGB-D surface by matching each target point to its
    # nearest transformed source point. This avoids forcing invisible source backside to target.
    for _ in range(iterations):
        p = transform(src_fit, r, s, t)
        tree = cKDTree(p)
        dist, idx = tree.query(tgt_fit, k=1, workers=-1)
        keep = dist <= min(max(float(np.percentile(dist, 82)), 0.018), 0.12)
        if keep.sum() < 200:
            keep = dist <= np.percentile(dist, 93)
        matched_src = src_fit[idx[keep]]
        matched_dst = tgt_fit[keep]
        rn, sn, tn = umeyama(matched_src, matched_dst)
        # Damp scale to prevent one small visible patch from shrinking the whole object.
        r = rn
        s = 0.72 * s + 0.28 * sn
        t = 0.72 * t + 0.28 * tn
    return r, float(s), t


def project(points, k):
    p = points[points[:, 2] > 1e-5]
    return np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def score_pose(src_eval, tgt_eval, r, s, t, k, mask, depth_obs):
    p = transform(src_eval, r, s, t)
    tree = cKDTree(p)
    dist, _ = tree.query(tgt_eval, k=1, workers=-1)
    chamfer_med = float(np.median(dist))
    chamfer_p90 = float(np.percentile(dist, 90))
    uv = project(p, k)
    h, w = mask.shape
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if inb.sum() < 100:
        return 1e9, {'reason': 'few projected'}
    uvi = uv[inb]
    px = np.clip(np.round(uvi[:, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(uvi[:, 1]).astype(np.int32), 0, h - 1)
    hit = mask[py, px] > 0
    valid = depth_obs[py, px] > 0.05
    diff = np.abs(p[inb, 2][hit & valid] - depth_obs[py[hit & valid], px[hit & valid]]) if np.any(hit & valid) else np.array([1.0])
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[py, px] = 255
    pred = cv2.dilate(pred, np.ones((5, 5), np.uint8), iterations=1) > 0
    target = mask > 0
    iou = float((pred & target).sum() / max(1, (pred | target).sum()))
    coverage = float((pred & target).sum() / max(1, target.sum()))
    leakage = float((pred & ~target).sum() / max(1, pred.sum()))
    lo = np.percentile(uvi, 1, axis=0)
    hi = np.percentile(uvi, 99, axis=0)
    bbox = np.r_[lo, hi]
    tb = mask_bbox(mask)
    ts = np.maximum(tb[2:] - tb[:2], 1)
    bbox_size_err = float(np.linalg.norm(((hi - lo) - ts) / ts))
    bbox_center_err = float(np.linalg.norm((((hi + lo) * 0.5) - ((tb[:2] + tb[2:]) * 0.5)) / ts))
    depth_med = float(np.median(diff))
    score = chamfer_med / 0.018 + 0.22 * chamfer_p90 / 0.055 + 0.75 * depth_med / 0.05 + 1.4 * (1 - iou) + 0.5 * (1 - coverage) + 0.65 * leakage + 0.8 * bbox_size_err + 1.2 * bbox_center_err
    return score, {
        'score': float(score),
        'target_to_source_chamfer_median_m': chamfer_med,
        'target_to_source_chamfer_p90_m': chamfer_p90,
        'projected_mask_iou': iou,
        'target_coverage': coverage,
        'leakage': leakage,
        'projected_mask_hit_ratio': float(hit.mean()),
        'depth_abs_median_m': depth_med,
        'bbox_xyxy_p01_p99': bbox.tolist(),
        'bbox_size_err': bbox_size_err,
        'bbox_center_err': bbox_center_err,
        'projected_points': int(inb.sum()),
    }


def draw_overlay(color, samples, k, output):
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
        mask = (cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, COLORS[name], 2)
    cv2.imwrite(str(output), img)


def fit_part(name, k, depth_obs, color):
    raw = load_mesh(RAW_MESHES[name])
    ref = load_mesh(REF_SURFACES[name])
    mask = (cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    src_surface, _ = trimesh.sample.sample_surface(raw, 70000 if name == 'base' else 50000)
    src_fit = src_surface[RNG.choice(len(src_surface), min(len(src_surface), 14000 if name == 'base' else 10000), replace=False)]
    src_eval = src_surface[RNG.choice(len(src_surface), min(len(src_surface), 28000 if name == 'base' else 22000), replace=False)]
    tgt_vertices = np.asarray(ref.vertices, dtype=np.float64)
    tgt_fit = tgt_vertices
    if len(tgt_fit) > 9000:
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), 9000, replace=False)]
    tgt_eval = tgt_vertices
    sc, sa, sv = pca_axes(src_fit)
    tc, ta, tv = pca_axes(tgt_fit)
    src_rms = math.sqrt(float(np.sum(np.var(src_fit - src_fit.mean(axis=0), axis=0))))
    tgt_rms = math.sqrt(float(np.sum(np.var(tgt_fit - tgt_fit.mean(axis=0), axis=0))))
    scale0 = tgt_rms / max(src_rms, 1e-9)
    candidates = []
    for cand in rotation_candidates(sa, ta):
        r0 = cand['r']
        t0 = tc - scale0 * (sc @ r0.T)
        r1, s1, t1 = refine_target_to_source(src_fit, tgt_fit, r0, scale0, t0)
        score, metrics = score_pose(src_eval, tgt_eval, r1, s1, t1, k, mask, depth_obs)
        candidates.append({'score': float(score), 'r': r1, 's': float(s1), 't': t1, 'metrics': metrics, 'init_perm': cand['perm'], 'init_signs': cand['signs']})
    candidates.sort(key=lambda c: c['score'])
    exports = []
    part_dir = OUT / name
    part_dir.mkdir(parents=True, exist_ok=True)
    for rank, c in enumerate(candidates[:4], start=1):
        mesh = raw.copy()
        mesh.vertices = transform(np.asarray(raw.vertices, dtype=np.float64), c['r'], c['s'], c['t'])
        glb = part_dir / f'{name}_sam3d_to_rgbd_surface_rank{rank:02d}.glb'
        mesh.export(glb)
        pts, _ = trimesh.sample.sample_surface(mesh, 32000)
        overlay = part_dir / f'{name}_rank{rank:02d}_overlay.png'
        draw_overlay(color, {name: pts}, k, overlay)
        exports.append({
            'rank': rank,
            'glb': str(glb),
            'overlay': str(overlay),
            'score': c['score'],
            'scale_uniform': c['s'],
            'rotation_matrix_source_to_camera': c['r'].tolist(),
            'translation_camera_m': c['t'].tolist(),
            'init_perm': list(c['init_perm']),
            'init_signs': list(c['init_signs']),
            'metrics': c['metrics'],
        })
    return raw, candidates, exports


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    k = np.asarray(meta['PVCamera']['k'], dtype=np.float64)
    depth_obs = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    report = {
        'method': 'Fit raw complete SAM3D meshes to the validated RGB-D visible-surface reference using multistart 7DoF similarity + target-to-source ICP. The RGB-D surface supplies pose/scale/relative placement; SAM3D supplies completion.',
        'reference_combined_glb': str(REF_DIR / 'cabinet_drawer_rgbd_mask_surface_groundtruth.glb'),
        'camera_k': k.tolist(),
        'parts': {},
        'outputs': {},
    }
    scene = trimesh.Scene()
    overlay_samples = {}
    for name in ('base', 'drawer'):
        raw, cands, exports = fit_part(name, k, depth_obs, color)
        best = exports[0]
        mesh = load_mesh(best['glb'])
        scene.add_geometry(mesh, geom_name=name, node_name=name)
        pts, _ = trimesh.sample.sample_surface(mesh, 40000 if name == 'base' else 30000)
        overlay_samples[name] = pts
        report['parts'][name] = {
            'raw_mesh': str(RAW_MESHES[name]),
            'reference_surface': str(REF_SURFACES[name]),
            'top_exports': exports,
        }
    combined = OUT / 'cabinet_drawer_sam3d_fitted_to_rgbd_surface_rank01.glb'
    scene.export(combined)
    overlay = OUT / 'combined_sam3d_to_rgbd_surface_rank01_overlay.png'
    draw_overlay(color, overlay_samples, k, overlay)
    # Keep a copy of the validated reference next to the fitted result for direct comparison.
    shutil.copy2(REF_DIR / 'cabinet_drawer_rgbd_mask_surface_groundtruth.glb', OUT / 'reference_rgbd_mask_surface_groundtruth.glb')
    report['outputs'] = {
        'combined_open_glb': str(combined),
        'projection_overlay': str(overlay),
        'reference_copy_glb': str(OUT / 'reference_rgbd_mask_surface_groundtruth.glb'),
    }
    report_path = OUT / 'sam3d_to_rgbd_surface_fit_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'combined_open_glb': str(combined), 'overlay': str(overlay), 'reference': report['outputs']['reference_copy_glb']}, indent=2))


if __name__ == '__main__':
    main()
