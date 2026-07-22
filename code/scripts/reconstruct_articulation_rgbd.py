from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree, ConvexHull
from sklearn.cluster import DBSCAN
import trimesh

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from task_json import save_task_json  # noqa: E402

BLENDER2OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _ordered_items(meta: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    grouped = list(meta.get('grouped_captures') or [])
    inputs = []
    for qtok in sorted(meta['inputs'].keys(), key=lambda x: float(x)):
        for frame_key in sorted(meta['inputs'][qtok].keys(), key=lambda x: int(x.rsplit('_', 1)[-1])):
            inputs.append((qtok, frame_key, meta['inputs'][qtok][frame_key]))
    if len(inputs) != len(grouped):
        raise ValueError(f'inputs/grouped_captures length mismatch: {len(inputs)} vs {len(grouped)}')
    return [(qtok, frame_key, rec, grouped[i]) for i, (qtok, frame_key, rec) in enumerate(inputs)]


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert('L')
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(mask) > 0
    arr = ndi.binary_opening(arr, structure=np.ones((3, 3), dtype=bool), iterations=1)
    arr = ndi.binary_closing(arr, structure=np.ones((5, 5), dtype=bool), iterations=1)
    arr = ndi.binary_fill_holes(arr)
    return arr.astype(bool)


def _resize_depth_to(depth_path: Path, size: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f'Depth must be single channel: {depth_path} shape={depth.shape}')
    if (depth.shape[1], depth.shape[0]) != size:
        depth_img = Image.fromarray(depth.astype(np.uint16), mode='I;16')
        depth = np.asarray(depth_img.resize(size, Image.Resampling.NEAREST), dtype=np.float32)
    return depth


def _depth_mask_to_camera_points(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    k: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
    max_points: int,
    seed: int,
) -> np.ndarray:
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    valid = eroded & np.isfinite(depth_mm) & (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(xs) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1)


def _pose_cv_from_larm_record(rec: dict[str, Any]) -> np.ndarray:
    return np.asarray(rec['transform_matrix'], dtype=np.float64) @ BLENDER2OPENCV


def _estimate_depth_to_sfm_scale(frame_infos: list[dict[str, Any]]) -> tuple[float, list[float]]:
    values = []
    for info in frame_infos:
        pts = info['points_cam']
        if len(pts) == 0:
            continue
        median_cam = np.median(pts, axis=0)
        pose = info['c2w_cv']
        a = pose[:3, :3] @ median_cam
        b = pose[:3, 3]
        denom = float(np.dot(a, a))
        if denom <= 1e-9:
            continue
        s = -float(np.dot(a, b)) / denom
        if np.isfinite(s) and s > 0:
            values.append(s)
    if not values:
        return 1.0, []
    return float(np.median(values)), [float(v) for v in values]


def _transform_points(points_cam: np.ndarray, pose: np.ndarray, scale: float) -> np.ndarray:
    if len(points_cam) == 0:
        return points_cam.copy()
    pts = points_cam * scale
    return pts @ pose[:3, :3].T + pose[:3, 3]


def _voxel_downsample(points: np.ndarray, voxel: float, max_points: int | None = None, seed: int = 0) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3)
    if voxel > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(idx)]
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def _robust_bbox_filter(points: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, pad: float = 0.05) -> np.ndarray:
    if len(points) == 0:
        return points
    lo = np.percentile(points, lo_pct, axis=0) - pad
    hi = np.percentile(points, hi_pct, axis=0) + pad
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf, dtype=np.float64)
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1, workers=-1)
    return d.astype(np.float64)


def _largest_cluster(points: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, dict[str, Any]]:
    if len(points) == 0:
        return points, {'clusters': 0, 'selected_cluster': None, 'noise_points': 0}
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    valid = labels >= 0
    if not np.any(valid):
        return points, {'clusters': 0, 'selected_cluster': None, 'noise_points': int(len(points))}
    ids, counts = np.unique(labels[valid], return_counts=True)
    selected = int(ids[np.argmax(counts)])
    return points[labels == selected], {
        'clusters': int(len(ids)),
        'selected_cluster': selected,
        'selected_points': int(np.max(counts)),
        'noise_points': int(np.sum(~valid)),
    }


def _write_point_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1))
    cloud = trimesh.points.PointCloud(points, colors=colors)
    cloud.export(path)


def _write_hull_mesh(path: Path, points: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 16:
        return False
    try:
        hull = ConvexHull(points)
        mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices, process=True)
        mesh.export(path)
        return True
    except Exception:
        return False


def _icp_rigid_transform(src: np.ndarray, dst: np.ndarray, max_iter: int = 40, trim_quantile: float = 0.75) -> tuple[np.ndarray, dict[str, Any]]:
    src0 = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src0) < 8 or len(dst) < 8:
        return np.eye(4), {'ok': False, 'reason': 'not_enough_points'}
    if len(src0) > 5000:
        rng = np.random.default_rng(7)
        src0 = src0[rng.choice(len(src0), size=5000, replace=False)]
    if len(dst) > 5000:
        rng = np.random.default_rng(8)
        dst = dst[rng.choice(len(dst), size=5000, replace=False)]
    tree = cKDTree(dst)
    moving = src0.copy()
    total = np.eye(4)
    last_rmse = None
    for _ in range(max_iter):
        d, idx = tree.query(moving, k=1, workers=-1)
        cutoff = np.quantile(d, trim_quantile)
        keep = d <= cutoff
        if np.sum(keep) < 8:
            break
        a = moving[keep]
        b = dst[idx[keep]]
        ca = a.mean(axis=0)
        cb = b.mean(axis=0)
        h = (a - ca).T @ (b - cb)
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = vt.T @ u.T
        t = cb - r @ ca
        moving = moving @ r.T + t
        delta = np.eye(4)
        delta[:3, :3] = r
        delta[:3, 3] = t
        total = delta @ total
        rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
        if last_rmse is not None and abs(last_rmse - rmse) < 1e-5:
            last_rmse = rmse
            break
        last_rmse = rmse
    return total, {'ok': True, 'rmse': last_rmse, 'src_points': int(len(src0)), 'dst_points': int(len(dst))}


def _axis_angle_from_rotation(r: np.ndarray) -> tuple[np.ndarray, float]:
    cos_angle = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
    axis /= max(2.0 * np.sin(angle), 1e-8)
    axis /= max(np.linalg.norm(axis), 1e-8)
    return axis, angle


def _axis_point_from_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    a = np.eye(3) - r
    try:
        p, *_ = np.linalg.lstsq(a, t, rcond=None)
        return p.astype(np.float64)
    except Exception:
        return np.zeros(3, dtype=np.float64)


def _write_urdf(path: Path, base_mesh: str, part_mesh: str, axis: np.ndarray, origin: np.ndarray, lower: float, upper: float) -> None:
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    text = f'''<?xml version="1.0" encoding="UTF-8"?>
<robot name="rgbd_articulation">
  <link name="base">
    <visual><geometry><mesh filename="{base_mesh}"/></geometry></visual>
    <collision><geometry><mesh filename="{base_mesh}"/></geometry></collision>
  </link>
  <link name="part">
    <visual><geometry><mesh filename="{part_mesh}"/></geometry></visual>
    <collision><geometry><mesh filename="{part_mesh}"/></geometry></collision>
  </link>
  <joint name="joint" type="revolute">
    <parent link="base"/>
    <child link="part"/>
    <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
    <limit lower="{lower}" upper="{upper}" effort="0" velocity="0"/>
  </joint>
</robot>
'''
    path.write_text(text, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-json', type=Path, default=Path('/workspace_whz/data/upload/larm/20260622_joint_0_white_outline_sfm/20260622_joint_0_white_outline_sfm.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('/workspace_whz/data/output/rgbd_articulation/20260622_joint_0_geometry'))
    parser.add_argument('--max-frame-points', type=int, default=25000)
    parser.add_argument('--voxel', type=float, default=0.01)
    parser.add_argument('--diff-threshold', type=float, default=0.045)
    parser.add_argument('--cluster-eps', type=float, default=0.055)
    parser.add_argument('--cluster-min-samples', type=int, default=20)
    parser.add_argument('--min-depth-mm', type=float, default=200.0)
    parser.add_argument('--max-depth-mm', type=float, default=7500.0)
    args = parser.parse_args()

    meta = _load_json(args.metadata_json.resolve())
    items = _ordered_items(meta)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frame_infos: list[dict[str, Any]] = []
    for index, (qtok, frame_key, rec, grouped) in enumerate(items):
        capture_dir = Path(grouped.get('capture', '') or '').resolve()
        cap_path = capture_dir / 'capture.json'
        cap = _load_json(cap_path) if cap_path.is_file() else {}
        raw_k = np.asarray(cap.get('intrinsics') or meta['intrinsics'], dtype=np.float64)
        rgb_path = Path(grouped.get('source_image') or rec['image_path']).resolve()
        raw_depth_path = Path(grouped.get('depth_path') or cap.get('depth_path')).resolve()
        meta_path = CODE_ROOT.parent / 'data' / 'upload' / f'{capture_dir.name}_meta.json'
        aligned_depth_path = None
        if meta_path.is_file():
            source_meta = _load_json(meta_path)
            align_name = (source_meta.get('DepthCamera') or {}).get('align_depth_name')
            if align_name:
                candidate = CODE_ROOT.parent / 'data' / 'output' / 'hololens2' / str(align_name)
                if candidate.is_file():
                    aligned_depth_path = candidate.resolve()
        depth_path = aligned_depth_path or raw_depth_path
        mask_path = Path(grouped.get('mask_path')).resolve()
        rgb = Image.open(rgb_path).convert('RGB')
        size = rgb.size
        mask = _load_mask(mask_path, size)
        depth = _resize_depth_to(depth_path, size)
        pts_cam = _depth_mask_to_camera_points(
            depth, mask, raw_k, args.min_depth_mm, args.max_depth_mm, args.max_frame_points, seed=index
        )
        c2w_cv = _pose_cv_from_larm_record(rec)
        frame_infos.append({
            'index': index,
            'qtok': qtok,
            'qpos': float(rec['qpos']),
            'frame_key': frame_key,
            'points_cam': pts_cam,
            'c2w_cv': c2w_cv,
            'raw_points': int(len(pts_cam)),
            'rgb_path': str(rgb_path),
            'depth_path': str(depth_path),
            'raw_depth_path': str(raw_depth_path),
            'used_aligned_depth': bool(aligned_depth_path is not None),
            'mask_path': str(mask_path),
        })

    scale, per_frame_scales = _estimate_depth_to_sfm_scale(frame_infos)
    by_qpos: dict[float, list[np.ndarray]] = {0.0: [], 1.0: []}
    for info in frame_infos:
        pts_world = _transform_points(info['points_cam'], info['c2w_cv'], scale)
        pts_world = _robust_bbox_filter(pts_world)
        pts_world = _voxel_downsample(pts_world, args.voxel, max_points=None, seed=info['index'])
        info['points_world'] = pts_world
        by_qpos[float(info['qpos'])].append(pts_world)
        _write_point_ply(out / f"frame_{info['index']}_{float(info['qpos']):.2f}_points.ply", pts_world, (180, 180, 180))

    q0 = _voxel_downsample(np.concatenate(by_qpos[0.0], axis=0), args.voxel, max_points=120000, seed=10)
    q1 = _voxel_downsample(np.concatenate(by_qpos[1.0], axis=0), args.voxel, max_points=120000, seed=11)
    q0 = _robust_bbox_filter(q0)
    q1 = _robust_bbox_filter(q1)
    d0 = _nearest_distances(q0, q1)
    d1 = _nearest_distances(q1, q0)

    static_q1 = q1[d1 <= args.diff_threshold]
    changed_q1 = q1[d1 > args.diff_threshold]
    changed_q0 = q0[d0 > args.diff_threshold]
    part_q1, cluster_q1 = _largest_cluster(changed_q1, args.cluster_eps, args.cluster_min_samples)
    part_q0, cluster_q0 = _largest_cluster(changed_q0, args.cluster_eps, args.cluster_min_samples)

    base_q1 = static_q1
    if len(base_q1) < 200:
        base_q1 = q1[d1 <= np.percentile(d1, 55.0)]
    base_q1 = _voxel_downsample(base_q1, args.voxel, max_points=80000, seed=12)
    part_q1 = _voxel_downsample(part_q1, args.voxel, max_points=60000, seed=13)
    part_q0 = _voxel_downsample(part_q0, args.voxel, max_points=60000, seed=14)

    _write_point_ply(out / 'qpos0_object_points.ply', q0, (150, 150, 150))
    _write_point_ply(out / 'qpos1_object_points.ply', q1, (190, 190, 190))
    _write_point_ply(out / 'base_points.ply', base_q1, (80, 130, 255))
    _write_point_ply(out / 'part_points.ply', part_q1, (255, 90, 70))
    _write_point_ply(out / 'part_qpos0_points.ply', part_q0, (255, 180, 70))

    base_mesh_ok = _write_hull_mesh(out / 'base_hull.ply', base_q1)
    part_mesh_ok = _write_hull_mesh(out / 'part_hull.ply', part_q1)

    tf, icp_diag = _icp_rigid_transform(part_q0, part_q1)
    r = tf[:3, :3]
    t = tf[:3, 3]
    axis, angle = _axis_angle_from_rotation(r)
    origin = _axis_point_from_transform(r, t)
    if angle > np.pi:
        angle -= 2 * np.pi
    lower = 0.0
    upper = float(angle)
    if upper < lower:
        lower, upper = upper, lower

    _write_urdf(
        out / 'rgbd_articulation.urdf',
        'base_hull.ply' if base_mesh_ok else 'base_points.ply',
        'part_hull.ply' if part_mesh_ok else 'part_points.ply',
        axis,
        origin,
        lower,
        upper,
    )

    finite_d1 = d1[np.isfinite(d1)]
    finite_d0 = d0[np.isfinite(d0)]
    diag = {
        'metadata_json': str(args.metadata_json.resolve()),
        'note': 'Uses calibrated PV-aligned depth when *_align_depth.png is available; falls back to raw depth resize only if aligned depth is missing.',
        'depth_to_sfm_scale': scale,
        'per_frame_depth_to_sfm_scales': per_frame_scales,
        'diff_threshold': args.diff_threshold,
        'cluster_eps': args.cluster_eps,
        'frame_points': [{k: v for k, v in info.items() if k not in {'points_cam', 'points_world', 'c2w_cv'}} for info in frame_infos],
        'qpos0_points': int(len(q0)),
        'qpos1_points': int(len(q1)),
        'static_qpos1_points': int(len(static_q1)),
        'changed_qpos1_points_before_cluster': int(len(changed_q1)),
        'changed_qpos0_points_before_cluster': int(len(changed_q0)),
        'part_qpos1_points': int(len(part_q1)),
        'part_qpos0_points': int(len(part_q0)),
        'base_points': int(len(base_q1)),
        'qpos1_changed_cluster': cluster_q1,
        'qpos0_changed_cluster': cluster_q0,
        'nearest_distance_q1_to_q0_percentiles': {str(p): float(np.percentile(finite_d1, p)) for p in [25, 50, 75, 90, 95, 99]},
        'nearest_distance_q0_to_q1_percentiles': {str(p): float(np.percentile(finite_d0, p)) for p in [25, 50, 75, 90, 95, 99]},
        'icp_part_q0_to_q1': icp_diag,
        'joint': {
            'type': 'revolute',
            'axis': [float(v) for v in axis.tolist()],
            'origin': [float(v) for v in origin.tolist()],
            'angle_rad': float(angle),
            'limit_lower': lower,
            'limit_upper': upper,
        },
        'outputs': {
            'urdf': str(out / 'rgbd_articulation.urdf'),
            'base_points': str(out / 'base_points.ply'),
            'part_points': str(out / 'part_points.ply'),
            'base_hull': str(out / 'base_hull.ply') if base_mesh_ok else None,
            'part_hull': str(out / 'part_hull.ply') if part_mesh_ok else None,
        },
    }
    save_task_json(out / 'diagnostics.json', diag)
    print(json.dumps(diag, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
