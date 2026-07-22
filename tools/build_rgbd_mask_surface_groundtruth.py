
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path('/workspace_whz')
CAPTURE = '20260622_081031_636398Z'
OUT = ROOT / 'data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth'
COLOR_PATH = ROOT / f'data/upload/larm_captures/{CAPTURE}/color.png'
DEPTH_PATH = ROOT / f'data/output/hololens2/{CAPTURE}_align_depth.png'
META_PATH = ROOT / f'data/upload/{CAPTURE}_meta.json'
MASKS = {
    'base': ROOT / 'data/output/geometric_joint_estimate_sam3door/base_mask.png',
    'drawer': ROOT / 'data/output/geometric_joint_estimate_sam3door/door_mask.png',
}
COLORS = {
    'base': [80, 220, 90, 255],
    'drawer': [240, 80, 80, 255],
}


def unproject_grid(depth_m, k):
    h, w = depth_m.shape
    ys, xs = np.indices((h, w))
    z = depth_m
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=-1)


def make_surface(name, mask, depth_m, color, k, stride=2, max_depth_jump=0.045):
    mask = (mask > 127) & (depth_m > 0.2) & (depth_m < 4.0)
    # Erode by a hair so SAM mask boundary does not create long torn triangles.
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    pts_grid = unproject_grid(depth_m, k)
    h, w = mask.shape
    valid_small = mask[::stride, ::stride]
    pts_small = pts_grid[::stride, ::stride]
    color_small = color[::stride, ::stride]
    hs, ws = valid_small.shape
    index = -np.ones((hs, ws), dtype=np.int64)
    coords = np.argwhere(valid_small)
    vertices = pts_small[valid_small].reshape(-1, 3)
    if color_small.shape[2] == 3:
        alpha = np.full(color_small.shape[:2] + (1,), 255, dtype=np.uint8)
        color_small = np.concatenate([color_small, alpha], axis=2)
    vcolors = color_small[valid_small].reshape(-1, 4)
    # Blend in a light semantic tint so the two parts remain separable in viewers.
    tint = np.asarray(COLORS[name], dtype=np.float32)
    vcolors = np.clip(0.82 * vcolors.astype(np.float32) + 0.18 * tint, 0, 255).astype(np.uint8)
    for idx, (yy, xx) in enumerate(coords):
        index[yy, xx] = idx
    faces = []
    for y in range(hs - 1):
        for x in range(ws - 1):
            ids = [index[y, x], index[y, x + 1], index[y + 1, x], index[y + 1, x + 1]]
            if min(ids) < 0:
                continue
            z = [vertices[i, 2] for i in ids]
            if max(z) - min(z) > max_depth_jump:
                continue
            faces.append([ids[0], ids[2], ids[1]])
            faces.append([ids[1], ids[2], ids[3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), vertex_colors=vcolors, process=False)
    return mesh, {
        'vertices': int(len(vertices)),
        'faces': int(len(faces)),
        'bounds_min': vertices.min(axis=0).tolist() if len(vertices) else None,
        'bounds_max': vertices.max(axis=0).tolist() if len(vertices) else None,
        'stride': stride,
        'max_depth_jump_m': max_depth_jump,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    k = np.asarray(meta['PVCamera']['k'], dtype=np.float64)
    depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)
    scene = trimesh.Scene()
    report = {
        'method': 'Ground-truth visible surface from aligned depth + SAM masks. No SAM3D pose/orientation is used, so the two parts are in the original camera coordinate frame with real RGB-D scale and relative placement. This is a visible-surface pose baseline, not a closed complete object model.',
        'camera_k': k.tolist(),
        'capture': CAPTURE,
        'parts': {},
        'outputs': {},
    }
    for name, path in MASKS.items():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mesh, stats = make_surface(name, mask, depth, color, k)
        glb = OUT / f'{name}_rgbd_mask_surface.glb'
        ply = OUT / f'{name}_rgbd_mask_surface.ply'
        mesh.export(glb)
        mesh.export(ply)
        scene.add_geometry(mesh, geom_name=name, node_name=name)
        report['parts'][name] = {'mask': str(path), 'outputs': {'glb': str(glb), 'ply': str(ply)}, **stats}
    combined = OUT / 'cabinet_drawer_rgbd_mask_surface_groundtruth.glb'
    scene.export(combined)
    report['outputs']['combined_open_glb'] = str(combined)
    report_path = OUT / 'rgbd_mask_surface_groundtruth_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'combined_open_glb': str(combined), 'parts': report['parts']}, indent=2))


if __name__ == '__main__':
    main()
