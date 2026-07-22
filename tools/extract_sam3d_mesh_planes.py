import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/mesh_plane_components"
MESHES = {
    "base": ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb",
    "drawer": ROOT / "data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb",
}
RNG = np.random.default_rng(20260624)


def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    return trimesh.util.concatenate([g.copy() for g in scene.geometry.values()])


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ center)
    return center, n, d


def extract_planes(mesh: trimesh.Trimesh, max_planes: int = 8, samples: int = 50000) -> list[dict]:
    pts, face_idx = trimesh.sample.sample_surface(mesh, samples)
    normals = mesh.face_normals[face_idx]
    remaining = np.ones(len(pts), dtype=bool)
    planes = []
    for plane_idx in range(max_planes):
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) < 3000:
            break
        best = None
        for _ in range(220):
            i = RNG.choice(rem_idx)
            n = normals[i]
            if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-6:
                continue
            n = n / np.linalg.norm(n)
            p = pts[i]
            dist = np.abs((pts[rem_idx] - p) @ n)
            normal_sim = np.abs(normals[rem_idx] @ n)
            in_local = (dist < 0.012) & (normal_sim > 0.90)
            count = int(in_local.sum())
            if best is None or count > best[0]:
                best = (count, n, p, rem_idx[in_local])
        if best is None or best[0] < 700:
            break
        inliers = best[3]
        center, n_fit, d = fit_plane(pts[inliers])
        # Orient normal to agree with average sampled normal.
        avg_n = normals[inliers].mean(axis=0)
        if n_fit @ avg_n < 0:
            n_fit = -n_fit
            d = -d
        ext = pts[inliers] - center
        _, singular, vh = np.linalg.svd(ext, full_matrices=False)
        axes = vh[:2]
        uv = ext @ axes.T
        min_uv = uv.min(axis=0)
        max_uv = uv.max(axis=0)
        # Approximate area by plane bbox; sampled area ratio is also reported.
        bbox_area = float(np.prod(max_uv - min_uv))
        planes.append(
            {
                "index": plane_idx,
                "sample_inliers": int(len(inliers)),
                "sample_fraction": float(len(inliers) / samples),
                "center": center.tolist(),
                "normal": n_fit.tolist(),
                "d": float(d),
                "plane_axes": axes.tolist(),
                "uv_min": min_uv.tolist(),
                "uv_max": max_uv.tolist(),
                "approx_bbox_area": bbox_area,
                "singular_values": singular.tolist(),
            }
        )
        remaining[inliers] = False
    return planes


def export_plane_patch(mesh: trimesh.Trimesh, plane: dict, path: Path) -> None:
    # Export a simple rectangle patch representing the detected plane extent.
    c = np.array(plane["center"], dtype=float)
    n = np.array(plane["normal"], dtype=float)
    axes = np.array(plane["plane_axes"], dtype=float)
    uv_min = np.array(plane["uv_min"], dtype=float)
    uv_max = np.array(plane["uv_max"], dtype=float)
    corners = []
    for u, v in [(uv_min[0], uv_min[1]), (uv_max[0], uv_min[1]), (uv_max[0], uv_max[1]), (uv_min[0], uv_max[1])]:
        corners.append(c + axes[0] * u + axes[1] * v)
    patch = trimesh.Trimesh(vertices=np.asarray(corners), faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    patch.export(path)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = {
        "method": "RANSAC-style extraction of large approximate planar components from raw SAM3DObject meshes. This inspects mesh geometry itself instead of assuming model coordinate axes are semantic axes.",
        "parts": {},
    }
    for name, path in MESHES.items():
        mesh = load_mesh(path)
        planes = extract_planes(mesh)
        part_dir = OUT / name
        part_dir.mkdir()
        for plane in planes:
            export_plane_patch(mesh, plane, part_dir / f"plane_{plane['index']:02d}_patch.ply")
        report["parts"][name] = {
            "source_mesh": str(path),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_vertices": int(len(mesh.vertices)),
            "planes": planes,
            "plane_patch_dir": str(part_dir),
        }
    report_path = OUT / "mesh_plane_components_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
