import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
SRC_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit/rgbd_similarity_fit_report.json"
OUT_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered"
COLOR_PATH = ROOT / "data/upload/larm_captures/20260622_081031_636398Z/color.png"

COLORS = {"base": (0, 220, 0), "drawer": (0, 0, 240)}


def load_mesh(path: str) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    return trimesh.util.concatenate([g.copy() for g in scene.geometry.values()])


def transform(points: np.ndarray, r: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    return s * (points @ r.T) + t


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = points[points[:, 2] > 1e-5]
    return np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))


def bbox_center_from_projected(uv: np.ndarray, width: int, height: int) -> tuple[np.ndarray, list[float]]:
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[inb]
    lo = np.percentile(uv, 1.0, axis=0)
    hi = np.percentile(uv, 99.0, axis=0)
    return (lo + hi) * 0.5, [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(SRC_REPORT.read_text(encoding="utf-8"))
    k = np.asarray(report["camera_k"], dtype=np.float64)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_UNCHANGED)
    if color.shape[2] == 4:
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    h, w = color.shape[:2]

    centered = {
        "method": "Post-process the RGB-D similarity fit with a camera-model 2D center snap. Rotation, uniform scale, and mesh topology are unchanged; only camera-frame x/y translation is adjusted so the projected p01-p99 bbox center matches the mask bbox center.",
        "source_report": str(SRC_REPORT),
        "camera_k": report["camera_k"],
        "parts": {},
    }
    combined = color.copy()

    for name, part in report["parts"].items():
        fit = part["fit"]
        mesh = load_mesh(part["source_mesh"])
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        r = np.asarray(fit["rotation_matrix_source_to_camera"], dtype=np.float64)
        s = float(fit["scale_uniform"])
        t = np.asarray(fit["translation_camera_m"], dtype=np.float64)

        sample, _ = trimesh.sample.sample_surface(mesh, 35000)
        p = transform(sample, r, s, t)
        uv = project(p, k)
        before_center, before_bbox = bbox_center_from_projected(uv, w, h)
        b = np.asarray(part["target"]["bbox_xyxy"], dtype=np.float64)
        target_center = (b[:2] + b[2:]) * 0.5
        z = float(np.median(p[:, 2]))
        duv = target_center - before_center
        delta = np.array([duv[0] * z / k[0, 0], duv[1] * z / k[1, 1], 0.0], dtype=np.float64)
        t2 = t + delta

        fitted = mesh.copy()
        fitted.vertices = transform(vertices, r, s, t2)
        glb_path = OUT_DIR / f"{name}_rgbd_similarity_centered.glb"
        ply_path = OUT_DIR / f"{name}_rgbd_similarity_centered.ply"
        fitted.export(glb_path)
        fitted.export(ply_path)

        pts, _ = trimesh.sample.sample_surface(fitted, 35000)
        uv2 = project(pts, k)
        after_center, after_bbox = bbox_center_from_projected(uv2, w, h)
        overlay = color.copy()
        inb = (uv2[:, 0] >= 0) & (uv2[:, 0] < w) & (uv2[:, 1] >= 0) & (uv2[:, 1] < h)
        pix = np.round(uv2[inb]).astype(np.int32)
        step = max(1, len(pix) // 35000)
        for x, y in pix[::step]:
            cv2.circle(overlay, (int(x), int(y)), 1, COLORS[name], -1)
            cv2.circle(combined, (int(x), int(y)), 1, COLORS[name], -1)
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLORS[name], 2)
        cv2.rectangle(combined, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLORS[name], 2)
        overlay_path = OUT_DIR / f"{name}_rgbd_similarity_centered_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        centered["parts"][name] = {
            "source_mesh": part["source_mesh"],
            "target_bbox_xyxy": b.tolist(),
            "scale_uniform": s,
            "rotation_matrix_source_to_camera": r.tolist(),
            "translation_camera_m_before": t.tolist(),
            "translation_camera_m_after": t2.tolist(),
            "xy_translation_delta_m": delta.tolist(),
            "bbox_center_before_px": before_center.tolist(),
            "bbox_center_after_px": after_center.tolist(),
            "target_bbox_center_px": target_center.tolist(),
            "bbox_before_xyxy": before_bbox,
            "bbox_after_xyxy": after_bbox,
            "outputs": {
                "glb": str(glb_path),
                "ply": str(ply_path),
                "projection_overlay": str(overlay_path),
            },
        }

    combined_path = OUT_DIR / "combined_rgbd_similarity_centered_overlay.png"
    cv2.imwrite(str(combined_path), combined)
    centered["outputs"] = {"combined_projection_overlay": str(combined_path)}
    out_report = OUT_DIR / "rgbd_similarity_centered_report.json"
    out_report.write_text(json.dumps(centered, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_report), "combined_overlay": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
