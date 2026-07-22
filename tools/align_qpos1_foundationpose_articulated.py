from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
QPOS1_CAPTURE = "20260622_081031_636398Z"
FIT_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_to_rgbd_surface_fit/sam3d_to_rgbd_surface_fit_report.json"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/foundationpose_qpos1_articulated"

COLOR_PATH = ROOT / f"data/upload/larm_captures/{QPOS1_CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{QPOS1_CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{QPOS1_CAPTURE}_meta.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
COLORS = {
    "base": (0, 220, 0),
    "drawer": (0, 0, 240),
    "drawer_closed": (245, 140, 0),
}
RNG = np.random.default_rng(20260703)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_k() -> np.ndarray:
    return np.asarray(read_json(META_PATH)["PVCamera"]["k"], dtype=np.float32).reshape(3, 3)


def load_depth_m(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) * 1e-3
    else:
        depth = depth.astype(np.float32)
    depth[(depth < 0.001) | ~np.isfinite(depth)] = 0
    return depth


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return trimesh.util.concatenate([geom.copy() for geom in loaded.geometry.values()])


def pose_from_fit_export(export: dict[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(export["rotation_matrix_source_to_camera"], dtype=np.float32).reshape(3, 3)
    pose[:3, 3] = np.asarray(export["translation_camera_m"], dtype=np.float32).reshape(3)
    return pose


def build_initial_candidates(part_report: dict[str, Any], max_initial_candidates: int) -> list[dict[str, Any]]:
    candidates = []
    for export in part_report["top_exports"][:max_initial_candidates]:
        candidates.append(
            {
                "source": "sam3d_to_rgbd_surface_fit",
                "rank": int(export["rank"]),
                "surface_fit_score": float(export["score"]),
                "model_scale": float(export["scale_uniform"]),
                "pose_cv": pose_from_fit_export(export).astype(float).tolist(),
                "surface_fit_glb": str(export["glb"]),
                "surface_fit_overlay": str(export["overlay"]),
            }
        )
    if not candidates:
        raise ValueError("no initial candidates in fit report")
    return candidates


def transform_mesh(raw_mesh: trimesh.Trimesh, scale: float, pose: np.ndarray) -> trimesh.Trimesh:
    mesh = raw_mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    translation = np.asarray(pose[:3, 3], dtype=np.float64)
    mesh.vertices = vertices @ rotation.T + translation
    return mesh


def translate_mesh(mesh: trimesh.Trimesh, translation: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = np.asarray(out.vertices, dtype=np.float64) + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return out


def project(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = points[:, 2] > 1e-5
    p = points[valid]
    uv = np.column_stack(
        [
            k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2],
            k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2],
        ]
    )
    return uv, valid


def sample_mesh_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    if len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, count)
        return np.asarray(pts, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) <= count:
        return vertices
    return vertices[RNG.choice(len(vertices), size=count, replace=False)]


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(4, dtype=np.float64)
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def score_projection(mesh: trimesh.Trimesh, k: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, sample_count: int) -> dict[str, Any]:
    points = sample_mesh_points(mesh, sample_count)
    uv, valid_z = project(points, k)
    points = points[valid_z]
    h, w = mask.shape
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if int(inb.sum()) == 0:
        return {"projected_points": 0, "reason": "no projected points in image"}

    uv_in = uv[inb]
    pts_in = points[inb]
    px = np.clip(np.round(uv_in[:, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(uv_in[:, 1]).astype(np.int32), 0, h - 1)
    hit = mask[py, px]
    valid_depth = depth_m[py, px] > 0.05
    if np.any(hit & valid_depth):
        depth_abs = np.abs(pts_in[:, 2][hit & valid_depth] - depth_m[py[hit & valid_depth], px[hit & valid_depth]])
        depth_median = float(np.median(depth_abs))
        depth_p90 = float(np.percentile(depth_abs, 90))
    else:
        depth_median = None
        depth_p90 = None

    pred = np.zeros((h, w), dtype=np.uint8)
    pred[py, px] = 255
    pred = cv2.dilate(pred, np.ones((5, 5), np.uint8), iterations=1) > 0
    target = mask > 0
    intersection = int((pred & target).sum())
    union = int((pred | target).sum())
    bbox_pred_lo = np.percentile(uv_in, 1, axis=0)
    bbox_pred_hi = np.percentile(uv_in, 99, axis=0)
    bbox_target = mask_bbox(mask)

    return {
        "projected_points": int(inb.sum()),
        "projected_mask_iou": float(intersection / max(1, union)),
        "target_coverage": float(intersection / max(1, int(target.sum()))),
        "leakage": float((pred & ~target).sum() / max(1, int(pred.sum()))),
        "projected_mask_hit_ratio": float(hit.mean()),
        "depth_abs_median_m": depth_median,
        "depth_abs_p90_m": depth_p90,
        "bbox_xyxy_p01_p99": [float(bbox_pred_lo[0]), float(bbox_pred_lo[1]), float(bbox_pred_hi[0]), float(bbox_pred_hi[1])],
        "target_bbox_xyxy": [float(v) for v in bbox_target],
    }


def draw_overlay(
    color_bgr: np.ndarray,
    k: np.ndarray,
    meshes: dict[str, trimesh.Trimesh],
    mask_paths: dict[str, Path],
    output_path: Path,
    sample_count: int = 45000,
) -> None:
    image = color_bgr.copy()
    for name, mask_path in mask_paths.items():
        mask = load_mask(mask_path).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, COLORS.get(name, (255, 255, 255)), 2)

    for name, mesh in meshes.items():
        points = sample_mesh_points(mesh, sample_count)
        uv, _valid = project(points, k)
        h, w = image.shape[:2]
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 55000)
        color = COLORS.get(name, (255, 255, 255))
        for x, y in pix[::step]:
            cv2.circle(image, (int(x), int(y)), 1, color, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def export_scene(path: Path, meshes: dict[str, trimesh.Trimesh]) -> None:
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path)


def run_foundationpose(
    runner: Any,
    part_name: str,
    foundationpose_mesh_path: Path,
    initial_candidates: list[dict[str, Any]],
    k: np.ndarray,
    iteration: int,
    debug_root: Path,
) -> dict[str, Any]:
    request = {
        "mesh_file": str(foundationpose_mesh_path),
        "color_file": str(COLOR_PATH),
        "depth_file": str(DEPTH_PATH),
        "mask_file": str(MASKS[part_name]),
        "k": k.astype(float).tolist(),
        "model_scale": float(initial_candidates[0]["model_scale"]),
        "iteration": int(iteration),
        "debug_dir": str(debug_root / part_name),
        "initial_pose_candidates_cv": initial_candidates,
    }
    return runner.run_alignment(request)


def make_joint_payload(joint_source: dict[str, Any], part_results: dict[str, Any]) -> dict[str, Any]:
    joint = joint_source["joint"]
    axis = np.asarray(joint["axis_camera_closed_to_open"], dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    displacement = float(joint["displacement_m"])
    open_to_closed = np.asarray(joint["translation_open_to_closed_camera_m"], dtype=np.float64)
    closed_to_open = np.asarray(joint["translation_closed_to_open_camera_m"], dtype=np.float64)

    return {
        "method": "qpos1 FoundationPose mesh alignment plus fixed selected_axis1d prismatic joint",
        "frame": joint_source.get("frame", "qpos1 OpenCV/PV camera frame"),
        "camera_axes": joint_source.get("camera_axes"),
        "joint": {
            "type": "prismatic",
            "axis_camera_closed_to_open": axis.astype(float).tolist(),
            "displacement_m": displacement,
            "translation_open_to_closed_camera_m": open_to_closed.astype(float).tolist(),
            "translation_closed_to_open_camera_m": closed_to_open.astype(float).tolist(),
            "qpos_closed_m": 0.0,
            "qpos_open_m": displacement,
            "source_json": str(JOINT_JSON),
            "source_status": joint_source.get("status"),
            "source_axis_label": (joint_source.get("selected_candidate") or {}).get("axis_label"),
        },
        "parts": {
            name: {
                "raw_mesh": str(result["raw_mesh"]),
                "foundationpose_pose_cv": np.asarray(result["pose_cv"], dtype=float).tolist(),
                "model_scale": float(result["model_scale"]),
                "initial_candidates": result["initial_candidates"],
                "foundationpose_initial_search": result["foundationpose"].get("initial_search"),
                "projection_metrics_qpos1": result["projection_metrics_qpos1"],
            }
            for name, result in part_results.items()
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-initial-candidates", type=int, default=2)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--clean", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    stage_root = ROOT / "code/stages/hololens3d_reconstruction"
    sys.path.insert(0, str(stage_root))
    from run_foundationpose_alignment_worker import FoundationPoseAlignmentRunner

    fit_report = read_json(FIT_REPORT)
    joint_source = read_json(JOINT_JSON)
    k = load_k()
    depth_m = load_depth_m(DEPTH_PATH)
    color_bgr = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(COLOR_PATH)

    os.environ.setdefault("FOUNDATIONPOSE_RENDER_BATCH_SIZE", "8")
    runner = FoundationPoseAlignmentRunner()

    part_results: dict[str, Any] = {}
    open_meshes: dict[str, trimesh.Trimesh] = {}
    debug_root = out / "foundationpose_debug"
    for part_name in ("base", "drawer"):
        part_report = fit_report["parts"][part_name]
        raw_mesh_path = Path(part_report["raw_mesh"])
        raw_mesh = load_mesh(raw_mesh_path)
        part_dir = out / part_name
        part_dir.mkdir(parents=True, exist_ok=True)
        foundationpose_mesh_path = part_dir / f"{part_name}_foundationpose_input.obj"
        raw_mesh.export(foundationpose_mesh_path)
        initial_candidates = build_initial_candidates(part_report, int(args.max_initial_candidates))
        fp_result = run_foundationpose(
            runner,
            part_name,
            foundationpose_mesh_path,
            initial_candidates,
            k,
            int(args.iteration),
            debug_root,
        )
        pose_cv = np.asarray(fp_result["pose"], dtype=np.float64).reshape(4, 4)
        model_scale = float(fp_result["model_scale"])
        aligned_mesh = transform_mesh(raw_mesh, model_scale, pose_cv)
        open_meshes[part_name] = aligned_mesh

        aligned_path = part_dir / f"{part_name}.glb"
        aligned_mesh.export(aligned_path)
        overlay_path = part_dir / f"{part_name}_foundationpose_qpos1_overlay.png"
        draw_overlay(color_bgr, k, {part_name: aligned_mesh}, {part_name: MASKS[part_name]}, overlay_path)

        metrics = score_projection(aligned_mesh, k, load_mask(MASKS[part_name]), depth_m, 30000)
        part_results[part_name] = {
            "raw_mesh": str(raw_mesh_path),
            "foundationpose_input_mesh": str(foundationpose_mesh_path),
            "glb": str(aligned_path),
            "overlay": str(overlay_path),
            "pose_cv": pose_cv.astype(float).tolist(),
            "model_scale": model_scale,
            "initial_candidates": initial_candidates,
            "foundationpose": fp_result,
            "projection_metrics_qpos1": metrics,
        }

    open_to_closed = np.asarray(joint_source["joint"]["translation_open_to_closed_camera_m"], dtype=np.float64)
    closed_drawer = translate_mesh(open_meshes["drawer"], open_to_closed)
    closed_drawer_path = out / "drawer" / "drawer_closed_from_selected_axis.glb"
    closed_drawer.export(closed_drawer_path)

    base_path = out / "base.glb"
    drawer_open_path = out / "drawer.glb"
    open_meshes["base"].export(base_path)
    open_meshes["drawer"].export(drawer_open_path)
    export_scene(out / "cabinet_drawer_articulated_open.glb", {"base": open_meshes["base"], "drawer": open_meshes["drawer"]})
    export_scene(out / "cabinet_drawer_articulated_closed.glb", {"base": open_meshes["base"], "drawer_closed": closed_drawer})
    export_scene(
        out / "cabinet_drawer_articulated_open_closed_overlay.glb",
        {"base": open_meshes["base"], "drawer": open_meshes["drawer"], "drawer_closed": closed_drawer},
    )

    draw_overlay(
        color_bgr,
        k,
        {"base": open_meshes["base"], "drawer": open_meshes["drawer"]},
        {"base": MASKS["base"], "drawer": MASKS["drawer"]},
        out / "combined_foundationpose_qpos1_overlay.png",
    )
    draw_overlay(
        color_bgr,
        k,
        {"base": open_meshes["base"], "drawer": open_meshes["drawer"], "drawer_closed": closed_drawer},
        {"base": MASKS["base"], "drawer": MASKS["drawer"]},
        out / "combined_foundationpose_qpos1_open_closed_overlay.png",
    )

    joint_payload = make_joint_payload(joint_source, part_results)
    joint_payload["outputs"] = {
        "base_glb": str(base_path),
        "drawer_open_glb": str(drawer_open_path),
        "drawer_closed_glb": str(closed_drawer_path),
        "open_scene_glb": str(out / "cabinet_drawer_articulated_open.glb"),
        "closed_scene_glb": str(out / "cabinet_drawer_articulated_closed.glb"),
        "open_closed_overlay_glb": str(out / "cabinet_drawer_articulated_open_closed_overlay.glb"),
        "combined_qpos1_overlay": str(out / "combined_foundationpose_qpos1_overlay.png"),
        "combined_qpos1_open_closed_overlay": str(out / "combined_foundationpose_qpos1_open_closed_overlay.png"),
    }
    joint_path = out / "joint.json"
    report_path = out / "foundationpose_qpos1_articulated_report.json"
    joint_path.write_text(json.dumps(joint_payload, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(joint_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "joint_json": str(joint_path),
                "report": str(report_path),
                "open_scene_glb": joint_payload["outputs"]["open_scene_glb"],
                "closed_scene_glb": joint_payload["outputs"]["closed_scene_glb"],
                "overlay": joint_payload["outputs"]["combined_qpos1_overlay"],
                "axis_camera_closed_to_open": joint_payload["joint"]["axis_camera_closed_to_open"],
                "displacement_m": joint_payload["joint"]["displacement_m"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
