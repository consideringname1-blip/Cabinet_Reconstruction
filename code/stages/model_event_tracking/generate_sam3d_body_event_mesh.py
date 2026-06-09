#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh

CODE_ROOT = Path(__file__).resolve().parents[2]
SAM3D_BODY_ROOT = CODE_ROOT / "reconstruction" / "sam3d-body"
if str(SAM3D_BODY_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3D_BODY_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prefer_cached_dinov3_torch_hub() -> None:
    original_parse = getattr(torch.hub, "_parse_repo_info", None)
    if original_parse is None or getattr(torch.hub, "_sam3d_cached_dinov3_patch", False):
        return

    def parse_repo_info(github: str):
        if github == "facebookresearch/dinov3":
            hub_dir = Path(torch.hub.get_dir())
            for ref in ("main", "master"):
                if (hub_dir / f"facebookresearch_dinov3_{ref}").exists():
                    return "facebookresearch", "dinov3", ref
        return original_parse(github)

    torch.hub._parse_repo_info = parse_repo_info
    torch.hub._sam3d_cached_dinov3_patch = True


def depth_image_to_m(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    depth = depth.astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and float(np.nanmedian(finite)) > 20.0:
        depth = depth / 1000.0
    return depth


def parse_camera_matrix(camera_info_path: Path) -> np.ndarray:
    payload = load_json(camera_info_path)
    msg = payload.get("message", payload)
    k = msg.get("k") or msg.get("K") or msg.get("camera_matrix")
    if isinstance(k, str):
        k = np.fromstring(k.replace("[", " ").replace("]", " ").replace(",", " "), sep=" ")
    matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return matrix


def clipped_box(box: np.ndarray, width: int, height: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    x0 = int(max(0, min(width - 1, math.floor(x0 - pad))))
    y0 = int(max(0, min(height - 1, math.floor(y0 - pad))))
    x1 = int(max(0, min(width, math.ceil(x1 + pad))))
    y1 = int(max(0, min(height, math.ceil(y1 + pad))))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def robust_depth_in_box(depth_m: np.ndarray, box: np.ndarray, *, pad: float = -0.15) -> float | None:
    height, width = depth_m.shape[:2]
    box = np.asarray(box, dtype=np.float64).reshape(4).copy()
    if -1.0 < pad < 0.0:
        x0f, y0f, x1f, y1f = box
        shrink = min(x1f - x0f, y1f - y0f) * abs(float(pad))
        box = np.array([x0f + shrink, y0f + shrink, x1f - shrink, y1f - shrink], dtype=np.float64)
        pad_px = 0.0
    else:
        pad_px = pad
    x0, y0, x1, y1 = clipped_box(box, width, height, pad=pad_px)
    roi = depth_m[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi > 0.2) & (roi < 8.0)]
    if valid.size < 25:
        return None
    lo, hi = np.percentile(valid, [15, 85])
    trimmed = valid[(valid >= lo) & (valid <= hi)]
    if trimmed.size < 25:
        trimmed = valid
    return float(np.median(trimmed))


def local_depth_near_pixel(depth_m: np.ndarray, pixel_xy: np.ndarray, radius: int = 12) -> float | None:
    height, width = depth_m.shape[:2]
    u = int(round(float(pixel_xy[0])))
    v = int(round(float(pixel_xy[1])))
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    roi = depth_m[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi > 0.2) & (roi < 8.0)]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def backproject_pixel(pixel_xy: np.ndarray, depth_m: float, camera_matrix: np.ndarray) -> np.ndarray:
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    z = float(depth_m)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def oriented_box_signed_distance(point: np.ndarray, corners: np.ndarray, margin_m: float = 0.0) -> float:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    corners = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    origin = corners[0]
    edges = np.vstack([corners[1] - origin, corners[3] - origin, corners[4] - origin])
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1.0e-9):
        return float(np.linalg.norm(point - corners.mean(axis=0)))
    axes = edges / lengths.reshape(3, 1)
    center = origin + 0.5 * edges.sum(axis=0)
    half_extents = lengths * 0.5 + max(0.0, float(margin_m))
    local = axes @ (point - center)
    delta = np.abs(local) - half_extents
    outside = np.maximum(delta, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance
    return float(np.max(delta))


def project_camera_point(point_camera_m: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray | None:
    point = np.asarray(point_camera_m, dtype=np.float64).reshape(3)
    if point[2] <= 0.0:
        return None
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    return np.array([fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy], dtype=np.float64)


def distance_to_object(point: np.ndarray, object_corners: np.ndarray | None, object_center: np.ndarray | None) -> float:
    if object_corners is not None:
        return max(0.0, oriented_box_signed_distance(point, object_corners))
    if object_center is not None:
        return float(np.linalg.norm(point - object_center))
    return math.inf


def build_estimator(device: str):
    prefer_cached_dinov3_torch_hub()
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_detector import HumanDetector

    checkpoint = SAM3D_BODY_ROOT / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
    mhr = SAM3D_BODY_ROOT / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_cfg = load_sam_3d_body(str(checkpoint), device=torch_device, mhr_path=str(mhr))
    detector = HumanDetector(name="vitdet", device=torch_device)
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
        human_segmentor=None,
        fov_estimator=None,
    )


def detect_boxes(estimator: Any, rgb_bgr: np.ndarray) -> np.ndarray:
    return estimator.detector.run_human_detection(
        rgb_bgr,
        det_cat_id=0,
        bbox_thr=0.35,
        nms_thr=0.3,
        default_to_full_image=False,
    ).astype(np.float32)


def prefilter_boxes(boxes: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray, object_corners: np.ndarray | None, max_distance_m: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0 or object_corners is None:
        return boxes, []
    object_center = object_corners.mean(axis=0)
    object_pixel = project_camera_point(object_center, camera_matrix)
    height, width = depth_m.shape[:2]
    records = []
    kept = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = clipped_box(box, width, height)
        observed_depth = robust_depth_in_box(depth_m, box)
        samples = []
        if observed_depth is not None:
            center_pixel = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
            center_point = backproject_pixel(center_pixel, observed_depth, camera_matrix)
            samples.append(("bbox_center_depth", center_pixel, observed_depth, center_point, distance_to_object(center_point, object_corners, object_center)))
        if object_pixel is not None:
            near_pixel = np.array([
                min(max(float(object_pixel[0]), float(x0)), float(max(x0, x1 - 1))),
                min(max(float(object_pixel[1]), float(y0)), float(max(y0, y1 - 1))),
            ], dtype=np.float64)
            near_depth = local_depth_near_pixel(depth_m, near_pixel) or observed_depth
            if near_depth is not None:
                near_point = backproject_pixel(near_pixel, near_depth, camera_matrix)
                samples.append(("closest_bbox_pixel_to_object_projection", near_pixel, near_depth, near_point, distance_to_object(near_point, object_corners, object_center)))
        if samples:
            best = min(samples, key=lambda item: item[4])
            best_distance = float(best[4])
        else:
            best = None
            best_distance = math.inf
        keep = bool(np.isfinite(best_distance) and best_distance <= max_distance_m)
        if keep:
            kept.append(index)
        records.append({
            "index": index,
            "bbox_xyxy": box.astype(float).tolist(),
            "kept": keep,
            "best_distance_m": best_distance if np.isfinite(best_distance) else None,
            "best_source": best[0] if best else None,
        })
    if not kept and records:
        finite = [r for r in records if r.get("best_distance_m") is not None]
        if finite:
            nearest = min(finite, key=lambda item: item["best_distance_m"])
            kept = [int(nearest["index"])]
            for record in records:
                if int(record["index"]) == kept[0]:
                    record["kept"] = True
                    record["reason"] = "fallback_keep_nearest_person_box"
    return boxes[kept].astype(np.float32), records


def align_output(output: dict[str, Any], depth_m: np.ndarray, camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vertices = np.asarray(output["pred_vertices"], dtype=np.float64) + np.asarray(output["pred_cam_t"], dtype=np.float64).reshape(1, 3)
    keypoints = np.asarray(output["pred_keypoints_3d"], dtype=np.float64) + np.asarray(output["pred_cam_t"], dtype=np.float64).reshape(1, 3)
    bbox = np.asarray(output["bbox"], dtype=np.float64).reshape(4)
    observed_depth = robust_depth_in_box(depth_m, bbox)
    mesh_depth = float(np.median(vertices[:, 2])) if vertices.size else None
    shift = np.zeros(3, dtype=np.float64)
    if observed_depth is not None and mesh_depth is not None and math.isfinite(mesh_depth):
        anchor_px = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)
        shift = backproject_pixel(anchor_px, observed_depth, camera_matrix) - backproject_pixel(anchor_px, mesh_depth, camera_matrix)
        vertices += shift.reshape(1, 3)
        keypoints += shift.reshape(1, 3)
    return vertices, keypoints, {
        "observed_depth_m": observed_depth,
        "mesh_median_depth_before_align_m": mesh_depth,
        "xyz_shift_m": shift.astype(float).tolist(),
        "alignment_method": "bbox_center_ray_translation",
    }


def project_to_panel(vertices_camera_m: np.ndarray, camera_matrix: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    z = np.maximum(vertices_camera_m[:, 2], 1.0e-6)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    u = fx * vertices_camera_m[:, 0] / z + cx
    v = fy * vertices_camera_m[:, 1] / z + cy
    panel_x = (u / max(1, image_width)) - 0.5
    panel_y = 0.5 - (v / max(1, image_height))
    panel_z = np.full_like(panel_x, 0.006)
    return np.stack([panel_x, panel_y, panel_z], axis=1).astype(np.float32)


def simplify_panel_mesh(vertices: np.ndarray, faces: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ratio = min(1.0, max(0.001, float(ratio)))
    raw_faces = int(len(faces))
    target_faces = max(64, int(round(raw_faces * ratio)))
    if raw_faces <= target_faces:
        return vertices.astype(np.float32), faces.astype(np.int32), {"method": "none", "ratio": 1.0, "raw_faces": raw_faces, "faces": raw_faces}
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        return np.asarray(simplified.vertices, dtype=np.float32), np.asarray(simplified.faces, dtype=np.int32), {
            "method": "trimesh_quadric_decimation",
            "ratio": ratio,
            "target_faces": target_faces,
            "raw_vertices": int(len(vertices)),
            "raw_faces": raw_faces,
            "vertices": int(len(simplified.vertices)),
            "faces": int(len(simplified.faces)),
        }
    except Exception as exc:
        areas = mesh.area_faces
        order = np.argsort(areas)[::-1][:target_faces]
        sampled_faces = faces[order]
        used = np.unique(sampled_faces.reshape(-1))
        remap = {int(old): idx for idx, old in enumerate(used)}
        compact_faces = np.vectorize(remap.__getitem__)(sampled_faces).astype(np.int32)
        compact_vertices = vertices[used].astype(np.float32)
        return compact_vertices, compact_faces, {
            "method": "area_largest_faces_fallback",
            "error": str(exc),
            "ratio": ratio,
            "target_faces": target_faces,
            "raw_vertices": int(len(vertices)),
            "raw_faces": raw_faces,
            "vertices": int(len(compact_vertices)),
            "faces": int(len(compact_faces)),
        }


def frame_paths_from_event(event: dict[str, Any], event_dir: Path) -> tuple[Path, Path, Path]:
    evidence = event.get("evidence_frame") if isinstance(event.get("evidence_frame"), dict) else {}
    rgb = Path(evidence.get("rgb_path") or event_dir / "rgb.png")
    depth = Path(evidence.get("depth_path") or event_dir / "debug" / "depth.png")
    camera = Path(evidence.get("camera_info_path") or event_dir / "debug" / "camera_info.json")
    return rgb, depth, camera


def generate(event_dir: Path, *, output: Path, decimate_ratio: float, max_distance_m: float, device: str) -> dict[str, Any]:
    event_json = event_dir / "event.json"
    event = load_json(event_json)
    rgb_path, depth_path, camera_info_path = frame_paths_from_event(event, event_dir)
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    depth_m = depth_image_to_m(depth_path)
    camera_matrix = parse_camera_matrix(camera_info_path)
    projected_box = event.get("projected_box") or {}
    corners = projected_box.get("corners_camera_m")
    object_corners = np.asarray(corners, dtype=np.float64).reshape(8, 3) if corners is not None else None

    estimator = build_estimator(device)
    boxes = detect_boxes(estimator, rgb_bgr)
    boxes, prefilter = prefilter_boxes(boxes, depth_m, camera_matrix, object_corners, max_distance_m)
    if boxes.size == 0:
        raise RuntimeError("no person boxes selected for body mesh")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    cam_int = torch.tensor(camera_matrix, dtype=torch.float32).reshape(1, 3, 3)
    outputs = estimator.process_one_image(rgb, bboxes=boxes, cam_int=cam_int, use_mask=False, inference_type="body")
    faces = np.asarray(estimator.faces, dtype=np.int32)
    people = []
    height, width = rgb_bgr.shape[:2]
    for index, output_item in enumerate(outputs):
        vertices_camera, _keypoints, align = align_output(output_item, depth_m, camera_matrix)
        panel_vertices = project_to_panel(vertices_camera, camera_matrix, width, height)
        low_vertices, low_faces, decimate = simplify_panel_mesh(panel_vertices, faces, decimate_ratio)
        people.append({
            "person_index": index,
            "bbox_xyxy": np.asarray(output_item["bbox"], dtype=float).reshape(4).tolist(),
            "vertices": np.round(low_vertices, 6).astype(float).tolist(),
            "triangles": low_faces.reshape(-1).astype(int).tolist(),
            "triangle_count": int(len(low_faces)),
            "vertex_count": int(len(low_vertices)),
            "alignment": align,
            "decimate": decimate,
        })

    payload = {
        "format": "h2ai_event_body_mesh_v1",
        "coordinate_system": "event_image_panel_normalized",
        "image_width": int(width),
        "image_height": int(height),
        "panel_vertex_scale": "Unity multiplies x by panel width and y by panel height; z is meters in popup local space.",
        "material": {"color": [0.55, 0.55, 0.55, 0.38], "transparent": True},
        "source": {
            "event_dir": str(event_dir),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera_info_path": str(camera_info_path),
        },
        "raw_bbox_count": int(len(prefilter)),
        "kept_bbox_count": int(len(boxes)),
        "person_prefilter": prefilter,
        "decimate_ratio": float(decimate_ratio),
        "people": people,
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decimate-ratio", type=float, default=1.0 / 8.0)
    parser.add_argument("--max-distance-m", type=float, default=1.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output or (args.event_dir / "body_mesh.json")
    payload = generate(
        args.event_dir,
        output=output,
        decimate_ratio=args.decimate_ratio,
        max_distance_m=args.max_distance_m,
        device=args.device,
    )
    print(json.dumps({"output": str(output), "people": len(payload["people"]), "faces": [p["triangle_count"] for p in payload["people"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
