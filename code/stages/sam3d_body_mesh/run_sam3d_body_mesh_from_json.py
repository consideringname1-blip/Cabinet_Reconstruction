from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from artifact_layout import SHIGURE_HISTORY_CACHE_ROOT, model_debug_dir, model_result_file, model_worker_dir
from path_config import BLENDER_BIN, SAM3D_BODY_FBX_EXPORT_SCRIPT, SAM3D_BODY_ROOT
from coordinate_systems import quat_xyzw_to_rotation_matrix
from spatial_transforms import aruco_points_to_shigure_camera, project_camera_points_to_pixels, shigure_camera_points_to_aruco
from task_json import load_task_json, resolve_task_json_path, save_task_json
from stages.shigure_history.marker_history import latest_marker_pose_path

from stages.sam3d_body_mesh import settings


OPENCV_TO_BLENDER_CAMERA_BASIS = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write('\n')


def _write_status(json_path: Path, task: dict[str, Any], status: str, **fields: Any) -> None:
    payload = dict(fields)
    payload['status'] = status
    payload['updated_at'] = _utc_now()
    task['SAM3DBodyMesh'] = _jsonable(payload)
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for SAM3D body status artifacts')
    _write_json(model_result_file(task_timestamp, 'body.result'), payload)
    save_task_json(json_path, task)


def _parse_float_array(value: Any, count: int, label: str) -> np.ndarray:
    if isinstance(value, str):
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)]
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != count:
        raise ValueError(f'{label} must contain {count} values')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{label} contains non-finite values')
    return array


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _load_camera_matrix(path: Path) -> tuple[np.ndarray, int, int]:
    payload = _load_json(path)
    msg = payload.get('message') if isinstance(payload.get('message'), Mapping) else payload
    k = msg.get('k') or msg.get('K') or msg.get('camera_matrix')
    matrix = _parse_float_array(k, 9, 'camera matrix').reshape(3, 3)
    width = int(msg.get('width') or 0)
    height = int(msg.get('height') or 0)
    return matrix.astype(np.float64), width, height


def _depth_image_to_m(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and float(np.nanmedian(finite)) > 20.0:
        depth = depth / 1000.0
    return depth.astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_sam3d_imports() -> None:
    root = Path(SAM3D_BODY_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f'SAM3D Body root not found: {root}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _prefer_cached_dinov3_torch_hub(torch_module: Any) -> None:
    if getattr(torch_module.hub, '_sam3d_cached_dinov3_patch', False):
        return
    original_parse = torch_module.hub._parse_repo_info

    def parse_repo_info(github: str):
        if github == 'facebookresearch/dinov3':
            hub_dir = Path(torch_module.hub.get_dir())
            for ref in ('main', 'master'):
                if (hub_dir / f'facebookresearch_dinov3_{ref}').exists():
                    return 'facebookresearch', 'dinov3', ref
        return original_parse(github)

    torch_module.hub._parse_repo_info = parse_repo_info
    torch_module.hub._sam3d_cached_dinov3_patch = True


def _build_estimator(detector_name: str, device: str):
    _ensure_sam3d_imports()
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_detector import HumanDetector

    checkpoint = Path(SAM3D_BODY_ROOT) / 'checkpoints' / 'sam-3d-body-dinov3' / 'model.ckpt'
    mhr = Path(SAM3D_BODY_ROOT) / 'checkpoints' / 'sam-3d-body-dinov3' / 'assets' / 'mhr_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not mhr.is_file():
        raise FileNotFoundError(mhr)
    _prefer_cached_dinov3_torch_hub(torch)
    torch_device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, model_cfg = load_sam_3d_body(str(checkpoint), device=torch_device, mhr_path=str(mhr))
    detector = None
    if detector_name and detector_name != 'none':
        detector = HumanDetector(name=detector_name, device=torch_device)
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
        human_segmentor=None,
        fov_estimator=None,
    )


def _detect_boxes(estimator: Any, rgb_bgr: np.ndarray) -> np.ndarray:
    if settings.SAM3D_BODY_DETECTOR_NAME == 'none' or estimator.detector is None:
        height, width = rgb_bgr.shape[:2]
        return np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
    boxes = estimator.detector.run_human_detection(
        rgb_bgr,
        det_cat_id=0,
        bbox_thr=settings.SAM3D_BODY_BBOX_THRESHOLD,
        nms_thr=settings.SAM3D_BODY_NMS_THRESHOLD,
        default_to_full_image=False,
    )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _run_sam3d_body(estimator: Any, rgb_bgr: np.ndarray, boxes: np.ndarray, camera_matrix: np.ndarray) -> list[dict[str, Any]]:
    import torch

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    cam_int = torch.tensor(camera_matrix, dtype=torch.float32).reshape(1, 3, 3)
    outputs = estimator.process_one_image(
        rgb,
        bboxes=boxes,
        cam_int=cam_int,
        use_mask=False,
        inference_type=settings.SAM3D_BODY_INFERENCE_TYPE,
    )
    return list(outputs or [])


def _clip_box(box: np.ndarray, width: int, height: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return (
        max(0, min(width - 1, int(math.floor(x0 - pad)))),
        max(0, min(height - 1, int(math.floor(y0 - pad)))),
        max(0, min(width, int(math.ceil(x1 + pad)))),
        max(0, min(height, int(math.ceil(y1 + pad)))),
    )


def _rasterize_mesh_depth(vertices: np.ndarray, faces: np.ndarray, camera_matrix: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    z = vertices[:, 2]
    valid_vertices = z > 0.02
    u = np.zeros(vertices.shape[0], dtype=np.float64)
    v = np.zeros(vertices.shape[0], dtype=np.float64)
    u[valid_vertices] = fx * vertices[valid_vertices, 0] / z[valid_vertices] + cx
    v[valid_vertices] = fy * vertices[valid_vertices, 1] / z[valid_vertices] + cy
    for face in faces[:, :3]:
        if not np.all(valid_vertices[face]):
            continue
        xs = u[face]
        ys = v[face]
        zs = z[face]
        x0 = max(0, int(math.floor(float(xs.min()))))
        x1 = min(width - 1, int(math.ceil(float(xs.max()))))
        y0 = max(0, int(math.floor(float(ys.min()))))
        y1 = min(height - 1, int(math.ceil(float(ys.max()))))
        if x1 < x0 or y1 < y0:
            continue
        x_grid = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
        y_grid = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(x_grid, y_grid)
        xa, xb, xc = xs
        ya, yb, yc = ys
        denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(float(denom)) < 1.0e-8:
            continue
        wa = ((yb - yc) * (xx - xc) + (xc - xb) * (yy - yc)) / denom
        wb = ((yc - ya) * (xx - xc) + (xa - xc) * (yy - yc)) / denom
        wc = 1.0 - wa - wb
        inside = (wa >= -1.0e-6) & (wb >= -1.0e-6) & (wc >= -1.0e-6)
        if not inside.any():
            continue
        zi = wa * zs[0] + wb * zs[1] + wc * zs[2]
        roi = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (zi > 0.02) & (zi < roi)
        roi[update] = zi.astype(np.float32)[update]
    zbuffer[~np.isfinite(zbuffer)] = 0.0
    return zbuffer


def _save_body_mesh_debug_overlay(
    task_timestamp: str,
    rgb_bgr: np.ndarray,
    vertices_camera_m: np.ndarray,
    faces: np.ndarray,
    camera_matrix: np.ndarray,
    bbox_xyxy: Any,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    debug_dir = model_debug_dir(task_timestamp)
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = debug_dir / '08_sam3d_body_mesh_on_taken_rgb.png'
    rendered_depth = _rasterize_mesh_depth(vertices_camera_m, faces, camera_matrix, rgb_bgr.shape[:2])
    mesh_mask = rendered_depth > 0.0
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.astype(np.float32).copy()
    if np.any(mesh_mask):
        mesh_color = np.asarray([0.0, 235.0, 185.0], dtype=np.float32)
        overlay[mesh_mask] = overlay[mesh_mask] * 0.52 + mesh_color * 0.48
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image, 'RGBA')
    if np.any(mesh_mask):
        contours, _ = cv2.findContours(mesh_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = [(int(p[0][0]), int(p[0][1])) for p in contour]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill=(255, 255, 255, 210), width=2)
    bbox_values = None
    try:
        bbox_values = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)[:4]]
    except Exception:
        bbox_values = None
    if bbox_values and len(bbox_values) == 4:
        x0, y0, x1, y1 = bbox_values
        draw.rectangle([x0, y0, x1, y1], outline=(0, 235, 185, 255), width=4)
        tag = label or 'sam3d body'
        tag_box = [x0, max(0.0, y0 - 24.0), min(float(rgb.shape[1]), x0 + 12.0 + len(tag) * 8.0), y0]
        draw.rectangle(tag_box, fill=(0, 0, 0, 150))
        draw.text((x0 + 6.0, max(0.0, y0 - 21.0)), tag, fill=(255, 255, 255, 255))
    image.save(overlay_path)
    return overlay_path, {
        'body_mesh_on_taken_rgb_path': str(overlay_path),
        'mesh_overlay_pixels': int(np.count_nonzero(mesh_mask)),
        'selected_person_bbox_xyxy': bbox_values,
    }


def _camera_ray(pixel_xy: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    ray = np.array([(float(pixel_xy[0]) - cx) / fx, (float(pixel_xy[1]) - cy) / fy, 1.0], dtype=np.float64)
    norm = np.linalg.norm(ray)
    return ray / norm if norm > 1.0e-9 else ray


def _align_person_depth(output: Mapping[str, Any], faces: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray) -> dict[str, Any]:
    cam_t = _to_numpy(output["pred_cam_t"]).astype(np.float64).reshape(1, 3)
    vertices = _to_numpy(output["pred_vertices"]).astype(np.float64) + cam_t
    keypoints = _to_numpy(output["pred_keypoints_3d"]).astype(np.float64) + cam_t
    bbox = _to_numpy(output["bbox"]).astype(np.float64).reshape(4)
    rendered = _rasterize_mesh_depth(vertices, faces, camera_matrix, depth_m.shape[:2])
    body_mask = rendered > 0.0
    x0, y0, x1, y1 = _clip_box(bbox, depth_m.shape[1], depth_m.shape[0], settings.BBOX_DEPTH_PAD_PX)
    roi_depth = depth_m[y0:y1, x0:x1]
    roi_rendered = rendered[y0:y1, x0:x1]
    valid = (roi_depth > 0.0) & np.isfinite(roi_depth) & (roi_rendered > 0.0) & np.isfinite(roi_rendered)
    valid_pixels = int(np.count_nonzero(valid))
    rendered_pixels = int(np.count_nonzero(roi_rendered > 0.0))
    ratio = valid_pixels / max(1, rendered_pixels)
    offset = 0.0
    distance_scale = 1.0
    original_distance = None
    adjusted_distance = None
    method = "body_mask_depth_median_offset_and_scale"
    sampled_pixels = 0
    if valid_pixels >= settings.DEPTH_OVERLAP_PIXELS and ratio >= settings.DEPTH_OVERLAP_RATIO:
        depth_values = roi_depth[valid].astype(np.float64)
        rendered_values = roi_rendered[valid].astype(np.float64)
        if depth_values.size > settings.DEPTH_SAMPLE_MAX:
            indices = np.linspace(0, depth_values.size - 1, settings.DEPTH_SAMPLE_MAX, dtype=np.int64)
            depth_values = depth_values[indices]
            rendered_values = rendered_values[indices]
        sampled_pixels = int(depth_values.size)
        residual = depth_values - rendered_values
        offset = float(np.median(residual))
        original_distance = float(np.median(rendered_values)) if rendered_values.size else None
        adjusted_distance = float(original_distance + offset) if original_distance is not None else None
        if original_distance is not None and original_distance > 1.0e-6 and adjusted_distance is not None and adjusted_distance > 1.0e-6:
            distance_scale = float(np.clip(adjusted_distance / original_distance, settings.DEPTH_SCALE_MIN, settings.DEPTH_SCALE_MAX))
    else:
        method = "insufficient_body_mask_depth_overlap_no_translation_or_scale"

    center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)
    mesh_center = np.nanmedian(vertices, axis=0)
    if np.all(np.isfinite(mesh_center)) and distance_scale != 1.0:
        vertices = (vertices - mesh_center.reshape(1, 3)) * distance_scale + mesh_center.reshape(1, 3)
        keypoints = (keypoints - mesh_center.reshape(1, 3)) * distance_scale + mesh_center.reshape(1, 3)
    shift = _camera_ray(center, camera_matrix) * offset
    vertices = vertices + shift.reshape(1, 3)
    keypoints = keypoints + shift.reshape(1, 3)
    rendered_aligned = _rasterize_mesh_depth(vertices, faces, camera_matrix, depth_m.shape[:2])
    aligned_body_mask = rendered_aligned > 0.0
    return {
        "vertices_camera_m": vertices,
        "keypoints_camera_m": keypoints,
        "bbox_xyxy": bbox,
        "body_mask": aligned_body_mask,
        "depth_offset_m": offset,
        "xyz_shift_m": shift,
        "distance_scale": distance_scale,
        "original_distance_m": original_distance,
        "adjusted_distance_m": adjusted_distance,
        "depth_overlap_pixels": valid_pixels,
        "depth_sampled_pixels": sampled_pixels,
        "depth_overlap_ratio": ratio,
        "depth_alignment_method": method,
    }


def _load_marker_camera_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(path)
    pose = payload.get('opencv_camera_pose') if isinstance(payload.get('opencv_camera_pose'), Mapping) else payload
    if not isinstance(pose, Mapping):
        raise ValueError('marker pose JSON does not contain opencv_camera_pose')
    if pose.get('rotation_matrix') is not None:
        rotation = _parse_float_array(pose.get('rotation_matrix'), 9, 'marker rotation_matrix').reshape(3, 3)
    elif pose.get('rotation_quaternion_xyzw') is not None:
        rotation = quat_xyzw_to_rotation_matrix(_parse_float_array(pose.get('rotation_quaternion_xyzw'), 4, 'marker quaternion'))
    else:
        raise ValueError('marker pose is missing rotation')
    translation = _parse_float_array(pose.get('tvec_m') if pose.get('tvec_m') is not None else pose.get('position'), 3, 'marker translation')
    return rotation.astype(np.float64), translation.astype(np.float64)


def _camera_to_armarker_points(points_camera_m: np.ndarray, marker_rotation_camera_marker_cv: np.ndarray, marker_translation_camera_marker_cv: np.ndarray) -> np.ndarray:
    return shigure_camera_points_to_aruco(
        points_camera_m,
        marker_rotation_camera_marker_cv,
        marker_translation_camera_marker_cv,
    )


def _object_center_aruco(task: Mapping[str, Any]) -> np.ndarray | None:
    bounds = task.get('ModelBounds') if isinstance(task.get('ModelBounds'), Mapping) else None
    if bounds and bounds.get('aabb_min_aruco') is not None and bounds.get('aabb_max_aruco') is not None:
        a = np.asarray(bounds.get('aabb_min_aruco'), dtype=np.float64).reshape(3)
        b = np.asarray(bounds.get('aabb_max_aruco'), dtype=np.float64).reshape(3)
        return (a + b) * 0.5
    obj = task.get('object_aruco') if isinstance(task.get('object_aruco'), Mapping) else None
    if obj and obj.get('position') is not None:
        return np.asarray(obj.get('position'), dtype=np.float64).reshape(3)
    return None


def _nearest_wrist(person_name: str, keypoints_aruco: np.ndarray, object_center: np.ndarray | None) -> dict[str, Any] | None:
    if object_center is None:
        return None
    records = []
    for wrist, idx in settings.WRIST_INDEXES.items():
        if idx >= len(keypoints_aruco):
            continue
        point = keypoints_aruco[idx]
        if not np.all(np.isfinite(point)):
            continue
        distance = float(np.linalg.norm(point - object_center))
        records.append({
            'person_name': person_name,
            'wrist': wrist,
            'joint_index': int(idx),
            'point_armarker': point.astype(float).tolist(),
            'distance_m': distance,
            'within_trusted_range': bool(distance <= settings.MAX_WRIST_DISTANCE_M),
        })
    if not records:
        return None
    return min(records, key=lambda item: item['distance_m'])


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with path.open('w', encoding='utf-8') as f:
        f.write('# SAM3D Body selected mesh in ArMarker/Unity coordinates\n')
        for v in vertices:
            f.write(f'v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n')
        for face in faces:
            a, b, c = [int(i) + 1 for i in face[:3]]
            f.write(f'f {a} {b} {c}\n')


def _export_fbx(obj_path: Path, fbx_path: Path) -> None:
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    blender = Path(BLENDER_BIN)
    if not blender.is_file():
        found = shutil.which('blender')
        if not found:
            raise FileNotFoundError(f'Blender executable not found: {BLENDER_BIN}')
        blender = Path(found)
    cfg_path = obj_path.with_suffix('.fbx_export.json')
    _write_json(cfg_path, {
        'obj_path': str(obj_path),
        'fbx_path': str(fbx_path),
        'decimate_ratio': settings.FBX_DECIMATE_RATIO,
        'material_color': settings.MATERIAL_COLOR,
        'material_alpha': settings.MATERIAL_ALPHA,
    })
    subprocess.run([str(blender), '--background', '--python', str(SAM3D_BODY_FBX_EXPORT_SCRIPT), '--', str(cfg_path)], check=True)


def _load_backup_paths(taken: Mapping[str, Any]) -> tuple[Path, Path, Path, Path | None]:
    backup_dir = Path(str(taken.get('backup_shigurei_dir') or ''))
    if not backup_dir.is_dir():
        raise FileNotFoundError(f'backup_shigurei_dir not found: {backup_dir}')
    rgb = backup_dir / 'rgb.png'
    depth = backup_dir / 'depth.png'
    camera = backup_dir / 'camera_info.json'
    marker = backup_dir / 'marker_6d_pose.json'
    for path in (rgb, depth, camera):
        if not path.is_file():
            raise FileNotFoundError(path)
    if marker.is_file():
        return rgb, depth, camera, marker
    return rgb, depth, camera, latest_marker_pose_path()


def _load_object_mask_from_taken(taken: Mapping[str, Any], image_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    history = taken.get("history_baseline") if isinstance(taken.get("history_baseline"), Mapping) else {}
    init = taken.get("init") if isinstance(taken.get("init"), Mapping) else {}
    if not history and isinstance(init.get("history_baseline"), Mapping):
        history = init.get("history_baseline")
    candidates = []
    if isinstance(history, Mapping) and history.get("old_mask_path"):
        candidates.append(("history_baseline.old_mask_path", Path(str(history.get("old_mask_path")))))
    debug_files = taken.get("debug_files") if isinstance(taken.get("debug_files"), Mapping) else {}
    init_debug = init.get("debug_files") if isinstance(init.get("debug_files"), Mapping) else {}
    for source, payload in (("taken.debug_files", debug_files), ("taken.init.debug_files", init_debug)):
        value = payload.get("init_trusted_shigure_mask_path")
        if value:
            candidates.append((f"{source}.init_trusted_shigure_mask_path", Path(str(value))))
    backup_dir = Path(str(taken.get("init_backup_shigurei_dir") or init.get("init_backup_shigurei_dir") or ""))
    if backup_dir.is_dir():
        candidates.append(("init_backup.old_mask", backup_dir / "old_mask.png"))
    for source, path in candidates:
        if not path.is_file():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        mask = raw > 0
        if mask.shape != image_shape:
            mask = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.any(mask):
            return mask, {"source": source, "path": str(path), "pixels": int(np.count_nonzero(mask))}
    return None, {"source": "missing", "checked": [{"source": src, "path": str(p)} for src, p in candidates]}



def _mask_bbox(mask: np.ndarray, pad: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(width, int(xs.max()) + 1 + pad)
    y1 = min(height, int(ys.max()) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _normalise_mask(mask: np.ndarray | None, image_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    h, w = image_shape
    normalized = np.asarray(mask).astype(bool)
    if normalized.shape != (h, w):
        normalized = cv2.resize(normalized.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return normalized if np.any(normalized) else None


def _mask_from_bbox(bbox_xyxy: Any, image_shape: tuple[int, int], pad: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    h, w = image_shape
    try:
        values = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)[:4]
    except Exception as exc:
        return None, {'source': 'selected_body_bbox', 'reason': 'invalid_bbox', 'error_message': str(exc)}
    if values.size != 4 or not np.all(np.isfinite(values)):
        return None, {'source': 'selected_body_bbox', 'reason': 'bbox_missing_or_non_finite'}
    x0, y0, x1, y1 = _clip_box(values, w, h, float(pad))
    if x1 <= x0 or y1 <= y0:
        return None, {'source': 'selected_body_bbox', 'reason': 'bbox_outside_image', 'bbox_xyxy': values.astype(float).tolist()}
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, {
        'source': 'selected_body_bbox',
        'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'pixels': int(np.count_nonzero(mask)),
    }


def _project_object_center_mask(
    object_center_aruco: np.ndarray | None,
    marker_rotation: np.ndarray,
    marker_translation: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, int],
    object_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if object_center_aruco is None:
        return None, {'source': 'object_center_armarker_projection', 'reason': 'object_center_missing'}
    h, w = image_shape
    try:
        object_center = np.asarray(object_center_aruco, dtype=np.float64).reshape(1, 3)
        camera_point = aruco_points_to_shigure_camera(object_center, marker_rotation, marker_translation).reshape(3)
        pixels, visible = project_camera_points_to_pixels(camera_point.reshape(1, 3), camera_matrix)
    except Exception as exc:
        return None, {'source': 'object_center_armarker_projection', 'reason': 'projection_failed', 'error_message': str(exc)}
    if not bool(visible[0]) or not np.all(np.isfinite(pixels[0])):
        return None, {
            'source': 'object_center_armarker_projection',
            'reason': 'object_center_behind_camera_or_non_finite',
            'object_center_camera_m': camera_point.astype(float).tolist(),
        }
    u = int(round(float(pixels[0, 0])))
    v = int(round(float(pixels[0, 1])))
    if u < 0 or u >= w or v < 0 or v >= h:
        return None, {
            'source': 'object_center_armarker_projection',
            'reason': 'object_center_outside_image',
            'pixel_xy': [float(pixels[0, 0]), float(pixels[0, 1])],
            'object_center_camera_m': camera_point.astype(float).tolist(),
        }
    radius = int(settings.SUBJECT_CROP_OBJECT_CENTER_RADIUS_PX)
    normalized_object = _normalise_mask(object_mask, image_shape)
    object_bbox = None
    if normalized_object is not None:
        object_bbox = _mask_bbox(normalized_object, 0, w, h)
        if object_bbox is not None:
            ox0, oy0, ox1, oy1 = object_bbox
            radius = max(radius, int(math.ceil(max(ox1 - ox0, oy1 - oy0) * 0.25)))
    radius = max(1, min(int(settings.SUBJECT_CROP_OBJECT_CENTER_MAX_RADIUS_PX), radius))
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_u8, (u, v), radius, 1, thickness=-1)
    mask = mask_u8 > 0
    return mask, {
        'source': 'object_center_armarker_projection',
        'pixel_xy': [int(u), int(v)],
        'radius_px': int(radius),
        'pixels': int(np.count_nonzero(mask)),
        'object_center_armarker': object_center.reshape(3).astype(float).tolist(),
        'object_center_camera_m': camera_point.astype(float).tolist(),
        'object_mask_bbox_xyxy': [int(v) for v in object_bbox] if object_bbox is not None else None,
    }


def _write_subject_crop(
    task_timestamp: str,
    rgb_bgr: np.ndarray,
    body_mask: np.ndarray | None,
    body_bbox_xyxy: Any,
    object_mask: np.ndarray | None,
    object_center_mask: np.ndarray | None,
) -> tuple[str | None, dict[str, Any]]:
    h, w = rgb_bgr.shape[:2]
    image_shape = (h, w)
    union = np.zeros((h, w), dtype=bool)
    sources = []

    body = _normalise_mask(body_mask, image_shape)
    bbox_mask, bbox_info = _mask_from_bbox(body_bbox_xyxy, image_shape, int(settings.SUBJECT_CROP_BODY_BBOX_PAD_PX))
    if body is not None and bbox_mask is not None:
        clipped_body = body & bbox_mask
        if np.any(clipped_body):
            union |= clipped_body
            sources.append({
                'source': 'selected_body_mesh_mask_clipped_by_body_bbox',
                'pixels': int(np.count_nonzero(clipped_body)),
                'raw_body_mask_pixels': int(np.count_nonzero(body)),
                'body_bbox': bbox_info,
            })
        else:
            union |= body
            sources.append({
                'source': 'selected_body_mesh_mask_bbox_clip_empty_fallback',
                'pixels': int(np.count_nonzero(body)),
                'body_bbox': bbox_info,
            })
    elif body is not None:
        union |= body
        sources.append({'source': 'selected_body_mesh_mask', 'pixels': int(np.count_nonzero(body)), 'body_bbox': bbox_info})
    elif bbox_mask is not None:
        union |= bbox_mask
        sources.append({'source': 'selected_body_bbox_fallback', 'pixels': int(np.count_nonzero(bbox_mask)), 'body_bbox': bbox_info})
    else:
        sources.append(bbox_info)

    obj = _normalise_mask(object_mask, image_shape)
    if obj is not None:
        union |= obj
        sources.append({'source': 'taken_object_old_mask', 'pixels': int(np.count_nonzero(obj))})

    center = _normalise_mask(object_center_mask, image_shape)
    if center is not None:
        union |= center
        sources.append({'source': 'object_center_armarker_projection_mask', 'pixels': int(np.count_nonzero(center))})

    bbox = _mask_bbox(union, int(settings.SUBJECT_CROP_PAD_PX), w, h)
    if bbox is None:
        return None, {'reason': 'empty_subject_crop_mask', 'sources': sources}
    x0, y0, x1, y1 = bbox
    crop = rgb_bgr[y0:y1, x0:x1].copy()
    crop_path = model_result_file(task_timestamp, 'body.subject_crop')
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_path), crop)
    return str(crop_path), {
        'reason': 'subject_crop_ready',
        'subject_crop_path': str(crop_path),
        'subject_crop_folder': 'model_result',
        'crop_bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'crop_size_px': [int(x1 - x0), int(y1 - y0)],
        'source_image_size_px': [int(w), int(h)],
        'display_resize_policy': 'fit_aspect_to_existing_window_max_scale',
        'sources': sources,
    }


def _write_latest_body_mesh_registry(payload: Mapping[str, Any]) -> str:
    path = SHIGURE_HISTORY_CACHE_ROOT / "latest_sam3d_body_mesh.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {**dict(payload), "updated_at": _utc_now()})
    return str(path)


def run_sam3d_body_mesh(json_path_arg: str | Path) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    taken = task.get('TakenObjectDetection') if isinstance(task.get('TakenObjectDetection'), Mapping) else {}
    task_name = str(task.get('task_name') or task.get('task_id') or json_path.stem)
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for SAM3D body artifacts')
    output_root = model_worker_dir(task_timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    if taken.get('status') != 'TAKEN':
        payload = {
            'result_timestamp': taken.get('result_timestamp'),
            'backup_shigurei_dir': taken.get('backup_shigurei_dir'),
            'reason': f"taken_object_detection_status={taken.get('status')}",
        }
        _write_status(json_path, task, 'SKIPPED_NOT_TAKEN', **payload)
        return {'status': 'SKIPPED_NOT_TAKEN'}

    rgb_path, depth_path, camera_info_path, marker_pose_path = _load_backup_paths(taken)
    result_timestamp = taken.get('result_timestamp')
    backup_dir = str(taken.get('backup_shigurei_dir'))
    if marker_pose_path is None:
        _write_status(
            json_path,
            task,
            'NO_VALID_WRIST_JOINT',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            reason='camera_to_armarker_marker_pose_missing',
            output_dir=str(output_root),
        )
        return {'status': 'NO_VALID_WRIST_JOINT', 'reason': 'camera_to_armarker_marker_pose_missing'}

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise RuntimeError(f'failed to read RGB: {rgb_path}')
    if depth_raw is None:
        raise RuntimeError(f'failed to read depth: {depth_path}')
    depth_m = _depth_image_to_m(depth_raw)
    if depth_m.shape[:2] != rgb_bgr.shape[:2]:
        depth_m = cv2.resize(depth_m, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    camera_matrix, width, height = _load_camera_matrix(camera_info_path)
    if width and height and (rgb_bgr.shape[1] != width or rgb_bgr.shape[0] != height):
        # Keep real image dimensions; CameraInfo intrinsics still define projection for this stream.
        pass

    estimator = _build_estimator(settings.SAM3D_BODY_DETECTOR_NAME, settings.SAM3D_BODY_DEVICE)
    boxes = _detect_boxes(estimator, rgb_bgr)
    if boxes.size == 0:
        _write_status(json_path, task, 'NO_PERSON_DETECTED', result_timestamp=result_timestamp, backup_shigurei_dir=backup_dir, output_dir=str(output_root))
        return {'status': 'NO_PERSON_DETECTED'}

    outputs = _run_sam3d_body(estimator, rgb_bgr, boxes, camera_matrix)
    if not outputs:
        _write_status(json_path, task, 'NO_VALID_BODY_MESH', result_timestamp=result_timestamp, backup_shigurei_dir=backup_dir, raw_bbox_count=int(len(boxes)), output_dir=str(output_root))
        return {'status': 'NO_VALID_BODY_MESH'}
    faces = np.asarray(estimator.faces, dtype=np.int64).reshape(-1, 3)
    marker_rotation, marker_translation = _load_marker_camera_pose(marker_pose_path)
    object_center = _object_center_aruco(task)

    people: list[dict[str, Any]] = []
    for idx, output in enumerate(outputs):
        person_name = f'person_{idx}'
        try:
            aligned = _align_person_depth(output, faces, depth_m, camera_matrix)
            vertices_aruco = _camera_to_armarker_points(aligned['vertices_camera_m'], marker_rotation, marker_translation)
            keypoints_aruco = _camera_to_armarker_points(aligned['keypoints_camera_m'], marker_rotation, marker_translation)
            nearest = _nearest_wrist(person_name, keypoints_aruco, object_center)
            mesh_npz = output_root / f'08_sam3d_body_{person_name}_armarker_mesh.npz'
            camera_mesh_npz = output_root / f'08_sam3d_body_{person_name}_camera_mesh.npz'
            np.savez_compressed(mesh_npz, vertices=vertices_aruco.astype(np.float32), faces=faces.astype(np.int32), keypoints=keypoints_aruco.astype(np.float32))
            np.savez_compressed(
                camera_mesh_npz,
                vertices=aligned['vertices_camera_m'].astype(np.float32),
                faces=faces.astype(np.int32),
                keypoints=aligned['keypoints_camera_m'].astype(np.float32),
                bbox_xyxy=aligned['bbox_xyxy'].astype(np.float32),
                body_mask=aligned['body_mask'].astype(np.uint8),
            )
            people.append({
                'person_name': person_name,
                'bbox_xyxy': aligned['bbox_xyxy'].astype(float).tolist(),
                'mesh_npz_path': str(mesh_npz),
                'camera_mesh_npz_path': str(camera_mesh_npz),
                'vertex_count': int(vertices_aruco.shape[0]),
                'face_count': int(faces.shape[0]),
                'depth_offset_m': float(aligned['depth_offset_m']),
                'xyz_shift_m': aligned['xyz_shift_m'].astype(float).tolist(),
                'distance_scale': float(aligned['distance_scale']),
                'original_distance_m': aligned.get('original_distance_m'),
                'adjusted_distance_m': aligned.get('adjusted_distance_m'),
                'depth_overlap_pixels': int(aligned['depth_overlap_pixels']),
                'depth_sampled_pixels': int(aligned['depth_sampled_pixels']),
                'depth_overlap_ratio': float(aligned['depth_overlap_ratio']),
                'depth_alignment_method': aligned['depth_alignment_method'],
                'nearest_wrist': nearest,
            })
        except Exception as exc:
            people.append({'person_name': person_name, 'error_message': str(exc)})

    people_json_path = model_result_file(task_timestamp, 'body.people')
    _write_json(people_json_path, {'people': people, 'object_center_armarker': object_center.tolist() if object_center is not None else None})
    valid_people = [p for p in people if isinstance(p.get('nearest_wrist'), Mapping)]
    if not valid_people:
        _write_status(
            json_path,
            task,
            'NO_VALID_WRIST_JOINT',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            people=people,
            output_dir=str(output_root),
            reason='no_sam3d_body_wrist_distance_to_object',
        )
        return {'status': 'NO_VALID_WRIST_JOINT'}

    selected = min(valid_people, key=lambda p: float((p.get('nearest_wrist') or {}).get('distance_m', math.inf)))
    selected_name = str(selected['person_name'])
    selected_npz = np.load(str(selected['mesh_npz_path']))
    work_obj_path = output_root / f'08_sam3d_body_{task_name}_{selected_name}_armarker.obj'
    obj_path = model_result_file(task_timestamp, 'body.selected_obj')
    fbx_path = model_result_file(task_timestamp, 'body.selected_fbx')
    _write_obj(work_obj_path, selected_npz['vertices'], selected_npz['faces'])

    debug_files: dict[str, Any] = {}
    selected_body_mask = None
    selected_body_bbox = selected.get('bbox_xyxy')
    selected_camera_mesh_npz_path = None
    camera_mesh_path = selected.get('camera_mesh_npz_path')
    if camera_mesh_path:
        try:
            selected_camera_mesh_npz_path = str(camera_mesh_path)
            selected_camera_npz = np.load(str(camera_mesh_path))
            if 'body_mask' in selected_camera_npz.files:
                selected_body_mask = selected_camera_npz['body_mask'].astype(bool)
            if 'bbox_xyxy' in selected_camera_npz.files:
                selected_body_bbox = selected_camera_npz['bbox_xyxy'].astype(float).tolist()
            _overlay_path, overlay_stats = _save_body_mesh_debug_overlay(
                task_timestamp,
                rgb_bgr,
                selected_camera_npz['vertices'],
                selected_camera_npz['faces'],
                camera_matrix,
                selected_body_bbox,
                f'sam3d body {selected_name}',
            )
            debug_files.update(overlay_stats)
        except Exception as exc:
            debug_files['body_mesh_on_taken_rgb_error'] = str(exc)
    object_mask, object_mask_info = _load_object_mask_from_taken(taken, rgb_bgr.shape[:2])
    object_center_mask, object_center_projection_info = _project_object_center_mask(
        object_center,
        marker_rotation,
        marker_translation,
        camera_matrix,
        rgb_bgr.shape[:2],
        object_mask,
    )
    subject_crop_path, subject_crop_info = _write_subject_crop(
        task_timestamp,
        rgb_bgr,
        selected_body_mask,
        selected_body_bbox,
        object_mask,
        object_center_mask,
    )
    debug_files['object_mask_for_subject_crop'] = object_mask_info
    debug_files['object_center_for_subject_crop'] = object_center_projection_info
    debug_files['subject_crop'] = subject_crop_info

    if obj_path != work_obj_path:
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work_obj_path, obj_path)
    try:
        _export_fbx(work_obj_path, fbx_path)
    except Exception as exc:
        _write_status(
            json_path,
            task,
            'NO_VALID_BODY_MESH',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            selected_person_name=selected_name,
            selected_person_obj_path=str(obj_path),
            people=people,
            output_dir=str(output_root),
            error_message=f'FBX export failed: {exc}',
        )
        return {'status': 'NO_VALID_BODY_MESH', 'reason': 'fbx_export_failed'}

    payload = {
        'result_timestamp': result_timestamp,
        'backup_shigurei_dir': backup_dir,
        'selected_person_name': selected_name,
        'selected_person_fbx_path': str(fbx_path),
        'selected_person_fbx_folder': 'model_result',
        'selected_person_pose_aruco': {
            'position': [0.0, 0.0, 0.0],
            'rotation_quaternion_xyzw': [0.0, 0.0, 0.0, 1.0],
            'scale': [1.0, 1.0, 1.0],
        },
        'selected_person_obj_path': str(obj_path),
        'selected_person_camera_mesh_npz_path': selected_camera_mesh_npz_path,
        'selected_person_bbox_xyxy': selected_body_bbox,
        'subject_crop_path': subject_crop_path,
        'subject_crop_folder': 'model_result' if subject_crop_path else None,
        'subject_crop': subject_crop_info,
        'material_color': settings.MATERIAL_COLOR,
        'material_alpha': settings.MATERIAL_ALPHA,
        'coordinate_space': 'aruco',
        'camera_to_armarker_basis': 'spatial_transforms.shigure_camera_points_to_aruco',
        'camera_to_armarker_source': str(marker_pose_path),
        'people_json_path': str(people_json_path),
        'people': people,
        'debug_files': debug_files,
        'object_center_armarker': object_center.tolist() if object_center is not None else None,
    }
    payload['latest_body_mesh_registry_path'] = _write_latest_body_mesh_registry({
        'task_id': str(task.get('task_id') or ''),
        'task_timestamp': task_timestamp,
        'result_timestamp': result_timestamp,
        'selected_person_name': selected_name,
        'selected_person_fbx_path': str(fbx_path),
        'selected_person_obj_path': str(obj_path),
        'selected_person_camera_mesh_npz_path': selected_camera_mesh_npz_path,
        'selected_person_bbox_xyxy': selected_body_bbox,
        'subject_crop_path': subject_crop_path,
        'people_json_path': str(people_json_path),
        'backup_shigurei_dir': backup_dir,
        'coordinate_space': 'aruco',
    })
    _write_status(json_path, task, 'SUCCESS', **payload)
    return {'status': 'SUCCESS', 'selected_person_name': selected_name, 'selected_person_fbx_path': str(fbx_path), 'subject_crop_path': subject_crop_path}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('Usage: python run_sam3d_body_mesh_from_json.py <task_meta.json>', file=sys.stderr)
        return 2
    try:
        result = run_sam3d_body_mesh(argv[1])
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        print('[OK] sam3d_body_mesh')
        return 0
    except Exception as exc:
        try:
            json_path = resolve_task_json_path(argv[1])
            task = load_task_json(json_path)
            _write_status(json_path, task, 'NO_VALID_BODY_MESH', error_message=str(exc))
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
