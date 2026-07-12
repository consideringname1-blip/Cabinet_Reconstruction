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

from artifact_layout import model_debug_dir, model_result_file, model_worker_dir
from path_config import BLENDER_BIN, SAM3D_BODY_FBX_EXPORT_SCRIPT, SAM3D_BODY_ROOT
from spatial_transforms import shigure_camera_points_to_aruco
from task_json import load_task_json, resolve_task_json_path, save_task_json

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
    matrix = _parse_float_array(payload.get('k'), 9, 'camera_info.k').reshape(3, 3)
    width = int(payload.get('width') or 0)
    height = int(payload.get('height') or 0)
    if width <= 0 or height <= 0:
        raise ValueError('camera_info width and height must be positive')
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError('camera_info.k is invalid')
    return matrix.astype(np.float64), width, height


def _depth_image_to_m(depth: np.ndarray) -> np.ndarray:
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Shigure depth must be a uint16 millimetre image')
    return depth.astype(np.float32) / 1000.0


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


def _build_estimator(device: str):
    _ensure_sam3d_imports()
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    checkpoint = Path(SAM3D_BODY_ROOT) / 'checkpoints' / 'sam-3d-body-dinov3' / 'model.ckpt'
    mhr = Path(SAM3D_BODY_ROOT) / 'checkpoints' / 'sam-3d-body-dinov3' / 'assets' / 'mhr_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not mhr.is_file():
        raise FileNotFoundError(mhr)
    _prefer_cached_dinov3_torch_hub(torch)
    torch_device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, model_cfg = load_sam_3d_body(str(checkpoint), device=torch_device, mhr_path=str(mhr))
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )


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
    pose = payload.get('opencv_camera_pose')
    if not isinstance(pose, Mapping):
        raise ValueError('marker pose JSON does not contain opencv_camera_pose')
    rotation = _parse_float_array(pose.get('rotation_matrix'), 9, 'opencv_camera_pose.rotation_matrix').reshape(3, 3)
    translation = _parse_float_array(pose.get('tvec_m'), 3, 'opencv_camera_pose.tvec_m')
    return rotation.astype(np.float64), translation.astype(np.float64)


def _camera_to_aruco_points(points_camera_m: np.ndarray, marker_rotation_camera_marker_cv: np.ndarray, marker_translation_camera_marker_cv: np.ndarray) -> np.ndarray:
    return shigure_camera_points_to_aruco(
        points_camera_m,
        marker_rotation_camera_marker_cv,
        marker_translation_camera_marker_cv,
    )


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
    return rgb, depth, camera, marker if marker.is_file() else None


def _contact_people_bbox(taken: Mapping[str, Any]) -> list[float] | None:
    direct = taken.get("people_bounding_box")
    if not isinstance(direct, Mapping):
        return None
    raw = direct.get("xyxy")
    try:
        values = [float(value) for value in np.asarray(raw, dtype=np.float64).reshape(4)]
    except Exception:
        return None
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        return None
    return values if values[2] > values[0] and values[3] > values[1] else None


def _load_object_mask_from_taken(taken: Mapping[str, Any], image_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    backup_dir = Path(str(taken.get("backup_shigurei_dir") or ""))
    path = backup_dir / "object_mask.png"
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if raw is None:
        return None, {"source": "missing", "path": str(path)}
    mask = raw > 0
    if mask.shape != image_shape:
        mask = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if not np.any(mask):
        return None, {"source": "empty", "path": str(path)}
    return mask, {"source": "remote_shigure_object_mask", "path": str(path), "pixels": int(np.count_nonzero(mask))}



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


def _write_subject_crop(
    task_timestamp: str,
    rgb_bgr: np.ndarray,
    body_mask: np.ndarray | None,
    body_bbox_xyxy: Any,
    object_mask: np.ndarray | None,
) -> tuple[str | None, dict[str, Any]]:
    h, w = rgb_bgr.shape[:2]
    image_shape = (h, w)
    union = np.zeros((h, w), dtype=bool)
    sources = []

    body = _normalise_mask(body_mask, image_shape)
    bbox_mask, bbox_info = _mask_from_bbox(body_bbox_xyxy, image_shape, int(settings.SUBJECT_CROP_BODY_BBOX_PAD_PX))
    if body is not None:
        union |= body
        sources.append({'source': 'selected_body_mesh_mask', 'pixels': int(np.count_nonzero(body)), 'body_bbox': bbox_info})
    elif bbox_mask is not None:
        union |= bbox_mask
        sources.append({'source': 'shigure_people_bbox', 'pixels': int(np.count_nonzero(bbox_mask)), 'body_bbox': bbox_info})
    else:
        sources.append(bbox_info)

    obj = _normalise_mask(object_mask, image_shape)
    if obj is None:
        return None, {'reason': 'shigure_object_mask_missing', 'sources': sources}
    union |= obj
    sources.append({'source': 'shigure_object_mask', 'pixels': int(np.count_nonzero(obj))})

    bbox = _mask_bbox(union, int(settings.SUBJECT_CROP_PAD_PX), w, h)
    if bbox is None:
        return None, {'reason': 'empty_subject_crop_mask', 'sources': sources}
    x0, y0, x1, y1 = bbox
    crop = rgb_bgr[y0:y1, x0:x1].copy()
    crop_path = model_result_file(task_timestamp, 'body.subject_crop')
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f'failed to write subject crop: {crop_path}')
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


def run_sam3d_body_mesh(json_path_arg: str | Path) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    taken = task.get('ShigureContactEvidence') if isinstance(task.get('ShigureContactEvidence'), Mapping) else {}
    task_name = str(task.get('task_name') or '').strip()
    if not task_name:
        raise ValueError('task_name is required for SAM3D body artifacts')
    task_timestamp = str(task.get('task_timestamp') or '').strip()
    if not task_timestamp:
        raise ValueError('task_timestamp is required for SAM3D body artifacts')
    output_root = model_worker_dir(task_timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    if taken.get('status') != 'TAKEN':
        payload = {
            'result_timestamp': taken.get('result_timestamp'),
            'backup_shigurei_dir': taken.get('backup_shigurei_dir'),
            'reason': f"shigure_contact_status={taken.get('status')}",
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
            'INPUT_MISSING',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            reason='camera_to_aruco_marker_pose_missing',
            output_dir=str(output_root),
        )
        return {'status': 'INPUT_MISSING', 'reason': 'camera_to_aruco_marker_pose_missing'}

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise RuntimeError(f'failed to read RGB: {rgb_path}')
    if depth_raw is None:
        raise RuntimeError(f'failed to read depth: {depth_path}')
    depth_m = _depth_image_to_m(depth_raw)
    if depth_m.shape[:2] != rgb_bgr.shape[:2]:
        raise ValueError('Shigure RGB and depth dimensions must match')
    camera_matrix, width, height = _load_camera_matrix(camera_info_path)
    if rgb_bgr.shape[1] != width or rgb_bgr.shape[0] != height:
        raise ValueError('Shigure RGB dimensions must match camera_info width and height')

    contact_bbox = _contact_people_bbox(taken)
    if contact_bbox is None:
        reason = 'remote_shigure_contact_people_bbox_missing_or_invalid'
        _write_status(
            json_path,
            task,
            'NO_PERSON_DETECTED',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            output_dir=str(output_root),
            reason=reason,
        )
        return {'status': 'NO_PERSON_DETECTED', 'reason': reason}

    estimator = _build_estimator(settings.SAM3D_BODY_DEVICE)
    boxes = np.asarray([contact_bbox], dtype=np.float32)

    outputs = _run_sam3d_body(estimator, rgb_bgr, boxes, camera_matrix)
    if not outputs:
        _write_status(json_path, task, 'NO_VALID_BODY_MESH', result_timestamp=result_timestamp, backup_shigurei_dir=backup_dir, raw_bbox_count=int(len(boxes)), output_dir=str(output_root))
        return {'status': 'NO_VALID_BODY_MESH'}
    faces = np.asarray(estimator.faces, dtype=np.int64).reshape(-1, 3)
    marker_rotation, marker_translation = _load_marker_camera_pose(marker_pose_path)

    people: list[dict[str, Any]] = []
    for idx, output in enumerate(outputs):
        person_name = f'person_{idx}'
        try:
            aligned = _align_person_depth(output, faces, depth_m, camera_matrix)
            vertices_aruco = _camera_to_aruco_points(aligned['vertices_camera_m'], marker_rotation, marker_translation)
            keypoints_aruco = _camera_to_aruco_points(aligned['keypoints_camera_m'], marker_rotation, marker_translation)
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
            })
        except Exception as exc:
            people.append({'person_name': person_name, 'error_message': str(exc)})

    people_json_path = model_result_file(task_timestamp, 'body.people')
    _write_json(people_json_path, {'people': people})
    valid_people = [p for p in people if p.get('mesh_npz_path') and not p.get('error_message')]
    if not valid_people:
        _write_status(
            json_path,
            task,
            'NO_VALID_BODY_MESH',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            people=people,
            output_dir=str(output_root),
            reason='no_valid_sam3d_body_for_shigure_contact_bbox',
        )
        return {'status': 'NO_VALID_BODY_MESH'}

    selected = valid_people[0]
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
    if object_mask is None:
        _write_status(
            json_path,
            task,
            'INPUT_MISSING',
            result_timestamp=result_timestamp,
            backup_shigurei_dir=backup_dir,
            reason='shigure_object_mask_missing_or_invalid',
            object_mask=object_mask_info,
        )
        return {'status': 'INPUT_MISSING', 'reason': 'shigure_object_mask_missing_or_invalid'}
    subject_crop_path, subject_crop_info = _write_subject_crop(
        task_timestamp,
        rgb_bgr,
        selected_body_mask,
        selected_body_bbox,
        object_mask,
    )
    if subject_crop_path is None:
        raise RuntimeError(f"subject crop failed: {subject_crop_info.get('reason')}")
    debug_files['object_mask_for_subject_crop'] = object_mask_info
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
        'person_selection_source': 'shigure_contact_people_bbox',
        'subject_crop_path': subject_crop_path,
        'subject_crop_folder': 'model_result' if subject_crop_path else None,
        'subject_crop': subject_crop_info,
        'material_color': settings.MATERIAL_COLOR,
        'material_alpha': settings.MATERIAL_ALPHA,
        'coordinate_space': 'aruco',
        'camera_to_aruco_basis': 'spatial_transforms.shigure_camera_points_to_aruco',
        'camera_to_aruco_source': str(marker_pose_path),
        'people_json_path': str(people_json_path),
        'people': people,
        'debug_files': debug_files,
    }
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
