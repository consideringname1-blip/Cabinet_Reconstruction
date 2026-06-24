from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from artifact_layout import ARUCO_TEMPLATE_PATH, SHIGURE_MARKER_HISTORY_PATH, SHIGURE_MARKER_HISTORY_ROOT
from coordinate_systems import (
    convert_opencv_camera_pose_to_unity_camera_pose,
    orthonormalize_rotation,
    rotation_matrix_to_quat_xyzw,
)
from stages.hololens_aruco_reference.aruco_common import load_aruco_template, resolve_marker_configs
from stages.shigure_history.cache import CachedRgbdSample, load_json, sample_key

try:
    from task_db import get_enabled_aruco_markers
except Exception:  # pragma: no cover - recorder should still work before DB init.
    get_enabled_aruco_markers = None  # type: ignore[assignment]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write('\n')


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def parse_float_array(value: Any, count: int, label: str) -> np.ndarray:
    if isinstance(value, str):
        normalized = re.sub(r'[\[\],]+', ' ', value)
        array = np.fromstring(normalized, sep=' ', dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != count:
        raise ValueError(f'{label} must contain {count} numeric values')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{label} contains non-finite values')
    return array


def load_camera_info_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int]:
    message = payload.get('message') if isinstance(payload.get('message'), dict) else payload
    camera_matrix = parse_float_array(
        message.get('k') or message.get('K') or message.get('camera_matrix'),
        9,
        'camera matrix',
    ).reshape(3, 3)
    distortion_value = message.get('d') or message.get('D') or message.get('distortion')
    if distortion_value is None:
        distortion = np.zeros((5, 1), dtype=np.float64)
    else:
        distortion = np.asarray(distortion_value, dtype=np.float64).reshape(-1, 1)
        if distortion.size == 0:
            distortion = np.zeros((5, 1), dtype=np.float64)
    width = int(message.get('width') or 0)
    height = int(message.get('height') or 0)
    return camera_matrix.astype(np.float64), distortion.astype(np.float64), width, height


def resolve_marker_configs_for_shigure() -> list[dict[str, Any]]:
    template = load_aruco_template()
    db_markers: list[dict[str, Any]] = []
    if get_enabled_aruco_markers is not None:
        try:
            db_markers = list(get_enabled_aruco_markers() or [])
        except Exception:
            db_markers = []
    return resolve_marker_configs(template, db_markers)


def resolve_dictionary(dictionary_name: str):
    dictionary_name = str(dictionary_name or '').strip()
    if not dictionary_name:
        raise ValueError('ArUco dictionary is empty')
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f'Unsupported ArUco dictionary: {dictionary_name}')
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters'):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    tuned = {
        'adaptiveThreshWinSizeMin': 3,
        'adaptiveThreshWinSizeMax': 53,
        'adaptiveThreshWinSizeStep': 4,
        'adaptiveThreshConstant': 7,
        'minMarkerPerimeterRate': 0.015,
        'maxMarkerPerimeterRate': 4.0,
        'polygonalApproxAccuracyRate': 0.05,
        'minCornerDistanceRate': 0.03,
        'minDistanceToBorder': 1,
    }
    for name, value in tuned.items():
        if hasattr(params, name):
            setattr(params, name, value)
    if hasattr(cv2.aruco, 'CORNER_REFINE_SUBPIX') and hasattr(params, 'cornerRefinementMethod'):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, 'cornerRefinementWinSize'):
        params.cornerRefinementWinSize = 5
    if hasattr(params, 'cornerRefinementMaxIterations'):
        params.cornerRefinementMaxIterations = 30
    if hasattr(params, 'cornerRefinementMinAccuracy'):
        params.cornerRefinementMinAccuracy = 0.01
    return params


def detect_markers(image_bgr: np.ndarray, dictionary):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    params = detector_parameters()
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected


def marker_object_points(marker_size_mm: float) -> np.ndarray:
    half = float(marker_size_mm) / 1000.0 * 0.5
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def pose_payload(rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    rotation = orthonormalize_rotation(rotation)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return {
        'position': translation.astype(float).tolist(),
        'rotation_quaternion_xyzw': rotation_matrix_to_quat_xyzw(rotation).astype(float).tolist(),
        'rotation_matrix': rotation.astype(float).tolist(),
    }


def estimate_from_sample(sample: CachedRgbdSample) -> dict[str, Any] | None:
    if sample.camera_info is not None:
        camera_matrix, distortion, width, height = load_camera_info_payload(sample.camera_info)
    elif sample.camera_info_path is not None and sample.camera_info_path.is_file():
        camera_matrix, distortion, width, height = load_camera_info(sample.camera_info_path)
    else:
        return None
    image = sample.rgb_bgr if sample.rgb_bgr is not None else None
    if image is None and sample.rgb_path is not None:
        image = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    markers = resolve_marker_configs_for_shigure()
    if not markers:
        return None

    detections: list[dict[str, Any]] = []
    detected_ids: set[int] = set()
    rejected_count = 0
    dictionaries: dict[str, Any] = {}
    for marker in markers:
        dictionary_name = str(marker['dictionary'])
        dictionary = dictionaries.get(dictionary_name)
        if dictionary is None:
            dictionary = resolve_dictionary(dictionary_name)
            dictionaries[dictionary_name] = dictionary
        corners_list, ids, rejected = detect_markers(image, dictionary)
        rejected_count += len(rejected) if rejected is not None else 0
        if ids is None or len(ids) == 0:
            continue
        ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
        for corners, marker_id in zip(corners_list, ids_flat):
            detected_ids.add(int(marker_id))
            if int(marker_id) != int(marker['marker_id']):
                continue
            corners_full = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            object_points = marker_object_points(float(marker['marker_size_mm']))
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                corners_full,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE') else cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            rotation_cv, _ = cv2.Rodrigues(rvec)
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
            projected = projected.reshape(-1, 2)
            reprojection_error = float(np.mean(np.linalg.norm(projected - corners_full, axis=1)))
            area = float(abs(cv2.contourArea(corners_full.astype(np.float32))))
            weight = max(area, 1.0) / max(reprojection_error + 1.0, 1.0)
            detections.append(
                {
                    'stamp': sample.stamp.to_dict(),
                    'sample_key': sample_key(sample.stamp),
                    'rgb_image': str(sample.rgb_path) if sample.rgb_path else f'chunk:{sample.chunk_id}:{sample.frame_index}',
                    'camera_info': str(sample.camera_info_path) if sample.camera_info_path else f'chunk:{sample.chunk_id}:camera_info',
                    'marker_id': int(marker['marker_id']),
                    'dictionary': dictionary_name,
                    'marker_size_mm': float(marker['marker_size_mm']),
                    'corners_px': corners_full.astype(float).tolist(),
                    'reprojection_error_px': reprojection_error,
                    'corner_area_px': area,
                    'weight': float(weight),
                    'rotation_matrix': rotation_cv.astype(float).tolist(),
                    'rvec': np.asarray(rvec, dtype=np.float64).reshape(3).astype(float).tolist(),
                    'tvec_m': np.asarray(tvec, dtype=np.float64).reshape(3).astype(float).tolist(),
                }
            )
    if not detections:
        return None
    best = max(detections, key=lambda item: item['weight'])
    rotation_cv = np.asarray(best['rotation_matrix'], dtype=np.float64).reshape(3, 3)
    translation_cv = np.asarray(best['tvec_m'], dtype=np.float64).reshape(3)
    unity_rotation, unity_translation = convert_opencv_camera_pose_to_unity_camera_pose(rotation_cv, translation_cv)
    payload = {
        'created_at': utc_now(),
        'source': 'shigure_history_marker_detection',
        'sample_key': best['sample_key'],
        'stamp': best['stamp'],
        'rgb_image': best['rgb_image'],
        'camera_info': best['camera_info'],
        'aruco_config': str(ARUCO_TEMPLATE_PATH),
        'marker_id': best['marker_id'],
        'dictionary': best['dictionary'],
        'marker_size_mm': best['marker_size_mm'],
        'corners_px': best['corners_px'],
        'reprojection_error_px': best['reprojection_error_px'],
        'corner_area_px': best['corner_area_px'],
        'opencv_camera_pose': {
            **pose_payload(rotation_cv, translation_cv),
            'rvec': best['rvec'],
            'tvec_m': best['tvec_m'],
            'coordinate_system': '+X right, +Y down, +Z forward; units are meters',
        },
        'unity_camera_pose': {
            **pose_payload(unity_rotation, unity_translation),
            'coordinate_system': '+X right, +Y up, +Z forward; HoloLens/Unity-compatible marker payload in this project',
        },
        'camera_matrix': camera_matrix.astype(float).tolist(),
        'distortion': distortion.reshape(-1).astype(float).tolist(),
        'detected_ids': sorted(detected_ids),
        'rejected_count': int(rejected_count),
    }
    return payload


def fuse_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    if not detections:
        raise ValueError('no Shigurei marker detections to fuse')
    weights = np.asarray([max(float(d.get('corner_area_px') or 1.0) / max(float(d.get('reprojection_error_px') or 0.0) + 1.0, 1.0), 1e-6) for d in detections], dtype=np.float64)
    weights = weights / weights.sum()
    translations = np.asarray([d['opencv_camera_pose']['tvec_m'] for d in detections], dtype=np.float64)
    fused_translation = np.sum(translations * weights[:, None], axis=0)
    rotation_acc = np.zeros((3, 3), dtype=np.float64)
    for detection, weight in zip(detections, weights):
        rotation_acc += np.asarray(detection['opencv_camera_pose']['rotation_matrix'], dtype=np.float64).reshape(3, 3) * float(weight)
    fused_rotation = orthonormalize_rotation(rotation_acc)
    latest = detections[-1]
    unity_rotation, unity_translation = convert_opencv_camera_pose_to_unity_camera_pose(fused_rotation, fused_translation)
    fused = {
        **latest,
        'created_at': utc_now(),
        'source': 'shigure_history_marker_warmup_fused',
        'observation_count': len(detections),
        'source_observations': [
            {
                'sample_key': d.get('sample_key'),
                'stamp': d.get('stamp'),
                'reprojection_error_px': d.get('reprojection_error_px'),
                'corner_area_px': d.get('corner_area_px'),
                'rgb_image': d.get('rgb_image'),
                'camera_info': d.get('camera_info'),
            }
            for d in detections
        ],
        'opencv_camera_pose': {
            **pose_payload(fused_rotation, fused_translation),
            'rvec': latest['opencv_camera_pose'].get('rvec'),
            'tvec_m': fused_translation.astype(float).tolist(),
            'coordinate_system': '+X right, +Y down, +Z forward; units are meters',
        },
        'unity_camera_pose': {
            **pose_payload(unity_rotation, unity_translation),
            'coordinate_system': '+X right, +Y up, +Z forward; HoloLens/Unity-compatible marker payload in this project',
        },
    }
    return fused


def write_marker_history(detection: dict[str, Any]) -> Path:
    root = Path(SHIGURE_MARKER_HISTORY_ROOT)
    history_dir = root / 'history'
    history_dir.mkdir(parents=True, exist_ok=True)
    sample_key_text = str(detection.get('sample_key') or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ'))
    history_path = history_dir / f'{sample_key_text}_marker_6d_pose.json'
    payload = {**detection, 'history_path': str(history_path), 'latest_path': str(SHIGURE_MARKER_HISTORY_PATH)}
    write_json(history_path, payload)
    write_json(Path(SHIGURE_MARKER_HISTORY_PATH), payload)
    return Path(SHIGURE_MARKER_HISTORY_PATH)


def latest_marker_pose_path() -> Path | None:
    path = Path(SHIGURE_MARKER_HISTORY_PATH)
    return path if path.is_file() else None


@dataclass
class MarkerHistoryWarmup:
    target_detections: int
    max_attempts: int
    max_reprojection_error_px: float
    min_corner_area_px: float
    attempts: int = 0
    detections: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    last_error: str | None = None

    def process_sample(self, sample: CachedRgbdSample) -> dict[str, Any]:
        if self.completed:
            return {'completed': True, 'updated': False, 'detection_count': len(self.detections), 'attempts': self.attempts}
        self.attempts += 1
        try:
            detection = estimate_from_sample(sample)
        except Exception as exc:
            self.last_error = str(exc)
            detection = None
        accepted = False
        if detection is not None:
            reprojection = float(detection.get('reprojection_error_px') or math.inf)
            area = float(detection.get('corner_area_px') or 0.0)
            accepted = reprojection <= self.max_reprojection_error_px and area >= self.min_corner_area_px
            if accepted:
                self.detections.append(detection)
        updated = False
        if len(self.detections) >= max(1, int(self.target_detections)):
            write_marker_history(fuse_detections(self.detections))
            self.completed = True
            updated = True
        elif self.attempts >= max(1, int(self.max_attempts)):
            if self.detections:
                write_marker_history(fuse_detections(self.detections))
                updated = True
            self.completed = True
        return {
            'completed': self.completed,
            'updated': updated,
            'attempts': self.attempts,
            'detection_count': len(self.detections),
            'accepted': accepted,
            'last_error': self.last_error,
            'latest_path': str(SHIGURE_MARKER_HISTORY_PATH) if Path(SHIGURE_MARKER_HISTORY_PATH).is_file() else None,
        }
