from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from config import (
    ARUCO_ANCHOR_MARKER_ID,
    ARUCO_ROI_PADDING_MIN_PX,
    ARUCO_ROI_PADDING_RATIO,
    ARUCO_TEMPLATE_PATH,
)
from hololens3d_reconstruction.pose_math import quat_xyzw_to_rotation_matrix
from task_db import (
    create_aruco_reference,
    get_aruco_marker_relation,
    get_enabled_aruco_markers,
    upsert_aruco_marker_relation,
)
from task_json import (
    load_task_json,
    normalize_path_for_storage,
    resolve_task_json_path,
    save_task_json,
)

try:
    from aruco_common import (
        ARUCO_LOCAL_COORDINATE_BASIS,
        OPENCV_CAMERA_TO_UNITY_TRANSFORM,
        UNITY_WORLD_COORDINATE_BASIS,
        WINDOWS_POSE_TO_UNITY_TRANSFORM,
        compose_world_pose,
        convert_cv_pose_to_unity_pose,
        ensure_raw_output_dir,
        invert_pose,
        load_aruco_template,
        load_json_payload,
        orthonormalize_rotation,
        pose_to_payload,
        resolve_marker_configs,
        resolve_pv_camera_matrix,
        resolve_pv_camera_world_pose,
        resolve_pv_frames,
        resolve_pv_image_path,
        resolve_selection_roi,
        resolve_task_name,
    )
    from run_aruco_sync_from_json import sync_completed_tasks_for_startup
except ModuleNotFoundError:
    from .aruco_common import (
        ARUCO_LOCAL_COORDINATE_BASIS,
        OPENCV_CAMERA_TO_UNITY_TRANSFORM,
        UNITY_WORLD_COORDINATE_BASIS,
        WINDOWS_POSE_TO_UNITY_TRANSFORM,
        compose_world_pose,
        convert_cv_pose_to_unity_pose,
        ensure_raw_output_dir,
        invert_pose,
        load_aruco_template,
        load_json_payload,
        orthonormalize_rotation,
        pose_to_payload,
        resolve_marker_configs,
        resolve_pv_camera_matrix,
        resolve_pv_camera_world_pose,
        resolve_pv_frames,
        resolve_pv_image_path,
        resolve_selection_roi,
        resolve_task_name,
    )
    from .run_aruco_sync_from_json import sync_completed_tasks_for_startup


def _load_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _resolve_dictionary(aruco_module, dictionary_name: str):
    dictionary_name = str(dictionary_name or "").strip()
    if not dictionary_name:
        raise ValueError("ArUco dictionary is empty")
    if not hasattr(aruco_module, dictionary_name):
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    return aruco_module.getPredefinedDictionary(getattr(aruco_module, dictionary_name))


def _build_detector_parameters(cv2):
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    tuned_values = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 53,
        "adaptiveThreshWinSizeStep": 4,
        "adaptiveThreshConstant": 7,
        "minMarkerPerimeterRate": 0.015,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.05,
        "minCornerDistanceRate": 0.03,
        "minDistanceToBorder": 1,
    }
    for name, value in tuned_values.items():
        if hasattr(parameters, name):
            setattr(parameters, name, value)

    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX") and hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(parameters, "cornerRefinementWinSize"):
        parameters.cornerRefinementWinSize = 5
    if hasattr(parameters, "cornerRefinementMaxIterations"):
        parameters.cornerRefinementMaxIterations = 30
    if hasattr(parameters, "cornerRefinementMinAccuracy"):
        parameters.cornerRefinementMinAccuracy = 0.01
    return parameters


def _expand_roi(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    pad_x = max(int(round(width * float(ARUCO_ROI_PADDING_RATIO))), int(ARUCO_ROI_PADDING_MIN_PX))
    pad_y = max(int(round(height * float(ARUCO_ROI_PADDING_RATIO))), int(ARUCO_ROI_PADDING_MIN_PX))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image_width, x1 + pad_x),
        min(image_height, y1 + pad_y),
    )


def _detect_markers(cv2, roi_image: np.ndarray, dictionary):
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if roi_image.ndim == 3 else roi_image
    parameters = _build_detector_parameters(cv2)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(gray)[:2]

    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids


def _marker_object_points(marker_size_mm: float) -> np.ndarray:
    marker_size_m = float(marker_size_mm) / 1000.0
    half = marker_size_m * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _estimate_marker_detection(
    cv2,
    *,
    frame: dict[str, Any],
    frame_index: int,
    marker: dict[str, Any],
    corners_full: np.ndarray,
    camera_matrix: np.ndarray,
) -> dict[str, Any] | None:
    object_points = _marker_object_points(float(marker["marker_size_mm"]))
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners_full.astype(np.float64),
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_cv, _ = cv2.Rodrigues(rvec)
    local_rotation, local_translation = convert_cv_pose_to_unity_pose(rotation_cv, tvec.reshape(3))
    camera_world_translation, camera_world_rotation = resolve_pv_camera_world_pose(frame)
    world_translation, world_rotation = compose_world_pose(
        camera_world_translation,
        camera_world_rotation,
        local_translation,
        local_rotation,
    )

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    projected = projected.reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(projected - corners_full, axis=1)))
    area = float(abs(cv2.contourArea(corners_full.astype(np.float32))))
    weight = max(area, 1.0) / max(reprojection_error + 1.0, 1.0)

    return {
        "frame_index": int(frame_index),
        "marker_id": int(marker["marker_id"]),
        "dictionary": str(marker["dictionary"]),
        "marker_size_mm": float(marker["marker_size_mm"]),
        "corners": corners_full.astype(float).tolist(),
        "local_translation": local_translation,
        "local_rotation": local_rotation,
        "world_translation": world_translation,
        "world_rotation": world_rotation,
        "local_pose": pose_to_payload(
            local_rotation,
            local_translation,
            ARUCO_LOCAL_COORDINATE_BASIS,
            scale=[1.0, 1.0, 1.0],
        ),
        "world_pose": pose_to_payload(
            world_rotation,
            world_translation,
            UNITY_WORLD_COORDINATE_BASIS,
            scale=[1.0, 1.0, 1.0],
        ),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "reprojection_error_px": reprojection_error,
        "corner_area_px": area,
        "weight": float(weight),
    }


def _pose_payload_to_rt(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(payload.get("position"), dtype=np.float64)
    quaternion = np.asarray(payload.get("rotation_quaternion_xyzw") or payload.get("rotation"), dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("pose payload must include position and quaternion")
    return quat_xyzw_to_rotation_matrix(quaternion), position


def _relation_from_anchor_to_marker(anchor_detection: dict[str, Any], marker_detection: dict[str, Any]) -> dict[str, Any]:
    anchor_inverse_rotation, anchor_inverse_translation = invert_pose(
        anchor_detection["world_rotation"],
        anchor_detection["world_translation"],
    )
    relation_translation = (anchor_inverse_rotation @ marker_detection["world_translation"]) + anchor_inverse_translation
    relation_rotation = anchor_inverse_rotation @ marker_detection["world_rotation"]
    return pose_to_payload(
        relation_rotation,
        relation_translation,
        ARUCO_LOCAL_COORDINATE_BASIS,
        scale=[1.0, 1.0, 1.0],
    )


def _anchor_candidate_from_marker_relation(
    marker_detection: dict[str, Any],
    relation_payload: dict[str, Any],
) -> dict[str, Any]:
    relation_rotation, relation_translation = _pose_payload_to_rt(relation_payload)
    anchor_rotation = marker_detection["world_rotation"] @ relation_rotation.T
    anchor_translation = marker_detection["world_translation"] - (anchor_rotation @ relation_translation)
    return {
        "translation": anchor_translation.astype(np.float64),
        "rotation": orthonormalize_rotation(anchor_rotation),
        "weight": float(marker_detection.get("weight") or 1.0),
        "source_marker_id": int(marker_detection["marker_id"]),
        "frame_index": int(marker_detection["frame_index"]),
        "source": "marker_relation",
    }


def _fuse_pose_candidates(candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("no pose candidates to fuse")
    weights = np.asarray([max(float(c.get("weight") or 0.0), 1e-6) for c in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    translations = np.asarray([c["translation"] for c in candidates], dtype=np.float64)
    fused_translation = np.sum(translations * weights[:, None], axis=0)
    rotation_accumulator = np.zeros((3, 3), dtype=np.float64)
    for candidate, weight in zip(candidates, weights):
        rotation_accumulator += np.asarray(candidate["rotation"], dtype=np.float64) * float(weight)
    fused_rotation = orthonormalize_rotation(rotation_accumulator)
    return fused_translation.astype(np.float64), fused_rotation.astype(np.float64)


def _build_detection_record(detection: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": detection["frame_index"],
        "marker_id": detection["marker_id"],
        "dictionary": detection["dictionary"],
        "marker_size_mm": detection["marker_size_mm"],
        "corners": detection["corners"],
        "marker_pose_local": detection["local_pose"],
        "marker_pose_world": detection["world_pose"],
        "reprojection_error_px": detection["reprojection_error_px"],
        "corner_area_px": detection["corner_area_px"],
        "weight": detection["weight"],
    }


def _write_debug(task: dict, aruco_stage: dict) -> None:
    debug_section = dict(task.get("debug") or {})
    pose_transform_stages = dict(debug_section.get("pose_transform_stages") or {})
    pose_transform_stages["aruco_stage"] = aruco_stage
    debug_section["pose_transform_stages"] = pose_transform_stages
    task["debug"] = debug_section


def _write_no_detection(
    *,
    json_path,
    task: dict,
    record_path,
    record: dict,
    aruco_stage: dict,
    reason: str,
) -> int:
    aruco_stage["config_reason"] = reason
    _write_debug(task, aruco_stage)
    save_task_json(json_path, task)
    record["aruco_stage"] = aruco_stage
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] aruco_detect : {reason}")
    print("[OK] aruco_detect")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: python code/stages/hololens_aruco_reference/run_aruco_detect_from_json.py <task_meta.json or filename>",
            file=sys.stderr,
        )
        raise SystemExit(2)

    json_path = resolve_task_json_path(argv[1])
    task = load_task_json(json_path)
    task_name = resolve_task_name(task, json_path.stem)
    task_id = str(task.get("task_id") or "")
    startup_session_id = str((task.get("device") or {}).get("startup_session_id") or "").strip()

    raw_dir = ensure_raw_output_dir(task_name)
    record_path = raw_dir / "record.json"
    legacy_roi_path = raw_dir / "roi.png"
    legacy_search_roi_path = raw_dir / "search_roi.png"
    legacy_annotated_path = raw_dir / "annotated.png"

    template = load_aruco_template()
    db_markers = get_enabled_aruco_markers()
    marker_configs = resolve_marker_configs(template, db_markers)
    marker_by_dict: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for marker in marker_configs:
        marker_by_dict[str(marker["dictionary"])][int(marker["marker_id"])] = marker

    aruco_stage = {
        "template_path": normalize_path_for_storage(ARUCO_TEMPLATE_PATH),
        "template_enabled": bool(template.get("enabled")),
        "configured": bool(marker_configs),
        "config_reason": "" if marker_configs else "no_enabled_markers",
        "anchor_marker_id": int(ARUCO_ANCHOR_MARKER_ID),
        "registered_marker_ids": [int(marker["marker_id"]) for marker in marker_configs],
        "detected": False,
        "detected_ids": [],
        "full_image_detected_ids": [],
        "matched_marker_id": None,
        "coordinate_basis_local": ARUCO_LOCAL_COORDINATE_BASIS,
        "coordinate_basis_world": UNITY_WORLD_COORDINATE_BASIS,
        "pv_pose_basis_transform": WINDOWS_POSE_TO_UNITY_TRANSFORM,
        "marker_camera_basis_transform": OPENCV_CAMERA_TO_UNITY_TRANSFORM,
        "marker_axes_definition": "origin=center, +x=marker right, +y=marker up, +z=marker front normal",
        "short_circuit": False,
        "marker_visible_outside_selection": False,
        "frame_count": 0,
        "detections": [],
        "raw_record_path": normalize_path_for_storage(record_path),
        "roi_image_path": normalize_path_for_storage(legacy_roi_path),
        "search_roi_image_path": normalize_path_for_storage(legacy_search_roi_path),
        "annotated_image_path": normalize_path_for_storage(legacy_annotated_path),
    }
    record = {
        "task_id": task_id,
        "task_name": task_name,
        "startup_session_id": startup_session_id,
        "template": template,
        "registered_markers": marker_configs,
    }

    cv2 = _load_cv2()
    if cv2 is None:
        aruco_stage["runtime_error"] = "opencv_not_available"
        return _write_no_detection(
            json_path=json_path,
            task=task,
            record_path=record_path,
            record=record,
            aruco_stage=aruco_stage,
            reason="opencv_not_available",
        )
    if not marker_configs:
        return _write_no_detection(
            json_path=json_path,
            task=task,
            record_path=record_path,
            record=record,
            aruco_stage=aruco_stage,
            reason="template disabled or no enabled marker registry entries",
        )

    frames = resolve_pv_frames(task)
    aruco_stage["frame_count"] = len(frames)
    if not frames:
        return _write_no_detection(
            json_path=json_path,
            task=task,
            record_path=record_path,
            record=record,
            aruco_stage=aruco_stage,
            reason="pv_frames_missing",
        )

    all_detections: list[dict[str, Any]] = []
    detections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    dictionaries = {}
    try:
        for dictionary_name in marker_by_dict:
            dictionaries[dictionary_name] = _resolve_dictionary(cv2.aruco, dictionary_name)

        for frame_index, frame in enumerate(frames):
            image_path = resolve_pv_image_path(frame)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                aruco_stage.setdefault("frame_errors", []).append(
                    {"frame_index": frame_index, "error": f"Failed to read PV image: {image_path}"}
                )
                continue

            image_height, image_width = image_bgr.shape[:2]
            x0, y0, x1, y1 = resolve_selection_roi(task, image_width, image_height)
            sx0, sy0, sx1, sy1 = _expand_roi(x0, y0, x1, y1, image_width, image_height)
            search_roi_image = image_bgr[sy0:sy1, sx0:sx1].copy()
            if frame_index == 0:
                cv2.imwrite(str(legacy_roi_path), image_bgr[y0:y1, x0:x1].copy())
                cv2.imwrite(str(legacy_search_roi_path), search_roi_image)
                aruco_stage["roi_pixel_bounds"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                aruco_stage["search_roi_pixel_bounds"] = {"x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1}

            annotated = search_roi_image.copy()
            frame_detected_ids: list[int] = []
            camera_matrix = resolve_pv_camera_matrix(frame)

            for dictionary_name, dictionary in dictionaries.items():
                corners, ids = _detect_markers(cv2, search_roi_image, dictionary)
                ids_list = [int(v) for v in ids.flatten().tolist()] if ids is not None else []
                full_corners, full_ids = _detect_markers(cv2, image_bgr, dictionary)
                full_ids_list = [int(v) for v in full_ids.flatten().tolist()] if full_ids is not None else []
                aruco_stage["full_image_detected_ids"].extend(full_ids_list)

                if corners:
                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

                for detected_index, marker_id in enumerate(ids_list):
                    marker = marker_by_dict[dictionary_name].get(marker_id)
                    if marker is None:
                        continue

                    frame_detected_ids.append(marker_id)
                    matched_roi_corners = np.asarray(corners[detected_index], dtype=np.float64).reshape(-1, 2)
                    matched_full_corners = matched_roi_corners.copy()
                    matched_full_corners[:, 0] += float(sx0)
                    matched_full_corners[:, 1] += float(sy0)

                    detection = _estimate_marker_detection(
                        cv2,
                        frame=frame,
                        frame_index=frame_index,
                        marker=marker,
                        corners_full=matched_full_corners,
                        camera_matrix=camera_matrix,
                    )
                    if detection is None:
                        continue
                    all_detections.append(detection)
                    detections_by_frame[frame_index].append(detection)

                    if hasattr(cv2, "drawFrameAxes"):
                        annotated_camera_matrix = camera_matrix.copy()
                        annotated_camera_matrix[0, 2] -= float(sx0)
                        annotated_camera_matrix[1, 2] -= float(sy0)
                        axis_length = (
                            float(np.linalg.norm(detection["tvec"])) * 0.5
                            if np.linalg.norm(detection["tvec"]) > 0
                            else 0.05
                        )
                        cv2.drawFrameAxes(
                            annotated,
                            annotated_camera_matrix,
                            np.zeros((5, 1), dtype=np.float64),
                            detection["rvec"],
                            detection["tvec"],
                            axis_length,
                        )

            annotated_path = raw_dir / f"annotated_{frame_index:03d}.png"
            cv2.imwrite(str(annotated_path), annotated)
            if frame_index == 0:
                cv2.imwrite(str(legacy_annotated_path), annotated)
            aruco_stage.setdefault("frames", []).append(
                {
                    "frame_index": frame_index,
                    "image_name": frame.get("name"),
                    "detected_ids": sorted(set(frame_detected_ids)),
                    "annotated_image_path": normalize_path_for_storage(annotated_path),
                }
            )
    except Exception as exc:
        aruco_stage["runtime_error"] = str(exc)

    aruco_stage["detections"] = [_build_detection_record(detection) for detection in all_detections]
    detected_ids = sorted({int(detection["marker_id"]) for detection in all_detections})
    aruco_stage["detected_ids"] = detected_ids
    aruco_stage["full_image_detected_ids"] = sorted(set(int(v) for v in aruco_stage["full_image_detected_ids"]))

    anchor_candidates: list[dict[str, Any]] = []
    learned_relations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame_index, frame_detections in detections_by_frame.items():
        anchor_detections = [
            detection
            for detection in frame_detections
            if int(detection["marker_id"]) == int(ARUCO_ANCHOR_MARKER_ID)
        ]
        for anchor_detection in anchor_detections:
            anchor_candidates.append(
                {
                    "translation": anchor_detection["world_translation"],
                    "rotation": anchor_detection["world_rotation"],
                    "weight": anchor_detection["weight"],
                    "source_marker_id": int(ARUCO_ANCHOR_MARKER_ID),
                    "frame_index": frame_index,
                    "source": "anchor_detection",
                }
            )

        if anchor_detections:
            anchor_detection = anchor_detections[0]
            for marker_detection in frame_detections:
                marker_id = int(marker_detection["marker_id"])
                if marker_id == int(ARUCO_ANCHOR_MARKER_ID):
                    continue
                relation_pose = _relation_from_anchor_to_marker(anchor_detection, marker_detection)
                relation_rotation, relation_translation = _pose_payload_to_rt(relation_pose)
                learned_relations[marker_id].append(
                    {
                        "translation": relation_translation,
                        "rotation": relation_rotation,
                        "weight": marker_detection["weight"],
                    }
                )

        for marker_detection in frame_detections:
            marker_id = int(marker_detection["marker_id"])
            if marker_id == int(ARUCO_ANCHOR_MARKER_ID):
                continue
            relation_row = get_aruco_marker_relation(marker_id)
            relation_payload = load_json_payload(relation_row.get("relation_pose_json")) if relation_row else None
            if relation_payload:
                anchor_candidates.append(_anchor_candidate_from_marker_relation(marker_detection, relation_payload))

    relation_updates = []
    for marker_id, relation_candidates in learned_relations.items():
        existing_row = get_aruco_marker_relation(marker_id)
        existing_payload = load_json_payload(existing_row.get("relation_pose_json")) if existing_row else None
        if existing_payload:
            try:
                existing_rotation, existing_translation = _pose_payload_to_rt(existing_payload)
                relation_candidates.append(
                    {
                        "translation": existing_translation,
                        "rotation": existing_rotation,
                        "weight": float(existing_row.get("sample_count") or 1),
                    }
                )
            except Exception:
                pass

        relation_translation, relation_rotation = _fuse_pose_candidates(relation_candidates)
        relation_pose = pose_to_payload(
            relation_rotation,
            relation_translation,
            ARUCO_LOCAL_COORDINATE_BASIS,
            scale=[1.0, 1.0, 1.0],
        )
        mean_error = float(
            np.mean(
                [
                    detection["reprojection_error_px"]
                    for detection in all_detections
                    if int(detection["marker_id"]) == int(marker_id)
                ]
            )
        )
        updated_row = upsert_aruco_marker_relation(
            marker_id=marker_id,
            relation_pose_json=relation_pose,
            sample_error=mean_error,
            task_id=task_id or None,
            raw_record_path=str(record_path),
        )
        relation_updates.append(
            {
                "marker_id": marker_id,
                "sample_count": updated_row.get("sample_count"),
                "mean_error": updated_row.get("mean_error"),
                "relation_pose": relation_pose,
            }
        )

    aruco_stage["relation_updates"] = relation_updates

    if anchor_candidates:
        anchor_translation, anchor_rotation = _fuse_pose_candidates(anchor_candidates)
        aruco_reference = pose_to_payload(
            anchor_rotation,
            anchor_translation,
            UNITY_WORLD_COORDINATE_BASIS,
            scale=[1.0, 1.0, 1.0],
        )
        aruco_stage["detected"] = True
        aruco_stage["matched_marker_id"] = int(ARUCO_ANCHOR_MARKER_ID)
        aruco_stage["short_circuit"] = True
        aruco_stage["marker_pose_world"] = aruco_reference
        aruco_stage["anchor_pose_candidates"] = [
            {
                "source": candidate.get("source"),
                "source_marker_id": candidate.get("source_marker_id"),
                "frame_index": candidate.get("frame_index"),
                "weight": candidate.get("weight"),
            }
            for candidate in anchor_candidates
        ]
        task["aruco_reference"] = aruco_reference
        task["object"] = None

        if startup_session_id:
            create_aruco_reference(
                startup_session_id=startup_session_id,
                task_id=task_id or None,
                marker_pose_json=aruco_reference,
                raw_record_path=str(record_path),
                config_snapshot_json={
                    "template": template,
                    "registered_markers": marker_configs,
                    "anchor_marker_id": int(ARUCO_ANCHOR_MARKER_ID),
                },
            )
            aruco_stage["retro_synced_completed_task_count"] = sync_completed_tasks_for_startup(
                startup_session_id
            )

    _write_debug(task, aruco_stage)
    save_task_json(json_path, task)

    record["aruco_stage"] = aruco_stage
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if aruco_stage["short_circuit"]:
        print(
            f"[INFO] aruco_detect : fused anchor marker {ARUCO_ANCHOR_MARKER_ID} from {len(anchor_candidates)} candidate(s)"
        )
    else:
        print("[INFO] aruco_detect : no usable anchor marker pose detected")
    print("[OK] aruco_detect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
