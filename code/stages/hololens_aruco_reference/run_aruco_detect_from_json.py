from __future__ import annotations

import json
import sys

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from config import (
    ARUCO_ROI_PADDING_MIN_PX,
    ARUCO_ROI_PADDING_RATIO,
    ARUCO_TEMPLATE_PATH,
)
from task_db import create_aruco_reference
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
        evaluate_aruco_template,
        load_aruco_template,
        pose_to_payload,
        resolve_pv_camera_matrix,
        resolve_pv_camera_world_pose,
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
        evaluate_aruco_template,
        load_aruco_template,
        pose_to_payload,
        resolve_pv_camera_matrix,
        resolve_pv_camera_world_pose,
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


def _annotate_image(
    cv2,
    roi_image: np.ndarray,
    corners,
    ids,
    matched_rvec,
    matched_tvec,
    camera_matrix,
) -> np.ndarray:
    annotated = roi_image.copy()
    if corners:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    if (
        matched_rvec is not None
        and matched_tvec is not None
        and camera_matrix is not None
        and hasattr(cv2, "drawFrameAxes")
    ):
        axis_length = float(np.linalg.norm(matched_tvec)) * 0.5 if np.linalg.norm(matched_tvec) > 0 else 0.05
        cv2.drawFrameAxes(
            annotated,
            camera_matrix,
            np.zeros((5, 1), dtype=np.float64),
            matched_rvec,
            matched_tvec,
            axis_length,
        )
    return annotated


def _write_debug(task: dict, aruco_stage: dict) -> None:
    debug_section = dict(task.get("debug") or {})
    pose_transform_stages = dict(debug_section.get("pose_transform_stages") or {})
    pose_transform_stages["aruco_stage"] = aruco_stage
    debug_section["pose_transform_stages"] = pose_transform_stages
    task["debug"] = debug_section


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
    roi_path = raw_dir / "roi.png"
    search_roi_path = raw_dir / "search_roi.png"
    annotated_path = raw_dir / "annotated.png"
    record_path = raw_dir / "record.json"

    template = load_aruco_template()
    template_state = evaluate_aruco_template(template)
    aruco_stage = {
        "template_path": normalize_path_for_storage(ARUCO_TEMPLATE_PATH),
        "template_enabled": template_state["enabled"],
        "configured": template_state["configured"],
        "config_reason": template_state["reason"],
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
        "raw_record_path": normalize_path_for_storage(record_path),
        "roi_image_path": normalize_path_for_storage(roi_path),
        "search_roi_image_path": normalize_path_for_storage(search_roi_path),
        "annotated_image_path": normalize_path_for_storage(annotated_path),
    }
    record = {
        "task_id": task_id,
        "task_name": task_name,
        "startup_session_id": startup_session_id,
        "template": template,
    }

    cv2 = _load_cv2()
    if cv2 is None:
        aruco_stage["runtime_error"] = "opencv_not_available"
        _write_debug(task, aruco_stage)
        save_task_json(json_path, task)
        record["aruco_stage"] = aruco_stage
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[INFO] aruco_detect : OpenCV unavailable, continuing without marker detection")
        print("[OK] aruco_detect")
        return 0

    image_path = resolve_pv_image_path(task)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read PV image: {image_path}")

    image_height, image_width = image_bgr.shape[:2]
    x0, y0, x1, y1 = resolve_selection_roi(task, image_width, image_height)
    sx0, sy0, sx1, sy1 = _expand_roi(x0, y0, x1, y1, image_width, image_height)
    aruco_stage["roi_pixel_bounds"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    aruco_stage["search_roi_pixel_bounds"] = {"x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1}

    roi_image = image_bgr[y0:y1, x0:x1].copy()
    search_roi_image = image_bgr[sy0:sy1, sx0:sx1].copy()
    cv2.imwrite(str(roi_path), roi_image)
    cv2.imwrite(str(search_roi_path), search_roi_image)
    cv2.imwrite(str(annotated_path), search_roi_image)

    if not template_state["configured"]:
        _write_debug(task, aruco_stage)
        save_task_json(json_path, task)
        record["aruco_stage"] = aruco_stage
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[INFO] aruco_detect : template disabled or incomplete, continuing")
        print("[OK] aruco_detect")
        return 0

    matched_rvec = None
    matched_tvec = None
    camera_matrix = None
    annotated_camera_matrix = None
    expected_marker_id = None

    try:
        expected_marker_id = int(template["marker_id"])
        dictionary = _resolve_dictionary(cv2.aruco, str(template.get("dictionary") or ""))
        corners, ids = _detect_markers(cv2, search_roi_image, dictionary)
        ids_list = [int(v) for v in ids.flatten().tolist()] if ids is not None else []
        aruco_stage["detected_ids"] = ids_list
        full_corners, full_ids = _detect_markers(cv2, image_bgr, dictionary)
        full_ids_list = [int(v) for v in full_ids.flatten().tolist()] if full_ids is not None else []
        aruco_stage["full_image_detected_ids"] = full_ids_list
        aruco_stage["marker_visible_outside_selection"] = bool(
            expected_marker_id in full_ids_list and expected_marker_id not in ids_list
        )
        matched_index = next((idx for idx, marker_id in enumerate(ids_list) if marker_id == expected_marker_id), None)

        if matched_index is not None and corners:
            matched_roi_corners = np.asarray(corners[matched_index], dtype=np.float64).reshape(-1, 2)
            matched_full_corners = matched_roi_corners.copy()
            matched_full_corners[:, 0] += float(sx0)
            matched_full_corners[:, 1] += float(sy0)

            marker_size_m = float(template["marker_size_mm"]) / 1000.0
            half = marker_size_m * 0.5
            object_points = np.array(
                [
                    [-half, half, 0.0],
                    [half, half, 0.0],
                    [half, -half, 0.0],
                    [-half, -half, 0.0],
                ],
                dtype=np.float64,
            )
            camera_matrix = resolve_pv_camera_matrix(task)
            annotated_camera_matrix = camera_matrix.copy()
            annotated_camera_matrix[0, 2] -= float(sx0)
            annotated_camera_matrix[1, 2] -= float(sy0)
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                matched_full_corners.astype(np.float64),
                camera_matrix,
                np.zeros((5, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                rotation_cv, _ = cv2.Rodrigues(rvec)
                local_rotation, local_translation = convert_cv_pose_to_unity_pose(rotation_cv, tvec.reshape(3))
                camera_world_translation, camera_world_rotation = resolve_pv_camera_world_pose(task)
                world_translation, world_rotation = compose_world_pose(
                    camera_world_translation,
                    camera_world_rotation,
                    local_translation,
                    local_rotation,
                )

                aruco_reference = pose_to_payload(
                    world_rotation,
                    world_translation,
                    UNITY_WORLD_COORDINATE_BASIS,
                    scale=[1.0, 1.0, 1.0],
                )
                local_pose = pose_to_payload(
                    local_rotation,
                    local_translation,
                    ARUCO_LOCAL_COORDINATE_BASIS,
                    scale=[1.0, 1.0, 1.0],
                )

                aruco_stage["detected"] = True
                aruco_stage["matched_marker_id"] = expected_marker_id
                aruco_stage["short_circuit"] = True
                aruco_stage["marker_pose_local"] = local_pose
                aruco_stage["marker_pose_world"] = aruco_reference
                task["aruco_reference"] = aruco_reference
                task["object"] = None

                if startup_session_id:
                    create_aruco_reference(
                        startup_session_id=startup_session_id,
                        task_id=task_id or None,
                        marker_pose_json=aruco_reference,
                        raw_record_path=str(record_path),
                        config_snapshot_json=template,
                    )
                    aruco_stage["retro_synced_completed_task_count"] = sync_completed_tasks_for_startup(
                        startup_session_id
                    )

                matched_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                matched_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    except Exception as exc:
        aruco_stage["runtime_error"] = str(exc)
        corners = []
        ids = None

    annotated = _annotate_image(
        cv2,
        search_roi_image,
        corners,
        ids,
        matched_rvec,
        matched_tvec,
        annotated_camera_matrix,
    )
    cv2.imwrite(str(annotated_path), annotated)

    _write_debug(task, aruco_stage)
    save_task_json(json_path, task)

    record["aruco_stage"] = aruco_stage
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if aruco_stage["short_circuit"]:
        print(f"[INFO] aruco_detect : marker {expected_marker_id} detected, short-circuiting task")
    else:
        print("[INFO] aruco_detect : no matching marker detected, continuing pipeline")
    print("[OK] aruco_detect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
