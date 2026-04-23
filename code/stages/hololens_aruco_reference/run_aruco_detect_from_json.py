from __future__ import annotations

import json
import sys

import numpy as np

try:
    import _bootstrap  # type: ignore
except ModuleNotFoundError:
    from . import _bootstrap  # type: ignore

from config import ARUCO_TEMPLATE_PATH
from task_db import create_aruco_reference
from task_json import load_task_json, resolve_task_json_path, save_task_json

try:
    from aruco_common import (
        ARUCO_LOCAL_COORDINATE_BASIS,
        UNITY_WORLD_COORDINATE_BASIS,
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
except ModuleNotFoundError:
    from .aruco_common import (
        ARUCO_LOCAL_COORDINATE_BASIS,
        UNITY_WORLD_COORDINATE_BASIS,
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


def _detect_markers(cv2, roi_image: np.ndarray, dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(roi_image)[:2]

    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _rejected = cv2.aruco.detectMarkers(roi_image, dictionary, parameters=parameters)
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
    annotated_path = raw_dir / "annotated.png"
    record_path = raw_dir / "record.json"

    template = load_aruco_template()
    template_state = evaluate_aruco_template(template)
    aruco_stage = {
        "template_path": str(ARUCO_TEMPLATE_PATH),
        "template_enabled": template_state["enabled"],
        "configured": template_state["configured"],
        "config_reason": template_state["reason"],
        "detected": False,
        "detected_ids": [],
        "matched_marker_id": None,
        "coordinate_basis_local": ARUCO_LOCAL_COORDINATE_BASIS,
        "coordinate_basis_world": UNITY_WORLD_COORDINATE_BASIS,
        "short_circuit": False,
        "raw_record_path": str(record_path),
        "roi_image_path": str(roi_path),
        "annotated_image_path": str(annotated_path),
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
    aruco_stage["roi_pixel_bounds"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    roi_image = image_bgr[y0:y1, x0:x1].copy()
    cv2.imwrite(str(roi_path), roi_image)
    cv2.imwrite(str(annotated_path), roi_image)

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
        corners, ids = _detect_markers(cv2, roi_image, dictionary)
        ids_list = [int(v) for v in ids.flatten().tolist()] if ids is not None else []
        aruco_stage["detected_ids"] = ids_list
        matched_index = next((idx for idx, marker_id in enumerate(ids_list) if marker_id == expected_marker_id), None)

        if matched_index is not None and corners:
            matched_roi_corners = np.asarray(corners[matched_index], dtype=np.float64).reshape(-1, 2)
            matched_full_corners = matched_roi_corners.copy()
            matched_full_corners[:, 0] += float(x0)
            matched_full_corners[:, 1] += float(y0)

            marker_size_m = float(template["marker_size_mm"]) / 1000.0
            half = marker_size_m * 0.5
            object_points = np.array(
                [
                    [-half, -half, 0.0],
                    [half, -half, 0.0],
                    [half, half, 0.0],
                    [-half, half, 0.0],
                ],
                dtype=np.float64,
            )
            camera_matrix = resolve_pv_camera_matrix(task)
            annotated_camera_matrix = camera_matrix.copy()
            annotated_camera_matrix[0, 2] -= float(x0)
            annotated_camera_matrix[1, 2] -= float(y0)
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

                matched_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                matched_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    except Exception as exc:
        aruco_stage["runtime_error"] = str(exc)
        corners = []
        ids = None

    annotated = _annotate_image(
        cv2,
        roi_image,
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
