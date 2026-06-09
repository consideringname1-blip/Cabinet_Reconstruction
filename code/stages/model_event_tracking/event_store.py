from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from . import settings
from .cache import safe_name
from .output_paths import safe_output_name
from .people import extract_people_skeletons
from .schemas import HandContact, ModelEventRecord, MovementDecision, ShigureFrame, to_jsonable


SKELETON_EDGES = (
    ("nose", "neck"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("neck", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)


def _copy_optional(path: Path | None, target: Path) -> Path | None:
    if path is None or not path.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def _save_mask(mask: np.ndarray | None, target: Path) -> Path | None:
    if mask is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(mask)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got {array.shape}")
    Image.fromarray((array > 0).astype(np.uint8) * 255).save(target)
    return target


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(dict(payload)), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _as_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(mask)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.shape != shape:
        return np.zeros(shape, dtype=bool)
    return array.astype(bool)


def _blend_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    rgb = np.asarray(color, dtype=np.float32)
    canvas[mask] = ((1.0 - alpha) * canvas[mask].astype(np.float32) + alpha * rgb).astype(np.uint8)


def _save_decision_overlay(
    rgb_path: Path | None,
    *,
    model_mask: np.ndarray | None,
    trusted_mask: np.ndarray | None,
    moved_mask: np.ndarray | None,
    target: Path,
) -> Path | None:
    if rgb_path is None or not rgb_path.is_file():
        return None
    with Image.open(rgb_path) as image:
        canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    shape = canvas.shape[:2]
    full = _as_mask(model_mask, shape)
    trusted = _as_mask(trusted_mask, shape)
    moved = _as_mask(moved_mask, shape)
    _blend_mask(canvas, full, (40, 235, 100), 0.35)
    _blend_mask(canvas, trusted, (255, 225, 40), 0.55)
    _blend_mask(canvas, moved, (255, 60, 60), 0.70)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(target, quality=95)
    return target


def _joint_map(skeleton: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    joints: dict[str, tuple[float, float]] = {}
    for joint in skeleton.get("joints") or []:
        if not isinstance(joint, Mapping):
            continue
        name = str(joint.get("body_part_name") or "")
        pixel = joint.get("pixel_xy")
        if not name or not isinstance(pixel, (list, tuple)) or len(pixel) < 2:
            continue
        try:
            x = float(pixel[0]); y = float(pixel[1])
        except Exception:
            continue
        if np.isfinite([x, y]).all():
            joints[name] = (x, y)
    return joints


def _save_skeleton_overlay(
    rgb_path: Path | None,
    skeletons: list[dict[str, Any]],
    hand_contact: HandContact | None,
    target: Path,
) -> Path | None:
    if rgb_path is None or not rgb_path.is_file():
        return None
    with Image.open(rgb_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for skeleton in skeletons:
        joints = _joint_map(skeleton)
        for start, end in SKELETON_EDGES:
            if start in joints and end in joints:
                draw.line((joints[start], joints[end]), fill=(40, 220, 255), width=4)
        for name, (x, y) in joints.items():
            radius = 5 if name.endswith("wrist") else 3
            color = (255, 230, 40) if name.endswith("wrist") else (40, 220, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    if hand_contact is not None and hand_contact.pixel_xy is not None:
        x, y = hand_contact.pixel_xy
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), outline=(255, 60, 60), width=5)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=95)
    return target



def _try_generate_body_mesh(output_dir: Path, debug_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    if not settings.SAM3D_BODY_EVENT_MESH_ENABLED:
        return None, {"status": "disabled"}
    script = Path(settings.SAM3D_BODY_EVENT_MESH_SCRIPT)
    python = Path(str(settings.SAM3D_BODY_PY))
    if not script.is_file() or not python.is_file():
        return None, {
            "status": "missing_runtime",
            "script": str(script),
            "python": str(python),
        }
    target = output_dir / "body_mesh.json"
    cmd = [
        str(python),
        str(script),
        str(output_dir),
        "--output",
        str(target),
        "--decimate-ratio",
        str(float(settings.SAM3D_BODY_EVENT_MESH_DECIMATE_RATIO)),
        "--max-distance-m",
        str(float(settings.SAM3D_BODY_EVENT_MESH_MAX_DISTANCE_M)),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(output_dir),
            text=True,
            capture_output=True,
            timeout=max(1.0, float(settings.SAM3D_BODY_EVENT_MESH_TIMEOUT_SEC)),
            check=False,
        )
    except Exception as exc:
        return None, {"status": "failed_to_start", "error": str(exc), "cmd": cmd}
    metadata = {
        "status": "ok" if proc.returncode == 0 and target.is_file() else "failed",
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "target": str(target),
        "decimate_ratio": float(settings.SAM3D_BODY_EVENT_MESH_DECIMATE_RATIO),
    }
    if proc.returncode == 0 and target.is_file():
        return target, metadata
    debug_dir.mkdir(parents=True, exist_ok=True)
    _write_json(debug_dir / "body_mesh_error.json", metadata)
    return None, metadata

def event_dir_for_task(
    task_id: str,
    *,
    event_type: str = "taken_away",
    task_output_name: str | None = None,
) -> Path:
    folder_name = safe_output_name(task_output_name or task_id)
    return Path(settings.MODEL_EVENT_OUTPUT_ROOT) / folder_name / safe_name(event_type)


def persist_taken_away_event(
    *,
    task_id: str,
    frame: ShigureFrame,
    decision: MovementDecision,
    hand_contact: HandContact | None = None,
    mask: np.ndarray | None = None,
    model_mask: np.ndarray | None = None,
    trusted_mask: np.ndarray | None = None,
    projected_box: Mapping[str, Any] | None = None,
    people_frame: ShigureFrame | None = None,
    task_output_name: str | None = None,
    replace: bool = False,
) -> ModelEventRecord:
    output_dir = event_dir_for_task(
        task_id,
        event_type="taken_away",
        task_output_name=task_output_name,
    )
    event_json = output_dir / "event.json"
    if event_json.is_file() and not replace:
        with event_json.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return ModelEventRecord(
            task_id=task_id,
            event_type="taken_away",
            event_timestamp=decision.timestamp,
            trigger_timestamp=(decision.trigger_contact.timestamp if decision.trigger_contact else None),
            decision=decision,
            hand_contact=hand_contact or decision.trigger_contact,
            output_dir=output_dir,
            files={key: Path(value) for key, value in dict(payload.get("files") or {}).items()},
        )

    debug_dir = output_dir / "debug"
    files: dict[str, Path] = {}
    debug_files: dict[str, Path] = {}
    contact = hand_contact or decision.trigger_contact

    copied_rgb = _copy_optional(frame.rgb_path, output_dir / "rgb.png")
    if copied_rgb is not None:
        files["rgb"] = copied_rgb

    skeletons: list[dict[str, Any]] = []
    skeleton_source_frame = people_frame or frame
    copied_people = _copy_optional(skeleton_source_frame.people_path, debug_dir / "people_detection.json")
    if copied_people is not None:
        debug_files["people_detection"] = copied_people
        detected_skeletons = extract_people_skeletons(copied_people, min_score=0.0)
        if contact is not None and contact.people_id:
            skeletons = [
                skeleton
                for skeleton in detected_skeletons
                if str(skeleton.get("people_id") or "") == contact.people_id
            ]
    files["skeletons"] = _write_json(output_dir / "skeletons.json", {"skeletons": skeletons})
    skeleton_overlay = _save_skeleton_overlay(copied_rgb, skeletons, contact, output_dir / "skeleton_overlay.png")
    if skeleton_overlay is not None:
        files["skeleton_overlay"] = skeleton_overlay

    copied_depth = _copy_optional(frame.depth_path, debug_dir / "depth.png")
    if copied_depth is not None:
        debug_files["depth"] = copied_depth
    copied_camera = _copy_optional(frame.camera_info_path, debug_dir / "camera_info.json")
    if copied_camera is not None:
        debug_files["camera_info"] = copied_camera
    copied_marker = _copy_optional(frame.marker_pose_path, debug_dir / "marker_pose.json")
    if copied_marker is not None:
        debug_files["marker_pose"] = copied_marker
    saved_mask = _save_mask(mask, debug_dir / "moved_mask.png")
    if saved_mask is not None:
        debug_files["moved_mask"] = saved_mask
    saved_overlay = _save_decision_overlay(
        copied_rgb,
        model_mask=model_mask,
        trusted_mask=trusted_mask,
        moved_mask=mask,
        target=debug_dir / "decision_overlay.png",
    )
    if saved_overlay is not None:
        debug_files["decision_overlay"] = saved_overlay
    if projected_box is not None:
        debug_files["projected_box"] = _write_json(debug_dir / "projected_box.json", {"projected_box": projected_box})

    event_record = ModelEventRecord(
        task_id=task_id,
        event_type="taken_away",
        event_timestamp=decision.timestamp,
        trigger_timestamp=(contact.timestamp if contact is not None else None),
        decision=decision,
        hand_contact=contact,
        output_dir=output_dir,
        files=files,
    )
    payload = event_record.to_dict()
    payload["projected_box"] = to_jsonable(projected_box or {})
    payload["evidence_frame"] = frame.to_dict()
    if people_frame is not None:
        payload["skeleton_source_frame"] = people_frame.to_dict()
    payload["debug_files"] = to_jsonable(debug_files)
    _write_json(event_json, payload)

    body_mesh_path, body_mesh_metadata = _try_generate_body_mesh(output_dir, debug_dir)
    if body_mesh_path is not None:
        files["body_mesh"] = body_mesh_path
        event_record = ModelEventRecord(
            task_id=task_id,
            event_type="taken_away",
            event_timestamp=decision.timestamp,
            trigger_timestamp=(contact.timestamp if contact is not None else None),
            decision=decision,
            hand_contact=contact,
            output_dir=output_dir,
            files=files,
        )
        payload = event_record.to_dict()
        payload["projected_box"] = to_jsonable(projected_box or {})
        payload["evidence_frame"] = frame.to_dict()
        if people_frame is not None:
            payload["skeleton_source_frame"] = people_frame.to_dict()
        payload["debug_files"] = to_jsonable(debug_files)
        payload["body_mesh_metadata"] = to_jsonable(body_mesh_metadata or {})
        _write_json(event_json, payload)
    elif body_mesh_metadata is not None:
        payload["body_mesh_metadata"] = to_jsonable(body_mesh_metadata)
        _write_json(event_json, payload)
    return event_record
