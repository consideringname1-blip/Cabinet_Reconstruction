from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from . import settings
from .cache import safe_name
from .output_paths import safe_output_name
from .people import extract_people_skeletons
from .schemas import HandContact, ModelEventRecord, MovementDecision, ShigureFrame, to_jsonable


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
    projected_box: Mapping[str, Any] | None = None,
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

    files: dict[str, Path] = {}
    copied_rgb = _copy_optional(frame.rgb_path, output_dir / "rgb.png")
    if copied_rgb is not None:
        files["rgb"] = copied_rgb
    copied_depth = _copy_optional(frame.depth_path, output_dir / "depth.png")
    if copied_depth is not None:
        files["depth"] = copied_depth
    copied_people = _copy_optional(frame.people_path, output_dir / "people_detection.json")
    if copied_people is not None:
        files["people_detection"] = copied_people
        skeletons = extract_people_skeletons(copied_people, min_score=0.0)
        files["skeletons"] = _write_json(output_dir / "skeletons.json", {"skeletons": skeletons})
    copied_camera = _copy_optional(frame.camera_info_path, output_dir / "camera_info.json")
    if copied_camera is not None:
        files["camera_info"] = copied_camera
    copied_marker = _copy_optional(frame.marker_pose_path, output_dir / "marker_pose.json")
    if copied_marker is not None:
        files["marker_pose"] = copied_marker
    saved_mask = _save_mask(mask, output_dir / "mask.png")
    if saved_mask is not None:
        files["mask"] = saved_mask

    contact = hand_contact or decision.trigger_contact
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
    _write_json(event_json, payload)
    return event_record
