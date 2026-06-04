from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import settings
from .geometry import point_to_oriented_box_signed_distance
from .schemas import HandContact, ProjectedBox, RosStamp, to_jsonable


HAND_PARTS = {
    "left_wrist": "left",
    "right_wrist": "right",
}


def _load_people_payload(path_or_payload: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(path_or_payload, Mapping):
        return dict(path_or_payload), None
    path = Path(path_or_payload)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file), path


def _message(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    message = payload.get("message")
    return message if isinstance(message, Mapping) else payload


def _point_xyz_m(point: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(point, Mapping):
        return None
    try:
        x = float(point.get("x", 0.0)) * 0.001
        y = float(point.get("y", 0.0)) * 0.001
        z = float(point.get("z", 0.0)) * 0.001
    except Exception:
        return None
    if not np.all(np.isfinite([x, y, z])) or z <= 0.0:
        return None
    return (x, y, z)


def _pixel_xy(point: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(point, Mapping):
        return None
    try:
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
    except Exception:
        return None
    if not np.all(np.isfinite([x, y])):
        return None
    return (x, y)


def iter_people(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    message = _message(payload)
    people = message.get("pose_key_points_list")
    if isinstance(people, list):
        for item in people:
            if isinstance(item, Mapping):
                yield item


def extract_people_skeletons(
    people_json: str | Path | Mapping[str, Any],
    *,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    payload, _source = _load_people_payload(people_json)
    skeletons: list[dict[str, Any]] = []
    for person in iter_people(payload):
        joints: list[dict[str, Any]] = []
        for point_data in person.get("point_data") or []:
            if not isinstance(point_data, Mapping):
                continue
            try:
                score = float(point_data.get("score", 0.0))
            except Exception:
                score = 0.0
            if score < float(min_score):
                continue
            projection_m = _point_xyz_m(point_data.get("projection_point"))
            joints.append(
                {
                    "body_part_name": str(point_data.get("body_part_name") or ""),
                    "score": score,
                    "pixel_xy": _pixel_xy(point_data.get("pixel_point")),
                    "projection_camera_m": projection_m,
                }
            )
        skeletons.append(
            {
                "people_id": str(person.get("people_id") or ""),
                "bounding_box": to_jsonable(person.get("bounding_box") or {}),
                "joints": joints,
            }
        )
    return skeletons


def find_hand_contacts(
    people_json: str | Path | Mapping[str, Any],
    projected_box: ProjectedBox,
    *,
    min_score: float = settings.HAND_SCORE_MIN,
    box_margin_m: float = settings.HAND_BOX_MARGIN_M,
) -> list[HandContact]:
    payload, source_path = _load_people_payload(people_json)
    timestamp = RosStamp.from_message_json(payload)
    contacts: list[HandContact] = []
    corners = np.asarray(projected_box.corners_camera_m, dtype=np.float64)

    for person in iter_people(payload):
        people_id = str(person.get("people_id") or "")
        for point_data in person.get("point_data") or []:
            if not isinstance(point_data, Mapping):
                continue
            body_part_name = str(point_data.get("body_part_name") or "")
            hand = HAND_PARTS.get(body_part_name)
            if hand is None:
                continue
            try:
                score = float(point_data.get("score", 0.0))
            except Exception:
                score = 0.0
            if score < float(min_score):
                continue
            point_camera_m = _point_xyz_m(point_data.get("projection_point"))
            if point_camera_m is None:
                continue

            signed_distance = point_to_oriented_box_signed_distance(
                np.asarray(point_camera_m, dtype=np.float64),
                corners,
                margin_m=box_margin_m,
            )
            contacts.append(
                HandContact(
                    timestamp=timestamp,
                    people_id=people_id,
                    hand=hand,
                    score=score,
                    point_camera_m=point_camera_m,
                    pixel_xy=_pixel_xy(point_data.get("pixel_point")),
                    signed_distance_m=float(signed_distance),
                    distance_m=max(0.0, float(signed_distance)),
                    inside_box=signed_distance <= 0.0,
                    source_path=source_path,
                )
            )

    contacts.sort(key=lambda item: (not item.inside_box, item.distance_m, -item.score))
    return contacts


def choose_event_start_contact(
    contacts: Iterable[HandContact],
    *,
    nearest_max_distance_m: float | None = settings.HAND_NEAREST_MAX_DISTANCE_M,
) -> HandContact | None:
    ordered = sorted(contacts, key=lambda item: (not item.inside_box, item.distance_m, -item.score))
    if not ordered:
        return None
    for contact in ordered:
        if contact.inside_box:
            return contact
    if nearest_max_distance_m is None:
        return ordered[0]
    close = [contact for contact in ordered if contact.distance_m <= float(nearest_max_distance_m)]
    return close[0] if close else ordered[0]
