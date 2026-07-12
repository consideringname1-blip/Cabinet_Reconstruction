"""Persistent display identity for the HoloLens reconstruction stage."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from task_db import (
    commit_display_object_capture_state,
    create_display_object,
    get_capture_instance,
    get_capture_instance_by_task_id,
    record_capture_binding_log,
    upsert_capture_instance,
)
from task_json import load_task_json, normalize_path_for_storage, resolve_task_json_path, save_task_json


DINO_IDENTITY_VERSION = 2
MATCHED_STATUSES = {"matched", "matched_force_new"}
IDENTITY_STATUSES = MATCHED_STATUSES | {"miss"}


def _task_id(task: dict[str, Any]) -> str:
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is missing")
    return task_id


def _capture_instance_id(task: dict[str, Any]) -> str:
    existing = task.get("DisplayIdentity") if isinstance(task.get("DisplayIdentity"), dict) else {}
    raw = existing.get("capture_instance_id")
    if raw:
        return str(raw)
    return f"capture_{_task_id(task)}"


def _resolve_existing_capture(capture_instance_id: str, task_id: str) -> dict[str, Any] | None:
    existing = get_capture_instance(capture_instance_id)
    if existing is not None:
        return existing
    return get_capture_instance_by_task_id(task_id)


def _resolve_existing_binding(existing_capture: dict[str, Any] | None) -> str | None:
    if not existing_capture:
        return None
    if str(existing_capture.get("binding_status") or "") != "bound":
        return None
    display_object_id = str(existing_capture.get("display_object_id") or "").strip()
    return display_object_id or None


def _existing_capture_uses_dinov2(existing_capture: dict[str, Any] | None) -> bool:
    if not existing_capture:
        return False
    raw = existing_capture.get("feature_json")
    try:
        feature = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return False
    if not isinstance(feature, dict):
        return False
    if str(feature.get("identity_source") or "") == "dinov2_historical_model_match":
        return True
    return isinstance(feature.get("dinov2"), dict)


def _historical_match_payload(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("HistoricalModelMatch")
    return payload if isinstance(payload, dict) else {}


def _historical_match_display_object_id(payload: dict[str, Any]) -> str | None:
    status = str(payload.get("status") or "").strip()
    if status not in MATCHED_STATUSES:
        return None
    display_object_id = str(payload.get("display_object_id") or "").strip()
    return display_object_id or None


def _historical_match_current_dinov2(payload: dict[str, Any]) -> dict[str, Any] | None:
    current = payload.get("current_dinov2") if isinstance(payload.get("current_dinov2"), dict) else None
    if not current:
        return None
    embedding = current.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return None
    return current


def _candidate_scores(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scores = payload.get("candidate_scores")
    if not isinstance(scores, list):
        return []
    return [item for item in scores if isinstance(item, dict)]


def _selected_distance(payload: dict[str, Any], display_object_id: str | None) -> float | None:
    value = payload.get("dinov2_distance")
    try:
        if value is not None:
            return float(value)
    except Exception:
        pass
    if display_object_id:
        for item in _candidate_scores(payload):
            if str(item.get("display_object_id") or "") != str(display_object_id):
                continue
            try:
                return float(item.get("dinov2_distance"))
            except Exception:
                return None
    return None


def _feature_from_historical_match(payload: dict[str, Any]) -> dict[str, Any]:
    feature: dict[str, Any] = {
        "version": DINO_IDENTITY_VERSION,
        "identity_source": "dinov2_historical_model_match",
        "historical_model_match_status": payload.get("status"),
        "historical_model_match_reason": payload.get("reason"),
    }
    current = _historical_match_current_dinov2(payload)
    if current is not None:
        feature["dinov2"] = current
    return feature


def _feature_summary(feature: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "version": feature.get("version"),
        "identity_source": feature.get("identity_source"),
        "historical_model_match_status": feature.get("historical_model_match_status"),
        "historical_model_match_reason": feature.get("historical_model_match_reason"),
    }
    dinov2 = feature.get("dinov2") if isinstance(feature.get("dinov2"), dict) else None
    if dinov2:
        embedding = dinov2.get("embedding")
        summary["dinov2_dim"] = len(embedding) if isinstance(embedding, list) else dinov2.get("dim")
        summary["dinov2_model_name"] = dinov2.get("model_name")
        summary["dinov2_device"] = dinov2.get("device")
    return {key: value for key, value in summary.items() if value is not None}


def _thresholds_payload(payload: dict[str, Any]) -> dict[str, Any]:
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    return {
        "identity_backend": "dinov2",
        "match_stage": "historical_model_match",
        **thresholds,
    }


def _evidence(json_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "json_path": normalize_path_for_storage(json_path),
        "identity_backend": "dinov2",
        "historical_model_match_status": payload.get("status"),
        "historical_model_match_reason": payload.get("reason"),
    }
    current = _historical_match_current_dinov2(payload)
    if current is not None:
        source = current.get("source") if isinstance(current.get("source"), dict) else {}
        evidence["dinov2_source"] = source
        evidence["dinov2_crop"] = current.get("crop") or {}
    return evidence


def _store_result_in_task_json(json_path: Path, task: dict[str, Any], result: dict[str, Any]) -> None:
    task["DisplayIdentity"] = result
    save_task_json(json_path, task)


def bind_capture_identity(
    json_path_arg: str | Path,
) -> dict[str, Any]:
    json_path = resolve_task_json_path(json_path_arg)
    task = load_task_json(json_path)
    task_id = _task_id(task)
    capture_instance_id = _capture_instance_id(task)
    existing_capture = _resolve_existing_capture(capture_instance_id, task_id)
    if existing_capture and existing_capture.get("capture_instance_id"):
        capture_instance_id = str(existing_capture["capture_instance_id"])
    existing_display_object_id = _resolve_existing_binding(existing_capture)
    timestamp_value = task.get("server_received_utc")
    if not isinstance(timestamp_value, str) or not timestamp_value.strip():
        raise ValueError("server_received_utc is required")
    timestamp = timestamp_value.strip()

    historical_payload = _historical_match_payload(task)
    historical_status = str(historical_payload.get("status") or "").strip()
    if historical_status not in IDENTITY_STATUSES:
        raise RuntimeError(f"invalid HistoricalModelMatch.status: {historical_status or 'missing'}")
    if _historical_match_current_dinov2(historical_payload) is None:
        raise RuntimeError("HistoricalModelMatch.current_dinov2 is missing or invalid")
    historical_display_object_id = _historical_match_display_object_id(historical_payload)
    if historical_status in MATCHED_STATUSES and historical_display_object_id is None:
        raise RuntimeError("matched HistoricalModelMatch is missing display_object_id")
    feature = _feature_from_historical_match(historical_payload)
    evidence = _evidence(json_path, historical_payload)
    candidate_scores = _candidate_scores(historical_payload)
    selected_distance = _selected_distance(historical_payload, historical_display_object_id)

    if historical_display_object_id:
        display_object_id = historical_display_object_id
        is_new_display_object = False
        binding_status = "bound"
        if historical_payload.get("reuse_model"):
            decision = "reuse_historical_model_match"
            reason = "dinov2_match_reused_historical_model"
        else:
            decision = "bind_historical_dinov2_match"
            reason = "dinov2_match_generate_new_model"
    elif existing_display_object_id and _existing_capture_uses_dinov2(existing_capture):
        display_object_id = existing_display_object_id
        is_new_display_object = False
        binding_status = "bound"
        decision = "reuse_existing_binding"
        reason = "existing_dinov2_capture_binding_reused"
    else:
        display_object_id = str(uuid.uuid4())
        create_display_object(display_object_id=display_object_id, canonical_capture_instance_id=capture_instance_id)
        is_new_display_object = True
        binding_status = "bound"
        decision = "create_new"
        reason = "dinov2_no_matched_display_object_create_new"

    upsert_capture_instance(
        capture_instance_id=capture_instance_id,
        task_id=task_id,
        display_object_id=display_object_id,
        source="hololens",
        timestamp=timestamp,
        binding_status=binding_status,
        binding_reason=reason,
        identity_distance=selected_distance,
        candidate_scores=candidate_scores,
        feature=feature,
        evidence=evidence,
    )

    result = {
        "capture_instance_id": capture_instance_id,
        "display_object_id": display_object_id,
        "is_new_display_object": bool(is_new_display_object),
        "binding_status": binding_status,
        "decision": decision,
        "binding_reason": reason,
        "identity_backend": "dinov2",
        "identity_distance": selected_distance,
        "candidate_scores": candidate_scores[:10],
        "candidate_scores_close": bool(historical_payload.get("second_margin_ok") is False),
        "close_candidates": candidate_scores[:3] if historical_payload.get("second_margin_ok") is False else [],
        "thresholds": _thresholds_payload(historical_payload),
        "feature_summary": _feature_summary(feature),
        "evidence": evidence,
    }
    record_capture_binding_log(
        capture_instance_id=capture_instance_id,
        task_id=task_id,
        display_object_id=display_object_id,
        decision=decision,
        binding_status=binding_status,
        reason=reason,
        candidate_scores=candidate_scores,
        detail=result,
    )
    object_aruco = task.get("object_aruco") if isinstance(task.get("object_aruco"), dict) else None
    if object_aruco is not None:
        reused_model = bool(historical_payload.get("reuse_model"))
        selected_model_task_id = str(historical_payload.get("selected_model_task_id") or "").strip() or None
        state = commit_display_object_capture_state(
            display_object_id=display_object_id,
            capture_task_id=task_id,
            pose_aruco=object_aruco,
            captured_at=timestamp,
            generated_new_model=not reused_model,
            active_model_task_id=selected_model_task_id if reused_model else task_id,
        )
        result["model_revision"] = int(state.get("active_model_revision") or 0)
        result["hololens_pose_revision"] = int(state.get("latest_hololens_pose_revision") or 0)
    _store_result_in_task_json(json_path, task, result)
    return result


def load_display_identity_for_task_json(task_json: dict[str, Any]) -> dict[str, Any] | None:
    value = task_json.get("DisplayIdentity")
    return value if isinstance(value, dict) else None
