from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap
import numpy as np

from artifact_layout import model_result_dir, model_result_file, model_worker_file
from config import (
    DINO_IDENTITY_CANDIDATE_LIMIT,
    DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD,
    DINO_IDENTITY_MATCH_REQUIRE_MARGIN,
    DINO_IDENTITY_MATCH_SECOND_MARGIN,
)
from stages.hololens3d_reconstruction.model_generation_common import (
    build_model_generation_payload,
    resolve_model_generation_source,
    resolve_runtime_mesh_source,
)
from stage_common import ensure_file, load_stage_task
from task_db import (
    get_latest_completed_task_for_display_object,
    get_task_by_task_id,
    list_identity_candidate_captures,
    update_capture_instance_feature,
)
from task_json import (
    load_task_json,
    resolve_task_json_path_from_record,
    save_task_json,
)


WORKER_RESPONSE_ENCODING = "utf-8"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        loaded = json.loads(str(value))
    except Exception:
        return default
    return loaded if loaded is not None else default


def _send_socket_request(socket_path: Path, payload: dict[str, Any], *, timeout: float = 180.0) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode(WORKER_RESPONSE_ENCODING))
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No response from DINOv2 identity worker: {socket_path}")
    response = json.loads(b"".join(chunks).decode(WORKER_RESPONSE_ENCODING).splitlines()[0])
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "DINOv2 identity worker failed"))
    return response


def _dinov2_from_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    value = feature.get("dinov2") if isinstance(feature, dict) else None
    if not isinstance(value, dict):
        return None
    embedding = value.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return None
    return value


def _embedding_array(payload: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(payload.get("embedding"), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("DINOv2 embedding is empty")
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        raise ValueError("DINOv2 embedding has zero norm")
    return arr / norm


def _cosine_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = _embedding_array(left)
    b = _embedding_array(right)
    if a.shape != b.shape:
        return 1.0
    return float(max(0.0, min(2.0, 1.0 - float(np.dot(a, b)))))


def _request_embedding(socket_path: Path, json_path: Path) -> dict[str, Any]:
    response = _send_socket_request(
        socket_path,
        {"action": "embed_task", "json_path": str(json_path)},
    )
    return {
        "embedding": response.get("embedding") or [],
        "dim": int(response.get("dim") or 0),
        "model_name": response.get("model_name"),
        "repo": response.get("repo"),
        "device": response.get("device"),
        "normalized": bool(response.get("normalized", True)),
        "crop": response.get("crop") or {},
        "source": response.get("source") or {},
    }


def _candidate_embedding(socket_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    feature = _json_loads(row.get("feature_json"), {})
    if not isinstance(feature, dict):
        feature = {}
    existing = _dinov2_from_feature(feature)
    if existing is not None:
        return existing

    task_id = str(row.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("identity candidate is missing task_id")
    task_row = get_task_by_task_id(task_id)
    if not task_row:
        raise ValueError(f"identity candidate task does not exist: {task_id}")
    candidate_json_path = resolve_task_json_path_from_record(task_row)
    embedding = _request_embedding(socket_path, candidate_json_path)
    _embedding_array(embedding)

    feature["dinov2"] = embedding
    capture_instance_id = str(row.get("capture_instance_id") or "").strip()
    if capture_instance_id:
        update_capture_instance_feature(capture_instance_id, feature=feature)
    return embedding


def _rewrite_obj_mtl_reference(obj_path: Path, mtl_name: str) -> None:
    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("mtllib "):
            rewritten.append(f"mtllib {mtl_name}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(0, f"mtllib {mtl_name}")
    obj_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _rewrite_mtl_texture_reference(mtl_path: Path, texture_name: str) -> None:
    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("map_Kd "):
            rewritten.append(f"map_Kd {texture_name}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.append(f"map_Kd {texture_name}")
    mtl_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _copy_checked_file(source_path: Path, target_path: Path, label: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ensure_file(source_path, label), target_path)


def _require_historical_model_payload(source_task: dict[str, Any]) -> dict[str, Any]:
    model_payload = source_task.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("historical model scale payload is missing")
    try:
        overall_scale = float(model_payload.get("overall_scale") or 0.0)
    except Exception as exc:
        raise ValueError("historical model overall_scale is invalid") from exc
    if overall_scale <= 0.0:
        raise ValueError("historical model overall_scale is missing")
    return dict(model_payload)


def _copy_historical_model_source(
    *,
    current_task: dict[str, Any],
    source_task: dict[str, Any],
    source_task_row: dict[str, Any],
    display_object_id: str,
) -> dict[str, Any]:
    source = resolve_model_generation_source(source_task, require_mtl_image=True)
    if source.mtl_path is None or source.image_path is None:
        raise ValueError(f"{source.source_stage}.mesh / mtl / image is missing")
    runtime_source = resolve_runtime_mesh_source(source_task, require_mtl_image=True)
    if runtime_source is None or runtime_source.mtl_path is None or runtime_source.image_path is None:
        raise ValueError("historical RuntimeMesh.mesh / mtl / image is missing")
    source_model_payload = _require_historical_model_payload(source_task)

    source_timestamp = str(source_task_row.get("task_timestamp") or source_task.get("task_timestamp") or "").strip()
    if not source_timestamp:
        raise ValueError("source task_timestamp is required for historical model reuse")
    blender_source = source_task.get("Blender") if isinstance(source_task.get("Blender"), dict) else {}
    source_fbx_name = str(blender_source.get("fbx") or "").strip()
    if not source_fbx_name or str(blender_source.get("artifact_root") or "") != "model_result":
        raise ValueError("historical Blender.fbx is missing")
    source_fbx_path = model_result_dir(source_timestamp) / source_fbx_name

    current_timestamp = str(current_task.get("task_timestamp") or "").strip()
    if not current_timestamp:
        raise ValueError("task_timestamp is required for historical model reuse")

    target_obj = model_worker_file(current_timestamp, "model.source_obj")
    target_mtl = model_worker_file(current_timestamp, "model.source_mtl")
    target_texture = model_worker_file(current_timestamp, "model.source_texture")
    _copy_checked_file(source.mesh_path, target_obj, f"{source.source_stage} mesh")
    _copy_checked_file(source.mtl_path, target_mtl, f"{source.source_stage} mtl")
    _copy_checked_file(source.image_path, target_texture, f"{source.source_stage} texture")
    _rewrite_obj_mtl_reference(target_obj, target_mtl.name)
    _rewrite_mtl_texture_reference(target_mtl, target_texture.name)

    target_runtime_obj = model_worker_file(current_timestamp, "model.runtime_obj")
    target_runtime_mtl = model_worker_file(current_timestamp, "model.runtime_mtl")
    target_runtime_texture = model_worker_file(current_timestamp, "model.runtime_texture")
    _copy_checked_file(runtime_source.mesh_path, target_runtime_obj, "historical runtime mesh obj")
    _copy_checked_file(runtime_source.mtl_path, target_runtime_mtl, "historical runtime mesh mtl")
    _copy_checked_file(runtime_source.image_path, target_runtime_texture, "historical runtime mesh texture")
    _rewrite_obj_mtl_reference(target_runtime_obj, target_runtime_mtl.name)
    _rewrite_mtl_texture_reference(target_runtime_mtl, target_runtime_texture.name)

    target_fbx = model_result_file(current_timestamp, "model.final_fbx")
    _copy_checked_file(source_fbx_path, target_fbx, "historical final fbx")

    reuse_info = {
        "source_task_id": source_task_row.get("task_id"),
        "source_task_timestamp": source_timestamp,
        "source_display_object_id": display_object_id,
        "source_mesh": source.mesh,
        "source_mtl": source.mtl,
        "source_image": source.image,
        "source_runtime_mesh": runtime_source.mesh,
        "source_runtime_mtl": runtime_source.mtl,
        "source_runtime_image": runtime_source.image,
        "source_fbx": source_fbx_name,
    }
    payload = build_model_generation_payload(
        backend=source.backend,
        mesh=target_obj.name,
        mtl=target_mtl.name,
        image=target_texture.name,
        artifact_root="model_worker",
        extra={
            "historical_reuse": reuse_info,
        },
    )
    current_task["ModelGeneration"] = payload

    model_payload = dict(source_model_payload)
    model_payload["historical_reuse"] = reuse_info
    current_task["model"] = model_payload

    runtime_payload = dict(runtime_source.payload)
    runtime_payload.update(
        {
            "mesh": target_runtime_obj.name,
            "mtl": target_runtime_mtl.name,
            "image": target_runtime_texture.name,
            "artifact_root": "model_worker",
            "historical_reuse": reuse_info,
        }
    )
    current_task["RuntimeMesh"] = runtime_payload

    blender_payload = dict(blender_source)
    blender_payload.update(
        {
            "fbx": target_fbx.name,
            "artifact_root": "model_result",
            "historical_reuse": reuse_info,
        }
    )
    current_task["Blender"] = blender_payload
    return reuse_info

def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "display_object_id": candidate.get("display_object_id"),
        "capture_instance_id": candidate.get("capture_instance_id"),
        "task_id": candidate.get("task_id"),
        "dinov2_distance": candidate.get("dinov2_distance"),
    }


def run_historical_model_match(json_path: Path) -> dict[str, Any]:
    task = load_task_json(json_path)
    force_new = _truthy(task.get("force_new_3d_model"))
    started = time.perf_counter()
    base_payload: dict[str, Any] = {
        "stage": "historical_model_match",
        "force_new_3d_model": bool(force_new),
        "thresholds": {
            "match_distance": float(DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD),
            "second_margin": float(DINO_IDENTITY_MATCH_SECOND_MARGIN),
            "require_second_margin": bool(DINO_IDENTITY_MATCH_REQUIRE_MARGIN),
        },
    }

    socket_path_text = str(os.environ.get("DINO_IDENTITY_WORKER_SOCKET") or "").strip()
    if not socket_path_text:
        raise RuntimeError("DINO_IDENTITY_WORKER_SOCKET_missing")
    socket_path = Path(socket_path_text)

    try:
        current_embedding = _request_embedding(socket_path, json_path)
    except Exception as exc:
        raise RuntimeError(f"current_embedding_failed:{exc}") from exc
    _embedding_array(current_embedding)

    current_task_id = str(task.get("task_id") or "").strip()
    rows = list_identity_candidate_captures(limit=int(DINO_IDENTITY_CANDIDATE_LIMIT))
    best_by_display: dict[str, dict[str, Any]] = {}
    for row in rows:
        if current_task_id and str(row.get("task_id") or "").strip() == current_task_id:
            continue
        display_object_id = str(row.get("display_object_id") or "").strip()
        if not display_object_id:
            raise ValueError("identity candidate is missing display_object_id")
        candidate_embedding = _candidate_embedding(socket_path, row)
        distance = _cosine_distance(current_embedding, candidate_embedding)
        candidate = {
            "display_object_id": display_object_id,
            "capture_instance_id": row.get("capture_instance_id"),
            "task_id": row.get("task_id"),
            "dinov2_distance": distance,
        }
        old = best_by_display.get(display_object_id)
        if old is None or distance < float(old.get("dinov2_distance") or 999.0):
            best_by_display[display_object_id] = candidate

    candidates = sorted(best_by_display.values(), key=lambda item: float(item.get("dinov2_distance") or 999.0))
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    second_margin_ok = True
    if best is not None and second is not None:
        second_margin_ok = (
            float(second.get("dinov2_distance") or 999.0) - float(best.get("dinov2_distance") or 999.0)
        ) >= float(DINO_IDENTITY_MATCH_SECOND_MARGIN)
    elif best is not None:
        second_margin_ok = True

    payload = {
        **base_payload,
        "status": "miss",
        "reuse_model": False,
        "reason": "no_candidate_below_threshold",
        "candidate_count": len(candidates),
        "candidate_scores": [_public_candidate(item) for item in candidates[:10]],
        "current_dinov2": current_embedding,
    }

    if best is not None:
        best_distance = float(best.get("dinov2_distance") or 999.0)
        payload["dinov2_distance"] = best_distance
        payload["second_margin_ok"] = bool(second_margin_ok)
        margin_required = bool(DINO_IDENTITY_MATCH_REQUIRE_MARGIN)
        if best_distance <= float(DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD) and (second_margin_ok or not margin_required):
            display_object_id = str(best["display_object_id"])
            latest_task = get_latest_completed_task_for_display_object(display_object_id)
            if force_new:
                payload.update(
                    {
                        "status": "matched_force_new",
                        "reuse_model": False,
                        "reason": "force_new_3d_model_generate_new_model_for_matched_display_object",
                        "display_object_id": display_object_id,
                        "selected_capture_instance_id": best.get("capture_instance_id"),
                        "selected_candidate_task_id": best.get("task_id"),
                        "selected_model_task_id": latest_task.get("task_id") if latest_task else None,
                        "selected_model_task_timestamp": latest_task.get("task_timestamp") if latest_task else None,
                        "dinov2_distance": best_distance,
                    }
                )
            elif latest_task is None:
                raise RuntimeError(f"matched display object has no completed model: {display_object_id}")
            else:
                try:
                    source_json_path = resolve_task_json_path_from_record(latest_task)
                    source_task = load_task_json(source_json_path)
                    reuse_info = _copy_historical_model_source(
                        current_task=task,
                        source_task=source_task,
                        source_task_row=latest_task,
                        display_object_id=display_object_id,
                    )
                    payload.update(
                        {
                            "status": "matched",
                            "reuse_model": True,
                            "reason": "dinov2_match_reuse_latest_completed_model",
                            "display_object_id": display_object_id,
                            "selected_capture_instance_id": best.get("capture_instance_id"),
                            "selected_candidate_task_id": best.get("task_id"),
                            "selected_model_task_id": latest_task.get("task_id"),
                            "selected_model_task_timestamp": latest_task.get("task_timestamp"),
                            "dinov2_distance": best_distance,
                            "historical_reuse": reuse_info,
                        }
                    )
                except Exception as exc:
                    raise RuntimeError(f"historical_model_copy_failed:{exc}") from exc
        elif best_distance <= float(DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD):
            raise RuntimeError("identity match is ambiguous: second candidate is too close")

    payload["duration_ms"] = (time.perf_counter() - started) * 1000.0
    task["HistoricalModelMatch"] = payload
    save_task_json(json_path, task)
    return payload

def main(argv: list[str]) -> int:
    try:
        json_path, _task = load_stage_task(
            argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_historical_model_match_from_json.py <task_meta.json or filename>",
            stage_name="historical_model_match",
        )
        result = run_historical_model_match(json_path)
        print(
            "[INFO] historical_model_match : "
            f"status={result.get('status')} reuse={result.get('reuse_model')} reason={result.get('reason')}"
        )
        if result.get("reuse_model"):
            print(
                "[INFO] historical_model_match : "
                f"display_object_id={result.get('display_object_id')} "
                f"model_task_id={result.get('selected_model_task_id')} "
                f"distance={result.get('dinov2_distance')}"
            )
        print("[OK] historical_model_match")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
