"""Parallel contact-evidence and body-mesh branch for Shigure events."""

from __future__ import annotations

import base64
import copy
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from artifact_layout import model_result_file, model_worker_dir
from config import SHIGURE_AUXILIARY_CONTACT_WAIT_SEC, SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD
from stages.shigure_history.shigure_identity import cosine_distance
from stages.shigure_history.cache import CachedRgbdSample, CachedShigureEvent, RosStamp, ShigureRgbdCache
from stages.shigure_history.marker_history import latest_marker_pose_path
from task_db import upsert_auxiliary_job
from task_json import load_task_json, save_task_json


DinoRequest = Callable[[dict[str, Any]], dict[str, Any]]
BodyRunner = Callable[[Path], None]
CompletionCallback = Callable[[str, Path], None]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _publish_taken_artifacts(task_timestamp: str, backup_dir: Path) -> dict[str, Any]:
    mappings = {
        "rgb.png": ("result_rgb", model_result_file(task_timestamp, "taken.result_rgb")),
        "depth.png": ("result_depth", model_result_file(task_timestamp, "taken.result_depth")),
        "object_mask.png": ("object_mask", model_result_file(task_timestamp, "taken.object_mask")),
        "camera_info.json": ("camera_info", model_result_file(task_timestamp, "taken.camera_info")),
        "object_detection.json": ("object_detection", model_result_file(task_timestamp, "taken.object_detection")),
        "marker_6d_pose.json": ("marker_pose", model_result_file(task_timestamp, "taken.marker_pose")),
    }
    published: dict[str, Any] = {"artifact_root": "model_result"}
    for source_name, (payload_key, destination) in mappings.items():
        source = backup_dir / source_name
        if not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        published[payload_key] = destination.name
    return published


def _watch_event_key(event: CachedShigureEvent) -> str:
    return (
        f"{int(event.source_stamp.sec):010d}_{int(event.source_stamp.nanosec):09d}_"
        f"{int(event.sequence):010d}"
    )


def _depth_for_png(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth)
    if not np.issubdtype(value.dtype, np.floating):
        return value
    finite = value[np.isfinite(value) & (value > 0)]
    scale = 1000.0 if finite.size and float(np.median(finite)) <= 20.0 else 1.0
    return np.clip(np.rint(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * scale), 0, 65535).astype(
        np.uint16
    )


def _freeze_watched_event(
    branch_root: Path,
    event: CachedShigureEvent,
    sample: CachedRgbdSample | None,
) -> Path:
    """Durably freeze sparse take-out evidence before the 60 s RGB-D cache evicts it."""

    event_dir = branch_root / "watched_events" / _watch_event_key(event)
    event_dir.mkdir(parents=True, exist_ok=True)
    _write_json(event_dir / "event.json", event.to_dict(include_masks=True))
    if sample is not None and sample.rgb_bgr.size:
        cv2.imwrite(str(event_dir / "rgb.png"), sample.rgb_bgr)
        cv2.imwrite(str(event_dir / "depth.png"), _depth_for_png(sample.depth))
        _write_json(event_dir / "camera_info.json", sample.camera_info)
    return event_dir


def _load_watched_events(
    branch_root: Path,
) -> dict[str, tuple[CachedShigureEvent, CachedRgbdSample | None]]:
    buffered: dict[str, tuple[CachedShigureEvent, CachedRgbdSample | None]] = {}
    watched_root = branch_root / "watched_events"
    if not watched_root.is_dir():
        return buffered
    for event_path in sorted(watched_root.glob("*/event.json")):
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            stamp_payload = payload["source_stamp"]
            contacted = payload["contacted"]
            object_detection = payload["object_detection"]
            matches = payload["contact_object_matches"]
            if contacted is not None and not isinstance(contacted, dict):
                raise ValueError("contacted must be an object or null")
            if object_detection is not None and not isinstance(object_detection, dict):
                raise ValueError("object_detection must be an object or null")
            if not isinstance(matches, list):
                raise ValueError("contact_object_matches must be an array")
            stamp = RosStamp.from_dict(stamp_payload)
            event = CachedShigureEvent(
                source_stamp=stamp,
                received_utc=str(payload["received_utc"]),
                received_monotonic=float(payload["received_monotonic"]),
                contacted_state=str(payload["contacted_state"]),
                object_detection_state=str(payload["object_detection_state"]),
                contacted=contacted,
                object_detection=object_detection,
                contact_object_matches=matches,
                sequence=int(payload["sequence"]),
            )
            event_dir = event_path.parent
            rgb_path = event_dir / "rgb.png"
            depth_path = event_dir / "depth.png"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path.is_file() else None
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
            sample: CachedRgbdSample | None = None
            if rgb is not None and depth is not None:
                camera_path = event_dir / "camera_info.json"
                camera_info = (
                    json.loads(camera_path.read_text(encoding="utf-8"))
                    if camera_path.is_file()
                    else {}
                )
                sample = CachedRgbdSample(
                    stamp=stamp,
                    rgb_bgr=rgb,
                    depth=depth,
                    camera_info=camera_info,
                )
            buffered[_watch_event_key(event)] = (event, sample)
        except Exception as exc:
            print(f"[auxiliary] ignored invalid frozen Shigure event {event_path}: {exc}")
    return buffered


def _decode_mask(value: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(str(value or ""))
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Shigure contact mask decode failed")
    if image.ndim == 3:
        image = image[:, :, 0]
    mask = image > 0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def _current_embedding(task: Mapping[str, Any]) -> list[float] | None:
    historical = task.get("HistoricalModelMatch") if isinstance(task.get("HistoricalModelMatch"), Mapping) else {}
    dino = historical.get("current_dinov2") if isinstance(historical.get("current_dinov2"), Mapping) else {}
    embedding = dino.get("embedding")
    return [float(value) for value in embedding] if isinstance(embedding, list) and embedding else None


def _find_detection(event: CachedShigureEvent, detection_id: str) -> dict[str, Any] | None:
    for item in ((event.object_detection or {}).get("objects") or []):
        if isinstance(item, dict) and str(item.get("object_id") or "") == detection_id:
            return item
    return None


class ShigureAuxiliaryBranchManager:
    """Run contact/taken and body processing beside the main model chain."""

    def __init__(
        self,
        *,
        dino_request: DinoRequest,
        body_runner: BodyRunner,
        on_complete: CompletionCallback | None = None,
    ) -> None:
        self.dino_request = dino_request
        self.body_runner = body_runner
        self.on_complete = on_complete
        self.cache = ShigureRgbdCache()
        self._lock = threading.RLock()
        self._running: set[str] = set()
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(str(task_id))
            return bool(str(task_id) in self._running and thread is not None and thread.is_alive())

    def stop(self, *, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._lock:
            threads = list(self._threads.values())
        deadline = time.monotonic() + max(0.0, float(timeout))
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def start(self, *, task_id: str, json_path: Path, start_event_sequence: int) -> None:
        with self._lock:
            existing = self._threads.get(task_id)
            if task_id in self._running and existing is not None and existing.is_alive():
                return
            self._stop.clear()
            self._running.discard(task_id)
            self._running.add(task_id)
        thread = threading.Thread(
            target=self._run_guarded,
            kwargs={
                "task_id": task_id,
                "json_path": json_path,
                "start_event_sequence": int(start_event_sequence),
            },
            daemon=True,
            name=f"shigure-aux-{task_id[:8]}",
        )
        with self._lock:
            self._threads[task_id] = thread
        thread.start()

    def _run_guarded(self, *, task_id: str, json_path: Path, start_event_sequence: int) -> None:
        try:
            self._run(task_id=task_id, json_path=json_path, start_event_sequence=start_event_sequence)
        except Exception as exc:
            upsert_auxiliary_job(
                task_id=task_id,
                branch_name="shigure_contact_body",
                status="failed",
                detail={"start_event_sequence": int(start_event_sequence)},
                error_message=str(exc),
            )
            print(f"[auxiliary] Shigure contact/body branch failed for {task_id}: {exc}")
        finally:
            with self._lock:
                self._running.discard(task_id)
                self._threads.pop(task_id, None)

    def _identity_distance(
        self,
        *,
        branch_root: Path,
        event: CachedShigureEvent,
        detection: Mapping[str, Any],
        sample: Any,
        mask: np.ndarray,
        reference_embedding: list[float],
    ) -> float:
        identity_dir = branch_root / f"identity_{int(event.sequence):010d}"
        identity_dir.mkdir(parents=True, exist_ok=True)
        color_path = identity_dir / "color.png"
        mask_path = identity_dir / "mask.png"
        cv2.imwrite(str(color_path), sample.rgb_bgr)
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        embedded = self.dino_request(
            {"action": "embed_files", "color_file": str(color_path), "mask_file": str(mask_path)}
        )
        return cosine_distance(reference_embedding, embedded.get("embedding") or [])

    def _run(self, *, task_id: str, json_path: Path, start_event_sequence: int) -> None:
        task = load_task_json(json_path)
        task_timestamp = str(task.get("task_timestamp") or "").strip()
        if not task_timestamp:
            raise ValueError("task_timestamp is required")
        reference_embedding = _current_embedding(task)
        branch_root = model_worker_dir(task_timestamp) / "auxiliary_shigure_contact_body"
        branch_root.mkdir(parents=True, exist_ok=True)
        branch_json_path = branch_root / "task.json"
        upsert_auxiliary_job(
            task_id=task_id,
            branch_name="shigure_contact_body",
            status="running",
            result_path=branch_json_path,
            detail={"start_event_sequence": int(start_event_sequence)},
        )

        deadline = time.monotonic() + max(1.0, float(SHIGURE_AUXILIARY_CONTACT_WAIT_SEC))
        cursor = max(0, int(start_event_sequence))
        selected: tuple[CachedShigureEvent, dict[str, Any], dict[str, Any], Any, np.ndarray, float] | None = None
        terminal_status: str | None = None
        terminal_reason: str | None = None
        terminal_event: CachedShigureEvent | None = None
        terminal_detail: dict[str, Any] = {}
        unresolved_source_stamps: set[tuple[int, int]] = set()
        saw_ambiguous_contact_match = False
        buffered_events = _load_watched_events(branch_root)
        while not self._stop.is_set() and time.monotonic() < deadline and selected is None and terminal_status is None:
            if reference_embedding is None:
                try:
                    task = load_task_json(json_path)
                    reference_embedding = _current_embedding(task)
                except Exception:
                    reference_embedding = None

            try:
                cache_status = self.cache.status() or {}
                if int(cache_status.get("latest_event_sequence") or 0) < cursor:
                    # The recorder is process-local and starts its event
                    # sequence again after a server/recorder restart.
                    cursor = 0
                updates = list(self.cache.iter_event_updates_after(cursor, include_masks=True))
            except Exception as exc:
                print(f"[auxiliary] waiting for Shigure event socket: {exc}")
                time.sleep(0.25)
                continue

            for event in updates:
                cursor = max(cursor, int(event.sequence))
                objects = [item for item in ((event.object_detection or {}).get("objects") or []) if isinstance(item, dict)]
                contacts = [item for item in ((event.contacted or {}).get("contacts") or []) if isinstance(item, dict)]
                has_take_out_detection = any(str(item.get("action") or "").strip().lower() == "take_out" for item in objects)
                has_take_out_contact = any(str(item.get("action") or "").strip().lower() == "take_out" for item in contacts)
                if not has_take_out_detection and not has_take_out_contact:
                    continue

                sample = self.cache.get_sample(event.source_stamp)
                _freeze_watched_event(branch_root, event, sample)
                buffered_events[_watch_event_key(event)] = (event, sample)

            if reference_embedding is None:
                time.sleep(0.25)
                continue

            processing = sorted(
                buffered_events.values(),
                key=lambda item: (
                    int(item[0].source_stamp.sec),
                    int(item[0].source_stamp.nanosec),
                    int(item[0].sequence),
                ),
            )
            buffered_events.clear()
            for event, frozen_sample in processing:
                source_key = (int(event.source_stamp.sec), int(event.source_stamp.nanosec))
                objects = [item for item in ((event.object_detection or {}).get("objects") or []) if isinstance(item, dict)]
                contacts = [item for item in ((event.contacted or {}).get("contacts") or []) if isinstance(item, dict)]
                has_take_out_detection = any(str(item.get("action") or "").strip().lower() == "take_out" for item in objects)
                has_take_out_contact = any(str(item.get("action") or "").strip().lower() == "take_out" for item in contacts)

                if event.contacted_state == "missing" or event.object_detection_state == "missing":
                    unresolved_source_stamps.add(source_key)
                    continue
                unresolved_source_stamps.discard(source_key)
                if has_take_out_detection and event.contacted_state == "explicit_empty":
                    sample = frozen_sample or self.cache.get_sample(event.source_stamp)
                    if sample is None or sample.rgb_bgr.size == 0:
                        unresolved_source_stamps.add(source_key)
                        continue
                    for detection in objects:
                        if str(detection.get("action") or "").strip().lower() != "take_out":
                            continue
                        mask = _decode_mask(str(detection.get("mask_b64") or ""), sample.rgb_bgr.shape[:2])
                        distance = self._identity_distance(
                            branch_root=branch_root,
                            event=event,
                            detection=detection,
                            sample=sample,
                            mask=mask,
                            reference_embedding=reference_embedding,
                        )
                        if distance <= float(SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD):
                            terminal_status = "WRONG_NO_CONTACTED_PERSON"
                            terminal_reason = "explicit_empty_contacted_list_for_matching_take_out"
                            terminal_event = event
                            terminal_detail = {
                                "identity_distance": float(distance),
                                "identity_threshold": float(SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD),
                                "object_detection": {
                                    key: value for key, value in detection.items() if key != "mask_b64"
                                },
                            }
                            break
                    if terminal_status is not None:
                        break
                    continue
                if has_take_out_contact and event.object_detection_state == "explicit_empty":
                    terminal_status = "INPUT_MISSING"
                    terminal_reason = "take_out_contact_has_no_object_detection_mask"
                    terminal_event = event
                    break
                if event.contacted_state != "present" or event.object_detection_state != "present":
                    continue
                for match in event.contact_object_matches or []:
                    contact_index = int(match.get("contact_index") or 0)
                    if not (0 <= contact_index < len(contacts)):
                        continue
                    contact = contacts[contact_index]
                    if str(contact.get("action") or "").strip().lower() != "take_out":
                        continue
                    if str(match.get("status") or "") != "matched_action_iou":
                        saw_ambiguous_contact_match = True
                        continue
                    best = match.get("best") if isinstance(match.get("best"), dict) else {}
                    detection = _find_detection(event, str(best.get("object_detection_id") or ""))
                    if detection is None:
                        continue
                    sample = frozen_sample or self.cache.get_sample(event.source_stamp)
                    if sample is None or sample.rgb_bgr.size == 0:
                        continue
                    mask = _decode_mask(str(detection.get("mask_b64") or ""), sample.rgb_bgr.shape[:2])
                    distance = self._identity_distance(
                        branch_root=branch_root,
                        event=event,
                        detection=detection,
                        sample=sample,
                        mask=mask,
                        reference_embedding=reference_embedding,
                    )
                    if distance <= float(SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD):
                        selected = (event, contact, detection, sample, mask, distance)
                        break
                if selected is not None:
                    break
            if selected is None and terminal_status is None:
                time.sleep(0.25)

        if self._stop.is_set() and selected is None and terminal_status is None:
            upsert_auxiliary_job(
                task_id=task_id,
                branch_name="shigure_contact_body",
                status="pending",
                result_path=branch_json_path,
                detail={
                    "start_event_sequence": int(start_event_sequence),
                    "last_event_sequence": int(cursor),
                    "reason": "server_shutdown",
                },
            )
            return

        branch_task = copy.deepcopy(task)
        branch_task["AuxiliaryBranch"] = {
            "branch_name": "shigure_contact_body",
            "start_event_sequence": int(start_event_sequence),
            "last_event_sequence": int(cursor),
            "source": "remote_shigure_contacted",
        }
        if selected is None:
            if terminal_status is None and reference_embedding is None:
                terminal_status = "INPUT_MISSING"
                terminal_reason = "hololens_identity_embedding_unavailable_before_timeout"
            if terminal_status is None and unresolved_source_stamps:
                terminal_status = "INPUT_MISSING"
                terminal_reason = "take_out_event_join_incomplete_before_timeout"
            if terminal_status is None and saw_ambiguous_contact_match:
                terminal_status = "WRONG_AMBIGUOUS_OBJECT"
                terminal_reason = "take_out_contact_mask_association_ambiguous"
            result_status = terminal_status or "NOT_TAKEN"
            result_reason = terminal_reason or "no_matching_shigure_take_out_before_timeout"
            branch_task["ShigureContactEvidence"] = {
                "status": result_status,
                "reason": result_reason,
                "source": "remote_shigure_contacted",
                "result_timestamp": terminal_event.source_stamp.to_dict() if terminal_event is not None else None,
                "event_sequence": int(terminal_event.sequence) if terminal_event is not None else None,
                **terminal_detail,
            }
            save_task_json(branch_json_path, branch_task)
            upsert_auxiliary_job(
                task_id=task_id,
                branch_name="shigure_contact_body",
                status="completed",
                result_path=branch_json_path,
                detail={"taken_status": result_status, "last_event_sequence": int(cursor)},
            )
            return

        event, contact, detection, sample, mask, distance = selected
        backup_dir = branch_root / f"snapshot_{int(event.sequence):010d}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(backup_dir / "rgb.png"), sample.rgb_bgr)
        depth = _depth_for_png(sample.depth)
        cv2.imwrite(str(backup_dir / "depth.png"), depth)
        cv2.imwrite(str(backup_dir / "object_mask.png"), mask.astype(np.uint8) * 255)
        (backup_dir / "camera_info.json").write_text(
            json.dumps(sample.camera_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_json(
            backup_dir / "object_detection.json",
            event.to_dict(include_masks=False).get("object_detection") or {},
        )
        marker_path = latest_marker_pose_path()
        if marker_path is not None:
            shutil.copy2(marker_path, backup_dir / "marker_6d_pose.json")

        published_artifacts = _publish_taken_artifacts(task_timestamp, backup_dir)

        taken_payload = {
            "status": "TAKEN",
            "source": "remote_shigure_contacted",
            "source_action": "take_out",
            "result_timestamp": event.source_stamp.to_dict(),
            "event_sequence": int(event.sequence),
            "backup_shigurei_dir": str(backup_dir),
            **published_artifacts,
            "people_bounding_box": contact.get("people_bounding_box"),
            "object_bounding_box": contact.get("object_bounding_box"),
            "shigure_object_id": contact.get("object_id"),
            "people_id": contact.get("people_id"),
            "event_id": contact.get("event_id"),
            "identity_distance": float(distance),
            "identity_threshold": float(SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD),
            "object_detection": {
                key: value for key, value in detection.items() if key != "mask_b64"
            },
        }
        branch_task["ShigureContactEvidence"] = taken_payload
        _write_json(model_result_file(task_timestamp, "taken.result"), taken_payload)
        save_task_json(branch_json_path, branch_task)
        if self._stop.is_set():
            upsert_auxiliary_job(
                task_id=task_id,
                branch_name="shigure_contact_body",
                status="pending",
                result_path=branch_json_path,
                detail={"start_event_sequence": int(start_event_sequence), "last_event_sequence": int(cursor)},
            )
            return
        self.body_runner(branch_json_path)
        completed_task = load_task_json(branch_json_path)
        body = completed_task.get("SAM3DBodyMesh") if isinstance(completed_task.get("SAM3DBodyMesh"), dict) else {}
        upsert_auxiliary_job(
            task_id=task_id,
            branch_name="shigure_contact_body",
            status="completed",
            result_path=branch_json_path,
            detail={
                "taken_status": "TAKEN",
                "body_status": body.get("status"),
                "event_sequence": int(event.sequence),
                "identity_distance": float(distance),
            },
        )
        if self.on_complete is not None:
            self.on_complete(task_id, branch_json_path)
