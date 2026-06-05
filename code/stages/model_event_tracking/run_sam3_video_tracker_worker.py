from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from config import SAM3_ROOT
except Exception:
    SAM3_ROOT = CODE_ROOT / "reconstruction" / "sam3"

sam3_root_str = str(Path(SAM3_ROOT).expanduser().resolve())
if sam3_root_str not in sys.path:
    sys.path.insert(0, sam3_root_str)

os.environ.setdefault("SAM3_DISABLE_TRITON_NMS", "1")
os.environ.setdefault("SAM3_DISABLE_TRITON_CONNECTED_COMPONENTS", "1")

import numpy as np
from PIL import Image

from stages.model_event_tracking import settings
from stages.model_event_tracking.event_store import persist_taken_away_event
from stages.model_event_tracking.movement import MaskDepthMovementTracker
from stages.model_event_tracking.people import choose_event_start_contact, find_hand_contacts
from stages.model_event_tracking.schemas import ProjectedBox, RosStamp, ShigureFrame
from stages.model_event_tracking.tracker import read_depth_image_m


class Sam3VideoWorker:
    def __init__(self) -> None:
        self.predictor = None

    def _build_predictor(self):
        if self.predictor is not None:
            return self.predictor
        import torch
        from sam3.model_builder import build_sam3_video_predictor

        compile_model = str(os.environ.get("SAM3_VIDEO_TRACKER_COMPILE", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        self.predictor = build_sam3_video_predictor(
            compile=compile_model,
            async_loading_frames=False,
        )
        return self.predictor

    def track_video(self, request: dict[str, Any]) -> dict[str, Any]:
        task_id = str(request.get("task_id") or uuid.uuid4())
        task_output_name = str(request.get("task_output_name") or task_id)
        frame_payloads = [dict(value) for value in request.get("frames") or [] if isinstance(value, dict)]
        if not frame_payloads:
            raise ValueError("frames is empty")
        frames = [self._frame_from_payload(value) for value in frame_payloads]
        for frame in frames:
            if frame.rgb_path is None or not frame.rgb_path.is_file():
                raise FileNotFoundError(str(frame.rgb_path))
            if frame.depth_path is None or not frame.depth_path.is_file():
                raise FileNotFoundError(str(frame.depth_path))

        output_dir = Path(request.get("output_dir") or f"/tmp/sam3_video_{task_id}")
        mask_dir = output_dir / "masks"
        image_width, image_height = self._resolve_image_size(request, frames[0].rgb_path)
        box_xywh = self._normalized_box_xywh(request.get("box_xyxy"), image_width, image_height)
        camera_matrix = np.asarray(request.get("camera_matrix"), dtype=np.float64).reshape(3, 3)
        projected_box = self._projected_box_from_payload(request.get("projected_box") or {})
        movement_tracker = MaskDepthMovementTracker()
        contact_frames = self._contact_frames(request.get("contact_frames") or [])
        contact_index = 0

        predictor = self._build_predictor()
        session_id = None
        masks: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        event_record = None
        last_mask: np.ndarray | None = None
        tracked_obj_id: int | None = None
        processed_indices: set[int] = set()
        diagnostics: dict[str, Any] = {
            "temporary_frame_cache": True,
            "prompt_saved": False,
            "processed_frame_count": 0,
            "propagation_attempts": 0,
            "retry_prompt_attempts": 0,
            "retry_prompt_saved": 0,
            "stream_saved_masks": 0,
            "empty_binary_mask_frames": 0,
            "missing_binary_mask_frames": 0,
            "stopped_after_taken_away": False,
        }

        def advance_contacts(until_seconds: float) -> None:
            nonlocal contact_index
            while contact_index < len(contact_frames):
                stamp, people_path = contact_frames[contact_index]
                if stamp.seconds > until_seconds:
                    break
                contact_index += 1
                if people_path is None or not people_path.is_file():
                    continue
                contacts = find_hand_contacts(people_path, projected_box)
                movement_tracker.note_hand_contact(choose_event_start_contact(contacts))

        def process_response(response: dict[str, Any], source: str) -> bool:
            nonlocal event_record, last_mask, tracked_obj_id
            frame_index = int(response.get("frame_index", -1))
            if frame_index < 0 or frame_index >= len(frames):
                return False
            mask_data = self._response_mask(
                response,
                diagnostics,
                frame=frames[frame_index],
                projected_box=projected_box,
                camera_matrix=camera_matrix,
                source=source,
                tracked_obj_id=tracked_obj_id,
            )
            if mask_data is None:
                return False
            mask, obj_id, score = mask_data
            if tracked_obj_id is None or source == "retry_prompt":
                tracked_obj_id = obj_id
            advance_contacts(frames[frame_index].stamp.seconds)
            frame = frames[frame_index]
            hand_contact = None
            if frame.people_path is not None and frame.people_path.is_file():
                hand_contact = choose_event_start_contact(find_hand_contacts(frame.people_path, projected_box))
            decision = movement_tracker.update(
                mask,
                read_depth_image_m(frame.depth_path),
                camera_matrix,
                timestamp=frame.stamp,
                hand_contact=hand_contact,
            )
            mask_path = self._save_mask(mask, mask_dir / f"{frame_index:06d}.png")
            if frame_index not in processed_indices:
                masks.append(
                    {
                        "frame_index": frame_index,
                        "obj_id": obj_id,
                        "score": score,
                        "mask_path": str(mask_path),
                        "source": source,
                    }
                )
                decisions.append(decision.to_dict())
                processed_indices.add(frame_index)
            last_mask = mask
            diagnostics["processed_frame_count"] = len(processed_indices)
            if source == "prompt":
                diagnostics["prompt_saved"] = True
            elif source == "retry_prompt":
                diagnostics["retry_prompt_saved"] += 1
            else:
                diagnostics["stream_saved_masks"] += 1

            if decision.moved and decision.should_stop_tracking:
                event_record = persist_taken_away_event(
                    task_id=task_id,
                    frame=frame,
                    decision=decision,
                    hand_contact=hand_contact,
                    mask=mask if mask is not None else last_mask,
                    projected_box=projected_box.to_dict(),
                    task_output_name=task_output_name,
                    replace=True,
                )
                diagnostics["stopped_after_taken_away"] = True
                return True
            return False

        temp_root = str(os.environ.get("SAM3_VIDEO_TMP_ROOT") or "/tmp")
        Path(temp_root).mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"sam3video_{task_id}_", dir=temp_root) as tmp_name:
                frame_dir = Path(tmp_name) / "frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                self._prepare_jpeg_frame_dir([frame.rgb_path for frame in frames], frame_dir)
                response = predictor.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": str(frame_dir),
                        "offload_video_to_cpu": bool(request.get("offload_video_to_cpu", True)),
                        "offload_state_to_cpu": bool(request.get("offload_state_to_cpu", False)),
                    }
                )
                session_id = response["session_id"]
                prompt_response = self._add_box_prompt(predictor, session_id, 0, box_xywh)
                if process_response(prompt_response, "prompt"):
                    pass
                else:
                    for frame_index in range(1, len(frames)):
                        diagnostics["propagation_attempts"] += 1
                        stream_request = {
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": "forward",
                            "start_frame_index": frame_index,
                            "max_frame_num_to_track": 1,
                        }
                        response_for_frame = None
                        for item in predictor.handle_stream_request(stream_request):
                            if int(item.get("frame_index", -1)) == frame_index:
                                response_for_frame = item
                                break
                            response_for_frame = item
                        stopped = False
                        if response_for_frame is not None:
                            stopped = process_response(response_for_frame, "propagation")
                        if stopped:
                            break
                        if frame_index in processed_indices:
                            continue
                        if bool(request.get("retry_box_prompt_on_empty", True)):
                            diagnostics["retry_prompt_attempts"] += 1
                            retry_box_xywh = (
                                self._normalized_mask_box_xywh(last_mask, image_width, image_height)
                                if last_mask is not None
                                else box_xywh
                            )
                            retry_response = self._add_box_prompt(
                                predictor,
                                session_id,
                                frame_index,
                                retry_box_xywh,
                            )
                            if process_response(retry_response, "retry_prompt"):
                                break
        finally:
            if session_id is not None:
                try:
                    predictor.handle_request(
                        {
                            "type": "close_session",
                            "session_id": session_id,
                            "run_gc_collect": True,
                        }
                    )
                except Exception:
                    traceback.print_exc()

        if event_record is not None:
            status = "taken_away"
        elif not masks:
            status = "no_masks"
        elif len(processed_indices) <= 1:
            status = "target_lost"
        else:
            status = "no_event"
        return {
            "status": status,
            "task_id": task_id,
            "session_id": session_id,
            "frame_count": len(frames),
            "mask_count": len(masks),
            "masks": masks,
            "decisions": decisions,
            "event_record": event_record.to_dict() if event_record is not None else None,
            "diagnostics": diagnostics,
            "output_dir": str(output_dir),
        }

    def shutdown(self) -> None:
        if self.predictor is not None:
            try:
                self.predictor.shutdown()
            except Exception:
                pass
            self.predictor = None

    @staticmethod
    def _frame_from_payload(payload: dict[str, Any]) -> ShigureFrame:
        stamp_payload = dict(payload.get("stamp") or {})
        stamp = RosStamp(sec=int(stamp_payload.get("sec", 0)), nanosec=int(stamp_payload.get("nanosec", 0)))

        def optional_path(key: str) -> Path | None:
            value = payload.get(key)
            return Path(value) if value else None

        return ShigureFrame(
            stamp=stamp,
            rgb_path=optional_path("rgb_path"),
            depth_path=optional_path("depth_path"),
            camera_info_path=optional_path("camera_info_path"),
            people_path=optional_path("people_path"),
            marker_pose_path=optional_path("marker_pose_path"),
        )

    @staticmethod
    def _contact_frames(payloads: list[Any]) -> list[tuple[RosStamp, Path | None]]:
        result: list[tuple[RosStamp, Path | None]] = []
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            stamp_payload = dict(payload.get("stamp") or {})
            stamp = RosStamp(sec=int(stamp_payload.get("sec", 0)), nanosec=int(stamp_payload.get("nanosec", 0)))
            people_value = payload.get("people_path")
            result.append((stamp, Path(people_value) if people_value else None))
        result.sort(key=lambda item: item[0].seconds)
        return result

    @staticmethod
    def _projected_box_from_payload(payload: dict[str, Any]) -> ProjectedBox:
        return ProjectedBox(
            corners_camera_m=np.asarray(payload.get("corners_camera_m"), dtype=np.float64),
            pixel_points=np.asarray(payload.get("pixel_points"), dtype=np.float64),
            bbox_xyxy=tuple(float(value) for value in payload.get("bbox_xyxy") or (0, 0, 1, 1)),
            coordinate_system=str(payload.get("coordinate_system") or "opencv_camera"),
        )

    @staticmethod
    def _add_box_prompt(predictor: Any, session_id: str, frame_index: int, box_xywh: list[float]) -> dict[str, Any]:
        return predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(frame_index),
                "bounding_boxes": [box_xywh],
                "bounding_box_labels": [1],
                "rel_coordinates": True,
            }
        )

    @staticmethod
    def _resolve_image_size(request: dict[str, Any], first_frame: Path) -> tuple[int, int]:
        value = request.get("image_size")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        with Image.open(first_frame) as image:
            return image.size

    @staticmethod
    def _prepare_jpeg_frame_dir(frame_paths: list[Path | None], frame_dir: Path) -> None:
        for index, source in enumerate(frame_paths):
            if source is None:
                raise ValueError(f"frame {index} is missing rgb_path")
            target = frame_dir / f"{index:06d}.jpg"
            with Image.open(source) as image:
                image.convert("RGB").save(target, quality=90, optimize=True)

    @staticmethod
    def _normalized_box_xywh(box_xyxy: Any, image_width: int, image_height: int) -> list[float]:
        if box_xyxy is None:
            raise ValueError("box_xyxy is required")
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
        if box.size < 4:
            raise ValueError("box_xyxy must have four values")
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        width = max(1.0, float(image_width))
        height = max(1.0, float(image_height))
        x0 = min(max(x0, 0.0), width - 1.0)
        x1 = min(max(x1, x0 + 1.0), width)
        y0 = min(max(y0, 0.0), height - 1.0)
        y1 = min(max(y1, y0 + 1.0), height)
        return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]

    @staticmethod
    def _normalized_mask_box_xywh(mask: np.ndarray, image_width: int, image_height: int) -> list[float]:
        mask_bool = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0:
            return [0.0, 0.0, 1.0, 1.0]
        span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        padding = max(4.0, float(span) * 0.15)
        box_xyxy = (
            float(xs.min()) - padding,
            float(ys.min()) - padding,
            float(xs.max() + 1) + padding,
            float(ys.max() + 1) + padding,
        )
        return Sam3VideoWorker._normalized_box_xywh(box_xyxy, image_width, image_height)

    @staticmethod
    def _response_mask(
        response: dict[str, Any],
        diagnostics: dict[str, Any],
        *,
        frame: ShigureFrame,
        projected_box: ProjectedBox,
        camera_matrix: np.ndarray,
        source: str,
        tracked_obj_id: int | None,
    ) -> tuple[np.ndarray, int, float | None] | None:
        outputs = response.get("outputs") or {}
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")
        scores = outputs.get("out_scores")
        if scores is None:
            scores = outputs.get("out_obj_scores")
        if binary_masks is None:
            diagnostics["missing_binary_mask_frames"] += 1
            return None

        import torch

        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.detach().cpu().numpy()
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        binary_masks = np.asarray(binary_masks)
        while binary_masks.ndim > 3 and 1 in binary_masks.shape:
            binary_masks = np.squeeze(binary_masks)
        if binary_masks.ndim == 2:
            binary_masks = binary_masks[None, ...]
        if binary_masks.size == 0 or binary_masks.ndim != 3:
            diagnostics["empty_binary_mask_frames"] += 1
            return None

        depth_m = read_depth_image_m(frame.depth_path)
        height, width = depth_m.shape[:2]
        x0, y0, x1, y1 = projected_box.bbox_xyxy
        ix0 = min(max(int(np.floor(x0)), 0), max(0, width - 1))
        iy0 = min(max(int(np.floor(y0)), 0), max(0, height - 1))
        ix1 = min(max(int(np.ceil(x1)), ix0 + 1), width)
        iy1 = min(max(int(np.ceil(y1)), iy0 + 1), height)
        box_area = max(1, (ix1 - ix0) * (iy1 - iy0))
        corners_z = np.asarray(projected_box.corners_camera_m, dtype=np.float64)[:, 2]
        model_min_depth = float(np.nanmin(corners_z)) - settings.OCCLUSION_FRONT_MARGIN_M
        model_max_depth = float(np.nanmax(corners_z)) + settings.OCCLUSION_MODEL_DEPTH_BAND_M
        expected_center = np.asarray(projected_box.center_camera_m, dtype=np.float64)
        expected_u = camera_matrix[0, 0] * expected_center[0] / expected_center[2] + camera_matrix[0, 2]
        expected_v = camera_matrix[1, 1] * expected_center[1] / expected_center[2] + camera_matrix[1, 2]
        box_diag = max(1.0, float(np.hypot(ix1 - ix0, iy1 - iy0)))

        obj_id_values = np.asarray(obj_ids).reshape(-1)
        score_values = np.asarray(scores).reshape(-1) if scores is not None else np.asarray([])
        candidates: list[dict[str, Any]] = []
        for index, raw_mask in enumerate(binary_masks):
            mask = np.asarray(raw_mask)
            while mask.ndim > 2 and 1 in mask.shape:
                mask = np.squeeze(mask)
            if mask.ndim != 2:
                continue
            if mask.shape != depth_m.shape:
                mask = np.asarray(Image.fromarray((mask > 0).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST))
            mask_bool = mask > 0
            area = int(np.count_nonzero(mask_bool))
            if area <= 0:
                continue
            inside_pixels = int(np.count_nonzero(mask_bool[iy0:iy1, ix0:ix1]))
            inside_ratio = float(inside_pixels) / float(area)
            box_coverage = float(inside_pixels) / float(box_area)
            ys, xs = np.nonzero(mask_bool)
            centroid_u = float(np.mean(xs))
            centroid_v = float(np.mean(ys))
            pixel_center_distance = float(np.hypot(centroid_u - expected_u, centroid_v - expected_v)) / box_diag

            valid = mask_bool & np.isfinite(depth_m) & (depth_m > 0.0)
            valid_depth = depth_m[valid]
            model_depth_ratio = 0.0
            center_distance_m = None
            median_depth_m = None
            if valid_depth.size:
                median_depth_m = float(np.nanmedian(valid_depth))
                model_depth_ratio = float(
                    np.count_nonzero((valid_depth >= model_min_depth) & (valid_depth <= model_max_depth))
                ) / float(valid_depth.size)
                valid_ys, valid_xs = np.nonzero(valid)
                z = depth_m[valid_ys, valid_xs]
                x = (valid_xs.astype(np.float64) - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
                y = (valid_ys.astype(np.float64) - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
                center = np.nanmedian(np.stack([x, y, z], axis=1), axis=0)
                center_distance_m = float(np.linalg.norm(center - expected_center))

            candidate_obj_id = int(obj_id_values[index]) if obj_id_values.size > index else 0
            target_matches = tracked_obj_id is None or candidate_obj_id == tracked_obj_id
            if source == "prompt":
                accepted = (
                    inside_ratio >= settings.MASK_MIN_INSIDE_BOX_RATIO
                    and model_depth_ratio >= settings.MASK_MIN_MODEL_DEPTH_RATIO
                )
            elif source == "retry_prompt":
                # A retry prompt is centered on the last valid mask. Some SAM
                # sessions allocate a fresh object id for that prompt.
                accepted = target_matches or binary_masks.shape[0] == 1
            else:
                # The projected box identifies the initial object only. During
                # video propagation the object is expected to leave that box,
                # so continuity is enforced by SAM's object id instead.
                accepted = target_matches
            selection_score = (
                4.0 * inside_ratio
                + 2.0 * model_depth_ratio
                + min(1.0, box_coverage)
                - pixel_center_distance
                - (center_distance_m if center_distance_m is not None else 2.0)
            )
            candidates.append(
                {
                    "index": index,
                    "obj_id": candidate_obj_id,
                    "target_matches": target_matches,
                    "accepted": accepted,
                    "area_px": area,
                    "inside_pixels": inside_pixels,
                    "inside_ratio": inside_ratio,
                    "box_coverage": box_coverage,
                    "model_depth_ratio": model_depth_ratio,
                    "median_depth_m": median_depth_m,
                    "pixel_center_distance": pixel_center_distance,
                    "center_distance_m": center_distance_m,
                    "selection_score": selection_score,
                    "mask": mask_bool,
                }
            )

        accepted = [item for item in candidates if item["accepted"]]
        diagnostics["last_mask_candidates"] = [
            {key: value for key, value in item.items() if key != "mask"}
            for item in candidates
        ]
        diagnostics["candidate_mask_count"] = int(diagnostics.get("candidate_mask_count", 0)) + len(candidates)
        diagnostics["rejected_mask_count"] = int(diagnostics.get("rejected_mask_count", 0)) + (len(candidates) - len(accepted))
        if not accepted:
            diagnostics["empty_binary_mask_frames"] += 1
            return None

        selected = max(accepted, key=lambda item: float(item["selection_score"]))
        diagnostics["last_selected_mask_index"] = int(selected["index"])
        index = int(selected["index"])
        obj_id = int(selected["obj_id"])
        score = float(score_values[index]) if score_values.size > index else None
        return np.asarray(selected["mask"], dtype=bool), obj_id, score

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(
            path,
            optimize=True,
            compress_level=9,
        )
        return path


def _read_socket_json(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        return {}
    return json.loads(b"".join(chunks).decode("utf-8").splitlines()[0])


def _send_socket_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def run_socket_server(socket_path: Path) -> None:
    socket_path = socket_path.expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    worker = Sam3VideoWorker()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(8)
    print(f"[SAM3 video worker] listening: {socket_path}", flush=True)
    try:
        while True:
            conn, _addr = server.accept()
            with conn:
                try:
                    request = _read_socket_json(conn)
                    action = str(request.get("action") or "track_video")
                    if action == "shutdown":
                        worker.shutdown()
                        _send_socket_json(conn, {"ok": True, "shutdown": True})
                        return
                    if action != "track_video":
                        raise ValueError(f"unsupported action: {action}")
                    result = worker.track_video(request)
                    _send_socket_json(conn, {"ok": True, "result": result})
                except Exception as exc:
                    traceback.print_exc()
                    _send_socket_json(conn, {"ok": False, "error": str(exc)})
    finally:
        worker.shutdown()
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--socket-server":
        run_socket_server(Path(argv[2]))
        return 0
    print("Usage: python run_sam3_video_tracker_worker.py --socket-server <socket_path>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
