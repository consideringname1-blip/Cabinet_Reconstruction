from __future__ import annotations

import json
import os
import socket
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
from PIL import Image


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
        frame_paths = [Path(value) for value in request.get("frames") or []]
        if not frame_paths:
            raise ValueError("frames is empty")
        for path in frame_paths:
            if not path.is_file():
                raise FileNotFoundError(str(path))

        output_dir = Path(request.get("output_dir") or f"/tmp/sam3_video_{task_id}")
        frame_dir = output_dir / "frames"
        mask_dir = output_dir / "masks"
        frame_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        image_width, image_height = self._resolve_image_size(request, frame_paths[0])
        self._prepare_jpeg_frame_dir(frame_paths, frame_dir)
        box_xywh = self._normalized_box_xywh(request.get("box_xyxy"), image_width, image_height)

        predictor = self._build_predictor()
        session_id = None
        masks: list[dict[str, Any]] = []
        try:
            response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_dir),
                    "offload_video_to_cpu": bool(request.get("offload_video_to_cpu", True)),
                    "offload_state_to_cpu": bool(request.get("offload_state_to_cpu", False)),
                }
            )
            session_id = response["session_id"]
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(request.get("prompt_frame_index", 0)),
                    "bounding_boxes": [box_xywh],
                    "bounding_box_labels": [1],
                    "rel_coordinates": True,
                }
            )
            max_frames = request.get("max_frame_num_to_track")
            stream_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": str(request.get("propagation_direction") or "forward"),
                "start_frame_index": int(request.get("start_frame_index", 0)),
            }
            if max_frames is not None:
                stream_request["max_frame_num_to_track"] = int(max_frames)

            for item in predictor.handle_stream_request(stream_request):
                saved = self._save_response_mask(item, mask_dir)
                if saved is not None:
                    masks.append(saved)
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

        return {
            "task_id": task_id,
            "session_id": session_id,
            "frame_count": len(frame_paths),
            "mask_count": len(masks),
            "masks": masks,
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
    def _resolve_image_size(request: dict[str, Any], first_frame: Path) -> tuple[int, int]:
        value = request.get("image_size")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        with Image.open(first_frame) as image:
            return image.size

    @staticmethod
    def _prepare_jpeg_frame_dir(frame_paths: list[Path], frame_dir: Path) -> None:
        for index, source in enumerate(frame_paths):
            target = frame_dir / f"{index:06d}.jpg"
            if target.is_file():
                continue
            with Image.open(source) as image:
                image.convert("RGB").save(target, quality=95)

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
    def _save_response_mask(response: dict[str, Any], mask_dir: Path) -> dict[str, Any] | None:
        frame_index = response.get("frame_index")
        if frame_index is None:
            return None
        outputs = response.get("outputs") or {}
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")
        scores = outputs.get("out_scores")
        if scores is None:
            scores = outputs.get("out_obj_scores")
        if binary_masks is None:
            return None

        import torch

        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.detach().cpu().numpy()
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        binary_masks = np.asarray(binary_masks)
        if binary_masks.size == 0:
            return None
        obj_index = 0
        mask = binary_masks[obj_index]
        while mask.ndim > 2 and 1 in mask.shape:
            mask = np.squeeze(mask)
        if mask.ndim == 3:
            mask = mask[0]
        mask_bool = mask > 0
        path = mask_dir / f"{int(frame_index):06d}.png"
        Image.fromarray(mask_bool.astype(np.uint8) * 255).save(path)
        obj_id = int(np.asarray(obj_ids).reshape(-1)[obj_index]) if np.asarray(obj_ids).size else 0
        score = None
        if scores is not None and np.asarray(scores).size:
            score = float(np.asarray(scores).reshape(-1)[obj_index])
        return {
            "frame_index": int(frame_index),
            "obj_id": obj_id,
            "score": score,
            "mask_path": str(path),
        }


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
