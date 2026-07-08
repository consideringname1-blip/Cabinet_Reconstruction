from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import _bootstrap
import cv2
import numpy as np

from artifact_layout import model_worker_file
from stage_common import ensure_file
from task_json import load_task_json, normalize_path_for_storage, resolve_task_json_path


WORKER_RESPONSE_ENCODING = "utf-8"
DEFAULT_DINOV2_REPO = "/root/.cache/torch/hub/facebookresearch_dinov2_main"
DEFAULT_DINOV2_MODEL = "dinov2_vitl14_reg"
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _read_socket_json(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    if not chunks:
        raise RuntimeError("empty DINOv2 identity worker request")
    raw = b"".join(chunks).splitlines()[0]
    return json.loads(raw.decode(WORKER_RESPONSE_ENCODING))


def _send_socket_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode(WORKER_RESPONSE_ENCODING))


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if image.ndim == 2:
        return image > 0
    if image.shape[2] == 4:
        return image[:, :, 3] > 0
    return np.any(image > 0, axis=2)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read color image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resolve_sam3_artifacts(task: dict[str, Any]) -> tuple[Path, Path]:
    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise ValueError("task_timestamp is required for DINOv2 identity artifacts")
    sam3 = task.get("sam3Name") if isinstance(task.get("sam3Name"), dict) else {}
    if not sam3.get("mask"):
        raise ValueError("sam3Name.mask is missing; run sam3mask before DINOv2 identity")
    color_path = model_worker_file(task_timestamp, "sam3.color")
    mask_path = model_worker_file(task_timestamp, "sam3.mask")
    return ensure_file(color_path, "SAM3 color image"), ensure_file(mask_path, "SAM3 mask image")


def _square_crop_bounds(mask: np.ndarray, padding_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(round(max(bw, bh) * float(padding_ratio)))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(bw, bh) + 2 * pad
    sx0 = int(round(cx - side / 2.0))
    sy0 = int(round(cy - side / 2.0))
    sx1 = sx0 + side
    sy1 = sy0 + side
    return max(0, sx0), max(0, sy0), min(w, sx1), min(h, sy1)


def _masked_crop_tensor(rgb: np.ndarray, mask: np.ndarray, image_size: int, padding_ratio: float):
    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"color/mask shape mismatch: {rgb.shape[:2]} vs {mask.shape[:2]}")
    x0, y0, x1, y1 = _square_crop_bounds(mask, padding_ratio)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop_rgb.size == 0 or not np.any(crop_mask):
        raise ValueError("empty masked crop")
    canvas = np.full_like(crop_rgb, 255, dtype=np.uint8)
    canvas[crop_mask] = crop_rgb[crop_mask]
    resized = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    normalized = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return normalized, {
        "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "mask_pixels": int(mask.sum()),
        "crop_mask_pixels": int(crop_mask.sum()),
        "image_size": int(image_size),
        "padding_ratio": float(padding_ratio),
    }


class Dinov2IdentityRunner:
    def __init__(self) -> None:
        self._loaded = False
        self.torch = None
        self.device = None
        self.model = None
        self.repo = None
        self.model_name = None
        self.image_size = int(os.environ.get("DINO_IDENTITY_IMAGE_SIZE", str(DEFAULT_IMAGE_SIZE)))
        self.padding_ratio = float(os.environ.get("DINO_IDENTITY_CROP_PADDING_RATIO", "0.08"))

    def _load_model(self) -> dict[str, Any]:
        if self._loaded:
            return {
                "loaded_now": False,
                "duration_ms": 0.0,
                "device": str(self.device),
                "repo": str(self.repo),
                "model_name": str(self.model_name),
            }

        start = time.perf_counter()
        import torch

        repo = Path(os.environ.get("DINO_IDENTITY_REPO", DEFAULT_DINOV2_REPO)).expanduser().resolve()
        model_name = str(os.environ.get("DINO_IDENTITY_MODEL", DEFAULT_DINOV2_MODEL)).strip() or DEFAULT_DINOV2_MODEL
        if not repo.is_dir():
            raise FileNotFoundError(f"DINOv2 local torch hub repo not found: {repo}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DINOv2 identity] loading {model_name} from {repo} on {device}", flush=True)
        model = torch.hub.load(str(repo), model_name, source="local", pretrained=True)
        model.to(device)
        model.eval()
        self.torch = torch
        self.device = device
        self.model = model
        self.repo = repo
        self.model_name = model_name
        self._loaded = True
        duration_ms = (time.perf_counter() - start) * 1000.0
        print(f"[DINOv2 identity] model ready in {duration_ms:.1f} ms", flush=True)
        return {
            "loaded_now": True,
            "duration_ms": duration_ms,
            "device": str(device),
            "repo": str(repo),
            "model_name": model_name,
        }

    def embed_task(self, json_path: Path) -> dict[str, Any]:
        timings: dict[str, Any] = {}
        start_total = time.perf_counter()
        model_info = self._load_model()
        timings["model_load"] = model_info
        assert self.torch is not None and self.model is not None and self.device is not None

        task = load_task_json(json_path)
        color_path, mask_path = _resolve_sam3_artifacts(task)
        rgb = _read_rgb(color_path)
        mask = _read_mask(mask_path)
        tensor_np, crop_info = _masked_crop_tensor(rgb, mask, self.image_size, self.padding_ratio)
        tensor = self.torch.from_numpy(tensor_np).unsqueeze(0).to(self.device)

        infer_start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, dict):
            for key in ("x_norm_clstoken", "x_norm_patchtokens", "last_hidden_state", "pooler_output"):
                value = output.get(key)
                if value is not None:
                    output = value
                    break
            else:
                output = next(iter(output.values()))
        output = output.detach().float().reshape(1, -1)
        output = output / output.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        embedding = output[0].cpu().numpy().astype(np.float32)
        timings["inference_ms"] = (time.perf_counter() - infer_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - start_total) * 1000.0

        return {
            "embedding": [float(v) for v in embedding.tolist()],
            "dim": int(embedding.size),
            "model_name": str(self.model_name),
            "repo": str(self.repo),
            "device": str(self.device),
            "normalized": True,
            "crop": crop_info,
            "source": {
                "json_path": normalize_path_for_storage(json_path),
                "color_path": normalize_path_for_storage(color_path),
                "mask_path": normalize_path_for_storage(mask_path),
            },
            "timings": timings,
        }


def run_socket_server(socket_path: Path) -> None:
    socket_path = socket_path.expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    runner = Dinov2IdentityRunner()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(8)
    print(f"[DINOv2 identity] listening: {socket_path}", flush=True)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = _read_socket_json(conn)
                    if request.get("action") == "shutdown":
                        _send_socket_json(conn, {"ok": True, "shutdown": True})
                        break
                    json_path = ensure_file(resolve_task_json_path(request["json_path"]), "JSON file")
                    result = runner.embed_task(json_path)
                    _send_socket_json(conn, {"ok": True, **result})
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    _send_socket_json(conn, {"ok": False, "error": str(exc)})
    finally:
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?")
    parser.add_argument("--socket-server", dest="socket_server")
    args = parser.parse_args()
    try:
        if args.socket_server:
            run_socket_server(Path(args.socket_server))
            return 0
        if not args.json_path:
            parser.error("json_path is required unless --socket-server is used")
        json_path = ensure_file(resolve_task_json_path(args.json_path), "JSON file")
        result = Dinov2IdentityRunner().embed_task(json_path)
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
