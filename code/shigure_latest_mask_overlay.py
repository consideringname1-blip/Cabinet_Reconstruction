#!/usr/bin/env python3
"""Render the latest Shigure object-detection masks on the RGB frame.

This is a standalone smoke-test tool for the online Shigure RGB-D/object cache.
It can be launched with any Python. If the current interpreter cannot import
the image stack, it re-execs itself in the server Python environment. If the
Shigure recorder socket is not running, it starts the recorder sidecar first.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = CODE_ROOT.parent
BOOTSTRAP_ENV_KEY = "SHIGURE_MASK_OVERLAY_BOOTSTRAPPED"
DEFAULT_SERVER_PY = Path(os.environ.get("SHIGURE_MASK_OVERLAY_PY", "/opt/miniconda/envs/server/bin/python"))


def _ensure_code_root() -> None:
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))


def _image_runtime_available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def _bootstrap_runtime() -> None:
    _ensure_code_root()
    if _image_runtime_available():
        return
    if os.environ.get(BOOTSTRAP_ENV_KEY) == "1":
        return
    if DEFAULT_SERVER_PY.exists() and DEFAULT_SERVER_PY.resolve() != Path(sys.executable).resolve():
        env = dict(os.environ)
        env[BOOTSTRAP_ENV_KEY] = "1"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(CODE_ROOT) if not existing else f"{CODE_ROOT}:{existing}"
        os.execve(str(DEFAULT_SERVER_PY), [str(DEFAULT_SERVER_PY), str(SCRIPT_PATH), *sys.argv[1:]], env)


_bootstrap_runtime()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from artifact_layout import DATA_ROOT, SHIGURE_HISTORY_CACHE_ROOT, SHIGURE_HISTORY_SOCKET_PATH, make_timestamp  # noqa: E402
from path_config import SHIGURE_HISTORY_RECORDER_RUN, SHIGURE_HISTORY_RECORDER_STAGE_PY  # noqa: E402
from stages.shigure_history import settings as shigure_settings  # noqa: E402
from stages.shigure_history.cache import CachedRgbdSample, ShigureRgbdCache  # noqa: E402


@dataclass(frozen=True)
class MaskObservation:
    index: int
    object_id: str
    action: str
    bbox_xyxy: tuple[int, int, int, int]
    center_xy: tuple[int, int]
    mask: np.ndarray
    pixel_count: int
    color_rgb: tuple[int, int, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "object_id": self.object_id,
            "action": self.action,
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "center_xy": [int(v) for v in self.center_xy],
            "pixel_count": int(self.pixel_count),
            "color_rgb": [int(v) for v in self.color_rgb],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _decode_mask(obj: Mapping[str, Any], image_shape: tuple[int, int]) -> np.ndarray | None:
    raw = obj.get("mask_b64")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        data = base64.b64decode(raw)
        with Image.open(BytesIO(data)) as image:
            mask = np.asarray(image.convert("L")) > 0
    except Exception:
        return None
    h, w = image_shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def _bbox_from_object(obj: Mapping[str, Any], image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    bbox = obj.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    h, w = image_shape
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    x0 = int(round(max(0.0, min(float(w - 1), x0))))
    y0 = int(round(max(0.0, min(float(h - 1), y0))))
    x1 = int(round(max(0.0, min(float(w - 1), x1))))
    y1 = int(round(max(0.0, min(float(h - 1), y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _center_from_object(obj: Mapping[str, Any], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    try:
        x = int(round(float(obj.get("x"))))
        y = int(round(float(obj.get("y"))))
    except Exception:
        x = int(round((x0 + x1) * 0.5))
        y = int(round((y0 + y1) * 0.5))
    return x, y


def _color_for_index(index: int, total: int) -> tuple[int, int, int]:
    total = max(1, int(total))
    hue = int(round((179 * (index % total)) / total))
    hsv = np.asarray([[[hue, 210, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _extract_observations(sample: CachedRgbdSample) -> list[MaskObservation]:
    payload = sample.yolo if isinstance(sample.yolo, Mapping) else {}
    objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    h, w = sample.rgb_bgr.shape[:2]
    observations: list[MaskObservation] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            continue
        mask = _decode_mask(obj, (h, w))
        bbox = _bbox_from_object(obj, (h, w))
        if mask is None or bbox is None:
            continue
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count <= 0:
            continue
        observations.append(
            MaskObservation(
                index=len(observations),
                object_id=str(obj.get("object_id") or f"object:{index}"),
                action=str(obj.get("action") or "object"),
                bbox_xyxy=bbox,
                center_xy=_center_from_object(obj, bbox),
                mask=mask,
                pixel_count=pixel_count,
                color_rgb=_color_for_index(len(observations), max(1, len(objects))),
            )
        )
    return observations


def _mask_relations(observations: list[MaskObservation], *, near_pixels: int) -> list[dict[str, Any]]:
    if len(observations) < 2:
        return []
    radius = max(0, int(near_pixels))
    kernel = None
    if radius > 0:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    relations: list[dict[str, Any]] = []
    for i, left in enumerate(observations):
        left_u8 = left.mask.astype(np.uint8)
        left_dilated = cv2.dilate(left_u8, kernel, iterations=1).astype(bool) if kernel is not None else left.mask
        for right in observations[i + 1 :]:
            overlap_pixels = int(np.count_nonzero(left.mask & right.mask))
            near_pixels_count = int(np.count_nonzero(left_dilated & right.mask))
            relation = "overlap" if overlap_pixels > 0 else "near" if near_pixels_count > 0 else "separate"
            if relation == "separate":
                continue
            relations.append(
                {
                    "left_index": int(left.index),
                    "right_index": int(right.index),
                    "left_object_id": left.object_id,
                    "right_object_id": right.object_id,
                    "relation": relation,
                    "overlap_pixels": overlap_pixels,
                    "near_pixels": near_pixels_count,
                }
            )
    return relations


def _draw_overlay(
    sample: CachedRgbdSample,
    observations: list[MaskObservation],
    relations: list[Mapping[str, Any]],
    *,
    alpha: float,
) -> np.ndarray:
    rgb = np.asarray(sample.rgb_bgr[:, :, ::-1], dtype=np.uint8)
    overlay = rgb.astype(np.float32).copy()
    alpha = max(0.0, min(1.0, float(alpha)))
    for obs in observations:
        color = np.asarray(obs.color_rgb, dtype=np.float32)
        overlay[obs.mask] = overlay[obs.mask] * (1.0 - alpha) + color * alpha
    image = np.clip(overlay, 0, 255).astype(np.uint8)
    for obs in observations:
        color = tuple(int(v) for v in obs.color_rgb)
        mask_u8 = obs.mask.astype(np.uint8) * 255
        contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(image, contours, -1, color, 2)
        x0, y0, x1, y1 = obs.bbox_xyxy
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        label = f"{obs.index}:{obs.action}"
        label_origin = (max(0, x0 + 4), max(18, y0 + 18))
        cv2.putText(image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    for relation in relations:
        left_index = int(relation.get("left_index", -1))
        right_index = int(relation.get("right_index", -1))
        if left_index < 0 or right_index < 0 or left_index >= len(observations) or right_index >= len(observations):
            continue
        left = observations[left_index]
        right = observations[right_index]
        line_color = (255, 255, 255) if relation.get("relation") == "overlap" else (255, 230, 0)
        cv2.line(image, left.center_xy, right.center_xy, line_color, 2, cv2.LINE_AA)
    if not observations:
        cv2.putText(image, "no Shigure object masks in latest sample", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, "no Shigure object masks in latest sample", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
    return image


def _depth_preview(depth: np.ndarray) -> np.ndarray | None:
    if depth.size == 0:
        return None
    values = depth.astype(np.float32)
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return None
    lo = float(np.percentile(valid, 2.0))
    hi = float(np.percentile(valid, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return colored_bgr[:, :, ::-1]


def _cache_status(cache: ShigureRgbdCache) -> dict[str, Any] | None:
    try:
        return cache.status()
    except Exception:
        return None


def _start_recorder(args: argparse.Namespace, output_dir: Path) -> subprocess.Popen[bytes]:
    python_bin = Path(os.environ.get("SHIGURE_HISTORY_RECORDER_PY") or str(SHIGURE_HISTORY_RECORDER_STAGE_PY))
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    socket_path = Path(args.socket_path)
    cache_root = Path(args.cache_root)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "shigure_history_recorder.log"
    command = [
        str(python_bin),
        str(SHIGURE_HISTORY_RECORDER_RUN),
        "--cache-root",
        str(cache_root),
        "--socket-server",
        str(socket_path),
        "--sample-hz",
        str(args.sample_hz),
        "--retention-seconds",
        str(args.retention_seconds),
    ]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(CODE_ROOT) if not existing_pythonpath else f"{CODE_ROOT}:{existing_pythonpath}"
    log_file = log_path.open("ab")
    return subprocess.Popen(
        command,
        cwd=str(SHIGURE_HISTORY_RECORDER_RUN.parent),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def _wait_for_cache(cache: ShigureRgbdCache, *, timeout_seconds: float, poll_interval: float) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while time.monotonic() <= deadline:
        status = _cache_status(cache)
        if status is not None:
            return status
        time.sleep(max(0.05, float(poll_interval)))
    return None


def _wait_for_sample(
    cache: ShigureRgbdCache,
    *,
    timeout_seconds: float,
    poll_interval: float,
    prefer_objects: bool,
) -> CachedRgbdSample | None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    newest: CachedRgbdSample | None = None
    while time.monotonic() <= deadline:
        sample = cache.newest_sample()
        if sample is not None:
            newest = sample
            observations = _extract_observations(sample)
            if observations or not prefer_objects:
                return sample
        time.sleep(max(0.05, float(poll_interval)))
    return newest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "shigure_mask_overlay_checks")
    parser.add_argument("--socket-path", type=Path, default=SHIGURE_HISTORY_SOCKET_PATH)
    parser.add_argument("--cache-root", type=Path, default=SHIGURE_HISTORY_CACHE_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--sample-hz", type=float, default=shigure_settings.SHIGURE_HISTORY_HZ)
    parser.add_argument("--retention-seconds", type=float, default=shigure_settings.SHIGURE_HISTORY_SECONDS)
    parser.add_argument("--near-pixels", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.52)
    parser.add_argument("--no-start-recorder", action="store_true", help="Fail instead of starting the recorder sidecar when the socket is missing.")
    parser.add_argument("--stop-started-recorder", action="store_true", help="Terminate the recorder sidecar if this script started it.")
    parser.add_argument("--allow-no-objects", action="store_true", help="Return success even if the latest sample has no object masks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_root) / make_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ShigureRgbdCache(socket_path=args.socket_path, timeout_seconds=min(5.0, max(1.0, args.timeout_seconds)))
    started_recorder: subprocess.Popen[bytes] | None = None
    try:
        status = _cache_status(cache)
        if status is None:
            if args.no_start_recorder:
                _write_json(
                    output_dir / "latest_object_mask_overlay.json",
                    {
                        "status": "socket_unavailable",
                        "updated_at": _utc_now(),
                        "socket_path": str(args.socket_path),
                        "last_error": cache.last_error,
                    },
                )
                print(f"[ERR] Shigure cache socket unavailable: {args.socket_path}", file=sys.stderr)
                return 1
            started_recorder = _start_recorder(args, output_dir)
            status = _wait_for_cache(cache, timeout_seconds=args.timeout_seconds, poll_interval=args.poll_interval)
            if status is None:
                _write_json(
                    output_dir / "latest_object_mask_overlay.json",
                    {
                        "status": "recorder_start_timeout",
                        "updated_at": _utc_now(),
                        "socket_path": str(args.socket_path),
                        "recorder_pid": started_recorder.pid if started_recorder else None,
                        "last_error": cache.last_error,
                    },
                )
                print(f"[ERR] Shigure recorder did not expose socket in time: {args.socket_path}", file=sys.stderr)
                return 1

        sample = _wait_for_sample(
            cache,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            prefer_objects=not args.allow_no_objects,
        )
        if sample is None:
            _write_json(
                output_dir / "latest_object_mask_overlay.json",
                {
                    "status": "no_rgbd_sample",
                    "updated_at": _utc_now(),
                    "socket_path": str(args.socket_path),
                    "cache_status": status,
                    "recorder_pid": started_recorder.pid if started_recorder else None,
                },
            )
            print("[ERR] No Shigure RGB-D sample available.", file=sys.stderr)
            return 1

        observations = _extract_observations(sample)
        relations = _mask_relations(observations, near_pixels=args.near_pixels)
        overlay_rgb = _draw_overlay(sample, observations, relations, alpha=args.alpha)
        rgb = np.asarray(sample.rgb_bgr[:, :, ::-1], dtype=np.uint8)
        overlay_path = output_dir / "latest_object_mask_overlay.png"
        rgb_path = output_dir / "latest_rgb.png"
        Image.fromarray(overlay_rgb).save(overlay_path)
        Image.fromarray(rgb).save(rgb_path)
        depth_preview = _depth_preview(sample.depth)
        depth_preview_path = None
        if depth_preview is not None:
            depth_preview_path = output_dir / "latest_depth_preview.png"
            Image.fromarray(depth_preview).save(depth_preview_path)

        result_status = "ok" if observations else "no_objects"
        payload = {
            "status": result_status,
            "updated_at": _utc_now(),
            "output_dir": str(output_dir),
            "overlay_path": str(overlay_path),
            "rgb_path": str(rgb_path),
            "depth_preview_path": str(depth_preview_path) if depth_preview_path else None,
            "socket_path": str(args.socket_path),
            "cache_root": str(args.cache_root),
            "started_recorder": started_recorder is not None,
            "recorder_pid": started_recorder.pid if started_recorder else None,
            "sample": sample.to_dict(),
            "object_count": len(observations),
            "objects": [obs.to_json() for obs in observations],
            "touching_or_near_relations": relations,
            "near_pixels": int(args.near_pixels),
            "alpha": float(args.alpha),
            "cache_status": _cache_status(cache),
        }
        _write_json(output_dir / "latest_object_mask_overlay.json", payload)
        print(f"[OK] Shigure latest mask overlay: {overlay_path}")
        print(f"[OK] Metadata: {output_dir / 'latest_object_mask_overlay.json'}")
        if not observations and not args.allow_no_objects:
            print("[WARN] Latest Shigure RGB-D sample has no object masks.", file=sys.stderr)
            return 2
        return 0
    finally:
        if started_recorder is not None and args.stop_started_recorder:
            started_recorder.terminate()
            try:
                started_recorder.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                started_recorder.kill()


if __name__ == "__main__":
    raise SystemExit(main())
