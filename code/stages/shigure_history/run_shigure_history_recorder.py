#!/usr/bin/env python3
"""Receive Shigurei ROS2 streams and expose a one-minute in-memory RGB-D cache."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[2]
ROS_SETUP_ENV = CODE_ROOT / "ros2" / "shigure_recv_ws" / "setup_env.sh"
ROS_PYTHON = Path(os.environ.get("SHIGURE_HISTORY_RECORDER_ROS_PYTHON", "/usr/bin/python3"))
SERVER_SITE_PACKAGES = Path(os.environ.get("SHIGURE_HISTORY_RECORDER_CV_SITE_PACKAGES", "/opt/miniconda/envs/server/lib/python3.10/site-packages"))
BOOTSTRAP_ENV_KEY = "SHIGURE_HISTORY_RECORDER_BOOTSTRAPPED"


def _site_packages_version(path: Path) -> tuple[int, int] | None:
    for part in path.parts:
        match = re.fullmatch(r"python(\d+)\.(\d+)", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _python_bin_version(python_bin: Path) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output(
            [str(python_bin), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        major, minor = output.split(".", 1)
        return int(major), int(minor)
    except Exception:
        return None


def _site_packages_match_current_python(path: Path) -> bool:
    version = _site_packages_version(path)
    if version is None:
        return True
    return version == (sys.version_info.major, sys.version_info.minor)


def _site_packages_match_python_bin(path: Path, python_bin: Path) -> bool:
    site_version = _site_packages_version(path)
    if site_version is None:
        return True
    return _python_bin_version(python_bin) == site_version


def bootstrap_runtime_environment() -> None:
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    if SERVER_SITE_PACKAGES.is_dir() and _site_packages_match_current_python(SERVER_SITE_PACKAGES) and str(SERVER_SITE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(SERVER_SITE_PACKAGES))
    if os.environ.get(BOOTSTRAP_ENV_KEY) == "1" or not ROS_SETUP_ENV.exists():
        return
    os.environ[BOOTSTRAP_ENV_KEY] = "1"
    python_bin = ROS_PYTHON if ROS_PYTHON.exists() else Path(sys.executable)
    env_export = ""
    if SERVER_SITE_PACKAGES.is_dir() and _site_packages_match_python_bin(SERVER_SITE_PACKAGES, python_bin):
        env_export = f"export PYTHONPATH={shlex.quote(str(SERVER_SITE_PACKAGES))}:${{PYTHONPATH}} && "
    command = (
        f"source {shlex.quote(str(ROS_SETUP_ENV))} && "
        f"{env_export}exec {shlex.quote(str(python_bin))} {shlex.quote(str(SCRIPT_PATH))} "
        f"{chr(32).join(shlex.quote(arg) for arg in sys.argv[1:])}"
    )
    os.execvp("bash", ["bash", "-lc", command])


bootstrap_runtime_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stages.shigure_history import settings  # noqa: E402
from stages.shigure_history.cache import (  # noqa: E402
    CachedRgbdSample,
    CachedShigureFrame,
    RosStamp,
    ShigureMemoryStore,
    sample_key,
    store_request,
)
from stages.shigure_history.debug_cache import (  # noqa: E402
    ShigureDebugDiskRing,
    validate_debug_cache_limits,
)
from stages.shigure_history.marker_history import MarkerHistoryWarmup  # noqa: E402
from stages.shigure_history.shigure_compatibility import ShigureCompatibilityAdapter  # noqa: E402

_running = True
REQUIRED_KEYS = ("rgb", "depth", "camera_info")


@dataclass(frozen=True)
class TopicSample:
    message: Any
    stamp: RosStamp | None
    received_at: float
    received_monotonic: float
    count: int


@dataclass
class TopicState:
    key: str
    topic: str
    type_name: str
    maxlen: int
    samples: deque[TopicSample] = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=max(1, int(self.maxlen)))

    def append(self, msg: Any) -> TopicSample:
        self.count += 1
        sample = TopicSample(
            message=msg,
            stamp=stamp_from_message(msg),
            received_at=time.time(),
            received_monotonic=time.monotonic(),
            count=self.count,
        )
        self.samples.append(sample)
        return sample

    def latest(self) -> TopicSample | None:
        return self.samples[-1] if self.samples else None

    def nearest(self, stamp: RosStamp, *, max_delta_seconds: float | None = None) -> TopicSample | None:
        if not self.samples:
            return None
        stamped = [sample for sample in self.samples if sample.stamp is not None]
        if not stamped:
            return self.latest()
        selected = min(stamped, key=lambda sample: abs(float(sample.stamp.seconds) - float(stamp.seconds)))
        if max_delta_seconds is not None and abs(float(selected.stamp.seconds) - float(stamp.seconds)) > float(max_delta_seconds):
            return None
        return selected

    def exact(self, stamp: RosStamp) -> TopicSample | None:
        for sample in reversed(self.samples):
            if sample.stamp == stamp:
                return sample
        return None


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    global _running
    _running = False


for _signum in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_signum, _handle_signal)
    except Exception:
        pass


def import_ros_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit(f"ROS2 Python modules are not available. Expected {ROS_SETUP_ENV}. Import error: {exc}") from exc
    return rclpy, Node, QoSProfile, ReliabilityPolicy, get_message


def stamp_from_message(msg: Any | None) -> RosStamp | None:
    header = getattr(msg, "header", None) if msg is not None else None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return RosStamp(sec=sec, nanosec=nanosec)


def bytes_to_json_blob(data: bytes | bytearray | array) -> dict[str, Any]:
    raw = bytes(data)
    return {"__encoding__": "base64", "length": len(raw)}


def message_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes_to_json_blob(value)
    if isinstance(value, array):
        if value.typecode in ("b", "B"):
            return bytes_to_json_blob(value)
        return [message_to_jsonable(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [message_to_jsonable(item) for item in value]
    if hasattr(value, "get_fields_and_field_types"):
        return {field_name: message_to_jsonable(getattr(value, field_name)) for field_name in value.get_fields_and_field_types().keys()}
    return str(value)


def camera_info_payload(sample: TopicSample) -> dict[str, Any]:
    msg = sample.message
    payload = message_to_jsonable(msg)
    if not isinstance(payload, dict):
        payload = {}
    try:
        payload["width"] = int(getattr(msg, "width"))
        payload["height"] = int(getattr(msg, "height"))
        payload["k"] = [float(value) for value in getattr(msg, "k")]
        payload["d"] = [float(value) for value in getattr(msg, "d", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError("CameraInfo message has invalid width, height, k, or d") from exc
    return payload


def compressed_image_payload(msg: Any, topic: str) -> tuple[bytes, int]:
    raw = bytes(getattr(msg, "data", b""))
    fmt = str(getattr(msg, "format", ""))
    is_depth = "compressedDepth" in fmt or "depth" in topic.lower()
    if is_depth and b"PNG" not in raw[:12]:
        return raw[12:], 12
    return raw, 0


def decode_compressed_image(sample: TopicSample, state: TopicState, *, color: bool) -> tuple[np.ndarray, dict[str, Any]]:
    payload, skipped_header_bytes = compressed_image_payload(sample.message, state.topic)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"cv2.imdecode returned None for {state.topic}")
    if not color and image.ndim == 3:
        image = image[:, :, 0]
    return image, {"shape": list(image.shape), "dtype": str(image.dtype), "skipped_header_bytes": skipped_header_bytes}


def _object_bbox_xyxy(obj: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(obj, "bounding_box", None)
    if bbox is None:
        return None
    try:
        x = float(getattr(bbox, "x"))
        y = float(getattr(bbox, "y"))
        width = float(getattr(bbox, "width"))
        height = float(getattr(bbox, "height"))
    except Exception:
        return None
    if not np.isfinite([x, y, width, height]).all() or width <= 0.0 or height <= 0.0:
        return None
    return x, y, x + width, y + height


def _bbox_payload(bbox: Any | None) -> dict[str, Any] | None:
    if bbox is None:
        return None
    try:
        x = float(getattr(bbox, "x"))
        y = float(getattr(bbox, "y"))
        width = float(getattr(bbox, "width"))
        height = float(getattr(bbox, "height"))
    except Exception:
        return None
    if not np.isfinite([x, y, width, height]).all() or width <= 0.0 or height <= 0.0:
        return None
    return {
        "xyxy": [x, y, x + width, y + height],
    }


def _cube_payload(cube: Any | None) -> dict[str, Any] | None:
    if cube is None:
        return None
    try:
        payload = {
            "x": float(getattr(cube, "x")),
            "y": float(getattr(cube, "y")),
            "z": float(getattr(cube, "z")),
            "width": float(getattr(cube, "width")),
            "height": float(getattr(cube, "height")),
            "depth": float(getattr(cube, "depth")),
        }
        values = list(payload.values())
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values[3:]):
            return None
        return payload
    except Exception:
        return None


def object_detection_payload(sample: TopicSample) -> dict[str, Any]:
    msg = sample.message
    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(getattr(msg, "object_list", []) or []):
        bbox = _object_bbox_xyxy(obj)
        if bbox is None:
            continue
        mask_msg = getattr(obj, "mask", None)
        mask_raw = bytes(getattr(mask_msg, "data", b"")) if mask_msg is not None else b""
        x0, y0, x1, y1 = bbox
        action = str(getattr(obj, "action", "object") or "object")
        objects.append(
            {
                "index": int(index),
                "action": action,
                "bbox_xyxy": [x0, y0, x1, y1],
                "mask_b64": base64.b64encode(mask_raw).decode("ascii"),
                "mask_format": str(getattr(mask_msg, "format", "")) if mask_msg is not None else "",
                "mask_bytes": len(mask_raw),
            }
        )
    return {
        "objects": objects,
        "object_count": len(objects),
    }


def object_tracking_payload(sample: TopicSample) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(getattr(sample.message, "tracked_object_list", []) or []):
        bbox = _object_bbox_xyxy(obj)
        object_id = str(getattr(obj, "object_id", "") or "").strip()
        action = str(getattr(obj, "action", "") or "").strip().lower()
        if bbox is None or not object_id:
            continue
        objects.append(
            {
                "index": int(index),
                "object_id": object_id,
                "action": action,
                "bbox_xyxy": [float(value) for value in bbox],
                "collider": _cube_payload(getattr(obj, "collider", None)),
            }
        )
    return {"objects": objects, "object_count": len(objects)}


def segments_payload(sample: TopicSample) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(getattr(sample.message, "segments", []) or []):
        try:
            x0 = float(getattr(segment, "xmin"))
            y0 = float(getattr(segment, "ymin"))
            x1 = float(getattr(segment, "xmax"))
            y1 = float(getattr(segment, "ymax"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
            continue
        segments.append(
            {
                "index": int(index),
                "class_id": str(getattr(segment, "class_id", "") or ""),
                "probability": float(getattr(segment, "probability", 0.0) or 0.0),
                "bbox_xyxy": [x0, y0, x1, y1],
                # Legacy bboxes_ex_msgs stores y in x_masks and x in y_masks.
                "x_masks": [int(value) for value in (getattr(segment, "x_masks", []) or [])],
                "y_masks": [int(value) for value in (getattr(segment, "y_masks", []) or [])],
            }
        )
    return {"segments": segments, "segment_count": len(segments)}


def _point_payload(point: Any | None) -> dict[str, float] | None:
    if point is None:
        return None
    try:
        values = {
            "x": float(getattr(point, "x")),
            "y": float(getattr(point, "y")),
            "z": float(getattr(point, "z")),
        }
    except (TypeError, ValueError, AttributeError):
        return None
    return values if np.isfinite(list(values.values())).all() else None


def people_payload(sample: TopicSample) -> dict[str, Any]:
    people: list[dict[str, Any]] = []
    for person in getattr(sample.message, "pose_key_points_list", []) or []:
        joints: list[dict[str, Any]] = []
        for point_data in getattr(person, "point_data", []) or []:
            joints.append(
                {
                    "body_part_name": str(getattr(point_data, "body_part_name", "") or ""),
                    "pixel_point": _point_payload(getattr(point_data, "pixel_point", None)),
                    "projection_point": _point_payload(getattr(point_data, "projection_point", None)),
                    "score": float(getattr(point_data, "score", 0.0) or 0.0),
                }
            )
        people.append(
            {
                "people_id": str(getattr(person, "people_id", "") or ""),
                "bounding_box": _bbox_payload(getattr(person, "bounding_box", None)),
                "joints": joints,
            }
        )
    return {"people": people, "people_count": len(people)}


def compatibility_payload(topic_key: str, sample: TopicSample) -> dict[str, Any]:
    if topic_key in {"rgb", "depth"}:
        return {"available": True}
    if topic_key == "camera_info":
        return camera_info_payload(sample)
    if topic_key == "object_detection":
        return object_detection_payload(sample)
    if topic_key == "object_tracking":
        return object_tracking_payload(sample)
    if topic_key == "segments":
        return segments_payload(sample)
    if topic_key == "people":
        return people_payload(sample)
    if topic_key == "contacted":
        return contacted_payload(sample)
    raise ValueError(f"unsupported canonical compatibility topic: {topic_key}")


def contacted_payload(sample: TopicSample) -> dict[str, Any]:
    msg = sample.message
    contacts: list[dict[str, Any]] = []
    for index, contacted in enumerate(getattr(msg, "contacted_list", []) or []):
        contacts.append(
            {
                "index": int(index),
                "event_id": str(getattr(contacted, "event_id", "") or ""),
                "people_id": str(getattr(contacted, "people_id", "") or ""),
                "object_id": str(getattr(contacted, "object_id", "") or ""),
                "action": str(getattr(contacted, "action", "") or ""),
                "people_bounding_box": _bbox_payload(getattr(contacted, "people_bounding_box", None)),
                "object_bounding_box": _bbox_payload(getattr(contacted, "object_bounding_box", None)),
                "object_cube": _cube_payload(getattr(contacted, "object_cube", None)),
            }
        )
    return {
        "contacts": contacts,
        "contact_count": len(contacts),
    }


def selected_rgb_stamp(rgb_sample: TopicSample) -> RosStamp:
    if rgb_sample.stamp is None:
        raise ValueError("RGB frame is missing its ROS source timestamp")
    return rgb_sample.stamp


def required_ready(states: dict[str, TopicState]) -> bool:
    return all(states[key].latest() is not None for key in REQUIRED_KEYS)


def latest_required_counts(states: dict[str, TopicState]) -> tuple[int, ...]:
    return tuple(states[key].count for key in REQUIRED_KEYS)



def append_aligned_sample(
    store: ShigureMemoryStore,
    states: dict[str, TopicState],
    *,
    last_key: str | None,
    rgb_depth_max_delta_seconds: float,
) -> tuple[str | None, bool, CachedRgbdSample | None, dict[str, Any]]:
    rgb_sample = states["rgb"].latest()
    if rgb_sample is None:
        return last_key, False, None, {"reason": "rgb_missing"}
    stamp = selected_rgb_stamp(rgb_sample)
    key = sample_key(stamp)
    if key == last_key:
        return last_key, False, None, {"reason": "duplicate_rgb_stamp"}

    depth_sample = states["depth"].nearest(stamp, max_delta_seconds=rgb_depth_max_delta_seconds)
    camera_info_sample = states["camera_info"].nearest(stamp, max_delta_seconds=None)
    if depth_sample is None or camera_info_sample is None:
        return last_key, False, None, {
            "reason": "required_aligned_topic_missing",
            "depth_available": depth_sample is not None,
            "camera_info_available": camera_info_sample is not None,
        }

    rgb, rgb_info = decode_compressed_image(rgb_sample, states["rgb"], color=True)
    depth, depth_info = decode_compressed_image(depth_sample, states["depth"], color=False)
    camera_info = camera_info_payload(camera_info_sample)

    sample = CachedRgbdSample(
        stamp=stamp,
        rgb_bgr=rgb,
        depth=depth,
        camera_info=camera_info,
    )
    store.append(sample)
    return key, True, sample, {
        "rgb_decode": rgb_info,
        "depth_decode": depth_info,
        "frame": sample.to_dict(),
    }


def attach_exact_canonical_rgbd(
    store: ShigureMemoryStore,
    states: dict[str, TopicState],
    frame: CachedShigureFrame,
    *,
    rgb_depth_max_delta_seconds: float,
) -> CachedShigureFrame:
    """Attach evidence for an event/Segments stamp without nearest-RGB reuse."""

    if not frame.events and not frame.recovery_candidates:
        return frame
    existing = store.get_sample(frame.source_stamp)
    if existing is not None and existing.stamp == frame.source_stamp:
        return frame
    diagnostic: dict[str, Any] = {
        "code": "CANONICAL_RGBD_UNAVAILABLE",
        "source_stamp": frame.source_stamp.to_dict(),
    }
    try:
        rgb_state = states["rgb"]
        depth_state = states["depth"]
        camera_state = states["camera_info"]
        # Never relabel a nearest RGB image as this canonical frame. Depth and
        # calibration follow the configured alignment policy.
        rgb_sample = rgb_state.exact(frame.source_stamp)
        depth_sample = depth_state.nearest(
            frame.source_stamp,
            max_delta_seconds=rgb_depth_max_delta_seconds,
        )
        camera_sample = camera_state.nearest(frame.source_stamp, max_delta_seconds=None)
        if rgb_sample is None or depth_sample is None or camera_sample is None:
            diagnostic.update(
                {
                    "reason": "aligned_topic_missing",
                    "exact_rgb_available": rgb_sample is not None,
                    "aligned_depth_available": depth_sample is not None,
                    "camera_info_available": camera_sample is not None,
                }
            )
            return replace(frame, diagnostics=[*frame.diagnostics, diagnostic])
        rgb, rgb_info = decode_compressed_image(rgb_sample, rgb_state, color=True)
        depth, depth_info = decode_compressed_image(depth_sample, depth_state, color=False)
        sample = CachedRgbdSample(
            stamp=frame.source_stamp,
            rgb_bgr=rgb,
            depth=depth,
            camera_info=camera_info_payload(camera_sample),
        )
        store.append(sample)
        return replace(
            frame,
            diagnostics=[
                *frame.diagnostics,
                {
                    "code": "CANONICAL_RGBD_ATTACHED",
                    "source_stamp": frame.source_stamp.to_dict(),
                    "rgb_decode": rgb_info,
                    "depth_decode": depth_info,
                    "depth_stamp": depth_sample.stamp.to_dict() if depth_sample.stamp else None,
                },
            ],
        )
    except Exception as exc:
        diagnostic.update({"reason": "decode_or_validation_failed", "error": str(exc)})
        return replace(frame, diagnostics=[*frame.diagnostics, diagnostic])


def _read_socket_json(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        return {}
    return json.loads(b"".join(chunks).decode("utf-8"))


def _send_socket_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


class ShigureHistorySocketServer:
    def __init__(self, socket_path: Path, store: ShigureMemoryStore) -> None:
        self.socket_path = socket_path.expanduser().resolve()
        self.store = store
        self.thread: threading.Thread | None = None
        self.server: socket.socket | None = None

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self.thread = threading.Thread(target=self._run, daemon=True, name="shigure-history-socket-server")
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _run(self) -> None:
        global _running
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server = server
        try:
            server.bind(str(self.socket_path))
            server.listen(8)
            server.settimeout(0.2)
            print(f"[shigure_history] listening: {self.socket_path}", flush=True)
            while _running:
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    try:
                        request = _read_socket_json(conn)
                        if request.get("action") == "shutdown":
                            _running = False
                            _send_socket_json(conn, {"ok": True, "shutdown": True})
                        else:
                            _send_socket_json(conn, store_request(self.store, request))
                    except Exception as exc:
                        try:
                            _send_socket_json(conn, {"ok": False, "error": str(exc)})
                        except Exception:
                            pass
        finally:
            try:
                server.close()
            except Exception:
                pass
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-server", type=Path, default=settings.SHIGURE_HISTORY_SOCKET_PATH)
    parser.add_argument("--sample-hz", type=float, default=settings.SHIGURE_HISTORY_HZ)
    parser.add_argument("--retention-seconds", type=float, default=settings.SHIGURE_HISTORY_SECONDS)
    parser.add_argument("--max-samples", type=int, default=settings.SHIGURE_HISTORY_MAX_SAMPLES)
    parser.add_argument("--max-frames", type=int, default=settings.SHIGURE_HISTORY_MAX_FRAMES)
    parser.add_argument("--log-interval", type=float, default=settings.SHIGURE_HISTORY_RECORDER_LOG_INTERVAL)
    parser.add_argument("--rgb-depth-max-delta-seconds", type=float, default=settings.SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS)
    debug_mode = parser.add_mutually_exclusive_group()
    debug_mode.add_argument("--debug-cache-enable", dest="debug_cache_enable", action="store_true")
    debug_mode.add_argument("--debug-cache-disable", dest="debug_cache_enable", action="store_false")
    parser.set_defaults(debug_cache_enable=settings.SHIGURE_DEBUG_CACHE_ENABLE)
    parser.add_argument("--debug-cache-root", type=Path, default=settings.SHIGURE_DEBUG_CACHE_ROOT)
    parser.add_argument(
        "--debug-cache-retention-seconds",
        type=float,
        default=settings.SHIGURE_DEBUG_CACHE_RETENTION_SECONDS,
    )
    parser.add_argument("--debug-cache-max-entries", type=int, default=settings.SHIGURE_DEBUG_CACHE_MAX_ENTRIES)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_debug_cache_limits(
        retention_seconds=args.debug_cache_retention_seconds,
        max_entries=args.debug_cache_max_entries,
    )
    rclpy, Node, QoSProfile, ReliabilityPolicy, get_message = import_ros_modules()
    rclpy.init(args=None)
    node = Node("shigure_memory_history_recorder")
    store = ShigureMemoryStore(
        max_seconds=args.retention_seconds,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
    )
    debug_ring: ShigureDebugDiskRing | None = None
    debug_rgbd_order: deque[str] = deque()
    debug_rgbd_keys: set[str] = set()

    def record_debug_rgbd_once(sample: CachedRgbdSample) -> bool:
        """Queue one RGB-D write per source stamp without affecting runtime data."""

        if debug_ring is None or not debug_ring.enabled:
            return False
        key = sample_key(sample.stamp)
        if key in debug_rgbd_keys or not debug_ring.record_rgbd(sample):
            return False
        limit = max(1, int(args.debug_cache_max_entries))
        while len(debug_rgbd_order) >= limit:
            debug_rgbd_keys.discard(debug_rgbd_order.popleft())
        debug_rgbd_order.append(key)
        debug_rgbd_keys.add(key)
        return True

    def append_canonical_frame(frame: CachedShigureFrame) -> CachedShigureFrame:
        frame = attach_exact_canonical_rgbd(
            store,
            states,
            frame,
            rgb_depth_max_delta_seconds=args.rgb_depth_max_delta_seconds,
        )
        exact_sample = store.get_sample(frame.source_stamp)
        if exact_sample is not None and exact_sample.stamp == frame.source_stamp:
            record_debug_rgbd_once(exact_sample)
        stored = store.append_canonical_frame(frame)
        if debug_ring is not None:
            debug_ring.record_canonical(stored)
        return stored

    adapter = ShigureCompatibilityAdapter(append_canonical_frame)
    debug_ring = ShigureDebugDiskRing(
        args.debug_cache_root,
        enabled=args.debug_cache_enable,
        retention_seconds=args.debug_cache_retention_seconds,
        max_entries=args.debug_cache_max_entries,
        session_id=adapter.source_incarnation_id,
    )
    socket_server = ShigureHistorySocketServer(args.socket_server, store)
    socket_server.start()
    states: dict[str, TopicState] = {}
    subscriptions = []
    try:
        best_effort_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        canonical_best_effort_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        history_len = max(16, int(round(args.sample_hz * max(args.retention_seconds, 1.0))) * 2)
        for key, (topic, type_name) in settings.TOPIC_SPECS.items():
            msg_type = get_message(type_name)
            state = TopicState(key=key, topic=topic, type_name=type_name, maxlen=history_len)
            states[key] = state

            def callback(msg: Any, topic_key: str = key) -> None:
                sample = states[topic_key].append(msg)
                if topic_key in settings.CANONICAL_TOPIC_KEYS and sample.stamp is not None:
                    try:
                        header = getattr(msg, "header", None)
                        frame_id = str(getattr(header, "frame_id", "") or "")
                        adapter.ingest(
                            topic_key,
                            sample.stamp,
                            compatibility_payload(topic_key, sample),
                            frame_id=frame_id,
                        )
                    except Exception as exc:
                        print(
                            f"[shigure_history] canonical adapter rejected {topic_key} "
                            f"at {sample.stamp.to_dict()}: {exc}",
                            flush=True,
                        )

            qos = canonical_best_effort_qos if key in settings.CANONICAL_TOPIC_KEYS else best_effort_qos
            subscriptions.append(node.create_subscription(msg_type, topic, callback, qos))

        interval = 1.0 / max(0.1, float(args.sample_hz))
        next_sample = time.monotonic() + interval
        last_key: str | None = None
        last_counts: tuple[int, ...] | None = None
        last_log = 0.0
        sample_count = 0
        marker_warmup = (
            MarkerHistoryWarmup(
                target_detections=settings.SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS,
                max_attempts=settings.SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS,
                max_reprojection_error_px=settings.SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX,
                min_corner_area_px=settings.SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX,
            )
            if settings.SHIGURE_MARKER_HISTORY_WARMUP_ENABLE
            else None
        )
        print(
            "[shigure_history] started memory recorder: "
            f"socket={args.socket_server} sample_hz={args.sample_hz} "
            f"retention_seconds={args.retention_seconds} max_samples={args.max_samples} "
            f"max_frames={args.max_frames} "
            f"source_incarnation_id={adapter.source_incarnation_id} "
            f"debug_cache={'enabled' if debug_ring.enabled else 'disabled'} "
            f"debug_retention_seconds={debug_ring.retention_seconds}",
            flush=True,
        )
        for key, state in states.items():
            print(f"[shigure_history] subscribe {key}: {state.topic} [{state.type_name}]", flush=True)

        while _running:
            try:
                rclpy.spin_once(node, timeout_sec=0.05)
            except Exception as exc:
                if exc.__class__.__name__ == "ExternalShutdownException":
                    break
                raise
            now = time.monotonic()
            if now < next_sample:
                continue
            while next_sample <= now:
                next_sample += interval
            if not required_ready(states):
                if now - last_log >= max(1.0, float(args.log_interval)):
                    missing = [key for key in REQUIRED_KEYS if states[key].latest() is None]
                    optional_missing = [
                        key
                        for key in ("object_detection", "object_tracking", "segments", "people", "contacted")
                        if states.get(key) is not None and states[key].latest() is None
                    ]
                    print(f"[shigure_history] waiting for required topics: {missing}; optional_missing={optional_missing}", flush=True)
                    last_log = now
                continue
            counts = latest_required_counts(states)
            if counts == last_counts:
                continue
            try:
                last_key, appended, sample, info = append_aligned_sample(
                    store,
                    states,
                    last_key=last_key,
                    rgb_depth_max_delta_seconds=args.rgb_depth_max_delta_seconds,
                )
                last_counts = counts
                if appended and sample is not None:
                    sample_count += 1
                    record_debug_rgbd_once(sample)
                    marker_status = None
                    if marker_warmup is not None and not marker_warmup.completed:
                        marker_status = marker_warmup.process_sample(sample)
                    if now - last_log >= max(1.0, float(args.log_interval)):
                        print(
                            f"[shigure_history] cached samples={sample_count} latest={last_key} "
                            f"objects={info.get('object_detection_object_count')}",
                            flush=True,
                        )
                        if marker_status and marker_status.get("updated"):
                            print("[shigure_history] updated Shigurei ArMarker history: " + str(marker_status.get("latest_path")), flush=True)
                        last_log = now
                elif now - last_log >= max(1.0, float(args.log_interval)):
                    print(f"[shigure_history] skipped sample: {info.get('reason')}", flush=True)
                    last_log = now
            except Exception as exc:
                if now - last_log >= max(1.0, float(args.log_interval)):
                    print(f"[shigure_history] failed to append sample: {exc}", flush=True)
                    last_log = now

        print("[shigure_history] stopped", flush=True)
        return 0
    finally:
        subscriptions.clear()
        debug_ring.close(timeout=5.0)
        socket_server.stop()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
