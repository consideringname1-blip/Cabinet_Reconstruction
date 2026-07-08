#!/usr/bin/env python3
"""Receive Shigurei ROS2 streams and expose a one-minute in-memory RGB-D cache."""

from __future__ import annotations

import argparse
import base64
import hashlib
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    RosStamp,
    ShigureMemoryStore,
    sample_key,
    store_request,
    write_json,
)
from stages.shigure_history.marker_history import MarkerHistoryWarmup  # noqa: E402

_running = True
REQUIRED_KEYS = ("rgb", "depth", "camera_info")


@dataclass(frozen=True)
class TopicSample:
    message: Any
    stamp: RosStamp | None
    received_at: float
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

    def append(self, msg: Any) -> None:
        self.count += 1
        self.samples.append(
            TopicSample(
                message=msg,
                stamp=stamp_from_message(msg),
                received_at=time.time(),
                count=self.count,
            )
        )

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


def now_stamp() -> RosStamp:
    now = time.time()
    sec = int(now)
    return RosStamp(sec=sec, nanosec=int((now - sec) * 1_000_000_000))


def stamp_to_dict(stamp: Any | None) -> dict[str, int]:
    return {"sec": int(getattr(stamp, "sec", 0)), "nanosec": int(getattr(stamp, "nanosec", 0))}


def header_to_dict(header: Any | None) -> dict[str, Any]:
    return {"stamp": stamp_to_dict(getattr(header, "stamp", None)), "frame_id": str(getattr(header, "frame_id", ""))}


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


def camera_info_payload(sample: TopicSample, state: TopicState) -> dict[str, Any]:
    msg = sample.message
    return {
        "topic": state.topic,
        "message_type": state.type_name,
        "received_at": datetime.fromtimestamp(sample.received_at, timezone.utc).isoformat(),
        "header": header_to_dict(getattr(msg, "header", None)),
        "message": message_to_jsonable(msg),
    }


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
    if width <= 0.0 or height <= 0.0:
        return None
    return x, y, x + width, y + height


def object_detection_payload(sample: TopicSample, state: TopicState) -> dict[str, Any]:
    msg = sample.message
    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(getattr(msg, "object_list", []) or []):
        bbox = _object_bbox_xyxy(obj)
        if bbox is None:
            continue
        mask_msg = getattr(obj, "mask", None)
        mask_raw = bytes(getattr(mask_msg, "data", b"")) if mask_msg is not None else b""
        if not mask_raw:
            continue
        x0, y0, x1, y1 = bbox
        action = str(getattr(obj, "action", "object") or "object")
        objects.append(
            {
                "object_id": f"{action}:{index}",
                "action": action,
                "bbox": [x0, y0, x1, y1],
                "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
                "x": (x0 + x1) * 0.5,
                "y": (y0 + y1) * 0.5,
                "mask_b64": base64.b64encode(mask_raw).decode("ascii"),
                "mask_format": str(getattr(mask_msg, "format", "")) if mask_msg is not None else "",
                "mask_bytes": len(mask_raw),
            }
        )
    header = getattr(msg, "header", None)
    return {
        "source": "shigure_object_detection",
        "topic": state.topic,
        "message_type": state.type_name,
        "received_at": datetime.fromtimestamp(sample.received_at, timezone.utc).isoformat(),
        "header": header_to_dict(header),
        "stamp": header_to_dict(header).get("stamp"),
        "objects": objects,
        "object_count": len(objects),
    }


def payload_hash(payload: dict[str, Any] | None) -> str | None:
    if not payload or not payload.get("objects"):
        return None
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def selected_rgb_stamp(rgb_sample: TopicSample) -> RosStamp:
    return rgb_sample.stamp or now_stamp()


def required_ready(states: dict[str, TopicState]) -> bool:
    return all(states[key].latest() is not None for key in REQUIRED_KEYS)


def latest_required_counts(states: dict[str, TopicState]) -> tuple[int, ...]:
    return tuple(states[key].count for key in REQUIRED_KEYS)


def write_status(root: Path, payload: dict[str, Any]) -> None:
    status = {"updated_at": datetime.now(timezone.utc).isoformat(), **payload}
    try:
        write_json(root / "recorder_status.json", status)
    except Exception as exc:
        print(f"[shigure_history] failed to write recorder_status.json: {exc}", flush=True)


def append_aligned_sample(
    store: ShigureMemoryStore,
    states: dict[str, TopicState],
    *,
    last_key: str | None,
    frame_index: int,
    rgb_depth_max_delta_seconds: float,
    object_detection_max_delta_seconds: float,
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
    camera_info = camera_info_payload(camera_info_sample, states["camera_info"])

    object_payload = None
    object_delta = None
    object_sample = states.get("object_detection").nearest(stamp, max_delta_seconds=object_detection_max_delta_seconds) if states.get("object_detection") else None
    if object_sample is not None:
        object_payload = object_detection_payload(object_sample, states["object_detection"])
        if object_sample.stamp is not None:
            object_delta = abs(float(object_sample.stamp.seconds) - float(stamp.seconds))

    yolo_hash = payload_hash(object_payload)
    sample = CachedRgbdSample(
        stamp=stamp,
        rgb_bgr=rgb,
        depth=depth,
        camera_info_path=None,
        camera_info=camera_info,
        yolo=object_payload,
        yolo_hash=yolo_hash,
        chunk_id="memory",
        frame_index=frame_index,
    )
    store.append(sample)
    return key, True, sample, {
        "rgb_decode": rgb_info,
        "depth_decode": depth_info,
        "frame": sample.to_dict(),
        "object_detection_paired": object_payload is not None,
        "object_detection_delta_seconds": object_delta,
        "object_detection_object_count": int((object_payload or {}).get("object_count") or 0),
    }


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
    parser.add_argument("--cache-root", type=Path, default=settings.SHIGURE_HISTORY_CACHE_ROOT)
    parser.add_argument("--socket-server", type=Path, default=settings.SHIGURE_HISTORY_SOCKET_PATH)
    parser.add_argument("--sample-hz", type=float, default=settings.SHIGURE_HISTORY_HZ)
    parser.add_argument("--retention-seconds", type=float, default=settings.SHIGURE_HISTORY_SECONDS)
    parser.add_argument("--max-samples", type=int, default=settings.SHIGURE_HISTORY_MAX_SAMPLES)
    parser.add_argument("--log-interval", type=float, default=settings.SHIGURE_HISTORY_RECORDER_LOG_INTERVAL)
    parser.add_argument("--rgb-depth-max-delta-seconds", type=float, default=settings.SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS)
    parser.add_argument("--object-detection-max-delta-seconds", type=float, default=settings.SHIGURE_HISTORY_OBJECT_DETECTION_MAX_DELTA_SECONDS)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rclpy, Node, QoSProfile, ReliabilityPolicy, get_message = import_ros_modules()
    rclpy.init(args=None)
    node = Node("shigure_memory_history_recorder")
    store = ShigureMemoryStore(max_seconds=args.retention_seconds, max_samples=args.max_samples)
    socket_server = ShigureHistorySocketServer(args.socket_server, store)
    socket_server.start()
    states: dict[str, TopicState] = {}
    subscriptions = []
    try:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        history_len = max(16, int(round(args.sample_hz * max(args.retention_seconds, 1.0))) * 2)
        for key, (topic, type_name) in settings.TOPIC_SPECS.items():
            msg_type = get_message(type_name)
            state = TopicState(key=key, topic=topic, type_name=type_name, maxlen=history_len)
            states[key] = state

            def callback(msg: Any, topic_key: str = key) -> None:
                states[topic_key].append(msg)

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
            f"retention_seconds={args.retention_seconds} max_samples={args.max_samples}",
            flush=True,
        )
        for key, state in states.items():
            print(f"[shigure_history] subscribe {key}: {state.topic} [{state.type_name}]", flush=True)

        while _running:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now < next_sample:
                continue
            while next_sample <= now:
                next_sample += interval
            if not required_ready(states):
                if now - last_log >= max(1.0, float(args.log_interval)):
                    missing = [key for key in REQUIRED_KEYS if states[key].latest() is None]
                    optional_missing = [key for key in ("object_detection",) if states.get(key) is not None and states[key].latest() is None]
                    print(f"[shigure_history] waiting for required topics: {missing}; optional_missing={optional_missing}", flush=True)
                    write_status(
                        args.cache_root,
                        {
                            "running": True,
                            "waiting_for": missing,
                            "optional_missing": optional_missing,
                            "topic_counts": {name: state.count for name, state in states.items()},
                            "store": store.status(),
                        },
                    )
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
                    frame_index=sample_count,
                    rgb_depth_max_delta_seconds=args.rgb_depth_max_delta_seconds,
                    object_detection_max_delta_seconds=args.object_detection_max_delta_seconds,
                )
                last_counts = counts
                if appended and sample is not None:
                    sample_count += 1
                    marker_status = None
                    if marker_warmup is not None and not marker_warmup.completed:
                        marker_status = marker_warmup.process_sample(sample)
                    write_status(
                        args.cache_root,
                        {
                            "running": True,
                            "last_sample_key": last_key,
                            "last_sample": sample.to_dict(),
                            "topic_counts": {name: state.count for name, state in states.items()},
                            "store": store.status(),
                            **info,
                        },
                    )
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
                    write_status(args.cache_root, {"running": True, "error": str(exc), "store": store.status()})
                    last_log = now

        write_status(args.cache_root, {"running": False, "stopped_at": datetime.now(timezone.utc).isoformat(), "store": store.status()})
        print("[shigure_history] stopped", flush=True)
        return 0
    finally:
        subscriptions.clear()
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
