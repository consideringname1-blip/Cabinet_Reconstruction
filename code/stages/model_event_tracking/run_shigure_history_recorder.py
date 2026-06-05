#!/usr/bin/env python3
"""Continuously cache the minimal Shigurei RGB-D + skeleton history for model events.

This process is owned by task_worker. It subscribes to the Shigurei/RealSense
ROS2 topics, decodes only the data needed for model-event tracking, and writes a
10 minute disk ring buffer through ShigureHistoryCache.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[2]
ROS_SETUP_ENV = CODE_ROOT / "ros2" / "shigure_recv_ws" / "setup_env.sh"
ROS_PYTHON = Path(os.environ.get("SHIGURE_EVENT_RECORDER_ROS_PYTHON", "/usr/bin/python3"))
SERVER_SITE_PACKAGES = Path(
    os.environ.get(
        "SHIGURE_EVENT_RECORDER_CV_SITE_PACKAGES",
        "/opt/miniconda/envs/server/lib/python3.10/site-packages",
    )
)
BOOTSTRAP_ENV_KEY = "SHIGURE_EVENT_RECORDER_BOOTSTRAPPED"


def _site_packages_version(path: Path) -> tuple[int, int] | None:
    for part in path.parts:
        match = re.fullmatch(r"python(\d+)\.(\d+)", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _site_packages_match_current_python(path: Path) -> bool:
    version = _site_packages_version(path)
    if version is None:
        return True
    return version == (sys.version_info.major, sys.version_info.minor)


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


def _site_packages_match_python_bin(path: Path, python_bin: Path) -> bool:
    site_version = _site_packages_version(path)
    if site_version is None:
        return True
    python_version = _python_bin_version(python_bin)
    return python_version == site_version


def bootstrap_runtime_environment() -> None:
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    if (
        SERVER_SITE_PACKAGES.is_dir()
        and _site_packages_match_current_python(SERVER_SITE_PACKAGES)
        and str(SERVER_SITE_PACKAGES) not in sys.path
    ):
        sys.path.insert(0, str(SERVER_SITE_PACKAGES))

    if os.environ.get(BOOTSTRAP_ENV_KEY) == "1":
        return
    if not ROS_SETUP_ENV.exists():
        return

    os.environ[BOOTSTRAP_ENV_KEY] = "1"
    python_bin = ROS_PYTHON if ROS_PYTHON.exists() else Path(sys.executable)
    env_export = ""
    if SERVER_SITE_PACKAGES.is_dir() and _site_packages_match_python_bin(SERVER_SITE_PACKAGES, python_bin):
        env_export = f"export PYTHONPATH={shlex.quote(str(SERVER_SITE_PACKAGES))}:${{PYTHONPATH}} && "
    command = (
        f"source {shlex.quote(str(ROS_SETUP_ENV))} && "
        f"{env_export}exec {shlex.quote(str(python_bin))} "
        f"{shlex.quote(str(SCRIPT_PATH))} "
        f"{chr(32).join(shlex.quote(arg) for arg in sys.argv[1:])}"
    )
    os.execvp("bash", ["bash", "-lc", command])


bootstrap_runtime_environment()

from stages.model_event_tracking import settings  # noqa: E402
from stages.model_event_tracking.cache import ShigureHistoryCache, frame_key, write_json  # noqa: E402
from stages.model_event_tracking.schemas import RosStamp  # noqa: E402


TOPIC_SPECS: dict[str, tuple[str, str]] = {
    "rgb": (
        os.environ.get("SHIGURE_EVENT_RGB_TOPIC", "/rs/color/compressed"),
        os.environ.get("SHIGURE_EVENT_RGB_TYPE", "sensor_msgs/msg/CompressedImage"),
    ),
    "depth": (
        os.environ.get("SHIGURE_EVENT_DEPTH_TOPIC", "/rs/aligned_depth_to_color/compressedDepth"),
        os.environ.get("SHIGURE_EVENT_DEPTH_TYPE", "sensor_msgs/msg/CompressedImage"),
    ),
    "camera_info": (
        os.environ.get("SHIGURE_EVENT_CAMERA_INFO_TOPIC", "/rs/aligned_depth_to_color/cameraInfo"),
        os.environ.get("SHIGURE_EVENT_CAMERA_INFO_TYPE", "sensor_msgs/msg/CameraInfo"),
    ),
    "people": (
        os.environ.get("SHIGURE_EVENT_PEOPLE_TOPIC", "/shigure/people_detection"),
        os.environ.get("SHIGURE_EVENT_PEOPLE_TYPE", "shigure_core_msgs/msg/PoseKeyPointsList"),
    ),
}
REQUIRED_KEYS = ("rgb", "depth", "camera_info")
_running = True


@dataclass
class TopicState:
    key: str
    topic: str
    type_name: str
    message: Any | None = None
    received_at: float | None = None
    count: int = 0


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    global _running
    _running = False


for _signum in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_signum, _handle_signal)
    except Exception:
        pass


def import_ros_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs.msg import CompressedImage
    except ImportError as exc:
        raise SystemExit(
            "ROS2 Python modules are not available. The recorder expects "
            f"{ROS_SETUP_ENV} to be sourceable. Import error: {exc}"
        ) from exc
    return rclpy, Node, QoSProfile, ReliabilityPolicy, get_message, CompressedImage


def stamp_to_dict(stamp: Any) -> dict[str, int]:
    return {
        "sec": int(getattr(stamp, "sec", 0)),
        "nanosec": int(getattr(stamp, "nanosec", 0)),
    }


def header_to_dict(header: Any) -> dict[str, Any]:
    return {
        "stamp": stamp_to_dict(getattr(header, "stamp", None)),
        "frame_id": str(getattr(header, "frame_id", "")),
    }


def stamp_from_message(msg: Any | None) -> RosStamp | None:
    if msg is None:
        return None
    header = getattr(msg, "header", None)
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


def bytes_to_json_blob(data: bytes | bytearray | array) -> dict[str, Any]:
    raw = bytes(data)
    return {
        "__encoding__": "base64",
        "length": len(raw),
        "data": base64.b64encode(raw).decode("ascii"),
    }


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
        return {
            field_name: message_to_jsonable(getattr(value, field_name))
            for field_name in value.get_fields_and_field_types().keys()
        }
    return str(value)


def write_message_json(path: Path, state: TopicState) -> None:
    write_json(
        path,
        {
            "topic": state.topic,
            "message_type": state.type_name,
            "received_at": datetime.fromtimestamp(state.received_at or time.time(), timezone.utc).isoformat(),
            "message": message_to_jsonable(state.message),
        },
    )


def compressed_image_payload(msg: Any, topic: str) -> tuple[bytes, int]:
    raw = bytes(msg.data)
    fmt = str(getattr(msg, "format", ""))
    is_depth = "compressedDepth" in fmt or "depth" in topic.lower()
    if is_depth and b"PNG" not in raw[:12]:
        return raw[12:], 12
    return raw, 0


def write_decoded_compressed_image(path: Path, state: TopicState) -> dict[str, Any]:
    import cv2
    import numpy as np

    payload, skipped_header_bytes = compressed_image_payload(state.message, state.topic)
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cv2.imdecode returned None for {state.topic}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"cv2.imwrite failed for {path}")
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "skipped_header_bytes": skipped_header_bytes,
    }


def latest_required_counts(states: dict[str, TopicState]) -> tuple[int, ...]:
    return tuple(states[key].count for key in REQUIRED_KEYS)


def required_ready(states: dict[str, TopicState]) -> bool:
    return all(states[key].message is not None for key in REQUIRED_KEYS)


def selected_stamp(states: dict[str, TopicState]) -> RosStamp:
    return (
        stamp_from_message(states["rgb"].message)
        or stamp_from_message(states["depth"].message)
        or stamp_from_message(states["camera_info"].message)
        or now_stamp()
    )


def marker_pose_path_from_env() -> Path | None:
    value = (
        os.environ.get("SHIGURE_EVENT_MARKER_POSE_JSON")
        or os.environ.get("MODEL_EVENT_MARKER_POSE_JSON")
        or ""
    ).strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def write_status(cache: ShigureHistoryCache, payload: dict[str, Any]) -> None:
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    try:
        write_json(cache.root / "recorder_status.json", status)
    except Exception as exc:
        print(f"[shigure_recorder] failed to write recorder_status.json: {exc}", flush=True)


def append_cached_frame(
    cache: ShigureHistoryCache,
    states: dict[str, TopicState],
    *,
    last_key: str | None,
) -> tuple[str | None, bool]:
    stamp = selected_stamp(states)
    key = frame_key(stamp)
    if key == last_key:
        return last_key, False

    tmp_root = cache.root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frame_", dir=str(tmp_root)) as tmp_name:
        tmp = Path(tmp_name)
        rgb_path = tmp / "rgb.png"
        depth_path = tmp / "depth.png"
        camera_info_path = tmp / "camera_info.json"
        people_path = tmp / "people_detection.json"

        rgb_info = write_decoded_compressed_image(rgb_path, states["rgb"])
        depth_info = write_decoded_compressed_image(depth_path, states["depth"])
        write_message_json(camera_info_path, states["camera_info"])

        people_source: Path | None = None
        if states["people"].message is not None:
            write_message_json(people_path, states["people"])
            people_source = people_path

        frame = cache.append_frame(
            stamp=stamp,
            rgb_path=rgb_path,
            depth_path=depth_path,
            camera_info_path=camera_info_path,
            people_path=people_source,
            marker_pose_path=marker_pose_path_from_env(),
        )
        write_status(
            cache,
            {
                "running": True,
                "last_frame_key": key,
                "last_frame": frame.to_dict(),
                "topic_counts": {name: state.count for name, state in states.items()},
                "rgb_decode": rgb_info,
                "depth_decode": depth_info,
            },
        )
    return key, True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=settings.SHIGURE_EVENT_CACHE_ROOT)
    parser.add_argument("--sample-hz", type=float, default=settings.SHIGURE_HISTORY_HZ)
    parser.add_argument("--retention-seconds", type=float, default=settings.SHIGURE_HISTORY_SECONDS)
    parser.add_argument("--log-interval", type=float, default=float(os.environ.get("SHIGURE_EVENT_RECORDER_LOG_INTERVAL", "10")))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rclpy, Node, QoSProfile, ReliabilityPolicy, get_message, _CompressedImage = import_ros_modules()

    rclpy.init(args=None)
    node = Node("shigure_event_history_recorder")
    cache = ShigureHistoryCache(
        args.cache_root,
        retention_seconds=args.retention_seconds,
        sample_hz=args.sample_hz,
    )
    states: dict[str, TopicState] = {}
    subscriptions = []

    try:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for key, (topic, type_name) in TOPIC_SPECS.items():
            try:
                msg_type = get_message(type_name)
            except Exception as exc:
                if key in REQUIRED_KEYS:
                    raise RuntimeError(f"required topic type unavailable: {topic} [{type_name}]: {exc}") from exc
                print(f"[shigure_recorder] skip optional topic {topic} [{type_name}]: {exc}", flush=True)
                continue

            state = TopicState(key=key, topic=topic, type_name=type_name)
            states[key] = state

            def callback(msg: Any, topic_key: str = key) -> None:
                current = states[topic_key]
                current.message = msg
                current.received_at = time.time()
                current.count += 1

            subscriptions.append(node.create_subscription(msg_type, topic, callback, qos))

        missing_required = [key for key in REQUIRED_KEYS if key not in states]
        if missing_required:
            raise RuntimeError(f"required recorder subscriptions are missing: {missing_required}")

        interval = 1.0 / max(0.1, float(args.sample_hz))
        next_sample = time.monotonic() + interval
        last_key: str | None = None
        last_counts: tuple[int, ...] | None = None
        last_log = 0.0
        frame_count = 0
        print(
            "[shigure_recorder] started: cache_root="
            + str(args.cache_root)
            + f" sample_hz={args.sample_hz} retention_seconds={args.retention_seconds}",
            flush=True,
        )
        for key, state in states.items():
            print(f"[shigure_recorder] subscribe {key}: {state.topic} [{state.type_name}]", flush=True)

        while _running:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now < next_sample:
                continue
            while next_sample <= now:
                next_sample += interval

            if not required_ready(states):
                if now - last_log >= max(1.0, float(args.log_interval)):
                    missing = [key for key in REQUIRED_KEYS if states[key].message is None]
                    print(f"[shigure_recorder] waiting for required topics: {missing}", flush=True)
                    write_status(
                        cache,
                        {
                            "running": True,
                            "waiting_for": missing,
                            "topic_counts": {name: state.count for name, state in states.items()},
                        },
                    )
                    last_log = now
                continue

            counts = latest_required_counts(states)
            if counts == last_counts:
                continue
            try:
                last_key, appended = append_cached_frame(cache, states, last_key=last_key)
                last_counts = counts
                if appended:
                    frame_count += 1
                    if now - last_log >= max(1.0, float(args.log_interval)):
                        print(
                            f"[shigure_recorder] cached frames={frame_count} latest={last_key}",
                            flush=True,
                        )
                        last_log = now
            except Exception as exc:
                if now - last_log >= max(1.0, float(args.log_interval)):
                    print(f"[shigure_recorder] failed to append frame: {exc}", flush=True)
                    write_status(cache, {"running": True, "error": str(exc)})
                    last_log = now

        write_status(cache, {"running": False, "stopped_at": datetime.now(timezone.utc).isoformat()})
        print("[shigure_recorder] stopped", flush=True)
        return 0
    finally:
        subscriptions.clear()
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
