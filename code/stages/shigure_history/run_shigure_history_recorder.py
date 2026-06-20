#!/usr/bin/env python3
"""Continuously cache Shigurei RGB-D and YOLO data as compressed chunks."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[2]
ROS_SETUP_ENV = CODE_ROOT / 'ros2' / 'shigure_recv_ws' / 'setup_env.sh'
ROS_PYTHON = Path(os.environ.get('SHIGURE_HISTORY_RECORDER_ROS_PYTHON', '/usr/bin/python3'))
SERVER_SITE_PACKAGES = Path(os.environ.get('SHIGURE_HISTORY_RECORDER_CV_SITE_PACKAGES', '/opt/miniconda/envs/server/lib/python3.10/site-packages'))
BOOTSTRAP_ENV_KEY = 'SHIGURE_HISTORY_RECORDER_BOOTSTRAPPED'


def _site_packages_version(path: Path) -> tuple[int, int] | None:
    for part in path.parts:
        match = re.fullmatch(r'python(\d+)\.(\d+)', part)
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
        output = subprocess.check_output([str(python_bin), '-c', "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True, stderr=subprocess.DEVNULL).strip()
        major, minor = output.split('.', 1)
        return int(major), int(minor)
    except Exception:
        return None


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
    if os.environ.get(BOOTSTRAP_ENV_KEY) == '1' or not ROS_SETUP_ENV.exists():
        return
    os.environ[BOOTSTRAP_ENV_KEY] = '1'
    python_bin = ROS_PYTHON if ROS_PYTHON.exists() else Path(sys.executable)
    env_export = ''
    if SERVER_SITE_PACKAGES.is_dir() and _site_packages_match_python_bin(SERVER_SITE_PACKAGES, python_bin):
        env_export = f'export PYTHONPATH={shlex.quote(str(SERVER_SITE_PACKAGES))}:${{PYTHONPATH}} && '
    command = f'source {shlex.quote(str(ROS_SETUP_ENV))} && {env_export}exec {shlex.quote(str(python_bin))} {shlex.quote(str(SCRIPT_PATH))} {chr(32).join(shlex.quote(arg) for arg in sys.argv[1:])}'
    os.execvp('bash', ['bash', '-lc', command])


bootstrap_runtime_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stages.shigure_history import settings  # noqa: E402
from stages.shigure_history.cache import CachedRgbdSample, ChunkedShigureHistoryWriter, RosStamp, sample_key, write_json  # noqa: E402
from stages.shigure_history.marker_history import MarkerHistoryWarmup  # noqa: E402

REQUIRED_KEYS = ('rgb', 'depth', 'camera_info', 'active_objects')
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


def import_ros_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit(f'ROS2 Python modules are not available. Expected {ROS_SETUP_ENV}. Import error: {exc}') from exc
    return rclpy, Node, QoSProfile, ReliabilityPolicy, get_message


def stamp_from_message(msg: Any | None) -> RosStamp | None:
    header = getattr(msg, 'header', None) if msg is not None else None
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, 'sec', 0))
    nanosec = int(getattr(stamp, 'nanosec', 0))
    if sec == 0 and nanosec == 0:
        return None
    return RosStamp(sec=sec, nanosec=nanosec)


def now_stamp() -> RosStamp:
    now = time.time()
    sec = int(now)
    return RosStamp(sec=sec, nanosec=int((now - sec) * 1_000_000_000))


def stamp_to_dict(stamp: Any | None) -> dict[str, int]:
    return {'sec': int(getattr(stamp, 'sec', 0)), 'nanosec': int(getattr(stamp, 'nanosec', 0))}


def header_to_dict(header: Any | None) -> dict[str, Any]:
    return {'stamp': stamp_to_dict(getattr(header, 'stamp', None)), 'frame_id': str(getattr(header, 'frame_id', ''))}


def bytes_to_json_blob(data: bytes | bytearray | array) -> dict[str, Any]:
    raw = bytes(data)
    return {'__encoding__': 'base64', 'length': len(raw)}


def message_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes_to_json_blob(value)
    if isinstance(value, array):
        if value.typecode in ('b', 'B'):
            return bytes_to_json_blob(value)
        return [message_to_jsonable(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [message_to_jsonable(item) for item in value]
    if hasattr(value, 'get_fields_and_field_types'):
        return {field_name: message_to_jsonable(getattr(value, field_name)) for field_name in value.get_fields_and_field_types().keys()}
    return str(value)


def camera_info_payload(state: TopicState) -> dict[str, Any]:
    msg = state.message
    return {
        'topic': state.topic,
        'message_type': state.type_name,
        'received_at': datetime.fromtimestamp(state.received_at or time.time(), timezone.utc).isoformat(),
        'header': header_to_dict(getattr(msg, 'header', None)),
        'message': message_to_jsonable(msg),
    }


def compressed_image_payload(msg: Any, topic: str) -> tuple[bytes, int]:
    raw = bytes(msg.data)
    fmt = str(getattr(msg, 'format', ''))
    is_depth = 'compressedDepth' in fmt or 'depth' in topic.lower()
    if is_depth and b'PNG' not in raw[:12]:
        return raw[12:], 12
    return raw, 0


def decode_compressed_image(state: TopicState, *, color: bool) -> tuple[np.ndarray, dict[str, Any]]:
    payload, skipped_header_bytes = compressed_image_payload(state.message, state.topic)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f'cv2.imdecode returned None for {state.topic}')
    if not color and image.ndim == 3:
        image = image[:, :, 0]
    return image, {'shape': list(image.shape), 'dtype': str(image.dtype), 'skipped_header_bytes': skipped_header_bytes}


def latest_required_counts(states: dict[str, TopicState]) -> tuple[int, ...]:
    return tuple(states[key].count for key in REQUIRED_KEYS)


def required_ready(states: dict[str, TopicState]) -> bool:
    return all(states[key].message is not None for key in REQUIRED_KEYS)


def selected_stamp(states: dict[str, TopicState]) -> RosStamp:
    return stamp_from_message(states['rgb'].message) or stamp_from_message(states['depth'].message) or stamp_from_message(states['camera_info'].message) or now_stamp()


def write_status(root: Path, payload: dict[str, Any]) -> None:
    status = {'updated_at': datetime.now(timezone.utc).isoformat(), **payload}
    try:
        write_json(root / 'recorder_status.json', status)
    except Exception as exc:
        print(f'[shigure_history] failed to write recorder_status.json: {exc}', flush=True)


def append_cached_sample(writer: ChunkedShigureHistoryWriter, states: dict[str, TopicState], *, last_key: str | None) -> tuple[str | None, bool, CachedRgbdSample | None, dict[str, Any]]:
    stamp = selected_stamp(states)
    key = sample_key(stamp)
    if key == last_key:
        return last_key, False, None, {}
    rgb, rgb_info = decode_compressed_image(states['rgb'], color=True)
    depth, depth_info = decode_compressed_image(states['depth'], color=False)
    camera_info = camera_info_payload(states['camera_info'])
    yolo_payload = str(getattr(states['active_objects'].message, 'data', ''))
    headers = {
        'rgb': header_to_dict(getattr(states['rgb'].message, 'header', None)),
        'depth': header_to_dict(getattr(states['depth'].message, 'header', None)),
        'camera_info': header_to_dict(getattr(states['camera_info'].message, 'header', None)),
    }
    topic_counts = {name: state.count for name, state in states.items()}
    frame = writer.append_sample(stamp=stamp, rgb_bgr=rgb, depth=depth, camera_info=camera_info, yolo_payload=yolo_payload, headers=headers, topic_counts=topic_counts)
    sample = CachedRgbdSample(stamp=stamp, rgb_bgr=rgb, depth=depth, camera_info_path=None, camera_info=camera_info, yolo=None, yolo_hash=frame.get('yolo_hash'), chunk_id=frame.get('chunk_id'), frame_index=int(frame.get('frame_index', 0)))
    return key, True, sample, {'rgb_decode': rgb_info, 'depth_decode': depth_info, 'frame': frame}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=settings.SHIGURE_HISTORY_CACHE_ROOT)
    parser.add_argument('--sample-hz', type=float, default=settings.SHIGURE_HISTORY_HZ)
    parser.add_argument('--retention-seconds', type=float, default=settings.SHIGURE_HISTORY_SECONDS)
    parser.add_argument('--chunk-seconds', type=float, default=settings.SHIGURE_HISTORY_CHUNK_SECONDS)
    parser.add_argument('--log-interval', type=float, default=settings.SHIGURE_HISTORY_RECORDER_LOG_INTERVAL)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rclpy, Node, QoSProfile, ReliabilityPolicy, get_message = import_ros_modules()
    rclpy.init(args=None)
    node = Node('shigure_chunked_history_recorder')
    writer = ChunkedShigureHistoryWriter(args.cache_root, retention_seconds=args.retention_seconds, sample_hz=args.sample_hz, chunk_seconds=args.chunk_seconds)
    states: dict[str, TopicState] = {}
    subscriptions = []
    try:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for key, (topic, type_name) in settings.TOPIC_SPECS.items():
            msg_type = get_message(type_name)
            state = TopicState(key=key, topic=topic, type_name=type_name)
            states[key] = state

            def callback(msg: Any, topic_key: str = key) -> None:
                current = states[topic_key]
                current.message = msg
                current.received_at = time.time()
                current.count += 1

            subscriptions.append(node.create_subscription(msg_type, topic, callback, qos))
        interval = 1.0 / max(0.1, float(args.sample_hz))
        next_sample = time.monotonic() + interval
        last_key: str | None = None
        last_counts: tuple[int, ...] | None = None
        last_log = 0.0
        sample_count = 0
        marker_warmup = MarkerHistoryWarmup(target_detections=settings.SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS, max_attempts=settings.SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS, max_reprojection_error_px=settings.SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX, min_corner_area_px=settings.SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX) if settings.SHIGURE_MARKER_HISTORY_WARMUP_ENABLE else None
        print(f'[shigure_history] started chunked recorder: cache_root={args.cache_root} sample_hz={args.sample_hz} chunk_seconds={args.chunk_seconds} retention_seconds={args.retention_seconds}', flush=True)
        for key, state in states.items():
            print(f'[shigure_history] subscribe {key}: {state.topic} [{state.type_name}]', flush=True)
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
                    print(f'[shigure_history] waiting for required topics: {missing}', flush=True)
                    write_status(args.cache_root, {'running': True, 'waiting_for': missing, 'topic_counts': {name: state.count for name, state in states.items()}})
                    last_log = now
                continue
            counts = latest_required_counts(states)
            if counts == last_counts:
                continue
            try:
                last_key, appended, sample, info = append_cached_sample(writer, states, last_key=last_key)
                last_counts = counts
                if appended and sample is not None:
                    sample_count += 1
                    marker_status = None
                    if marker_warmup is not None and not marker_warmup.completed:
                        marker_status = marker_warmup.process_sample(sample)
                    if sample_count % max(1, writer.max_frames_per_chunk) == 0:
                        writer.prune(newest_stamp=sample.stamp)
                    write_status(args.cache_root, {'running': True, 'last_sample_key': last_key, 'last_sample': sample.to_dict(), 'topic_counts': {name: state.count for name, state in states.items()}, **info})
                    if now - last_log >= max(1.0, float(args.log_interval)):
                        print(f'[shigure_history] cached samples={sample_count} latest={last_key} chunk={sample.chunk_id}', flush=True)
                        if marker_status and marker_status.get('updated'):
                            print('[shigure_history] updated Shigurei ArMarker history: ' + str(marker_status.get('latest_path')), flush=True)
                        last_log = now
            except Exception as exc:
                if now - last_log >= max(1.0, float(args.log_interval)):
                    print(f'[shigure_history] failed to append sample: {exc}', flush=True)
                    write_status(args.cache_root, {'running': True, 'error': str(exc)})
                    last_log = now
        writer.close()
        write_status(args.cache_root, {'running': False, 'stopped_at': datetime.now(timezone.utc).isoformat()})
        print('[shigure_history] stopped', flush=True)
        return 0
    finally:
        try:
            writer.close()
        except Exception:
            pass
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


if __name__ == '__main__':
    raise SystemExit(main())
