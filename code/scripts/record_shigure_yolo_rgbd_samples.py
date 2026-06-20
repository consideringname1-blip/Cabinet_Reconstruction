#!/usr/bin/env python3
"""Record YOLO active-object JSON together with Shigurei RGB-D samples."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
ROS_SETUP_ENV = CODE_ROOT / 'ros2' / 'shigure_recv_ws' / 'setup_env.sh'
ROS_PYTHON = Path(os.environ.get('SHIGURE_YOLO_RGBD_ROS_PYTHON', '/usr/bin/python3'))
SERVER_SITE_PACKAGES = Path(
    os.environ.get(
        'SHIGURE_YOLO_RGBD_CV_SITE_PACKAGES',
        '/opt/miniconda/envs/server/lib/python3.10/site-packages',
    )
)
BOOTSTRAP_ENV_KEY = 'SHIGURE_YOLO_RGBD_BOOTSTRAPPED'


def _site_packages_version(path: Path) -> tuple[int, int] | None:
    for part in path.parts:
        match = re.fullmatch(r'python(\d+)\.(\d+)', part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _python_bin_version(python_bin: Path) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output(
            [str(python_bin), '-c', "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        major, minor = output.split('.', 1)
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

    if os.environ.get(BOOTSTRAP_ENV_KEY) == '1':
        return
    if not ROS_SETUP_ENV.exists():
        return

    os.environ[BOOTSTRAP_ENV_KEY] = '1'
    python_bin = ROS_PYTHON if ROS_PYTHON.exists() else Path(sys.executable)
    env_export = ''
    if SERVER_SITE_PACKAGES.is_dir() and _site_packages_match_python_bin(SERVER_SITE_PACKAGES, python_bin):
        env_export = f'export PYTHONPATH={shlex.quote(str(SERVER_SITE_PACKAGES))}:${{PYTHONPATH}} && '
    command = (
        f'source {shlex.quote(str(ROS_SETUP_ENV))} && '
        f'{env_export}exec {shlex.quote(str(python_bin))} '
        f'{shlex.quote(str(SCRIPT_PATH))} '
        f'{chr(32).join(shlex.quote(arg) for arg in sys.argv[1:])}'
    )
    os.execvp('bash', ['bash', '-lc', command])


bootstrap_runtime_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class TopicState:
    message: Any | None = None
    received_at: float | None = None
    count: int = 0


@dataclass
class LatestState:
    rgb: TopicState
    depth: TopicState
    camera_info: TopicState
    active_objects: TopicState


def import_ros_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, CompressedImage
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit(
            'ROS2 Python modules are not available. Source '
            f'{ROS_SETUP_ENV} or run this script through its bootstrap. Import error: {exc}'
        ) from exc
    return rclpy, Node, QoSProfile, ReliabilityPolicy, CameraInfo, CompressedImage, String


def stamp_to_dict(stamp: Any | None) -> dict[str, int]:
    return {
        'sec': int(getattr(stamp, 'sec', 0)),
        'nanosec': int(getattr(stamp, 'nanosec', 0)),
    }


def header_to_dict(header: Any | None) -> dict[str, Any]:
    return {
        'stamp': stamp_to_dict(getattr(header, 'stamp', None)),
        'frame_id': str(getattr(header, 'frame_id', '')),
    }


def stamp_key(stamp: Any | None) -> str:
    payload = stamp_to_dict(stamp)
    return f"{payload['sec']:010d}_{payload['nanosec']:09d}"


def now_key() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')


def bytes_to_json_blob(value: bytes | bytearray | array) -> dict[str, Any]:
    raw = bytes(value)
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
        return {
            field_name: message_to_jsonable(getattr(value, field_name))
            for field_name in value.get_fields_and_field_types().keys()
        }
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def compressed_image_payload(msg: Any, topic: str) -> tuple[bytes, int]:
    raw = bytes(msg.data)
    fmt = str(getattr(msg, 'format', ''))
    is_depth = 'compressedDepth' in fmt or 'depth' in topic.lower()
    if is_depth and b'PNG' not in raw[:12]:
        return raw[12:], 12
    return raw, 0


def decode_compressed_image(msg: Any, topic: str, *, color: bool) -> tuple[np.ndarray, dict[str, Any]]:
    payload, skipped_header_bytes = compressed_image_payload(msg, topic)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f'cv2.imdecode failed for {topic}')
    if not color and image.ndim == 3:
        image = image[:, :, 0]
    return image, {
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'skipped_header_bytes': skipped_header_bytes,
        'compressed_format': str(getattr(msg, 'format', '')),
        'compressed_length': len(bytes(msg.data)),
    }


def parse_active_objects(payload: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(payload), None
    except Exception as exc:
        return None, str(exc)


def message_age_seconds(state: TopicState, now: float) -> float | None:
    if state.received_at is None:
        return None
    return float(now - state.received_at)


def build_sample_dir(output_dir: Path, index: int, key: str) -> Path:
    return output_dir / f'{index:06d}_{key}'


def write_sample(
    output_dir: Path,
    manifest_path: Path,
    state: LatestState,
    args: argparse.Namespace,
    *,
    index: int,
) -> dict[str, Any]:
    if (
        state.rgb.message is None
        or state.depth.message is None
        or state.camera_info.message is None
        or state.active_objects.message is None
    ):
        raise ValueError('cannot write sample before all required messages are ready')

    now = time.time()
    rgb_msg = state.rgb.message
    depth_msg = state.depth.message
    camera_msg = state.camera_info.message
    active_msg = state.active_objects.message
    rgb_key = stamp_key(getattr(getattr(rgb_msg, 'header', None), 'stamp', None))
    sample_dir = build_sample_dir(output_dir, index, rgb_key if rgb_key != '0000000000_000000000' else now_key())

    rgb_image, rgb_decode = decode_compressed_image(rgb_msg, args.rgb_topic, color=True)
    depth_image, depth_decode = decode_compressed_image(depth_msg, args.depth_topic, color=False)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_dir / 'rgb.png'
    depth_path = sample_dir / 'depth.png'
    if not cv2.imwrite(str(rgb_path), rgb_image):
        raise ValueError(f'cv2.imwrite failed for {rgb_path}')
    if not cv2.imwrite(str(depth_path), depth_image):
        raise ValueError(f'cv2.imwrite failed for {depth_path}')

    active_payload = str(active_msg.data)
    active_json, active_parse_error = parse_active_objects(active_payload)
    active_json_path = sample_dir / 'active_objects.json'
    active_raw_path = sample_dir / 'active_objects_raw.txt'
    if active_json is not None:
        write_json(active_json_path, active_json)
    else:
        active_raw_path.write_text(active_payload, encoding='utf-8')

    camera_info_path = sample_dir / 'camera_info.json'
    write_json(
        camera_info_path,
        {
            'topic': args.camera_info_topic,
            'received_at': datetime.fromtimestamp(state.camera_info.received_at or now, timezone.utc).isoformat(),
            'header': header_to_dict(getattr(camera_msg, 'header', None)),
            'message': message_to_jsonable(camera_msg),
        },
    )

    active_object_count = None
    active_frame_w = None
    active_frame_h = None
    active_timestamp = None
    if isinstance(active_json, dict):
        active_object_count = len(active_json.get('objects') or [])
        active_frame_w = active_json.get('frame_w')
        active_frame_h = active_json.get('frame_h')
        active_timestamp = active_json.get('timestamp')

    counts = {
        'rgb': state.rgb.count,
        'depth': state.depth.count,
        'camera_info': state.camera_info.count,
        'active_objects': state.active_objects.count,
    }
    ages = {
        'rgb': message_age_seconds(state.rgb, now),
        'depth': message_age_seconds(state.depth, now),
        'camera_info': message_age_seconds(state.camera_info, now),
        'active_objects': message_age_seconds(state.active_objects, now),
    }
    meta = {
        'sample_index': index,
        'written_at': datetime.now(timezone.utc).isoformat(),
        'topics': {
            'rgb': args.rgb_topic,
            'depth': args.depth_topic,
            'camera_info': args.camera_info_topic,
            'active_objects': args.active_objects_topic,
        },
        'message_counts': counts,
        'message_age_seconds': ages,
        'headers': {
            'rgb': header_to_dict(getattr(rgb_msg, 'header', None)),
            'depth': header_to_dict(getattr(depth_msg, 'header', None)),
            'camera_info': header_to_dict(getattr(camera_msg, 'header', None)),
        },
        'paths': {
            'rgb': rgb_path.name,
            'depth': depth_path.name,
            'camera_info': camera_info_path.name,
            'active_objects': active_json_path.name if active_json is not None else active_raw_path.name,
        },
        'decode': {
            'rgb': rgb_decode,
            'depth': depth_decode,
        },
        'active_objects': {
            'parse_error': active_parse_error,
            'object_count': active_object_count,
            'frame_w': active_frame_w,
            'frame_h': active_frame_h,
            'timestamp': active_timestamp,
            'payload_length': len(active_payload),
        },
    }
    write_json(sample_dir / 'meta.json', meta)

    manifest_record = {
        'sample_index': index,
        'sample_dir': sample_dir.name,
        'written_at': meta['written_at'],
        'message_counts': counts,
        'message_age_seconds': ages,
        'headers': meta['headers'],
        'active_objects': meta['active_objects'],
    }
    with manifest_path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(manifest_record, ensure_ascii=False) + '\n')
    return manifest_record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-topic', default=os.environ.get('SHIGURE_YOLO_RGBD_RGB_TOPIC', '/rs/color/compressed'))
    parser.add_argument(
        '--depth-topic',
        default=os.environ.get('SHIGURE_YOLO_RGBD_DEPTH_TOPIC', '/rs/aligned_depth_to_color/compressedDepth'),
    )
    parser.add_argument(
        '--camera-info-topic',
        default=os.environ.get('SHIGURE_YOLO_RGBD_CAMERA_INFO_TOPIC', '/rs/aligned_depth_to_color/cameraInfo'),
    )
    parser.add_argument(
        '--active-objects-topic',
        default=os.environ.get('SHIGURE_YOLO_RGBD_ACTIVE_OBJECTS_TOPIC', '/tracking/active_objects'),
    )
    parser.add_argument('--output-dir', type=Path, default=Path('.test/shigure_yolo_rgbd_samples'))
    parser.add_argument('--run-id', default=None, help='Subdirectory name under --output-dir. Defaults to a UTC timestamp.')
    parser.add_argument('--duration', type=float, default=30.0, help='Seconds to record after all topics are ready.')
    parser.add_argument('--sample-hz', type=float, default=5.0, help='Sampling rate for paired latest messages.')
    parser.add_argument('--timeout', type=float, default=20.0, help='Seconds to wait for the first messages.')
    parser.add_argument('--count', type=int, default=0, help='Optional sample count limit; 0 derives it from duration * sample-hz.')
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rclpy, Node, QoSProfile, ReliabilityPolicy, CameraInfo, CompressedImage, String = import_ros_modules()

    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.jsonl'
    if manifest_path.exists():
        manifest_path.unlink()
    write_json(
        output_dir / 'run_config.json',
        {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'duration': float(args.duration),
            'sample_hz': float(args.sample_hz),
            'topics': {
                'rgb': args.rgb_topic,
                'depth': args.depth_topic,
                'camera_info': args.camera_info_topic,
                'active_objects': args.active_objects_topic,
            },
        },
    )

    rclpy.init(args=None)
    node = Node('shigure_yolo_rgbd_sample_recorder')
    state = LatestState(
        rgb=TopicState(),
        depth=TopicState(),
        camera_info=TopicState(),
        active_objects=TopicState(),
    )

    def update(topic_state: TopicState, msg: Any) -> None:
        topic_state.message = msg
        topic_state.received_at = time.time()
        topic_state.count += 1

    rgb_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    active_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    subscriptions = [
        node.create_subscription(CompressedImage, args.rgb_topic, lambda msg: update(state.rgb, msg), rgb_qos),
        node.create_subscription(CompressedImage, args.depth_topic, lambda msg: update(state.depth, msg), rgb_qos),
        node.create_subscription(CameraInfo, args.camera_info_topic, lambda msg: update(state.camera_info, msg), rgb_qos),
        node.create_subscription(String, args.active_objects_topic, lambda msg: update(state.active_objects, msg), active_qos),
    ]

    print(f'[yolo_rgbd] output: {output_dir}', flush=True)
    print(f'[yolo_rgbd] subscribe rgb: {args.rgb_topic}', flush=True)
    print(f'[yolo_rgbd] subscribe depth: {args.depth_topic}', flush=True)
    print(f'[yolo_rgbd] subscribe camera_info: {args.camera_info_topic}', flush=True)
    print(f'[yolo_rgbd] subscribe active_objects: {args.active_objects_topic}', flush=True)

    required = {
        'rgb': state.rgb,
        'depth': state.depth,
        'camera_info': state.camera_info,
        'active_objects': state.active_objects,
    }
    start_wait = time.monotonic()
    record_start: float | None = None
    next_sample = 0.0
    sample_index = 0
    target_count = int(round(float(args.duration) * float(args.sample_hz))) if args.count <= 0 else int(args.count)
    interval = 1.0 / max(0.1, float(args.sample_hz))

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.03)
            now = time.monotonic()
            missing = [name for name, topic_state in required.items() if topic_state.message is None]
            if missing:
                if args.timeout > 0 and now - start_wait > float(args.timeout):
                    raise TimeoutError('timed out waiting for topics: ' + ', '.join(missing))
                continue
            if record_start is None:
                record_start = now
                next_sample = now
                print(
                    f'[yolo_rgbd] all topics ready; recording {target_count} samples at {args.sample_hz:g} fps',
                    flush=True,
                )

            if sample_index >= target_count:
                break
            if now < next_sample:
                continue
            while next_sample <= now:
                next_sample += interval

            sample_index += 1
            record = write_sample(output_dir, manifest_path, state, args, index=sample_index)
            active = record['active_objects']
            print(
                '[yolo_rgbd] sample '
                f'{sample_index}/{target_count} objects={active.get("object_count")} '
                f'dir={record["sample_dir"]}',
                flush=True,
            )

        write_json(
            output_dir / 'run_summary.json',
            {
                'finished_at': datetime.now(timezone.utc).isoformat(),
                'sample_count': sample_index,
                'target_count': target_count,
                'output_dir': str(output_dir),
                'manifest_path': str(manifest_path),
                'final_message_counts': {name: topic_state.count for name, topic_state in required.items()},
            },
        )
        print(f'[yolo_rgbd] finished: samples={sample_index} output={output_dir}', flush=True)
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


if __name__ == '__main__':
    raise SystemExit(main())
