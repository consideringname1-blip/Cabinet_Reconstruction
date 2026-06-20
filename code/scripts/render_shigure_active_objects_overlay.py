#!/usr/bin/env python3
"""Render Shigurei active object masks over a tabletop-filtered RGB frame."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
ROS_SETUP_ENV = CODE_ROOT / 'ros2' / 'shigure_recv_ws' / 'setup_env.sh'
ROS_PYTHON = Path(os.environ.get('SHIGURE_OVERLAY_ROS_PYTHON', '/usr/bin/python3'))
SERVER_SITE_PACKAGES = Path(
    os.environ.get(
        'SHIGURE_OVERLAY_CV_SITE_PACKAGES',
        '/opt/miniconda/envs/server/lib/python3.10/site-packages',
    )
)
BOOTSTRAP_ENV_KEY = 'SHIGURE_OVERLAY_BOOTSTRAPPED'


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
class ActiveObject:
    object_id: str
    center_xy: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    mask_pixels: int
    color_bgr: tuple[int, int, int] = (255, 255, 255)
    median_depth: float | None = None
    depth_pixels: int = 0


@dataclass
class LatestState:
    rgb_msg: Any | None = None
    depth_msg: Any | None = None
    active_msg: Any | None = None
    rgb_count: int = 0
    depth_count: int = 0
    active_count: int = 0
    last_render_key: tuple[int, int, int] | None = None


def import_ros_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit(
            'ROS2 Python modules are not available. Source '
            f'{ROS_SETUP_ENV} or run this script through its bootstrap. Import error: {exc}'
        ) from exc
    return rclpy, Node, QoSProfile, ReliabilityPolicy, CompressedImage, String


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


def compressed_image_to_bgr(msg: Any) -> np.ndarray:
    raw = bytes(msg.data)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('cv2.imdecode failed for RGB compressed image')
    return image


def compressed_depth_to_image(msg: Any) -> np.ndarray:
    raw = bytes(msg.data)
    fmt = str(getattr(msg, 'format', ''))
    if ('compressedDepth' in fmt or 'depth' in fmt.lower()) and b'PNG' not in raw[:12]:
        raw = raw[12:]
    depth = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError('cv2.imdecode failed for depth compressed image')
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def normalize_bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return 0, 0, max(0, width - 1), max(0, height - 1)
    x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def decode_object_mask(mask_b64: str, bbox: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    raw = base64.b64decode(mask_b64)
    mask_img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError('cv2.imdecode failed for mask_b64')
    if mask_img.shape[:2] == (height, width):
        return mask_img > 0

    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    full = np.zeros((height, width), dtype=bool)
    if mask_img.shape[:2] != (box_h, box_w):
        mask_img = cv2.resize(mask_img, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
    full[y0 : y1 + 1, x0 : x1 + 1] = mask_img[:box_h, :box_w] > 0
    return full


def parse_active_objects(payload: str, width: int, height: int, min_mask_pixels: int) -> tuple[dict[str, Any], list[ActiveObject]]:
    data = json.loads(payload)
    objects: list[ActiveObject] = []
    for raw_obj in data.get('objects') or []:
        mask_b64 = raw_obj.get('mask_b64') or ''
        if not mask_b64:
            continue
        bbox = normalize_bbox(raw_obj.get('bbox'), width, height)
        try:
            mask = decode_object_mask(mask_b64, bbox, width, height)
        except Exception as exc:
            print(f"[shigure_overlay] skip object {raw_obj.get('object_id')}: {exc}", flush=True)
            continue
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < min_mask_pixels:
            continue
        center = (
            int(round(float(raw_obj.get('x', (bbox[0] + bbox[2]) * 0.5)))),
            int(round(float(raw_obj.get('y', (bbox[1] + bbox[3]) * 0.5)))),
        )
        objects.append(
            ActiveObject(
                object_id=str(raw_obj.get('object_id', len(objects))),
                center_xy=center,
                bbox_xyxy=bbox,
                mask=mask,
                mask_pixels=mask_pixels,
            )
        )
    return data, objects


def bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0] + 1) * max(0, box[3] - box[1] + 1)


def bbox_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 < x0 or y1 < y0:
        return 0
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def object_depth_stats(obj: ActiveObject, depth: np.ndarray) -> tuple[float | None, int]:
    if depth.shape[:2] != obj.mask.shape[:2]:
        depth = cv2.resize(depth, (obj.mask.shape[1], obj.mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    values = depth[obj.mask]
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0:
        return None, 0
    return float(np.median(values)), int(values.size)


def select_table_object(objects: list[ActiveObject], width: int, height: int) -> ActiveObject | None:
    best: ActiveObject | None = None
    best_score = -1.0
    for obj in objects:
        x0, y0, x1, y1 = obj.bbox_xyxy
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        if y1 < height * 0.48:
            continue
        if cx < width * 0.10 or cx > width * 0.90:
            continue
        center_weight = 1.0 - min(1.0, abs(cx - width * 0.5) / (width * 0.5))
        bottom_weight = max(0.0, y1 / max(1.0, float(height)))
        vertical_weight = max(0.0, cy / max(1.0, float(height)))
        area_score = max(float(obj.mask_pixels), float(bbox_area(obj.bbox_xyxy)) * 0.35)
        score = area_score * (0.35 + center_weight) * (0.5 + bottom_weight + vertical_weight * 0.4)
        if score > best_score:
            best_score = score
            best = obj
    return best


def filter_tabletop_objects(
    objects: list[ActiveObject],
    table: ActiveObject | None,
    depth: np.ndarray | None,
    *,
    closer_margin: float,
    table_depth_low_percentile: float,
    table_depth_high_percentile: float,
    min_table_overlap: float,
) -> tuple[list[ActiveObject], dict[str, Any] | None]:
    if table is None or depth is None:
        return objects, None
    if depth.shape[:2] != table.mask.shape[:2]:
        depth = cv2.resize(depth, (table.mask.shape[1], table.mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    table_values = depth[table.mask]
    table_values = table_values[np.isfinite(table_values)]
    table_values = table_values[table_values > 0]
    if table_values.size == 0:
        return objects, {
            'table_object_id': table.object_id,
            'table_bbox_xyxy': list(table.bbox_xyxy),
            'error': 'no valid depth in table mask',
        }

    low = float(np.percentile(table_values, table_depth_low_percentile))
    high = float(np.percentile(table_values, table_depth_high_percentile))
    # The tabletop can span a wide depth range because of perspective. Objects on
    # the table should be closer than the far side of that table-depth band.
    near_threshold = max(0.0, high - float(closer_margin))
    table_box = table.bbox_xyxy
    selected: list[ActiveObject] = []
    rejected: list[dict[str, Any]] = []

    for obj in objects:
        if obj.object_id == table.object_id:
            continue
        obj.median_depth, obj.depth_pixels = object_depth_stats(obj, depth)
        inter_area = bbox_intersection_area(obj.bbox_xyxy, table_box)
        overlap_ratio = inter_area / max(1, bbox_area(obj.bbox_xyxy))
        mask_on_table_pixels = int(np.count_nonzero(obj.mask & table.mask))
        mask_on_table_ratio = mask_on_table_pixels / max(1, obj.mask_pixels)
        inside_table = overlap_ratio >= min_table_overlap or mask_on_table_ratio >= min_table_overlap
        closer_than_table = obj.median_depth is not None and obj.median_depth < near_threshold
        if inside_table and closer_than_table:
            selected.append(obj)
        else:
            rejected.append(
                {
                    'object_id': obj.object_id,
                    'median_depth': obj.median_depth,
                    'overlap_ratio': overlap_ratio,
                    'mask_on_table_ratio': mask_on_table_ratio,
                    'inside_table': inside_table,
                    'closer_than_table': closer_than_table,
                }
            )

    table_info = {
        'table_object_id': table.object_id,
        'table_bbox_xyxy': list(table.bbox_xyxy),
        'table_mask_pixels': table.mask_pixels,
        'table_depth_pixels': int(table_values.size),
        'table_depth_percentiles': {
            str(table_depth_low_percentile): low,
            str(table_depth_high_percentile): high,
        },
        'closer_depth_threshold': near_threshold,
        'closer_depth_rule': 'object median depth < table high percentile - margin',
        'closer_margin': float(closer_margin),
        'selected_count': len(selected),
        'rejected_count': len(rejected),
        'rejected': rejected[:200],
    }
    return selected, table_info


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(av - bv))


def generate_palette(size: int = 96) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    golden = 0.618033988749895
    hue = 0.02
    for index in range(size):
        hue = (hue + golden) % 1.0
        saturation = 0.78 if index % 3 else 0.92
        value = 0.95 if index % 4 else 0.82
        hsv = np.uint8([[[int(hue * 179), int(saturation * 255), int(value * 255)]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def assign_distinct_colors(objects: list[ActiveObject]) -> None:
    palette = generate_palette(max(96, len(objects) * 3))
    assigned: dict[int, tuple[int, int, int]] = {}
    order = sorted(range(len(objects)), key=lambda idx: (-objects[idx].mask_pixels, objects[idx].object_id))
    for idx in order:
        neighbors = [
            other_idx
            for other_idx in assigned
            if bbox_intersects(objects[idx].bbox_xyxy, objects[other_idx].bbox_xyxy)
        ]
        best_color = palette[0]
        best_score = -1.0
        for color_index, color in enumerate(palette):
            if neighbors:
                neighbor_score = min(color_distance(color, assigned[other_idx]) for other_idx in neighbors)
            else:
                neighbor_score = 255.0
            if assigned:
                global_score = min(color_distance(color, used) for used in assigned.values())
            else:
                global_score = 255.0
            score = neighbor_score * 4.0 + global_score - (color_index * 0.001)
            if score > best_score:
                best_score = score
                best_color = color
        assigned[idx] = best_color
        objects[idx].color_bgr = best_color


def label_origin(x0: int, y0: int, text_w: int, text_h: int, width: int, height: int) -> tuple[int, int, int, int]:
    pad = 4
    label_w = text_w + pad * 2
    label_h = text_h + pad * 2
    lx = max(0, min(width - label_w - 1, x0))
    ly = y0 - label_h - 2
    if ly < 0:
        ly = min(height - label_h - 1, y0 + 2)
    return lx, ly, lx + label_w, ly + label_h


def draw_label(out: np.ndarray, label: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.55) -> None:
    height, width = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 2)
    lx0, ly0, lx1, ly1 = label_origin(xy[0], xy[1], text_w, text_h + baseline, width, height)
    cv2.rectangle(out, (lx0, ly0), (lx1, ly1), color, -1)
    text_color = (0, 0, 0) if sum(color) > 420 else (255, 255, 255)
    cv2.putText(out, label, (lx0 + 4, ly1 - baseline - 3), font, scale, text_color, 2, lineType=cv2.LINE_AA)


def render_overlay(
    rgb_bgr: np.ndarray,
    objects: list[ActiveObject],
    *,
    alpha: float,
    draw_contours: bool,
    table: ActiveObject | None = None,
) -> np.ndarray:
    height, width = rgb_bgr.shape[:2]
    assign_distinct_colors(objects)

    base = rgb_bgr.astype(np.float32)
    color_sum = np.zeros_like(base, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for obj in objects:
        mask = obj.mask
        color = np.asarray(obj.color_bgr, dtype=np.float32)
        color_sum[mask] += color
        count[mask] += 1.0

    out = rgb_bgr.copy()
    mask_any = count > 0
    if np.any(mask_any):
        avg_color = np.zeros_like(base, dtype=np.float32)
        avg_color[mask_any] = color_sum[mask_any] / count[mask_any, None]
        blended = base.copy()
        blended[mask_any] = (1.0 - alpha) * base[mask_any] + alpha * avg_color[mask_any]
        out = np.clip(blended, 0, 255).astype(np.uint8)

    if table is not None:
        table_color = (255, 255, 255)
        x0, y0, x1, y1 = table.bbox_xyxy
        cv2.rectangle(out, (x0, y0), (x1, y1), table_color, 2, lineType=cv2.LINE_AA)
        draw_label(out, f'table roi id:{table.object_id}', (x0, y0), table_color, scale=0.6)

    for obj in sorted(objects, key=lambda item: item.mask_pixels):
        color = obj.color_bgr
        x0, y0, x1, y1 = obj.bbox_xyxy
        thickness = 2 if max(x1 - x0, y1 - y0) < 160 else 3
        if draw_contours:
            contours, _ = cv2.findContours(obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 1, lineType=cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
        label = f'id:{obj.object_id}'
        if obj.median_depth is not None:
            label += f' z:{obj.median_depth:.0f}'
        draw_label(out, label, (x0, y0), color)
    return out


def make_output_path(output_dir: Path, count: int, explicit_output: Path | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if explicit_output is not None and count == 1:
        image_path = explicit_output
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
        image_path = output_dir / f'{stamp}_active_objects_overlay.png'
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def render_from_state(state: LatestState, args: argparse.Namespace, render_count: int) -> Path | None:
    if state.rgb_msg is None or state.active_msg is None:
        return None
    if args.tabletop_filter and state.depth_msg is None:
        return None
    render_key = (state.rgb_count, state.depth_count, state.active_count)
    if render_key == state.last_render_key:
        return None

    rgb_bgr = compressed_image_to_bgr(state.rgb_msg)
    depth = compressed_depth_to_image(state.depth_msg) if state.depth_msg is not None else None
    height, width = rgb_bgr.shape[:2]
    active_payload, objects = parse_active_objects(
        state.active_msg.data,
        width,
        height,
        min_mask_pixels=max(0, int(args.min_mask_pixels)),
    )
    table = select_table_object(objects, width, height) if args.tabletop_filter else None
    draw_objects = objects
    table_info = None
    if args.tabletop_filter:
        draw_objects, table_info = filter_tabletop_objects(
            objects,
            table,
            depth,
            closer_margin=float(args.table_closer_margin),
            table_depth_low_percentile=float(args.table_depth_low_percentile),
            table_depth_high_percentile=float(args.table_depth_high_percentile),
            min_table_overlap=float(args.min_table_overlap),
        )
    overlay = render_overlay(
        rgb_bgr,
        draw_objects,
        alpha=max(0.0, min(1.0, float(args.alpha))),
        draw_contours=not args.no_contours,
        table=table,
    )
    image_path = make_output_path(args.output_dir, render_count, args.output)
    ok = cv2.imwrite(str(image_path), overlay)
    if not ok:
        raise ValueError(f'cv2.imwrite failed: {image_path}')

    summary = {
        'written_at': datetime.now(timezone.utc).isoformat(),
        'image_path': str(image_path),
        'rgb_topic': args.rgb_topic,
        'depth_topic': args.depth_topic if args.tabletop_filter else None,
        'active_objects_topic': args.active_objects_topic,
        'rgb_header': header_to_dict(getattr(state.rgb_msg, 'header', None)),
        'depth_header': header_to_dict(getattr(state.depth_msg, 'header', None)) if state.depth_msg is not None else None,
        'active_event': active_payload.get('event'),
        'active_timestamp': active_payload.get('timestamp'),
        'frame_w': width,
        'frame_h': height,
        'raw_object_count': len(objects),
        'drawn_object_count': len(draw_objects),
        'tabletop_filter': bool(args.tabletop_filter),
        'table': table_info,
        'objects': [
            {
                'object_id': obj.object_id,
                'center_xy': list(obj.center_xy),
                'bbox_xyxy': list(obj.bbox_xyxy),
                'mask_pixels': obj.mask_pixels,
                'median_depth': obj.median_depth,
                'depth_pixels': obj.depth_pixels,
                'color_bgr': list(obj.color_bgr),
            }
            for obj in draw_objects
        ],
    }
    write_json(image_path.with_suffix('.json'), summary)
    state.last_render_key = render_key
    print(
        f"[shigure_overlay] wrote {image_path} objects={len(draw_objects)}/{len(objects)}"
        + (f" table={table.object_id}" if table is not None else ''),
        flush=True,
    )
    return image_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-topic', default=os.environ.get('SHIGURE_OVERLAY_RGB_TOPIC', '/rs/color/compressed'))
    parser.add_argument(
        '--depth-topic',
        default=os.environ.get('SHIGURE_OVERLAY_DEPTH_TOPIC', '/rs/aligned_depth_to_color/compressedDepth'),
    )
    parser.add_argument(
        '--active-objects-topic',
        default=os.environ.get('SHIGURE_OVERLAY_ACTIVE_OBJECTS_TOPIC', '/tracking/active_objects'),
    )
    parser.add_argument('--output-dir', type=Path, default=Path('.test/shigure_active_objects_overlay'))
    parser.add_argument('--output', type=Path, default=None, help='Exact output image path when --count is 1.')
    parser.add_argument('--count', type=int, default=1, help='Number of timestamped overlays to write. Use 0 to run forever.')
    parser.add_argument('--alpha', type=float, default=0.45, help='Mask overlay opacity.')
    parser.add_argument('--sample-hz', type=float, default=2.0, help='Maximum render rate when running continuously.')
    parser.add_argument('--timeout', type=float, default=15.0, help='Seconds to wait for the first paired messages.')
    parser.add_argument('--min-mask-pixels', type=int, default=1, help='Skip tiny masks below this pixel count.')
    parser.add_argument('--no-tabletop-filter', dest='tabletop_filter', action='store_false')
    parser.set_defaults(tabletop_filter=True)
    parser.add_argument('--table-closer-margin', type=float, default=20.0, help='Depth margin in depth-image units; lower depth is closer.')
    parser.add_argument('--table-depth-low-percentile', type=float, default=10.0)
    parser.add_argument('--table-depth-high-percentile', type=float, default=90.0)
    parser.add_argument('--min-table-overlap', type=float, default=0.05)
    parser.add_argument('--no-contours', action='store_true', help='Draw only masks, boxes and IDs.')
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rclpy, Node, QoSProfile, ReliabilityPolicy, CompressedImage, String = import_ros_modules()

    rclpy.init(args=None)
    node = Node('shigure_active_objects_overlay_renderer')
    state = LatestState()
    rendered = 0
    start = time.monotonic()
    next_render = start

    rgb_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    string_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

    def rgb_callback(msg: Any) -> None:
        state.rgb_msg = msg
        state.rgb_count += 1

    def depth_callback(msg: Any) -> None:
        state.depth_msg = msg
        state.depth_count += 1

    def active_callback(msg: Any) -> None:
        state.active_msg = msg
        state.active_count += 1

    rgb_sub = node.create_subscription(CompressedImage, args.rgb_topic, rgb_callback, rgb_qos)
    depth_sub = node.create_subscription(CompressedImage, args.depth_topic, depth_callback, rgb_qos) if args.tabletop_filter else None
    active_sub = node.create_subscription(String, args.active_objects_topic, active_callback, string_qos)
    print(f'[shigure_overlay] subscribe rgb: {args.rgb_topic}', flush=True)
    if args.tabletop_filter:
        print(f'[shigure_overlay] subscribe depth: {args.depth_topic}', flush=True)
    print(f'[shigure_overlay] subscribe active objects: {args.active_objects_topic}', flush=True)

    try:
        interval = 1.0 / max(0.1, float(args.sample_hz))
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if state.rgb_msg is None or state.active_msg is None or (args.tabletop_filter and state.depth_msg is None):
                if args.timeout > 0 and now - start > args.timeout:
                    missing = []
                    if state.rgb_msg is None:
                        missing.append(args.rgb_topic)
                    if args.tabletop_filter and state.depth_msg is None:
                        missing.append(args.depth_topic)
                    if state.active_msg is None:
                        missing.append(args.active_objects_topic)
                    raise TimeoutError('timed out waiting for topics: ' + ', '.join(missing))
                continue
            if now < next_render:
                continue
            next_render = now + interval
            path = render_from_state(state, args, rendered + 1)
            if path is None:
                continue
            rendered += 1
            if args.count > 0 and rendered >= args.count:
                return 0
    finally:
        try:
            node.destroy_subscription(rgb_sub)
            if depth_sub is not None:
                node.destroy_subscription(depth_sub)
            node.destroy_subscription(active_sub)
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
