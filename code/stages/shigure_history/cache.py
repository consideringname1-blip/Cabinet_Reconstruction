from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import settings


@dataclass(frozen=True)
class RosStamp:
    sec: int
    nanosec: int = 0

    @property
    def seconds(self) -> float:
        return float(self.sec) + float(self.nanosec) / 1_000_000_000.0

    def to_dict(self) -> dict[str, int]:
        return {'sec': int(self.sec), 'nanosec': int(self.nanosec)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'RosStamp':
        return cls(sec=int(payload.get('sec', 0)), nanosec=int(payload.get('nanosec', 0)))


@dataclass(frozen=True)
class CachedRgbdSample:
    stamp: RosStamp
    rgb_bgr: np.ndarray
    depth: np.ndarray
    camera_info_path: Path | None
    camera_info: dict[str, Any] | None = None
    yolo_path: Path | None = None
    yolo: dict[str, Any] | None = None
    yolo_hash: str | None = None
    chunk_id: str | None = None
    frame_index: int = 0
    rgb_path: Path | None = None
    depth_path: Path | None = None

    @property
    def key(self) -> str:
        return sample_key(self.stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            'stamp': self.stamp.to_dict(),
            'chunk_id': self.chunk_id,
            'frame_index': self.frame_index,
            'camera_info_path': str(self.camera_info_path) if self.camera_info_path else None,
            'yolo_hash': self.yolo_hash,
            'yolo_path': str(self.yolo_path) if self.yolo_path else None,
        }


def sample_key(stamp: RosStamp) -> str:
    return f'{int(stamp.sec):010d}_{int(stamp.nanosec):09d}'


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)
        file.write('\n')
    tmp.replace(target)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoChunkEncoder:
    def __init__(self, chunk_dir: Path, *, width: int, height: int, fps: float, keyframe_interval: int) -> None:
        self.chunk_dir = chunk_dir
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.keyframe_interval = max(1, int(keyframe_interval))
        self.rgb_path = chunk_dir / 'rgb.mp4'
        self.depth_path = chunk_dir / 'depth.mkv'
        self._rgb_process: subprocess.Popen[bytes] | None = None
        self._depth_process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        size = f'{self.width}x{self.height}'
        rgb_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', size, '-r', str(self.fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', settings.SHIGURE_HISTORY_RGB_PRESET,
            '-crf', str(settings.SHIGURE_HISTORY_RGB_CRF), '-g', str(self.keyframe_interval),
            '-pix_fmt', 'yuv420p', str(self.rgb_path),
        ]
        depth_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-s', size, '-r', str(self.fps), '-i', '-',
            '-an', '-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1',
            '-g', '1', '-slices', '4', '-slicecrc', '0', str(self.depth_path),
        ]
        self._rgb_process = subprocess.Popen(rgb_cmd, stdin=subprocess.PIPE)
        self._depth_process = subprocess.Popen(depth_cmd, stdin=subprocess.PIPE)

    def write(self, rgb_bgr: np.ndarray, depth: np.ndarray) -> None:
        if self._rgb_process is None or self._depth_process is None:
            raise RuntimeError('chunk encoder is not started')
        if self._rgb_process.stdin is None or self._depth_process.stdin is None:
            raise RuntimeError('chunk encoder stdin is unavailable')
        rgb = np.ascontiguousarray(rgb_bgr, dtype=np.uint8)
        depth_u16 = np.ascontiguousarray(depth, dtype=np.uint16)
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(f'RGB shape {rgb.shape} does not match {(self.height, self.width)}')
        if depth_u16.shape[:2] != (self.height, self.width):
            raise ValueError(f'Depth shape {depth_u16.shape} does not match {(self.height, self.width)}')
        self._rgb_process.stdin.write(rgb.tobytes())
        self._depth_process.stdin.write(depth_u16.tobytes())

    def close(self) -> None:
        errors = []
        for name, process in (('rgb', self._rgb_process), ('depth', self._depth_process)):
            if process is None:
                continue
            try:
                if process.stdin is not None:
                    process.stdin.close()
                code = process.wait(timeout=30)
                if code != 0:
                    errors.append(f'{name} ffmpeg exited with {code}')
            except Exception as exc:
                errors.append(f'{name} ffmpeg close failed: {exc}')
        self._rgb_process = None
        self._depth_process = None
        if errors:
            raise RuntimeError('; '.join(errors))


class ChunkedShigureHistoryWriter:
    def __init__(self, root: str | Path, *, retention_seconds: float, sample_hz: float, chunk_seconds: float) -> None:
        self.root = Path(root)
        self.chunks_root = self.root / 'chunks'
        self.yolo_root = self.root / 'yolo_payloads'
        self.retention_seconds = max(1.0, float(retention_seconds))
        self.sample_hz = max(0.1, float(sample_hz))
        self.chunk_seconds = max(1.0, float(chunk_seconds))
        self.max_frames_per_chunk = max(1, int(round(self.sample_hz * self.chunk_seconds)))
        self.chunks_root.mkdir(parents=True, exist_ok=True)
        self.yolo_root.mkdir(parents=True, exist_ok=True)
        self._encoder: VideoChunkEncoder | None = None
        self._chunk_dir: Path | None = None
        self._chunk_id: str | None = None
        self._frames: list[dict[str, Any]] = []
        self._camera_info: dict[str, Any] | None = None
        self._width: int | None = None
        self._height: int | None = None

    def append_sample(self, *, stamp: RosStamp, rgb_bgr: np.ndarray, depth: np.ndarray, camera_info: Mapping[str, Any], yolo_payload: str | None, headers: Mapping[str, Any], topic_counts: Mapping[str, int]) -> dict[str, Any]:
        if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
            raise ValueError(f'expected BGR image, got shape {rgb_bgr.shape}')
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        height, width = rgb_bgr.shape[:2]
        if self._encoder is None or len(self._frames) >= self.max_frames_per_chunk:
            self.finalize_current_chunk()
            self._start_chunk(stamp, width=width, height=height)
        assert self._encoder is not None
        assert self._chunk_id is not None
        yolo_hash = self._store_yolo(yolo_payload) if yolo_payload else None
        frame_index = len(self._frames)
        self._encoder.write(rgb_bgr, depth)
        self._camera_info = dict(camera_info)
        frame = {'frame_index': frame_index, 'stamp': stamp.to_dict(), 'sample_key': sample_key(stamp), 'headers': dict(headers), 'topic_counts': dict(topic_counts), 'yolo_hash': yolo_hash}
        self._frames.append(frame)
        return {'chunk_id': self._chunk_id, **frame}

    def finalize_current_chunk(self) -> Path | None:
        if self._encoder is None or self._chunk_dir is None or self._chunk_id is None:
            return None
        encoder = self._encoder
        chunk_dir = self._chunk_dir
        frames = self._frames
        chunk_id = self._chunk_id
        camera_info = self._camera_info
        width = self._width
        height = self._height
        self._encoder = None
        self._chunk_dir = None
        self._chunk_id = None
        self._frames = []
        self._camera_info = None
        self._width = None
        self._height = None
        encoder.close()
        if camera_info is not None:
            write_json(chunk_dir / 'camera_info.json', camera_info)
        start_seconds = RosStamp.from_dict(frames[0]['stamp']).seconds if frames else None
        end_seconds = RosStamp.from_dict(frames[-1]['stamp']).seconds if frames else None
        manifest = {'chunk_id': chunk_id, 'created_at': utc_now(), 'fps': self.sample_hz, 'chunk_seconds': self.chunk_seconds, 'width': width, 'height': height, 'frame_count': len(frames), 'start_seconds': start_seconds, 'end_seconds': end_seconds, 'rgb_video': 'rgb.mp4', 'depth_video': 'depth.mkv', 'camera_info': 'camera_info.json' if camera_info is not None else None, 'frames': frames}
        write_json(chunk_dir / 'chunk_manifest.json', manifest)
        return chunk_dir

    def close(self) -> None:
        self.finalize_current_chunk()

    def prune(self, *, newest_stamp: RosStamp | None = None) -> None:
        self.finalize_current_chunk()
        ShigureRgbdCache(self.root).prune(newest_stamp=newest_stamp, retention_seconds=self.retention_seconds)

    def _start_chunk(self, stamp: RosStamp, *, width: int, height: int) -> None:
        chunk_id = sample_key(stamp)
        chunk_dir = self.chunks_root / chunk_id
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        keyframe_interval = max(1, int(round(self.sample_hz * settings.SHIGURE_HISTORY_RGB_KEYFRAME_SECONDS)))
        encoder = VideoChunkEncoder(chunk_dir, width=width, height=height, fps=self.sample_hz, keyframe_interval=keyframe_interval)
        encoder.start()
        self._encoder = encoder
        self._chunk_dir = chunk_dir
        self._chunk_id = chunk_id
        self._width = width
        self._height = height

    def _store_yolo(self, payload: str) -> str:
        raw = payload.encode('utf-8')
        digest = hashlib.sha256(raw).hexdigest()
        path = self.yolo_root / f'{digest}.json'
        if not path.exists():
            try:
                parsed = json.loads(payload)
                write_json(path, parsed)
            except Exception:
                path.write_text(payload, encoding='utf-8')
        return digest


class ShigureRgbdCache:
    """Chunked Shigurei RGB-D cache with in-memory decoded chunk LRU."""

    def __init__(self, root: str | Path, *, decoded_chunk_cache_max: int = settings.SHIGURE_HISTORY_DECODED_CHUNK_CACHE_MAX) -> None:
        self.root = Path(root)
        self.chunks_root = self.root / 'chunks'
        self.yolo_root = self.root / 'yolo_payloads'
        self.decoded_chunk_cache_max = max(0, int(decoded_chunk_cache_max))
        self._decoded_chunks: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def clear_decoded_cache(self) -> None:
        self._decoded_chunks.clear()

    def iter_samples(self, *, start: RosStamp | None = None, end: RosStamp | None = None) -> Iterable[CachedRgbdSample]:
        start_seconds = start.seconds if start is not None else None
        end_seconds = end.seconds if end is not None else None
        for chunk_dir, manifest in self._iter_chunk_manifests(start_seconds=start_seconds, end_seconds=end_seconds):
            rgb_frames, depth_frames = self._decode_chunk(chunk_dir, manifest)
            camera_info_path = chunk_dir / str(manifest.get('camera_info') or 'camera_info.json')
            camera_info = load_json(camera_info_path) if camera_info_path.is_file() else None
            for frame in manifest.get('frames') or []:
                stamp = RosStamp.from_dict(frame.get('stamp') or {})
                seconds = stamp.seconds
                if start_seconds is not None and seconds < start_seconds:
                    continue
                if end_seconds is not None and seconds > end_seconds:
                    continue
                index = int(frame.get('frame_index', 0))
                if index < 0 or index >= len(rgb_frames) or index >= len(depth_frames):
                    continue
                yolo_hash = frame.get('yolo_hash')
                yolo_path = self.yolo_root / f'{yolo_hash}.json' if yolo_hash else None
                yolo = load_json(yolo_path) if yolo_path and yolo_path.is_file() else None
                yield CachedRgbdSample(stamp=stamp, rgb_bgr=rgb_frames[index], depth=depth_frames[index], camera_info_path=camera_info_path if camera_info_path.is_file() else None, camera_info=camera_info, yolo_path=yolo_path if yolo_path and yolo_path.is_file() else None, yolo=yolo, yolo_hash=str(yolo_hash) if yolo_hash else None, chunk_id=str(manifest.get('chunk_id') or chunk_dir.name), frame_index=index)

    def iter_samples_after(self, stamp: RosStamp | None) -> Iterable[CachedRgbdSample]:
        minimum = stamp.seconds if stamp is not None else None
        for sample in self.iter_samples(start=stamp):
            if minimum is not None and sample.stamp.seconds <= minimum:
                continue
            yield sample

    def newest_sample(self) -> CachedRgbdSample | None:
        newest = None
        for sample in self.iter_samples():
            newest = sample
        return newest

    def get_sample(self, stamp: RosStamp, *, mode: str = 'nearest') -> CachedRgbdSample | None:
        samples = list(self.iter_samples())
        if not samples:
            return None
        target = stamp.seconds
        if mode == 'before':
            candidates = [sample for sample in samples if sample.stamp.seconds <= target]
            return candidates[-1] if candidates else None
        if mode == 'after':
            candidates = [sample for sample in samples if sample.stamp.seconds >= target]
            return candidates[0] if candidates else None
        return min(samples, key=lambda sample: abs(sample.stamp.seconds - target))

    def prune(self, *, newest_stamp: RosStamp | None = None, retention_seconds: float | None = None) -> None:
        retention = float(retention_seconds if retention_seconds is not None else settings.SHIGURE_HISTORY_SECONDS)
        manifests = list(self._iter_chunk_manifests())
        if newest_stamp is None:
            end_values = [float(manifest.get('end_seconds')) for _chunk, manifest in manifests if manifest.get('end_seconds') is not None]
            newest_seconds = max(end_values) if end_values else None
        else:
            newest_seconds = newest_stamp.seconds
        if newest_seconds is None:
            return
        cutoff = newest_seconds - retention
        removed = set()
        for chunk_dir, manifest in manifests:
            end_seconds = manifest.get('end_seconds')
            if end_seconds is not None and float(end_seconds) < cutoff:
                shutil.rmtree(chunk_dir, ignore_errors=True)
                removed.add(chunk_dir.name)
        if removed:
            self.clear_decoded_cache()
        self.prune_yolo_payloads()

    def prune_yolo_payloads(self) -> None:
        referenced: set[str] = set()
        for _chunk_dir, manifest in self._iter_chunk_manifests():
            for frame in manifest.get('frames') or []:
                yolo_hash = frame.get('yolo_hash')
                if yolo_hash:
                    referenced.add(str(yolo_hash))
        for payload in self.yolo_root.glob('*.json'):
            if payload.stem not in referenced:
                try:
                    payload.unlink()
                except FileNotFoundError:
                    pass

    def _iter_chunk_manifests(self, *, start_seconds: float | None = None, end_seconds: float | None = None) -> Iterable[tuple[Path, dict[str, Any]]]:
        if not self.chunks_root.is_dir():
            return
        for manifest_path in sorted(self.chunks_root.glob('*/chunk_manifest.json')):
            try:
                manifest = load_json(manifest_path)
            except Exception:
                continue
            chunk_start = manifest.get('start_seconds')
            chunk_end = manifest.get('end_seconds')
            if start_seconds is not None and chunk_end is not None and float(chunk_end) < start_seconds:
                continue
            if end_seconds is not None and chunk_start is not None and float(chunk_start) > end_seconds:
                continue
            yield manifest_path.parent, manifest

    def _decode_chunk(self, chunk_dir: Path, manifest: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        chunk_id = str(manifest.get('chunk_id') or chunk_dir.name)
        cached = self._decoded_chunks.get(chunk_id)
        if cached is not None:
            self._decoded_chunks.move_to_end(chunk_id)
            return cached
        frame_count = int(manifest.get('frame_count') or len(manifest.get('frames') or []))
        width = int(manifest['width'])
        height = int(manifest['height'])
        rgb_path = chunk_dir / str(manifest.get('rgb_video') or 'rgb.mp4')
        depth_path = chunk_dir / str(manifest.get('depth_video') or 'depth.mkv')
        rgb = self._decode_raw_video(rgb_path, pix_fmt='bgr24', dtype=np.uint8, shape=(frame_count, height, width, 3))
        depth = self._decode_raw_video(depth_path, pix_fmt='gray16le', dtype=np.uint16, shape=(frame_count, height, width))
        decoded = (rgb, depth)
        if self.decoded_chunk_cache_max > 0:
            self._decoded_chunks[chunk_id] = decoded
            self._decoded_chunks.move_to_end(chunk_id)
            while len(self._decoded_chunks) > self.decoded_chunk_cache_max:
                self._decoded_chunks.popitem(last=False)
        return decoded

    @staticmethod
    def _decode_raw_video(path: Path, *, pix_fmt: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-']
        raw = subprocess.check_output(cmd)
        array = np.frombuffer(raw, dtype=dtype)
        expected = int(np.prod(shape))
        if array.size != expected:
            raise ValueError(f'decoded {array.size} values from {path}, expected {expected}')
        return array.reshape(shape).copy()
