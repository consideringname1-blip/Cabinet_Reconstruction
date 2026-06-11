from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    rgb_path: Path
    depth_path: Path
    camera_info_path: Path | None
    camera_info_sha256: str | None = None

    @property
    def key(self) -> str:
        return sample_key(self.stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            'stamp': self.stamp.to_dict(),
            'rgb_path': self.rgb_path.name,
            'depth_path': self.depth_path.name,
            'camera_info_path': self.camera_info_path.name if self.camera_info_path else None,
            'camera_info_sha256': self.camera_info_sha256,
        }


def sample_key(stamp: RosStamp) -> str:
    return f'{int(stamp.sec):010d}_{int(stamp.nanosec):09d}'


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)
        file.write('\n')


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ShigureRgbdCache:
    """Flat disk cache for Shigurei RGB-D frames and deduplicated camera info."""

    def __init__(
        self,
        root: str | Path,
        *,
        retention_seconds: float = settings.SHIGURE_HISTORY_SECONDS,
        sample_hz: float = settings.SHIGURE_HISTORY_HZ,
    ) -> None:
        self.root = Path(root)
        self.retention_seconds = max(1.0, float(retention_seconds))
        self.sample_hz = max(0.1, float(sample_hz))
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def max_samples(self) -> int:
        return max(1, int(round(self.retention_seconds * self.sample_hz)))

    def append_sample(
        self,
        *,
        stamp: RosStamp,
        rgb_path: str | Path,
        depth_path: str | Path,
        camera_info_path: str | Path,
    ) -> CachedRgbdSample:
        key = sample_key(stamp)
        rgb_target = self.root / f'{key}_rgb.png'
        depth_target = self.root / f'{key}_depth.png'
        self._copy_required(rgb_path, rgb_target)
        self._copy_required(depth_path, depth_target)

        camera_source = Path(camera_info_path)
        if not camera_source.is_file():
            raise FileNotFoundError(str(camera_source))
        camera_digest = file_sha256(camera_source)
        latest_camera_info = self.find_latest_camera_info(key)
        latest_digest = file_sha256(latest_camera_info) if latest_camera_info and latest_camera_info.is_file() else None
        camera_reused = latest_digest == camera_digest
        camera_target = latest_camera_info if camera_reused else self.root / f'{key}_camera_info.json'
        if not camera_reused:
            self._copy_required(camera_source, camera_target)

        sample = CachedRgbdSample(
            stamp=stamp,
            rgb_path=rgb_target,
            depth_path=depth_target,
            camera_info_path=camera_target,
            camera_info_sha256=camera_digest,
        )
        write_json(
            self.root / f'{key}_meta.json',
            {
                **sample.to_dict(),
                'camera_info_reused': bool(camera_reused),
                'written_at': datetime.now(timezone.utc).isoformat(),
            },
        )
        self.prune(newest_stamp=stamp)
        return sample

    def iter_samples(
        self,
        *,
        start: RosStamp | None = None,
        end: RosStamp | None = None,
    ) -> Iterable[CachedRgbdSample]:
        start_seconds = start.seconds if start is not None else None
        end_seconds = end.seconds if end is not None else None
        for meta_path in sorted(self.root.glob('*_meta.json')):
            try:
                sample = self._load_sample(meta_path)
            except Exception:
                continue
            seconds = sample.stamp.seconds
            if start_seconds is not None and seconds < start_seconds:
                continue
            if end_seconds is not None and seconds > end_seconds:
                continue
            yield sample

    def iter_samples_after(self, stamp: RosStamp | None) -> Iterable[CachedRgbdSample]:
        minimum_key = sample_key(stamp) if stamp is not None else None
        for meta_path in sorted(self.root.glob('*_meta.json')):
            key = meta_path.name.removesuffix('_meta.json')
            if minimum_key is not None and key <= minimum_key:
                continue
            try:
                sample = self._load_sample(meta_path)
            except Exception:
                continue
            if stamp is not None and sample.stamp.seconds <= stamp.seconds:
                continue
            yield sample

    def newest_sample(self) -> CachedRgbdSample | None:
        newest: CachedRgbdSample | None = None
        for sample in self.iter_samples():
            newest = sample
        return newest

    def find_latest_camera_info(self, key: str) -> Path | None:
        candidates = [path for path in sorted(self.root.glob('*_camera_info.json')) if path.name[:20] <= key]
        return candidates[-1] if candidates else None

    def prune(self, *, newest_stamp: RosStamp | None = None) -> None:
        entries: list[tuple[Path, CachedRgbdSample | None, float]] = []
        for meta_path in sorted(self.root.glob('*_meta.json')):
            sample: CachedRgbdSample | None = None
            try:
                sample = self._load_sample(meta_path)
            except Exception:
                sample = None
            sort_seconds = sample.stamp.seconds if sample is not None else meta_path.stat().st_mtime
            entries.append((meta_path, sample, sort_seconds))

        entries.sort(key=lambda item: item[2])
        parsed_samples = [sample for _path, sample, _seconds in entries if sample is not None]
        if newest_stamp is None and parsed_samples:
            newest_stamp = parsed_samples[-1].stamp

        removable: list[Path] = []
        cutoff = newest_stamp.seconds - self.retention_seconds if newest_stamp is not None else None
        for meta_path, sample, seconds in entries:
            if cutoff is not None and seconds < cutoff:
                removable.append(meta_path)

        removable_set = set(removable)
        remaining = [entry for entry in entries if entry[0] not in removable_set]
        overflow = max(0, len(remaining) - self.max_samples)
        removable.extend(meta_path for meta_path, _sample, _seconds in remaining[:overflow])

        for meta_path in dict.fromkeys(removable):
            key = meta_path.name.removesuffix('_meta.json')
            for suffix in ('_rgb.png', '_depth.png', '_meta.json'):
                try:
                    (self.root / f'{key}{suffix}').unlink()
                except FileNotFoundError:
                    pass

        referenced_camera_info = set()
        for meta_path in sorted(self.root.glob('*_meta.json')):
            try:
                sample = self._load_sample(meta_path)
            except Exception:
                continue
            if sample.camera_info_path is not None:
                referenced_camera_info.add(sample.camera_info_path.resolve())

        for camera_info in self.root.glob('*_camera_info.json'):
            if camera_info.resolve() not in referenced_camera_info:
                try:
                    camera_info.unlink()
                except FileNotFoundError:
                    pass

    def _copy_required(self, source_path: str | Path, target_path: Path) -> Path:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target_path.resolve():
            shutil.copy2(source, target_path)
        return target_path

    def _load_sample(self, meta_path: Path) -> CachedRgbdSample:
        payload = load_json(meta_path)
        stamp = RosStamp.from_dict(payload.get('stamp') or {})
        camera_info_name = payload.get('camera_info_path')
        camera_info_path = self.root / camera_info_name if camera_info_name else self.find_latest_camera_info(sample_key(stamp))
        return CachedRgbdSample(
            stamp=stamp,
            rgb_path=self.root / str(payload['rgb_path']),
            depth_path=self.root / str(payload['depth_path']),
            camera_info_path=camera_info_path,
            camera_info_sha256=payload.get('camera_info_sha256'),
        )
