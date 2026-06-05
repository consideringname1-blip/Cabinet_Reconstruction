from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import settings
from .schemas import RosStamp, ShigureFrame, to_jsonable


REQUIRED_SNAPSHOT_FILES = {
    "rgb": "messages/rs_color_compressed.png",
    "depth": "messages/rs_aligned_depth_to_color_compressedDepth.png",
    "camera_info": "messages/rs_aligned_depth_to_color_cameraInfo.json",
    "people": "messages/shigure_people_detection.json",
}


def safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    out = out.strip("._")
    return out or "frame"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(dict(payload)), file, ensure_ascii=False, indent=2)
        file.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_from_json(path: str | Path) -> RosStamp | None:
    try:
        return RosStamp.from_message_json(load_json(path))
    except Exception:
        return None


def frame_key(stamp: RosStamp) -> str:
    return f"{int(stamp.sec):010d}_{int(stamp.nanosec):09d}"


class ShigureHistoryCache:
    """Disk ring buffer for the minimum data needed by model-event tracking."""

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
        self.frames_root = self.root / "frames"
        self.camera_info_root = self.root / "camera_info"
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.camera_info_root.mkdir(parents=True, exist_ok=True)

    @property
    def max_frames(self) -> int:
        return max(1, int(round(self.retention_seconds * self.sample_hz)))

    def append_frame(
        self,
        *,
        stamp: RosStamp,
        rgb_path: str | Path,
        depth_path: str | Path,
        camera_info_path: str | Path,
        people_path: str | Path | None = None,
        marker_pose_path: str | Path | None = None,
        extra_node_paths: Mapping[str, str | Path] | None = None,
    ) -> ShigureFrame:
        frame_dir = self.frames_root / frame_key(stamp)
        frame_dir.mkdir(parents=True, exist_ok=True)

        rgb_copy = self._copy_required(rgb_path, frame_dir / "rgb.png")
        depth_copy = self._copy_required(depth_path, frame_dir / "depth.png")
        camera_info_copy = self._copy_camera_info(camera_info_path)
        people_copy = self._copy_optional(people_path, frame_dir / "people_detection.json")
        marker_copy = self._copy_optional(marker_pose_path, frame_dir / "marker_pose.json")

        node_paths: dict[str, Path] = {}
        for name, path in dict(extra_node_paths or {}).items():
            source = Path(path)
            if not source.is_file():
                continue
            suffix = source.suffix or ".dat"
            node_paths[safe_name(name)] = self._copy_required(
                source,
                frame_dir / f"node_{safe_name(name)}{suffix}",
            )

        frame = ShigureFrame(
            stamp=stamp,
            rgb_path=rgb_copy,
            depth_path=depth_copy,
            camera_info_path=camera_info_copy,
            people_path=people_copy,
            marker_pose_path=marker_copy,
            node_paths=node_paths,
        )
        write_json(frame_dir / "frame.json", frame.to_dict())
        self.prune()
        return frame

    def append_snapshot_dir(
        self,
        snapshot_dir: str | Path,
        *,
        marker_pose_path: str | Path | None = None,
    ) -> ShigureFrame:
        snapshot = Path(snapshot_dir)
        paths = {name: snapshot / rel for name, rel in REQUIRED_SNAPSHOT_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"snapshot is missing required cached files: {missing}")
        stamp = stamp_from_json(paths["people"]) or stamp_from_json(paths["camera_info"])
        if stamp is None:
            raise ValueError("snapshot does not contain a ROS stamp in people/camera_info JSON")
        return self.append_frame(
            stamp=stamp,
            rgb_path=paths["rgb"],
            depth_path=paths["depth"],
            camera_info_path=paths["camera_info"],
            people_path=paths["people"],
            marker_pose_path=marker_pose_path,
        )

    def iter_frames(
        self,
        *,
        start: RosStamp | None = None,
        end: RosStamp | None = None,
    ) -> Iterable[ShigureFrame]:
        start_seconds = start.seconds if start is not None else None
        end_seconds = end.seconds if end is not None else None
        for frame_dir in sorted(self.frames_root.glob("*")):
            frame_json = frame_dir / "frame.json"
            if not frame_json.is_file():
                continue
            try:
                frame = self._load_frame(frame_json)
            except Exception:
                continue
            seconds = frame.stamp.seconds
            if start_seconds is not None and seconds < start_seconds:
                continue
            if end_seconds is not None and seconds > end_seconds:
                continue
            yield frame

    def iter_frames_after(self, stamp: RosStamp | None) -> Iterable[ShigureFrame]:
        """Yield newer frames without reopening older frame JSON files."""
        minimum_key = frame_key(stamp) if stamp is not None else None
        for frame_dir in sorted(self.frames_root.glob("*")):
            if minimum_key is not None and frame_dir.name <= minimum_key:
                continue
            frame_json = frame_dir / "frame.json"
            if not frame_json.is_file():
                continue
            try:
                frame = self._load_frame(frame_json)
            except Exception:
                continue
            if stamp is not None and frame.stamp.seconds <= stamp.seconds:
                continue
            yield frame

    def newest_frame(self) -> ShigureFrame | None:
        newest: ShigureFrame | None = None
        for frame in self.iter_frames():
            newest = frame
        return newest

    def prune(self, *, newest_stamp: RosStamp | None = None) -> None:
        frame_dirs = sorted(path for path in self.frames_root.iterdir() if path.is_dir())
        entries: list[tuple[Path, ShigureFrame | None, float]] = []
        for frame_dir in frame_dirs:
            frame: ShigureFrame | None = None
            try:
                frame_json = frame_dir / "frame.json"
                if frame_json.is_file():
                    frame = self._load_frame(frame_json)
            except Exception:
                frame = None
            sort_seconds = frame.stamp.seconds if frame is not None else frame_dir.stat().st_mtime
            entries.append((frame_dir, frame, sort_seconds))

        entries.sort(key=lambda item: item[2])
        parsed_frames = [frame for _, frame, _ in entries if frame is not None]
        if newest_stamp is None and parsed_frames:
            newest_stamp = parsed_frames[-1].stamp
        cutoff = None
        if newest_stamp is not None:
            cutoff = newest_stamp.seconds - self.retention_seconds

        removable: list[Path] = []
        for frame_dir, frame, _sort_seconds in entries:
            if frame is not None and cutoff is not None and frame.stamp.seconds < cutoff:
                removable.append(frame_dir)

        removable_set = set(removable)
        remaining = [entry for entry in entries if entry[0] not in removable_set]
        overflow = max(0, len(remaining) - self.max_frames)
        removable.extend(frame_dir for frame_dir, _frame, _sort_seconds in remaining[:overflow])

        seen: set[Path] = set()
        for path in removable:
            if path in seen:
                continue
            seen.add(path)
            shutil.rmtree(path, ignore_errors=True)

    def _copy_required(self, source_path: str | Path, target_path: Path) -> Path:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target_path.resolve():
            shutil.copy2(source, target_path)
        return target_path

    def _copy_optional(self, source_path: str | Path | None, target_path: Path) -> Path | None:
        if source_path is None or not str(source_path).strip():
            return None
        source = Path(source_path)
        if not source.is_file():
            return None
        return self._copy_required(source, target_path)

    def _copy_camera_info(self, source_path: str | Path) -> Path:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        digest = file_sha256(source)
        target = self.camera_info_root / f"{digest}.json"
        if not target.is_file():
            shutil.copy2(source, target)
        return target

    def _load_frame(self, frame_json: Path) -> ShigureFrame:
        payload = load_json(frame_json)
        stamp_payload = payload.get("stamp") or {}
        stamp = RosStamp(sec=int(stamp_payload["sec"]), nanosec=int(stamp_payload.get("nanosec", 0)))

        def _optional_path(key: str) -> Path | None:
            value = payload.get(key)
            return Path(value) if value else None

        node_paths = {
            str(key): Path(value)
            for key, value in dict(payload.get("node_paths") or {}).items()
            if value
        }
        return ShigureFrame(
            stamp=stamp,
            rgb_path=_optional_path("rgb_path"),
            depth_path=_optional_path("depth_path"),
            camera_info_path=_optional_path("camera_info_path"),
            people_path=_optional_path("people_path"),
            marker_pose_path=_optional_path("marker_pose_path"),
            node_paths=node_paths,
        )
