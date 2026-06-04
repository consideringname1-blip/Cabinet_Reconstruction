from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image

from .cache import ShigureHistoryCache
from .event_store import persist_taken_away_event
from .geometry import load_camera_matrix, project_model_bounds_to_shigurei
from .movement import MaskDepthMovementTracker
from .people import choose_event_start_contact, find_hand_contacts
from .schemas import ModelEventRecord, MovementDecision, ProjectedBox, RosStamp, ShigureFrame


@dataclass(frozen=True)
class VideoTrackerPrompt:
    task_id: str
    initial_box_xyxy: tuple[float, float, float, float]
    initial_frame: ShigureFrame
    projected_box: ProjectedBox


@dataclass(frozen=True)
class TrackingUpdate:
    frame: ShigureFrame
    mask: np.ndarray
    score: float | None = None


class Sam3VideoTracker(Protocol):
    """Protocol for the real SAM3 video worker.

    Implementations should keep one independent tracker state per model and
    release it when stop() is called after the taken-away event is decided.
    """

    def start(self, prompt: VideoTrackerPrompt) -> None:
        ...

    def update(self, frame: ShigureFrame) -> TrackingUpdate | None:
        ...

    def stop(self, task_id: str) -> None:
        ...


class NoopSam3VideoTracker:
    def start(self, prompt: VideoTrackerPrompt) -> None:
        raise RuntimeError("SAM3 video tracker implementation is not connected")

    def update(self, frame: ShigureFrame) -> TrackingUpdate | None:
        raise RuntimeError("SAM3 video tracker implementation is not connected")

    def stop(self, task_id: str) -> None:
        return None


def read_depth_image_m(path: str | Path) -> np.ndarray:
    depth = np.asarray(Image.open(path))
    return depth.astype(np.float64) * 0.001


@dataclass
class TrackingFrameResult:
    frame: ShigureFrame
    decision: MovementDecision
    persisted_event: ModelEventRecord | None = None


class ModelEventTrackingSession:
    def __init__(
        self,
        *,
        task_id: str,
        projected_box: ProjectedBox,
        camera_matrix: np.ndarray,
        tracker: Sam3VideoTracker,
        movement_tracker: MaskDepthMovementTracker | None = None,
    ) -> None:
        self.task_id = task_id
        self.projected_box = projected_box
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.tracker = tracker
        self.movement_tracker = movement_tracker or MaskDepthMovementTracker()
        self.started = False
        self.stopped = False

    def start(self, initial_frame: ShigureFrame) -> None:
        prompt = VideoTrackerPrompt(
            task_id=self.task_id,
            initial_box_xyxy=self.projected_box.bbox_xyxy,
            initial_frame=initial_frame,
            projected_box=self.projected_box,
        )
        self.tracker.start(prompt)
        self.started = True

    def process_update(self, update: TrackingUpdate) -> TrackingFrameResult:
        if update.frame.depth_path is None:
            raise ValueError("tracking frame is missing aligned depth")
        depth_m = read_depth_image_m(update.frame.depth_path)

        hand_contact = None
        if update.frame.people_path is not None:
            contacts = find_hand_contacts(update.frame.people_path, self.projected_box)
            hand_contact = choose_event_start_contact(contacts)

        decision = self.movement_tracker.update(
            update.mask,
            depth_m,
            self.camera_matrix,
            timestamp=update.frame.stamp,
            hand_contact=hand_contact,
        )

        event_record = None
        if decision.moved and decision.should_stop_tracking:
            event_record = persist_taken_away_event(
                task_id=self.task_id,
                frame=update.frame,
                decision=decision,
                hand_contact=hand_contact,
                mask=update.mask,
                projected_box=self.projected_box.to_dict(),
            )
            self.stop()
        return TrackingFrameResult(frame=update.frame, decision=decision, persisted_event=event_record)

    def replay_cached_frames(self, frames: Iterable[ShigureFrame]) -> list[TrackingFrameResult]:
        results: list[TrackingFrameResult] = []
        iterator = iter(frames)
        try:
            first_frame = next(iterator)
        except StopIteration:
            return results
        if not self.started:
            self.start(first_frame)

        for frame in [first_frame, *list(iterator)]:
            if self.stopped:
                break
            update = self.tracker.update(frame)
            if update is None:
                continue
            result = self.process_update(update)
            results.append(result)
            if result.persisted_event is not None:
                break
        return results

    def stop(self) -> None:
        if self.stopped:
            return
        self.tracker.stop(self.task_id)
        self.stopped = True


def build_session_from_task(
    *,
    task_id: str,
    task_json: dict,
    marker_pose_json: str | Path,
    camera_info_json: str | Path,
    tracker: Sam3VideoTracker,
    image_size: tuple[int, int] | None = None,
) -> ModelEventTrackingSession:
    camera_matrix = load_camera_matrix(camera_info_json)
    projected_box = project_model_bounds_to_shigurei(
        task_json,
        marker_pose_json,
        camera_matrix,
        image_size=image_size,
    )
    return ModelEventTrackingSession(
        task_id=task_id,
        projected_box=projected_box,
        camera_matrix=camera_matrix,
        tracker=tracker,
    )


def replay_from_history_after_model_ready(
    *,
    task_id: str,
    task_json: dict,
    cache: ShigureHistoryCache,
    tracker: Sam3VideoTracker,
    marker_pose_json: str | Path,
    start_stamp: RosStamp | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[TrackingFrameResult]:
    first_frame = next(cache.iter_frames(start=start_stamp), None)
    if first_frame is None:
        return []
    camera_info = first_frame.camera_info_path
    if camera_info is None:
        raise ValueError("cached frame is missing camera_info")
    session = build_session_from_task(
        task_id=task_id,
        task_json=task_json,
        marker_pose_json=marker_pose_json,
        camera_info_json=camera_info,
        tracker=tracker,
        image_size=image_size,
    )
    return session.replay_cached_frames(cache.iter_frames(start=start_stamp))
