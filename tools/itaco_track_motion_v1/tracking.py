"""Short 3D tracks from fixed rebased device poses and forward/backward LK."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .geometry import transform_point, unproject_pixel
from .proposals import proposal_at


def _depth_uncertainty(depth: np.ndarray, u: int, v: int, radius: int) -> float:
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    values = depth[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 3:
        return float("inf")
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _observation(track_id: int, record: dict, payload: dict, proposals: list[dict], uv: np.ndarray,
                 pose_reference_camera: np.ndarray, intrinsic: np.ndarray, tracking_confidence: float,
                 config: dict) -> dict | None:
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    height, width = payload["depth"].shape
    if not (0 <= u < width and 0 <= v < height) or not payload["valid_for_tracking"][v, u]:
        return None
    if payload["hand"][v, u] or payload["occlusion"][v, u] or not payload["fb_consistent"][v, u]:
        return None
    depth = float(payload["depth"][v, u])
    point_camera = unproject_pixel(np.asarray([u, v]), depth, intrinsic)
    point_world = transform_point(pose_reference_camera, point_camera)
    depth_uncertainty = _depth_uncertainty(payload["depth"], u, v, int(config["depth_uncertainty_patch_radius_px"]))
    depth_confidence = float(payload["depth_confidence"][v, u])
    bgr = payload["rgb"][v, u]
    return {
        "track_id": track_id, "processing_index": int(record["processing_index"]),
        "original_frame_id": int(record["original_frame_id"]), "timestamp": int(record["timestamp"]),
        "pixel_uv": [float(uv[0]), float(uv[1])], "depth": depth,
        "point_camera": point_camera.tolist(), "point_world": point_world.tolist(),
        "tracking_confidence": float(tracking_confidence), "depth_confidence": depth_confidence,
        "depth_uncertainty_m": depth_uncertainty, "visibility": True,
        "occlusion_flag": False, "boundary_distance": float(payload["boundary_distance"][v, u]),
        "hand_mask_flag": False, "proposal_id": proposal_at(proposals, u, v),
        "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
    }


def _detect(gray: np.ndarray, valid: np.ndarray, active_uv: list[np.ndarray], config: dict, capacity: int) -> np.ndarray:
    if capacity <= 0:
        return np.empty((0, 2), dtype=np.float32)
    mask = valid.astype(np.uint8) * 255
    exclusion = int(config["active_exclusion_radius_px"])
    for uv in active_uv:
        cv2.circle(mask, (int(round(float(uv[0]))), int(round(float(uv[1])))), exclusion, 0, -1)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=capacity, qualityLevel=float(config["quality_level"]),
                                     minDistance=float(config["min_feature_distance_px"]), mask=mask,
                                     blockSize=int(config["block_size_px"]), useHarrisDetector=False)
    return np.empty((0, 2), dtype=np.float32) if points is None else points[:, 0].astype(np.float32)


def build_tracks(records: list[dict], payloads: list[dict], proposals: list[list[dict]], poses_rebased: np.ndarray,
                 config: dict) -> tuple[list[dict], list[dict]]:
    intrinsic = np.asarray(records[0]["intrinsics"], dtype=np.float64)
    grays = [cv2.cvtColor(payload["rgb"], cv2.COLOR_BGR2GRAY) for payload in payloads]
    tracks: dict[int, dict] = {}
    active: dict[int, np.ndarray] = {}
    next_id = 0

    def add_new(frame_index: int) -> None:
        nonlocal next_id
        capacity = min(int(config["max_new_features_per_frame"]), int(config["max_total_tracks"]) - next_id)
        points = _detect(grays[frame_index], payloads[frame_index]["valid_for_tracking"], list(active.values()), config, capacity)
        for uv in points:
            observation = _observation(next_id, records[frame_index], payloads[frame_index], proposals[frame_index], uv,
                                       poses_rebased[frame_index], intrinsic, 1.0, config)
            if observation is None:
                continue
            tracks[next_id] = {"track_id": next_id, "observations": [observation], "attempted_transitions": 0,
                               "occlusion_failures": 0, "boundary_failures": 0, "fb_failures": 0,
                               "termination_reason": None}
            active[next_id] = uv
            next_id += 1

    add_new(0)
    lk = dict(winSize=(int(config["lk_window_px"]), int(config["lk_window_px"])), maxLevel=int(config["lk_max_level"]),
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(config["lk_iterations"]), float(config["lk_epsilon"])))
    for frame_index in range(1, len(records)):
        ids = list(active)
        if ids:
            previous_points = np.asarray([active[track_id] for track_id in ids], dtype=np.float32).reshape(-1, 1, 2)
            forward, status_f, error_f = cv2.calcOpticalFlowPyrLK(grays[frame_index - 1], grays[frame_index], previous_points, None, **lk)
            backward, status_b, _ = cv2.calcOpticalFlowPyrLK(grays[frame_index], grays[frame_index - 1], forward, None, **lk)
            fb_error = np.linalg.norm(previous_points[:, 0] - backward[:, 0], axis=1)
            for row, track_id in enumerate(ids):
                track = tracks[track_id]
                track["attempted_transitions"] += 1
                uv = forward[row, 0]
                u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
                inside = 0 <= u < grays[frame_index].shape[1] and 0 <= v < grays[frame_index].shape[0]
                if not status_f[row, 0] or not status_b[row, 0] or fb_error[row] > float(config["fb_max_error_px"]):
                    track["fb_failures"] += 1; track["termination_reason"] = "forward_backward_inconsistent"; active.pop(track_id, None); continue
                if not inside:
                    track["termination_reason"] = "outside_image"; active.pop(track_id, None); continue
                if payloads[frame_index]["occlusion"][v, u]:
                    track["occlusion_failures"] += 1; track["termination_reason"] = "occluded"; active.pop(track_id, None); continue
                if payloads[frame_index]["boundary_distance"][v, u] < float(config["min_boundary_distance_px"]):
                    track["boundary_failures"] += 1; track["termination_reason"] = "depth_boundary"; active.pop(track_id, None); continue
                confidence = float(np.exp(-fb_error[row] / max(float(config["fb_confidence_scale_px"]), 1e-9)) *
                                   np.exp(-float(error_f[row, 0]) / max(float(config["lk_error_scale"]), 1e-9)))
                observation = _observation(track_id, records[frame_index], payloads[frame_index], proposals[frame_index], uv,
                                           poses_rebased[frame_index], intrinsic, confidence, config)
                if observation is None:
                    track["termination_reason"] = "invalid_tracking_pixel"; active.pop(track_id, None); continue
                track["observations"].append(observation); active[track_id] = uv
        add_new(frame_index)
    for track_id in active:
        tracks[track_id]["termination_reason"] = "end_of_view"
    raw = [tracks[track_id] for track_id in sorted(tracks)]
    filtered = []
    for track in raw:
        observations = track["observations"]
        temporal_span = observations[-1]["processing_index"] - observations[0]["processing_index"] + 1
        if len(observations) >= int(config["min_observations"]) and temporal_span >= int(config["min_temporal_span_frames"]):
            track["temporal_span_frames"] = temporal_span
            track["coverage"] = len(observations) / temporal_span
            filtered.append(track)
    assert all(not obs["hand_mask_flag"] for track in raw for obs in track["observations"])
    return raw, filtered


def save_tracks_npz(path: Path, tracks: list[dict]) -> None:
    observations = [obs for track in tracks for obs in track["observations"]]
    offsets = [0]
    for track in tracks:
        offsets.append(offsets[-1] + len(track["observations"]))
    fields = {
        "track_offsets": np.asarray(offsets, dtype=np.int64),
        "track_ids": np.asarray([track["track_id"] for track in tracks], dtype=np.int64),
        "observation_track_id": np.asarray([obs["track_id"] for obs in observations], dtype=np.int64),
        "original_frame_id": np.asarray([obs["original_frame_id"] for obs in observations], dtype=np.int64),
        "timestamp": np.asarray([obs["timestamp"] for obs in observations], dtype=np.int64),
        "processing_index": np.asarray([obs["processing_index"] for obs in observations], dtype=np.int64),
        "pixel_uv": np.asarray([obs["pixel_uv"] for obs in observations], dtype=np.float32).reshape(-1, 2),
        "depth": np.asarray([obs["depth"] for obs in observations], dtype=np.float32),
        "point_camera": np.asarray([obs["point_camera"] for obs in observations], dtype=np.float32).reshape(-1, 3),
        "point_world": np.asarray([obs["point_world"] for obs in observations], dtype=np.float32).reshape(-1, 3),
        "tracking_confidence": np.asarray([obs["tracking_confidence"] for obs in observations], dtype=np.float32),
        "depth_confidence": np.asarray([obs["depth_confidence"] for obs in observations], dtype=np.float32),
        "visibility": np.asarray([obs["visibility"] for obs in observations], dtype=bool),
        "occlusion_flag": np.asarray([obs["occlusion_flag"] for obs in observations], dtype=bool),
        "boundary_distance": np.asarray([obs["boundary_distance"] for obs in observations], dtype=np.float32),
        "hand_mask_flag": np.asarray([obs["hand_mask_flag"] for obs in observations], dtype=bool),
        "proposal_id": np.asarray([obs["proposal_id"] for obs in observations], dtype="U64"),
        "rgb": np.asarray([obs["rgb"] for obs in observations], dtype=np.uint8).reshape(-1, 3),
    }
    optional_float = (
        "measured_sensor_confidence", "derived_depth_quality", "binary_validity_fallback",
        "local_depth_gradient", "neighborhood_valid_fraction", "reprojection_depth_consistency",
        "incidence_quality", "time_delta_seconds", "occlusion_edge_flag",
    )
    for name in optional_float:
        if observations and all(name in obs for obs in observations):
            fields[name] = np.asarray([obs[name] for obs in observations], dtype=np.float32)
    np.savez_compressed(path, **fields)


def load_tracks_npz(path: Path) -> list[dict]:
    data = np.load(path)
    offsets, ids = data["track_offsets"], data["track_ids"]
    tracks = []
    scalar_fields = ("original_frame_id", "timestamp", "processing_index", "depth", "tracking_confidence",
                     "depth_confidence", "visibility", "occlusion_flag", "boundary_distance", "hand_mask_flag", "proposal_id")
    vector_fields = ("pixel_uv", "point_camera", "point_world", "rgb")
    optional = ("measured_sensor_confidence", "derived_depth_quality", "binary_validity_fallback", "local_depth_gradient",
                "neighborhood_valid_fraction", "reprojection_depth_consistency", "incidence_quality", "time_delta_seconds", "occlusion_edge_flag")
    for row, track_id in enumerate(ids):
        observations = []
        for index in range(int(offsets[row]), int(offsets[row+1])):
            obs = {"track_id": int(track_id)}
            for name in scalar_fields:
                value = data[name][index]
                obs[name] = value.item() if hasattr(value, "item") else value
            for name in vector_fields:
                obs[name] = data[name][index].tolist()
            for name in optional:
                if name in data: obs[name] = float(data[name][index])
            obs["depth_uncertainty_m"] = 0.0
            observations.append(obs)
        tracks.append({"track_id": int(track_id), "observations": observations, "attempted_transitions": max(len(observations)-1,0),
                       "occlusion_failures": 0, "boundary_failures": 0, "fb_failures": 0, "termination_reason": "loaded_npz"})
    return tracks
