"""Immutable frame manifest construction and strict validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import Failure, ValidationErrors
from .geometry import load_intrinsics, load_odometry_log, validate_pose


def _read_index(path: Path, root: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed index line {line_number + 1}: {path}")
        relative = parts[1].replace("\\", "/")
        records.append({"timestamp": int(parts[0]), "path": str((root / relative).resolve())})
    return records


def _image_shape(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if image is None else tuple(int(x) for x in image.shape[:2])


def build_manifest(config: dict) -> tuple[list[dict], np.ndarray, dict]:
    dataset = config["dataset"]
    if dataset.get("adapter") != "hololens_pinhole_v1":
        raise ValidationErrors([Failure("manifest", "unsupported_adapter", "Only hololens_pinhole_v1 is implemented in stage 1")])
    source_root = Path(dataset["source_root"])
    mapping_path = Path(dataset["processing_view_manifest_path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rgb_index = _read_index(Path(dataset["rgb_index_path"]), Path(dataset["pinhole_root"]))
    depth_index = _read_index(Path(dataset["depth_index_path"]), Path(dataset["pinhole_root"]))
    pose_ids, all_poses = load_odometry_log(Path(dataset["odometry_path"]))
    intrinsic = load_intrinsics(Path(dataset["intrinsics_path"]))
    hand_paths = sorted(Path(dataset["hand_mask_dir"]).glob(dataset["hand_mask_glob"]))
    autoseg_paths = sorted(Path(dataset["autoseg_dir"]).glob(dataset["autoseg_glob"]))
    if dataset.get("autoseg_order") == "reverse_processing":
        autoseg_paths = autoseg_paths[::-1]

    failures: list[Failure] = []
    warnings: list[dict] = []
    count = len(mapping)
    counts = {
        "processing_view": count,
        "rgb_index": len(rgb_index),
        "depth_index": len(depth_index),
        "odometry": len(all_poses),
        "hand_masks": len(hand_paths),
        "autoseg_frames": len(autoseg_paths),
    }
    if len(hand_paths) != count:
        failures.append(Failure("frame_validation", "hand_count_mismatch", "Hand-mask count does not match processing view", details=counts))
    if len(autoseg_paths) != count:
        failures.append(Failure("frame_validation", "autoseg_count_mismatch", "AutoSeg count does not match processing view", details=counts))
    if len(pose_ids) != len(all_poses):
        failures.append(Failure("frame_validation", "pose_id_count_mismatch", "Pose IDs and matrices differ in count", details=counts))

    records: list[dict] = []
    expected_shape: tuple[int, int] | None = None
    pose_tolerance = float(config["validity"]["pose_matrix_tolerance"])
    for processing_index, item in enumerate(mapping):
        original_id = int(item.get("source_index", item.get("original_frame_id", -1)))
        if original_id < 0 or original_id >= min(len(rgb_index), len(depth_index), len(all_poses)):
            failures.append(Failure("frame_validation", "source_index_out_of_range", "Original frame ID is outside source arrays", original_id, {"processing_index": processing_index}))
            continue
        rgb_path = Path(rgb_index[original_id]["path"])
        depth_path = Path(depth_index[original_id]["path"])
        hand_path = hand_paths[processing_index] if processing_index < len(hand_paths) else Path("__missing__")
        autoseg_path = autoseg_paths[processing_index] if processing_index < len(autoseg_paths) else Path("__missing__")
        paths = {"rgb_path": rgb_path, "depth_path": depth_path, "hand_mask_path": hand_path, "autoseg_path": autoseg_path}
        for key, path in paths.items():
            if not path.is_file():
                failures.append(Failure("frame_validation", "file_missing", f"{key} does not exist", original_id, {"path": str(path), "processing_index": processing_index}))

        rgb_source_id, depth_source_id = rgb_path.stem, depth_path.stem
        if rgb_source_id != depth_source_id:
            failures.append(Failure("frame_validation", "rgb_depth_source_id_mismatch", "RGB and depth filenames do not share source ID", original_id, {"rgb": rgb_source_id, "depth": depth_source_id}))
        mapped_rgb = Path(item.get("rgb", item.get("linked_rgb", str(rgb_path)))).resolve()
        if mapped_rgb != rgb_path.resolve():
            failures.append(Failure("frame_validation", "mapping_rgb_mismatch", "Processing-view RGB does not match source index", original_id, {"mapping": str(mapped_rgb), "source": str(rgb_path)}))
        if int(pose_ids[original_id]) != original_id:
            failures.append(Failure("frame_validation", "pose_source_id_mismatch", "Odometry source ID does not match original frame ID", original_id, {"pose_id": int(pose_ids[original_id])}))

        rgb_shape, depth_shape = _image_shape(rgb_path), _image_shape(depth_path)
        if rgb_shape is None or depth_shape is None:
            failures.append(Failure("frame_validation", "image_decode_failed", "RGB or depth image could not be decoded", original_id))
        elif rgb_shape != depth_shape:
            failures.append(Failure("frame_validation", "rgb_depth_shape_mismatch", "RGB and depth shapes differ", original_id, {"rgb_hw": rgb_shape, "depth_hw": depth_shape}))
        elif expected_shape is None:
            expected_shape = rgb_shape
        elif rgb_shape != expected_shape:
            failures.append(Failure("frame_validation", "image_shape_changed", "Image dimensions changed within view", original_id, {"expected_hw": expected_shape, "actual_hw": rgb_shape}))

        if hand_path.is_file():
            hand = np.load(hand_path).squeeze()
            if rgb_shape is not None and hand.shape != rgb_shape:
                failures.append(Failure("frame_validation", "hand_shape_mismatch", "Hand mask shape differs from RGB", original_id, {"hand_hw": hand.shape, "rgb_hw": rgb_shape}))
        if autoseg_path.is_file():
            with np.load(autoseg_path) as archive:
                masks = archive["a"]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim != 3 or (rgb_shape is not None and tuple(masks.shape[-2:]) != rgb_shape):
                failures.append(Failure("frame_validation", "autoseg_shape_mismatch", "AutoSeg archive does not contain NxHxW masks matching RGB", original_id, {"shape": list(masks.shape), "rgb_hw": rgb_shape}))

        pose_ok, pose_metrics = validate_pose(all_poses[original_id], pose_tolerance)
        if not pose_ok:
            failures.append(Failure("frame_validation", "invalid_camera_pose", "T_world_camera is not a valid rigid transform", original_id, pose_metrics))
        timestamp = int(rgb_index[original_id]["timestamp"])
        if timestamp != int(depth_index[original_id]["timestamp"]):
            failures.append(Failure("frame_validation", "rgb_depth_timestamp_mismatch", "RGB/depth index timestamps differ", original_id, {"rgb": timestamp, "depth": int(depth_index[original_id]["timestamp"])}))

        records.append({
            "original_frame_id": original_id,
            "timestamp": timestamp,
            "processing_index": processing_index,
            "view_order": processing_index,
            "rgb_path": str(rgb_path.resolve()),
            "depth_path": str(depth_path.resolve()),
            "intrinsics": intrinsic.tolist(),
            "camera_pose_init": all_poses[original_id].tolist(),
            "pose_source": dataset["pose_source"],
            "hand_mask_path": str(hand_path.resolve()),
            "rgb_valid_path": None,
            "depth_valid_path": None,
            "autoseg_path": str(autoseg_path.resolve()),
            "rgb_source_id": rgb_source_id,
            "depth_source_id": depth_source_id,
            "source_root": str(source_root.resolve()),
        })

    original_ids = [record["original_frame_id"] for record in records]
    timestamps = [record["timestamp"] for record in records]
    if len(set(original_ids)) != len(original_ids):
        failures.append(Failure("frame_validation", "duplicate_original_frame_id", "Processing view contains duplicate original frame IDs"))
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        failures.append(Failure("frame_validation", "non_monotonic_timestamp", "Processing-view timestamps are not strictly increasing"))
    if expected_shape is not None:
        height, width = expected_shape
        if not (0 <= intrinsic[0, 2] < width and 0 <= intrinsic[1, 2] < height and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0):
            failures.append(Failure("frame_validation", "intrinsics_image_mismatch", "Intrinsics are incompatible with image dimensions", details={"K": intrinsic.tolist(), "image_hw": expected_shape}))
    if dataset.get("depth_confidence_dir") is None:
        warnings.append({"code": "depth_confidence_unavailable", "message": "No registered confidence map was supplied; configured binary measured-depth confidence fallback will be recorded and used."})
    report = {
        "valid": not failures,
        "adapter": dataset["adapter"],
        "frame_count": len(records),
        "source_counts": counts,
        "image_size_hw": list(expected_shape) if expected_shape else None,
        "intrinsics": intrinsic.tolist(),
        "pose_convention": "T_world_camera",
        "checks": ["files_exist", "counts", "timestamp_monotonic", "source_ids", "image_depth_mask_shapes", "intrinsics", "rigid_camera_pose", "traceability"],
        "warnings": warnings,
        "errors": [failure.to_dict() for failure in failures],
    }
    if failures:
        raise ValidationErrors(failures)
    poses = np.stack([np.asarray(record["camera_pose_init"], dtype=np.float64) for record in records])
    return records, poses, report


def save_manifest(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
