"""Explicit AutoSeg proposal adapter with identity independent of NPZ layer order."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _load_layers(path: str) -> np.ndarray:
    with np.load(path) as archive:
        layers = archive["a"]
    if layers.ndim == 4 and layers.shape[1] == 1:
        layers = layers[:, 0]
    return layers.astype(bool)


def _descriptor(mask: np.ndarray) -> dict:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return {"area": 0, "centroid_uv": [float("nan"), float("nan")], "bbox_xyxy": [0, 0, 0, 0]}
    return {
        "area": int(len(xx)), "centroid_uv": [float(np.mean(xx)), float(np.mean(yy))],
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
    }


def _similarity(previous: np.ndarray, current: np.ndarray, config: dict) -> float:
    intersection = int(np.logical_and(previous, current).sum())
    union = int(np.logical_or(previous, current).sum())
    iou = intersection / max(union, 1)
    prev_desc, curr_desc = _descriptor(previous), _descriptor(current)
    distance = np.linalg.norm(np.asarray(prev_desc["centroid_uv"]) - np.asarray(curr_desc["centroid_uv"]))
    centroid_score = np.exp(-distance / float(config["centroid_scale_px"]))
    return float(config["iou_weight"]) * iou + float(config["centroid_weight"]) * centroid_score


def build_explicit_proposals(records: list[dict], config: dict, output_dir: Path) -> tuple[list[dict], list[list[dict]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    next_id = 0
    previous: list[dict] = []
    proposal_records: list[dict] = []
    by_frame: list[list[dict]] = []
    for record in records:
        layers = _load_layers(record["autoseg_path"])
        current = []
        for layer_index, mask in enumerate(layers):
            descriptor = _descriptor(mask)
            if descriptor["area"] < int(config["min_area_px"]):
                continue
            current.append({"mask": mask, "legacy_layer_index": layer_index, **descriptor})
        assignments: dict[int, tuple[str, float]] = {}
        if previous and current:
            matrix = np.asarray([[_similarity(old["mask"], new["mask"], config) for new in current] for old in previous])
            old_indices, new_indices = linear_sum_assignment(-matrix)
            for old_index, new_index in zip(old_indices, new_indices):
                score = float(matrix[old_index, new_index])
                if score >= float(config["association_min_score"]):
                    assignments[int(new_index)] = (previous[int(old_index)]["proposal_id"], score)
        frame_payload = {}
        frame_entries = []
        for current_index, proposal in enumerate(current):
            if current_index in assignments:
                proposal_id, association_score = assignments[current_index]
            else:
                proposal_id = f"proposal_{next_id:06d}"
                next_id += 1
                association_score = None
            mask_key = proposal_id
            if mask_key in frame_payload:
                mask_key = f"{proposal_id}_component_{current_index}"
            frame_payload[mask_key] = proposal["mask"]
            entry = {
                "proposal_id": proposal_id,
                "original_frame_id": int(record["original_frame_id"]),
                "processing_index": int(record["processing_index"]),
                "mask": {"path": None, "key": mask_key},
                "score": float(config["legacy_default_score"]),
                "score_source": "configured_default_source_score_missing",
                "association_score": association_score,
                "source": {"type": "legacy_autoseg_npz_adapter", "path": record["autoseg_path"], "layer_index_provenance_only": int(proposal["legacy_layer_index"]), "layer_order_used_as_uid": False},
                "area_px": proposal["area"], "centroid_uv": proposal["centroid_uv"], "bbox_xyxy": proposal["bbox_xyxy"],
            }
            frame_entries.append({**entry, "mask_array": proposal["mask"]})
        mask_path = output_dir / f"frame_{int(record['original_frame_id'])}.npz"
        np.savez_compressed(mask_path, **frame_payload)
        for entry in frame_entries:
            entry["mask"]["path"] = str(mask_path.resolve())
            serializable = {key: value for key, value in entry.items() if key != "mask_array"}
            proposal_records.append(serializable)
        by_frame.append(frame_entries)
        previous = [{"proposal_id": entry["proposal_id"], "mask": entry["mask_array"]} for entry in frame_entries]
    return proposal_records, by_frame


def proposal_at(proposals: list[dict], u: int, v: int) -> str:
    candidates = [item for item in proposals if item["mask_array"][v, u]]
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["area_px"]), item["proposal_id"]))
    return str(candidates[0]["proposal_id"])
