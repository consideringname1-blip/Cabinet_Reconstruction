"""Temporal hand-mask diagnosis and conservative correction for stage 1.5."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def _centroid(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    return np.asarray([xx.mean(), yy.mean()], dtype=np.float64) if len(xx) else np.asarray([np.nan, np.nan])


def _centroid_motion(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = _centroid(a), _centroid(b)
    if not np.isfinite(ca).all() or not np.isfinite(cb).all():
        return 0.0 if int(a.sum()) == int(b.sum()) == 0 else 1.0
    diagonal = float(np.hypot(*a.shape))
    return float(np.linalg.norm(ca - cb) / max(diagonal, 1.0))


def _boundary_contact(mask: np.ndarray, band: int) -> float:
    if not mask.any():
        return 0.0
    border = np.zeros_like(mask)
    border[:band] = True; border[-band:] = True; border[:, :band] = True; border[:, -band:] = True
    return float(np.logical_and(mask, border).sum() / mask.sum())


def _load_detector_manifest(path: Path, records: list[dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("records", [])
    by_image = {str(Path(item["image"]).resolve()): item for item in source}
    result = []
    for index, record in enumerate(records):
        item = by_image.get(str(Path(record["rgb_path"]).resolve()))
        if item is None and index < len(source):
            item = source[index]
        item = item or {}
        result.append({
            "detector_confidence": float(max(item.get("grounding_confidences", [0.0]), default=0.0)),
            "segmentation_confidence": float(max(item.get("sam_scores", [0.0]), default=0.0)),
            "detector_record": item,
        })
    return result


def diagnose(records: list[dict], footprints: list[np.ndarray], manifest_path: Path, config: dict) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    masks = [np.load(record["hand_mask_path"]).squeeze().astype(bool) for record in records]
    detector = _load_detector_manifest(manifest_path, records)
    rows, anomalies = [], []
    band = int(config["image_boundary_band_px"])
    for index, (record, mask, footprint) in enumerate(zip(records, masks, footprints)):
        footprint_count = max(int(footprint.sum()), 1)
        prev = masks[index - 1] if index else mask
        nxt = masks[index + 1] if index + 1 < len(masks) else mask
        prev_area = max(int(prev.sum()), 1)
        next_area = max(int(nxt.sum()), 1)
        area_change_prev = abs(int(mask.sum()) - int(prev.sum())) / prev_area
        area_change_next = abs(int(mask.sum()) - int(nxt.sum())) / next_area
        iou_prev, iou_next = _iou(mask, prev), _iou(mask, nxt)
        motion_prev, motion_next = _centroid_motion(mask, prev), _centroid_motion(mask, nxt)
        boundary = _boundary_contact(mask, band)
        area_in = float(np.logical_and(mask, footprint).sum() / footprint_count)
        det_conf = detector[index]["detector_confidence"]
        reasons = []
        if max(area_change_prev, area_change_next) > float(config["max_neighbor_area_change_ratio"]):
            reasons.append("abrupt_area_change")
        if area_in > float(config["max_rgb_footprint_coverage"]):
            reasons.append("excessive_rgb_footprint_coverage")
        if min(iou_prev, iou_next) < float(config["min_neighbor_iou"]):
            reasons.append("low_temporal_iou")
        if max(motion_prev, motion_next) > float(config["max_centroid_motion_fraction"]):
            reasons.append("discontinuous_centroid_motion")
        if boundary > float(config["max_boundary_contact_ratio"]):
            reasons.append("large_image_boundary_attachment")
        if det_conf < float(config["low_detector_confidence"]) and area_in > float(config["large_area_with_low_confidence"]):
            reasons.append("large_mask_with_low_detector_confidence")
        row = {
            "processing_index": index, "original_frame_id": int(record["original_frame_id"]), "timestamp": int(record["timestamp"]),
            "hand_area_ratio": float(mask.mean()), "hand_area_ratio_in_rgb_footprint": area_in,
            "hand_mask_iou_with_previous": iou_prev, "hand_mask_iou_with_next": iou_next,
            "hand_mask_centroid_motion": motion_prev, "hand_mask_centroid_motion_to_next": motion_next,
            "hand_mask_area_change_ratio": area_change_prev, "hand_mask_area_change_ratio_to_next": area_change_next,
            "hand_mask_boundary_contact_ratio": boundary, "hand_detector_confidence": det_conf,
            "hand_segmentation_confidence": detector[index]["segmentation_confidence"],
            "hand_propagation_confidence": 0.0, "hand_mask_temporal_anomaly": bool(reasons),
            "anomaly_reasons": ";".join(reasons),
        }
        rows.append(row)
        if reasons:
            anomalies.append({"processing_index": index, "original_frame_id": int(record["original_frame_id"]), "timestamp": int(record["timestamp"]), "reasons": reasons})
    return masks, rows, anomalies


def _flow_params(config: dict) -> dict:
    return {"pyr_scale": float(config["pyr_scale"]), "levels": int(config["levels"]), "winsize": int(config["winsize"]),
            "iterations": int(config["iterations"]), "poly_n": int(config["poly_n"]), "poly_sigma": float(config["poly_sigma"]), "flags": 0}


def _warp_source_to_target(source_mask: np.ndarray, source_gray: np.ndarray, target_gray: np.ndarray, config: dict) -> np.ndarray:
    # Backward flow maps every target pixel to its source sampling coordinate.
    backward = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, **_flow_params(config))
    height, width = target_gray.shape
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped = cv2.remap(source_mask.astype(np.uint8), xx + backward[..., 0], yy + backward[..., 1], cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(bool)


def _propagate(masks: list[np.ndarray], grays: list[np.ndarray], source: int, target: int, config: dict) -> np.ndarray:
    mask = masks[source].copy()
    step = 1 if target > source else -1
    index = source
    while index != target:
        nxt = index + step
        mask = _warp_source_to_target(mask, grays[index], grays[nxt], config)
        index = nxt
    return mask


def correct(records: list[dict], masks: list[np.ndarray], rows: list[dict], config: dict, output_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    original_dir = output_dir / "hand_masks_original"
    corrected_dir = output_dir / "hand_masks_corrected"
    uncertain_dir = output_dir / "hand_masks_uncertain"
    vis_dir = output_dir / "visualization" / "hand_mask_before_after"
    for directory in (original_dir, corrected_dir, uncertain_dir, vis_dir): directory.mkdir(parents=True, exist_ok=True)
    grays = [cv2.cvtColor(cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY) for record in records]
    trusted = [not row["hand_mask_temporal_anomaly"] and row["hand_detector_confidence"] >= float(config["trusted_detector_confidence"]) for row in rows]
    corrected, uncertain, correction_records = [], [], []
    kernel = np.ones((3, 3), np.uint8)
    max_distance = int(config["max_keyframe_distance"])
    for index, (record, original, row) in enumerate(zip(records, masks, rows)):
        np.save(original_dir / f"{record['original_frame_id']}.npy", original)
        if not row["hand_mask_temporal_anomaly"]:
            final, uncertainty, source, prop_conf = original.copy(), np.zeros_like(original), "original_trusted_per_frame_detection", 1.0
        else:
            previous = next((j for j in range(index - 1, max(-1, index - max_distance - 1), -1) if trusted[j]), None)
            following = next((j for j in range(index + 1, min(len(records), index + max_distance + 1)) if trusted[j]), None)
            forward = _propagate(masks, grays, previous, index, config) if previous is not None else None
            backward = _propagate(masks, grays, following, index, config) if following is not None else None
            redetection_dir = Path(config["redetection_mask_dir"])
            redetection_path = redetection_dir / f"{index:06d}.npy"
            detector = np.load(redetection_path).squeeze().astype(bool) if redetection_path.exists() else original
            redetection_available = redetection_path.exists()
            if forward is not None and backward is not None:
                agreement = forward & backward
                support = detector & (forward | backward)
                final = agreement | support
                consistency = _iou(forward, backward)
                prop_conf = float(consistency * np.sqrt(max(rows[previous]["hand_detector_confidence"], 0.0) * max(rows[following]["hand_detector_confidence"], 0.0)))
                source = ("current_frame_redetection" if redetection_available else "stored_per_frame_detection") + "_supported_by_bidirectional_optical_flow"
                envelope = forward | backward | detector
            elif forward is not None or backward is not None:
                propagated = forward if forward is not None else backward
                final = detector & propagated
                prop_conf = float(_iou(detector, propagated) * rows[previous if previous is not None else following]["hand_detector_confidence"])
                source = ("current_frame_redetection" if redetection_available else "stored_per_frame_detection") + "_supported_by_unidirectional_optical_flow"
                envelope = detector | propagated
            else:
                final = np.zeros_like(original)
                prop_conf = 0.0
                source = "no_trusted_propagation_hand_neighborhood_uncertain"
                envelope = original
            if final.any():
                final = cv2.morphologyEx(final.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=int(config["mask_close_iterations"])).astype(bool)
            dilation = int(config["uncertain_dilation_px"])
            uncertainty = cv2.dilate(envelope.astype(np.uint8), np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)).astype(bool) & (~final)
        row["hand_propagation_confidence"] = prop_conf
        row["corrected_hand_area_ratio"] = float(final.mean())
        row["correction_source"] = source
        corrected.append(final); uncertain.append(uncertainty)
        corrected_path = corrected_dir / f"{record['original_frame_id']}.npy"
        uncertain_path = uncertain_dir / f"{record['original_frame_id']}.npy"
        np.save(corrected_path, final); np.save(uncertain_path, uncertainty)
        record["hand_mask_original_path"] = record["hand_mask_path"]
        record["hand_mask_path"] = str(corrected_path.resolve())
        record["hand_uncertain_path"] = str(uncertain_path.resolve())
        correction_records.append({
            "processing_index": index, "original_frame_id": int(record["original_frame_id"]), "timestamp": int(record["timestamp"]),
            "anomaly_reasons": row["anomaly_reasons"].split(";") if row["anomaly_reasons"] else [], "correction_source": source,
            "original_area_ratio": row["hand_area_ratio"], "corrected_area_ratio": float(final.mean()),
            "uncertain_area_ratio": float(uncertainty.mean()), "propagation_confidence": prop_conf,
        })
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        left, right = image.copy(), image.copy()
        left[original] = (0, 165, 255); right[final] = (0, 165, 255); right[uncertainty] = (255, 0, 255)
        panel = np.hstack([cv2.addWeighted(image, .55, left, .45, 0), cv2.addWeighted(image, .55, right, .45, 0)])
        cv2.putText(panel, f"original={record['original_frame_id']} left=raw right=corrected/uncertain", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(str(vis_dir / f"{record['original_frame_id']}.png"), panel)
    return corrected, uncertain, correction_records


def write_quality_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_anomalies(path: Path, anomalies: list[dict], corrections: list[dict], config: dict) -> None:
    path.write_text(json.dumps({"thresholds": config, "anomalies": anomalies, "corrections": corrections}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
