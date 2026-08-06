"""Unified camera/tracking/fusion validity masks."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .reference import depth_components, rgb_footprint

def compose_coarse_static_mask(dynamic_mask: np.ndarray, object_mask: np.ndarray,
                               valid_for_camera: np.ndarray) -> np.ndarray:
    """Correct replacement for the baseline mask that was overwritten."""
    if dynamic_mask.shape != object_mask.shape or dynamic_mask.shape != valid_for_camera.shape:
        raise ValueError("coarse mask shapes must match")
    return (~dynamic_mask.astype(bool)) & object_mask.astype(bool) & valid_for_camera.astype(bool)



def _dense_fb(gray_a: np.ndarray, gray_b: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    params = dict(
        pyr_scale=float(config["pyr_scale"]), levels=int(config["levels"]),
        winsize=int(config["winsize"]), iterations=int(config["iterations"]),
        poly_n=int(config["poly_n"]), poly_sigma=float(config["poly_sigma"]), flags=0,
    )
    forward = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **params)
    backward = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, **params)
    height, width = gray_a.shape
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = xx + forward[..., 0], yy + forward[..., 1]
    back_at = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    error_a = np.linalg.norm(forward + back_at, axis=2)
    inside_a = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    map_x_b, map_y_b = xx + backward[..., 0], yy + backward[..., 1]
    fwd_at = cv2.remap(forward, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    error_b = np.linalg.norm(backward + fwd_at, axis=2)
    inside_b = (map_x_b >= 0) & (map_x_b <= width - 1) & (map_y_b >= 0) & (map_y_b <= height - 1)
    threshold = float(config["max_error_px"])
    return inside_a & np.isfinite(error_a) & (error_a <= threshold), inside_b & np.isfinite(error_b) & (error_b <= threshold)


def build_validity(records: list[dict], config: dict, output_dir: Path) -> tuple[list[dict], list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_valid_dir, depth_valid_dir = output_dir / "rgb_valid", output_dir / "depth_valid"
    rgb_valid_dir.mkdir(exist_ok=True); depth_valid_dir.mkdir(exist_ok=True)
    rgbs, grays, depths, footprints, depth_valids, boundary_distances, hands = [], [], [], [], [], [], []
    for record in records:
        rgb = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
        depth = raw_depth.astype(np.float32) * float(config["depth_scale_to_m"])
        footprint = rgb_footprint(rgb, config)
        depth_valid, boundary_distance = depth_components(depth, config)
        hand = np.load(record["hand_mask_path"]).squeeze().astype(bool)
        if record.get("hand_uncertain_path"):
            hand = hand | np.load(record["hand_uncertain_path"]).squeeze().astype(bool)
        rgbs.append(rgb); grays.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)); depths.append(depth)
        footprints.append(footprint); depth_valids.append(depth_valid); boundary_distances.append(boundary_distance); hands.append(hand)
    count = len(records)
    fb_masks = [np.ones_like(footprints[0], dtype=bool) for _ in records]
    if count > 1:
        outgoing, incoming = [], []
        for index in range(count - 1):
            a, b = _dense_fb(grays[index], grays[index + 1], config["dense_fb"])
            outgoing.append(a); incoming.append(b)
        fb_masks[0] = outgoing[0]
        fb_masks[-1] = incoming[-1]
        for index in range(1, count - 1):
            fb_masks[index] = incoming[index - 1] | outgoing[index]

    payloads, usage = [], []
    fallback_confidence = float(config["depth_confidence_fallback"])
    min_confidence = float(config["min_depth_confidence"])
    boundary_min = float(config["depth_boundary_erosion_px"])
    for index, record in enumerate(records):
        depth_confidence = np.where(depth_valids[index], fallback_confidence, 0.0).astype(np.float32)
        confidence_valid = depth_confidence >= min_confidence
        boundary_valid = boundary_distances[index] >= boundary_min
        occlusion = (~fb_masks[index]) & footprints[index]
        common = footprints[index] & depth_valids[index] & confidence_valid & (~hands[index]) & boundary_valid & (~occlusion) & fb_masks[index]
        valid_for_camera = common.copy()
        valid_for_tracking = common.copy()
        valid_for_fusion = footprints[index] & depth_valids[index] & confidence_valid & (~hands[index]) & boundary_valid
        original_id = int(record["original_frame_id"])
        rgb_valid_path = rgb_valid_dir / f"{original_id}.npy"
        depth_valid_path = depth_valid_dir / f"{original_id}.npy"
        np.save(rgb_valid_path, footprints[index]); np.save(depth_valid_path, depth_valids[index])
        record["rgb_valid_path"] = str(rgb_valid_path.resolve())
        record["depth_valid_path"] = str(depth_valid_path.resolve())
        mask_path = output_dir / f"{original_id}.npz"
        np.savez_compressed(mask_path, rgb_footprint=footprints[index], depth_valid=depth_valids[index], depth_confidence=depth_confidence,
                            hand_mask=hands[index], boundary_distance=boundary_distances[index], fb_consistent=fb_masks[index],
                            occlusion=occlusion, valid_for_camera=valid_for_camera, valid_for_tracking=valid_for_tracking, valid_for_fusion=valid_for_fusion)
        footprint_count = max(int(footprints[index].sum()), 1)
        hand_fraction = float(hands[index].sum() / footprint_count)
        tracking_fraction = float(valid_for_tracking.sum() / footprint_count)
        fusion_fraction = float(valid_for_fusion.sum() / footprint_count)
        hand_ok = hand_fraction <= float(config["frame_max_hand_fraction"])
        usage.append({
            "processing_index": index, "original_frame_id": original_id, "timestamp": record["timestamp"],
            "usable_for_pose_prior": True,
            "usable_for_static_tracking": bool(hand_ok and tracking_fraction >= float(config["frame_min_tracking_fraction"])),
            "usable_for_geometry": bool(hand_ok and fusion_fraction >= float(config["frame_min_geometry_fraction"])),
            "usable_for_joint_estimation": bool(hand_ok and tracking_fraction >= float(config["frame_min_joint_fraction"])),
            "hand_fraction": hand_fraction, "rgb_valid_fraction": float(footprints[index].mean()),
            "depth_valid_fraction": float(depth_valids[index].sum() / footprint_count),
            "valid_for_tracking_fraction": tracking_fraction, "valid_for_fusion_fraction": fusion_fraction,
            "mask_path": str(mask_path.resolve()),
        })
        payloads.append({"rgb": rgbs[index], "depth": depths[index], "hand": hands[index], "boundary_distance": boundary_distances[index],
                         "depth_confidence": depth_confidence, "occlusion": occlusion, "fb_consistent": fb_masks[index],
                         "valid_for_camera": valid_for_camera, "valid_for_tracking": valid_for_tracking, "valid_for_fusion": valid_for_fusion})
    return payloads, usage
