"""Explicit derived depth-quality features; never labels heuristics as sensor confidence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def inspect_sensor_metadata(records: list[dict], config: dict) -> dict:
    source_root = Path(records[0]["source_root"])
    native = source_root / "Depth Long Throw"
    source_index = Path(config["source_depth_index_path"])
    depth_lines = [line.split() for line in source_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapped = {Path(parts[-1].replace("\\", "/")).stem: parts[0] for parts in depth_lines}
    rows = []
    for record in records:
        timestamp = mapped.get(record["depth_source_id"])
        rows.append({"original_frame_id": int(record["original_frame_id"]), "long_throw_timestamp": timestamp,
                     "depth_native_exists": bool(timestamp and (native / f"{timestamp}.pgm").exists()),
                     "active_brightness_native_exists": bool(timestamp and (native / f"{timestamp}_ab.pgm").exists())})
    return {
        "sigma_available": False, "invalidity_available": False,
        "active_brightness_native_available_for_all_frames": all(row["active_brightness_native_exists"] for row in rows),
        "active_brightness_registered_to_current_pv_depth": False,
        "measured_sensor_confidence_status": "unavailable_for_current_registered_depth",
        "reason": "Long Throw active-brightness rasters exist, but the current PV depth was z-buffered from world PLY vertices; the PLY has no native raster index, so brightness cannot be associated unambiguously after projection.",
        "algorithm_input_policy": "derived_depth_quality plus binary_validity_fallback; active brightness is provenance only",
        "records": rows,
    }


def _maps(payload: dict, intrinsic: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    depth = payload["depth"]
    valid = np.isfinite(depth) & (depth > 0)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    radius = int(config["neighborhood_radius_px"])
    size = 2 * radius + 1
    neighbor_valid = cv2.boxFilter(valid.astype(np.float32), -1, (size, size), normalize=True)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    # Approximate camera-space tangents and incidence from the local depth surface.
    tx = np.stack([depth / fx + gx * 0, np.zeros_like(depth), gx], axis=-1)
    ty = np.stack([np.zeros_like(depth), depth / fy + gy * 0, gy], axis=-1)
    normal = np.cross(tx, ty)
    normal_norm = np.linalg.norm(normal, axis=2)
    incidence = np.abs(normal[..., 2]) / np.maximum(normal_norm, 1e-9)
    boundary_q = np.clip(payload["boundary_distance"] / max(float(config["boundary_full_quality_px"]), 1e-9), 0.0, 1.0)
    gradient_q = np.exp(-gradient / max(float(config["gradient_scale_m_per_px"]), 1e-9))
    validity_q = valid.astype(np.float32)
    occlusion_edge = payload["occlusion"] | (~payload["fb_consistent"])
    occlusion_q = (~occlusion_edge).astype(np.float32)
    return {"valid": valid, "gradient": gradient, "neighbor_valid": neighbor_valid, "incidence": incidence,
            "occlusion_edge": occlusion_edge, "boundary_quality": boundary_q, "gradient_quality": gradient_q,
            "occlusion_quality": occlusion_q, "validity_quality": validity_q}


def annotate_tracks(tracks: list[dict], payloads: list[dict], records: list[dict], config: dict) -> dict:
    intrinsic = np.asarray(records[0]["intrinsics"], dtype=np.float64)
    maps = [_maps(payload, intrinsic, config) for payload in payloads]
    timestamp_scale = float(config["timestamp_scale_seconds"])
    quality_values = []
    for track in tracks:
        observations = track["observations"]
        for row, obs in enumerate(observations):
            index = int(obs["processing_index"]); u, v = (int(round(x)) for x in obs["pixel_uv"])
            local = maps[index]
            if row:
                prev = observations[row - 1]
                consistency_error = float(np.linalg.norm(np.asarray(obs["point_world"])-np.asarray(prev["point_world"])))
                dt = abs(int(obs["timestamp"]) - int(prev["timestamp"])) * timestamp_scale
            else:
                consistency_error = 0.0; dt = 0.0
            reprojection_q = float(np.exp(-consistency_error / max(float(config["reprojection_depth_scale_m"]), 1e-9)))
            time_q = float(np.exp(-dt / max(float(config["time_delta_scale_seconds"]), 1e-9))) if row else 1.0
            components={"boundary":float(local["boundary_quality"][v,u]),"gradient":float(local["gradient_quality"][v,u]),
                        "neighborhood":float(local["neighbor_valid"][v,u]),"incidence":float(local["incidence"][v,u]),
                        "reprojection":reprojection_q,"time_delta":time_q,"occlusion":float(local["occlusion_quality"][v,u])}
            weights=config["quality_weights"]; total=sum(float(weights[name]) for name in components)
            quality=float(local["validity_quality"][v,u] * sum(float(weights[name])*value for name,value in components.items())/max(total,1e-9))
            obs.update({
                "measured_sensor_confidence": float("nan"),
                "derived_depth_quality": quality, "binary_validity_fallback": float(local["valid"][v, u]),
                "local_depth_gradient": float(local["gradient"][v, u]),
                "neighborhood_valid_fraction": float(local["neighbor_valid"][v, u]),
                "reprojection_depth_consistency": reprojection_q,
                "incidence_quality": float(local["incidence"][v, u]), "time_delta_seconds": dt,
                "occlusion_edge_flag": float(local["occlusion_edge"][v, u]),
            })
            # Fixed stage-1 classifier reads this field; only its observational input provenance changes.
            obs["depth_confidence"] = quality
            quality_values.append(quality)
    return {"observation_count": len(quality_values), "derived_depth_quality_percentiles": np.percentile(quality_values, [0,10,25,50,75,90,100]).tolist() if quality_values else [],
            "measured_sensor_confidence_used": False, "binary_validity_fallback_used": True,
            "quality_aggregation":"configured weighted arithmetic mean after hard binary validity",
            "quality_weights":config["quality_weights"],"classifier_thresholds_changed": False}


def save_quality_npz(path: Path, tracks: list[dict]) -> None:
    observations = [obs for track in tracks for obs in track["observations"]]
    names = ["derived_depth_quality", "binary_validity_fallback", "local_depth_gradient", "neighborhood_valid_fraction",
             "reprojection_depth_consistency", "incidence_quality", "time_delta_seconds", "occlusion_edge_flag"]
    values = {name: np.asarray([obs[name] for obs in observations], dtype=np.float32) for name in names}
    values["track_id"] = np.asarray([obs["track_id"] for obs in observations], dtype=np.int64)
    values["original_frame_id"] = np.asarray([obs["original_frame_id"] for obs in observations], dtype=np.int64)
    values["measured_sensor_confidence"] = np.full(len(observations), np.nan, dtype=np.float32)
    np.savez_compressed(path, **values)


def write_report(path: Path, sensor: dict, derived: dict) -> None:
    path.write_text(json.dumps({"sensor_metadata": sensor, "derived_quality": derived}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
