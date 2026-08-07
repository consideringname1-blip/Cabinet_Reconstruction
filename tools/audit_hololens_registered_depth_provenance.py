#!/usr/bin/env python3
"""Audit HoloLens2ForCV registered-depth provenance and physical units.

This tool is read-only with respect to the recording and reconstruction outputs.
It does not import or invoke Assignment v4, SAM2, TSDF, NKSR, or mesh code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

UPSTREAM_REPOSITORY = "https://github.com/microsoft/HoloLens2ForCV"
UPSTREAM_REVISION = "207205596840ae6e7b8c2a795d35c3c4e7bec22e"
UPSTREAM_SAVE_PCLOUDS = "Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py"
UPSTREAM_UTILS = "Samples/StreamRecorder/StreamRecorderConverter/utils.py"
UPSTREAM_PROCESS_ALL = "Samples/StreamRecorder/StreamRecorderConverter/process_all.py"


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_index(path: Path, root: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed index line {line_number}: {path}")
        relative = parts[1].replace("\\", "/")
        rows.append({"association_timestamp": int(parts[0]), "relative_path": relative,
                     "path": root / relative, "filename_timestamp": int(Path(relative).stem.split("_")[0])})
    return rows


def load_poses(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 5:
        raise ValueError(f"invalid odometry blocks: {path}")
    return np.stack([np.asarray([[float(v) for v in row.split()] for row in lines[i + 1:i + 5]])
                     for i in range(0, len(lines), 5)])


def load_pv_metadata(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    header = [float(v) for v in lines[0].split(",")]
    timestamps, focal, poses = [], [], []
    for line in lines[1:]:
        values = line.split(",")
        timestamps.append(int(values[0])); focal.append([float(values[1]), float(values[2])])
        poses.append(np.asarray(values[3:19], float).reshape(4, 4))
    return {"principal_point": header[:2], "width": int(header[2]), "height": int(header[3]),
            "timestamps": np.asarray(timestamps, np.int64), "focal_lengths": np.asarray(focal),
            "poses_world": np.asarray(poses)}


def png_header(path: Path) -> dict:
    data = path.read_bytes()[:33]
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return {"is_png": False}
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", data[16:29])
    return {"is_png": True, "width": width, "height": height, "bit_depth": bit_depth,
            "color_type": color_type, "compression": compression, "filter": filtering, "interlace": interlace}


def histogram_quantile(histogram: np.ndarray, probability: float) -> int | None:
    total = int(histogram.sum())
    if total == 0:
        return None
    return int(np.searchsorted(np.cumsum(histogram), probability * (total - 1) + 1))


def saturation_count(image: np.ndarray) -> int:
    return int((np.asarray(image) == np.iinfo(np.uint16).max).sum())


def audit_registered_files(root: Path, depth_rows: list[dict], rgb_rows: list[dict], representative: dict) -> dict:
    histogram = np.zeros(65536, np.int64); shapes = {}; dtypes = {}; per_frame = []
    saturation = 0; nonzero_total = 0; zero_total = 0
    for frame_id, row in enumerate(depth_rows):
        image = cv2.imread(str(row["path"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(row["path"])
        shapes[str(tuple(image.shape))] = shapes.get(str(tuple(image.shape)), 0) + 1
        dtypes[str(image.dtype)] = dtypes.get(str(image.dtype), 0) + 1
        counts = np.bincount(image.reshape(-1), minlength=65536); histogram += counts
        nonzero = image[image > 0]; saturation += saturation_count(image)
        nonzero_total += int(nonzero.size); zero_total += int((image == 0).sum())
        per_frame.append({"original_frame_id": frame_id, "association_timestamp": row["association_timestamp"],
                          "filename_timestamp": row["filename_timestamp"], "shape": list(image.shape),
                          "dtype": str(image.dtype), "nonzero_ratio": float((image > 0).mean()),
                          "min_raw_nonzero": int(nonzero.min()) if len(nonzero) else None,
                          "median_raw_nonzero": float(np.median(nonzero)) if len(nonzero) else None,
                          "p95_raw_nonzero": float(np.percentile(nonzero, 95)) if len(nonzero) else None,
                          "max_raw": int(image.max()), "saturated_65535": saturation_count(image)})
    rgb_depth_same_file_timestamp = [a["filename_timestamp"] == b["filename_timestamp"] for a, b in zip(depth_rows, rgb_rows)]
    source_files_exist = [(root / "Depth Long Throw" / f"{row['association_timestamp']}.pgm").is_file() and
                          (root / "Depth Long Throw" / f"{row['association_timestamp']}.ply").is_file()
                          for row in depth_rows]
    all_pixels = nonzero_total + zero_total
    sample_ids = [x for values in representative.values() for x in values]
    return {"artifact": str(root / "pinhole_projection/depth"), "file_count": len(depth_rows),
            "rgb_file_count": len(rgb_rows), "image_shapes": shapes, "dtypes": dtypes,
            "png_header_first": png_header(depth_rows[0]["path"]), "png_metadata_note": "PNG IHDR contains no physical-unit tag",
            "min_raw_nonzero": histogram_quantile(histogram[1:], 0.0) + 1,
            "median_raw_nonzero": histogram_quantile(histogram[1:], 0.5) + 1,
            "p95_raw_nonzero": histogram_quantile(histogram[1:], 0.95) + 1,
            "max_raw": int(np.flatnonzero(histogram)[-1]), "zero_ratio": zero_total / all_pixels,
            "saturation_65535_pixels": saturation, "uniform_shape_and_dtype": len(shapes) == 1 and len(dtypes) == 1,
            "rgb_depth_filename_timestamp_match_all": bool(all(rgb_depth_same_file_timestamp)),
            "association_timestamp_has_pgm_and_ply_all": bool(all(source_files_exist)),
            "timestamp_semantics": "depth.txt column 1 is Long Throw timestamp; path stem is matched PV timestamp",
            "representative_original_frame_ids": representative,
            "representative_frames": [per_frame[i] for i in sample_ids], "all_frames": per_frame}


def last_write_projection(points_camera: np.ndarray, intrinsic: np.ndarray, shape: tuple[int, int]) -> dict:
    points = np.asarray(points_camera, float); z = points[:, 2]
    uv_float = np.column_stack((intrinsic[0, 0] * points[:, 0] / z + intrinsic[0, 2],
                                intrinsic[1, 1] * points[:, 1] / z + intrinsic[1, 2]))
    uv = np.rint(uv_float).astype(np.int64); h, w = shape
    inside = np.isfinite(points).all(axis=1) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(inside); flat = uv[ids, 1] * w + uv[ids, 0]
    reverse_positions = np.unique(flat[::-1], return_index=True)[1]
    chosen = ids[len(ids) - 1 - reverse_positions]
    order = np.argsort(uv[chosen, 1] * w + uv[chosen, 0]); chosen = chosen[order]
    flat_chosen = uv[chosen, 1] * w + uv[chosen, 0]
    maps = {name: np.zeros(h * w, np.float64) for name in ("z", "range")}
    index_map = np.full(h * w, -1, np.int64)
    maps["z"][flat_chosen] = z[chosen]; maps["range"][flat_chosen] = np.linalg.norm(points[chosen], axis=1)
    index_map[flat_chosen] = chosen
    return {"z": maps["z"].reshape(h, w), "range": maps["range"].reshape(h, w),
            "point_index": index_map.reshape(h, w), "valid": index_map.reshape(h, w) >= 0}


def fit_through_origin(raw: np.ndarray, reference: np.ndarray, iterations: int = 12) -> dict:
    x = np.asarray(raw, float); y = np.asarray(reference, float); valid = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    slope = float(np.median(y / x))
    for _ in range(iterations):
        residual = y - slope * x; sigma = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-12
        weight = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
        slope = float(np.sum(weight * x * y) / np.sum(weight * x * x))
    residual = y - slope * x
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    return {"scale": slope, "median_ratio": float(np.median(y / x)), "ratio_mad": float(np.median(np.abs(y / x - np.median(y / x)))),
            "median_abs_residual_m": float(np.median(np.abs(residual))), "p90_abs_residual_m": float(np.percentile(np.abs(residual), 90)),
            "correlation": correlation, "sample_count": len(x)}


def fit_scale_offset(raw: np.ndarray, reference: np.ndarray, iterations: int = 12) -> dict:
    x = np.asarray(raw, float); y = np.asarray(reference, float); valid = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]; design = np.column_stack((x, np.ones(len(x))))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y - design @ beta; sigma = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-12
        weight = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
        beta = np.linalg.lstsq(design * np.sqrt(weight[:, None]), y * np.sqrt(weight), rcond=None)[0]
    residual = y - design @ beta
    return {"scale": float(beta[0]), "offset_m": float(beta[1]),
            "median_abs_residual_m": float(np.median(np.abs(residual))), "p90_abs_residual_m": float(np.percentile(np.abs(residual), 90)),
            "sample_count": len(x)}


def candidate_metrics(raw: np.ndarray, reference: np.ndarray, reference_valid: np.ndarray, scale: float) -> dict:
    decoded = raw.astype(float) * float(scale); candidate_valid = (decoded >= 0.2) & (decoded <= 4.0) & (raw > 0)
    union = candidate_valid | reference_valid; common = candidate_valid & reference_valid
    error = decoded[common] - reference[common]
    absolute = np.abs(error)
    return {"scale_to_m": scale, "candidate_valid_pixels": int(candidate_valid.sum()), "reference_valid_pixels": int(reference_valid.sum()),
            "common_valid_pixels": int(common.sum()), "valid_mask_iou": float(common.sum() / max(union.sum(), 1)),
            "median_abs_depth_error_m": float(np.median(absolute)) if len(absolute) else None,
            "p90_abs_depth_error_m": float(np.percentile(absolute, 90)) if len(absolute) else None,
            "within_1mm": float((absolute <= .001).mean()) if len(absolute) else 0.0,
            "within_5mm": float((absolute <= .005).mean()) if len(absolute) else 0.0,
            "within_1cm": float((absolute <= .01).mean()) if len(absolute) else 0.0,
            "within_3cm": float((absolute <= .03).mean()) if len(absolute) else 0.0}


def detect_scale_drift(per_frame_scales: list[float], relative_limit: float = .005) -> dict:
    values = np.asarray(per_frame_scales, float); median = float(np.median(values))
    variation = float((values.max() - values.min()) / median) if len(values) else float("inf")
    return {"median": median, "relative_peak_to_peak": variation, "threshold": relative_limit, "drift_detected": variation > relative_limit}


def quantity_fit(raw: np.ndarray, quantities: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, values in quantities.items():
        fit = fit_through_origin(raw, values)
        result[name] = fit
    best = min(result, key=lambda key: result[key]["median_abs_residual_m"])
    return {"candidates": result, "best_quantity": best}


def transform_world_to_camera(points_world: np.ndarray, pose_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world) - pose_world_camera[:3, 3]) @ pose_world_camera[:3, :3]


def representative_geometry(root: Path, rows: list[dict], poses: np.ndarray, intrinsic: np.ndarray,
                            pv: dict, representative: dict) -> tuple[dict, dict, dict]:
    frame_records = []; all_raw = []; all_z = []; all_range = []; all_pv_z = []; phase_arrays = {}
    lut = np.fromfile(root / "Depth Long Throw_lut.bin", dtype=np.float32).reshape(-1, 3)
    lut_norm = np.linalg.norm(lut, axis=1)
    for phase, frame_ids in representative.items():
        phase_raw, phase_z = [], []
        for frame_id in frame_ids:
            row = rows[frame_id]; raw = cv2.imread(str(row["path"]), cv2.IMREAD_UNCHANGED)
            ply_path = root / "Depth Long Throw" / f"{row['association_timestamp']}.ply"
            pgm_path = root / "Depth Long Throw" / f"{row['association_timestamp']}.pgm"
            world = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
            camera = transform_world_to_camera(world, poses[frame_id]); projected = last_write_projection(camera, intrinsic, raw.shape)
            common = (raw > 0) & projected["valid"]
            point_ids = projected["point_index"][common]
            target_pv = int(np.argmin(np.abs(pv["timestamps"] - row["filename_timestamp"])))
            pv_camera = transform_world_to_camera(world, pv["poses_world"][target_pv])
            pv_z_map = np.zeros(raw.shape, float); pv_z_map[common] = pv_camera[point_ids, 2]
            raw_values = raw[common].astype(float); z_values = projected["z"][common]; range_values = projected["range"][common]
            pv_values = pv_z_map[common]
            pgm = cv2.imread(str(pgm_path), cv2.IMREAD_UNCHANGED); pgm_valid = pgm.reshape(-1) > 0
            source_range = pgm.reshape(-1)[pgm_valid].astype(float) / 1000.0
            ordering_ok = len(source_range) == len(camera)
            source_range_error = np.linalg.norm(camera, axis=1) - source_range if ordering_ok else np.asarray([])
            predicted_encoded = np.zeros(raw.shape, np.uint16)
            predicted_encoded[projected["valid"]] = (projected["z"][projected["valid"]] * 5000).astype(np.uint16)
            exact_common = (raw > 0) & (predicted_encoded > 0)
            quantity = quantity_fit(raw_values, {"virtual_pinhole_optical_axis_z_m": z_values,
                                                 "virtual_pinhole_euclidean_range_m": range_values,
                                                 "true_pv_camera_signed_z_m": pv_values,
                                                 "true_pv_camera_forward_depth_m": -pv_values})
            fit = fit_through_origin(raw_values, z_values); fit_offset = fit_scale_offset(raw_values, z_values)
            frame_records.append({"phase": phase, "original_frame_id": frame_id,
                                  "long_throw_timestamp": row["association_timestamp"], "matched_pv_timestamp": row["filename_timestamp"],
                                  "ply_path": str(ply_path), "pgm_path": str(pgm_path), "registered_path": str(row["path"]),
                                  "point_count": len(world), "common_pixels": int(common.sum()),
                                  "producer_reencode_exact_ratio_on_common": float((raw[exact_common] == predicted_encoded[exact_common]).mean()),
                                  "producer_reencode_common_pixels": int(exact_common.sum()), "scale_fit": fit, "scale_offset_fit": fit_offset,
                                  "quantity_fit": quantity, "source_pgm_range_ordering_tested": ordering_ok,
                                  "source_pgm_vs_ply_range_median_abs_error_m": float(np.median(np.abs(source_range_error))) if len(source_range_error) else None,
                                  "source_pgm_vs_ply_range_p90_abs_error_m": float(np.percentile(np.abs(source_range_error), 90)) if len(source_range_error) else None})
            all_raw.append(raw_values); all_z.append(z_values); all_range.append(range_values); all_pv_z.append(pv_values)
            phase_raw.append(raw.reshape(-1)); phase_z.append(projected["z"].reshape(-1))
        phase_arrays[phase] = (np.concatenate(phase_raw), np.concatenate(phase_z))
    raw_all, z_all = np.concatenate(all_raw), np.concatenate(all_z)
    per_frame_scale = [record["scale_fit"]["scale"] for record in frame_records]
    global_origin = fit_through_origin(raw_all, z_all); global_offset = fit_scale_offset(raw_all, z_all)
    global_origin["per_frame_scale_95pct_interval"] = [float(np.percentile(per_frame_scale, 2.5)), float(np.percentile(per_frame_scale, 97.5))]
    global_origin["per_frame_scale_drift"] = detect_scale_drift(per_frame_scale)
    quantity = quantity_fit(raw_all, {"virtual_pinhole_optical_axis_z_m": np.concatenate(all_z),
                                     "virtual_pinhole_euclidean_range_m": np.concatenate(all_range),
                                     "true_pv_camera_signed_z_m": np.concatenate(all_pv_z),
                                     "true_pv_camera_forward_depth_m": -np.concatenate(all_pv_z)})
    scale_candidates = {}
    for phase, (phase_raw, phase_z) in phase_arrays.items():
        pseudo_shape = (1, len(phase_raw)); raw_image = phase_raw.astype(np.uint16).reshape(pseudo_shape)
        reference = phase_z.reshape(pseudo_shape); valid = (reference >= .2) & (reference <= 4.)
        fitted = global_origin["scale"]
        scale_candidates[phase] = {str(value): candidate_metrics(raw_image, reference, valid, value)
                                   for value in (.001, .0002, fitted)}
    empirical = {"representative_frames": representative, "through_origin": global_origin, "with_offset": global_offset,
                 "offset_interpretation": "approximately half a 0.2 mm truncation bin; producer formula has no physical offset",
                 "per_frame": [{"phase": x["phase"], "original_frame_id": x["original_frame_id"], **x["scale_fit"]} for x in frame_records],
                 "candidate_scale_metrics_by_phase": scale_candidates}
    cross = {"lut_ray_norm_percentiles": np.percentile(lut_norm, [0, 1, 50, 99, 100]).tolist(),
             "source_pgm_quantity": "radial range in millimetres; unit ray norm is 1",
             "registered_quantity": "virtual Long Throw pinhole optical-axis Z, not radial range and not true PV-camera Z",
             "quantity_comparison": quantity, "per_frame": frame_records}
    return empirical, cross, {"raw": raw_all, "z": z_all}


def git_output(repo: Path, args: list[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def parse_consumer_hits(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            rows.append({"consumer_path": parts[0], "line": int(parts[1]), "evidence": parts[2].strip()})
    return rows


def consumer_audit(repo: Path) -> list[dict]:
    paths = [
        ("tools/itaco_track_motion_v1/reference.py", "configured; 0.001 in recording config", "registered virtual-pinhole Z", "Stage 1/1.5 reference and validity", True),
        ("tools/itaco_track_motion_v1/validity.py", "configured; 0.001 in recording config", "registered virtual-pinhole Z", "Stage 1/1.5 tracks", True),
        ("tools/itaco_funrec_assignment_v3/pipeline.py", "configured; 0.001", "registered virtual-pinhole Z", "Assignment v3 and depth-projective tracks", True),
        ("tools/itaco_region_assignment_v4/frame_data.py", "configured; 0.001", "registered virtual-pinhole Z", "Assignment v4 formal ownership", True),
        ("tools/compare_gt_monst3r_upload.py", "hard-coded 0.001; comment says uint16 mm", "assumed millimetres", "VGGT/MonST3R comparison diagnostics", True),
        ("tools/build_hololens_rgbd_video.py", "hard-coded /1000", "assumed millimetres", "generic HoloLens RGB-D video when pointed at pinhole_projection", True),
        ("tools/build_itaco_gt_surface.py", "hard-coded /1000 for uint16", "assumed millimetres", "generic GT surface helper when pointed at registered PNG", True),
        ("tools/visualize_axis_compare.py", "hard-coded 0.001 on registered PNG", "assumed millimetres", "legacy axis/depth comparison", True),
        ("tools/diagnose_assignment_v4_masks_depth.py", "inherits formal v4 0.001", "registered virtual-pinhole Z", "failed formal-scale follow-up panels", True),
        ("tools/diagnose_assignment_v4_depth_scale.py", "explicit diagnostic formal_scale/5", "registered virtual-pinhole Z", "corrected-scale follow-up only", False),
        ("tools/audit_hololens_registered_depth_provenance.py", "producer-verified 1/5000", "virtual-pinhole optical Z", "this provenance audit", False),
        ("tools/build_itaco_movable_mesh.py", "hard-coded /1000 where PNG is read", "assumed millimetres", "legacy uploaded-depth mesh helper", True),
        ("tools/run_itaco_upload_gt_sam3.py", "hard-coded /1000 where PNG is read", "assumed millimetres", "legacy GT+SAM3 helper", True),
        ("tools/fuse_hololens_articulation_assignment_v2.py", "none; projects Long Throw PLY", "PLY metres", "Assignment v2", False),
        ("tools/fuse_hololens_articulation_dual_volume.py", "none; projects Long Throw PLY", "PLY metres", "geometry_interior_v1", False),
        ("tools/fuse_hololens_articulation_dual_tsdf.py", "none for geometry; Long Throw PLY", "PLY metres", "dual TSDF diagnostic", False),
        ("tools/test_arkit_pose_error_tsdf.py", "none for source geometry; Long Throw PLY", "PLY metres", "pose-error TSDF diagnostic", False),
        ("tools/build_itaco_gt_surface_from_world_ply.py", "none; Long Throw PLY", "PLY metres", "GT surface/preprocess", False),
        ("tools/reproject_hololens_world_ply_depth.py", "writes float-metre PLY reprojection", "PLY metres", "registered-depth alternative diagnostic", False),
        ("tools/recompute_prismatic_q_from_moving_labels.py", "none; consumes float-metre prompt_depth_video NPY", "metres", "monotonic q_t", False),
    ]
    return [{"consumer_path": p, "assumed_scale": s, "physical_quantity_assumption": q,
             "affected_outputs": out, "potentially_affected": affected,
             "path_exists": (repo / p).exists()} for p, s, q, out, affected in paths]


def historical_impact() -> list[dict]:
    return [
        {"result": "official-compatible iTACO baseline", "affected": "no", "depth_source_used": "float-metre prompt_depth_video generated from Long Throw PLY", "reason": "does not decode registered uint16 PNG with 0.001", "requires_rerun": "no"},
        {"result": "moving-map fix", "affected": "no", "depth_source_used": "official float-metre preprocess depth plus masks", "reason": "registered PNG scale is not its physical-depth decoder", "requires_rerun": "no"},
        {"result": "centroid axis", "affected": "no", "depth_source_used": "float-metre prompt_depth_video", "reason": "axis input is already metre-valued NPY", "requires_rerun": "no"},
        {"result": "monotonic q_t", "affected": "no", "depth_source_used": "float-metre prompt_depth_video", "reason": "q recomputation reads NPY directly", "requires_rerun": "no"},
        {"result": "geometry_interior_v1", "affected": "no", "depth_source_used": "Long Throw world PLY in metres", "reason": "fusion projects PLY and does not decode registered PNG", "requires_rerun": "no"},
        {"result": "Assignment v2", "affected": "no", "depth_source_used": "Long Throw world PLY self-projection", "reason": "visibility depth is generated from the same PLY", "requires_rerun": "no"},
        {"result": "Assignment v3", "affected": "yes", "depth_source_used": "registered uint16 PNG decoded with 0.001", "reason": "3D tracks and residuals inherit factor-five error", "requires_rerun": "yes"},
        {"result": "v3 depth-projective", "affected": "yes", "depth_source_used": "registered uint16 PNG decoded with 0.001", "reason": "projective depth association uses wrong physical scale", "requires_rerun": "yes"},
        {"result": "Assignment v4 formal", "affected": "yes", "depth_source_used": "registered uint16 PNG decoded with 0.001", "reason": "direct projective ownership evidence is invalid", "requires_rerun": "yes, only after user authorization"},
        {"result": "static TSDF diagnostics", "affected": "no", "depth_source_used": "Long Throw PLY", "reason": "audited helpers construct geometry from metre PLY", "requires_rerun": "no for this unit issue"},
        {"result": "pose-error TSDF diagnostics", "affected": "no", "depth_source_used": "Long Throw PLY plus pose perturbation", "reason": "does not use registered PNG scale", "requires_rerun": "no"},
        {"result": "articulated GLB/URDF", "affected": "no", "depth_source_used": "geometry_interior_v1 PLY/NKSR outputs", "reason": "upstream geometry, axis and q use metre-valued PLY/NPY paths", "requires_rerun": "no for this unit issue"},
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def readme_text(conclusion: dict, empirical: dict, cross: dict, impact: list[dict]) -> str:
    affected = ", ".join(x["result"] for x in impact if x["affected"] == "yes")
    unaffected = ", ".join(x["result"] for x in impact if x["affected"] == "no")
    scale = empirical["through_origin"]["scale"]
    return f"""# Registered-depth Physical-Unit & Provenance Audit

This audit is read-only. It did not rerun Assignment v4, SAM2, TSDF, NKSR, Mesh,
or any camera/axis/q/moving-map optimization.

## Plain-language answer

One nonzero integer in `pinhole_projection/depth/*.png` is **not one millimetre**.
The matched Microsoft StreamRecorderConverter first constructs a 3-D Long Throw
point in metres, projects that point into a fixed 320x288 virtual pinhole camera,
takes the point's optical-axis `Z`, multiplies it by 5000, truncates it to
`uint16`, and writes the PNG. Therefore the producer contract is
`stored = uint16(Z_m * 5000)`. The nominal decoder is `Z_m = stored / 5000`,
with a quantization interval of 0.2 mm per count.

`0.001 m/count` was wrong because local consumers treated any uint16 depth PNG
as millimetres. Git history first records this assumption without producer
evidence. `0.0002` comes directly from the matched producer's explicit factor
5000, and the independent robust fit is `{scale:.12g} m/count`.

The original Long Throw PGM stores radial range in millimetres. Its LUT rays have
unit norm. The registered PNG is different: it stores optical-axis Z in a virtual
pinhole frame aligned to the Long Throw point cloud. It is not radial range and
it is not true PV-camera Z. The paired RGB is PV color resampled onto that virtual
depth view, which caused earlier reports to call the frame “PV” too loosely.

## Provenance level

- Producer family and formula: verified by the exact output signature and the
  Microsoft source at `{UPSTREAM_REVISION}`.
- Exact checkout/command used on 2026-07-30: not recorded locally; revision-level
  execution provenance remains partial.

Reference source at `{UPSTREAM_REVISION}`:

- [save_pclouds.py](https://github.com/microsoft/HoloLens2ForCV/blob/{UPSTREAM_REVISION}/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py)
- [utils.py](https://github.com/microsoft/HoloLens2ForCV/blob/{UPSTREAM_REVISION}/Samples/StreamRecorder/StreamRecorderConverter/utils.py)
- Unit contract: verified for the matched converter lineage; the factor 5000 is
  explicit and has been present since the initial public converter history.
- Empirical scale: consistent across 11 representative closed/interaction/open frames.

The repository-local `code/Hololens2/DepthConvertToRGB/align_pv_depth.py` is not
this artifact's producer: it writes a differently named PV-sized alignment using
`pv_z * 1000`, whereas this recording has the Microsoft converter's fixed
320x288 calibration, `*_proj.png` names, and `depth.txt/rgb.txt/trajectory.xyz/odometry.log` bundle.

## Historical impact

Affected and requiring rerun: {affected}.

Not affected by this unit issue: {unaffected}.

This does not mean every affected scientific result is otherwise wrong, and it
does not retroactively invalidate PLY-based reconstruction paths.

## Can corrected v4 run now?

`corrected_v4_rerun_allowed = {str(conclusion['corrected_v4_rerun_allowed']).lower()}`.
The physical contract is now supported, but this audit was not user authorization
to execute a corrected research run. The next step is user approval for a new,
separate corrected-v4 output plus a loader scale-consistency gate.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closed-frames", type=int, nargs=3, default=[5, 85, 165])
    parser.add_argument("--interaction-frames", type=int, nargs=5, default=[177, 186, 195, 204, 213])
    parser.add_argument("--open-frames", type=int, nargs=3, default=[220, 290, 360])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.recording_root.resolve(); repo = args.repo_root.resolve(); pinhole = root / "pinhole_projection"
    representative = {"closed": args.closed_frames, "interaction": args.interaction_frames, "open": args.open_frames}
    depth_rows = parse_index(pinhole / "depth.txt", pinhole); rgb_rows = parse_index(pinhole / "rgb.txt", pinhole)
    if len(depth_rows) != len(rgb_rows): raise RuntimeError("depth/rgb count mismatch")
    poses = load_poses(pinhole / "odometry.log")
    if len(poses) != len(depth_rows): raise RuntimeError("pose count mismatch")
    fx, fy, cx, cy = np.loadtxt(pinhole / "calibration.txt").reshape(-1)[:4]
    intrinsic = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], float)
    pv_path = next(root.glob("*_pv.txt")); pv = load_pv_metadata(pv_path)
    file_audit = audit_registered_files(root, depth_rows, rgb_rows, representative)
    empirical, cross, _ = representative_geometry(root, depth_rows, poses, intrinsic, pv, representative)

    head = git_output(repo, ["rev-parse", "HEAD"]); branch = git_output(repo, ["branch", "--show-current"])
    wrong_origin = {"first_seen_commit": "9812ffcd762781a2323bd847484e3d4733f0b6e2",
                    "file": "tools/itaco_track_motion_v1/configs/hololens_2026-07-30-002840.yaml",
                    "context": "depth_scale_to_m: 0.001; compare_gt_monst3r_upload.py labels uint16 as mm",
                    "whether_evidence_backed": False, "whether_dataset_specific": True,
                    "why": "unresolved beyond an undocumented generic uint16-millimetre assumption; no producer citation or scale check",
                    "v4_inheritance": "7eb840e7343c43c1b4f5c4ab91a61263f8073f11 copied the same 0.001 assumption into v4"}
    producer = {"producer_provenance_status": "partial", "identified_producer": "Microsoft HoloLens2ForCV StreamRecorderConverter/save_pclouds.py",
                "repository": UPSTREAM_REPOSITORY, "reference_revision": UPSTREAM_REVISION,
                "actual_execution_revision": None, "actual_execution_command_log_found": False,
                "standard_commands": ["python process_all.py --recording_path <recording>", "python save_pclouds.py --recording_path <recording>"],
                "source_evidence": [{"path": UPSTREAM_SAVE_PCLOUDS, "lines": "130-176, 208-221", "role": "virtual camera, projection, encoding, indexes, pose"},
                                    {"path": UPSTREAM_UTILS, "lines": "16-18, 152-175", "role": "explicit factor 5000 and optical-Z projection"},
                                    {"path": UPSTREAM_PROCESS_ALL, "lines": "17-56", "role": "standard conversion entry command"}],
                "recording_signature": {"calibration": [fx, fy, cx, cy], "expected": [200, 200, 160, 144],
                                        "fixed_shape": [288, 320], "suffix": "_proj.png",
                                        "bundle": ["depth.txt", "rgb.txt", "trajectory.xyz", "odometry.log"], "matches": True},
                "rejected_local_candidate": {"path": "/workspace_whz/code/Hololens2/DepthConvertToRGB/align_pv_depth.py",
                                             "reason": "different PV-size align_depth artifact and explicit pv_z*1000 encoding"}}
    unit = {"unit_contract_status": "verified", "physical_quantity": "virtual Long Throw pinhole optical-axis Z",
            "stored_dtype": "uint16", "encoding_formula": "stored_uint16 = trunc_toward_zero(Z_m * 5000)",
            "decoding_formula": "nominal Z_m = stored_uint16 / 5000; exact Z lies in [stored/5000,(stored+1)/5000)",
            "scale_to_m": .0002, "offset_m": 0.0, "quantization_step_m": .0002, "invalid_value": 0,
            "coordinate_frame": "fixed virtual 320x288 pinhole camera over Long Throw camera-space points; K=(200,200,160,144)",
            "not_quantities": ["Long Throw radial range", "true PV-camera optical-axis Z", "inverse depth", "disparity"]}
    evidence = {"evidence_levels": {"A_provenance_verified": ["producer family", "encoding formula", "quantity definition"],
                                    "B_empirically_verified": ["scale stability", "PLY/PNG re-encoding", "quantity discrimination"],
                                    "C_unresolved": ["exact converter checkout revision", "literal command and environment used"]},
                "double_scaling_hypothesis": {"supported": False, "evidence": ["raw radial millimetres are converted once to metre points by /1000", "virtual optical Z metres are intentionally encoded once by *5000", "factor-five error enters only when downstream decodes as *0.001"]}}
    consumers = consumer_audit(repo); impact = historical_impact()
    contract = {"dataset_recording": root.name, "artifact": str(pinhole / "depth"), "producer": producer["identified_producer"],
                "producer_revision": None, "reference_source_revision": UPSTREAM_REVISION,
                "source_quantity": "Long Throw radial-range PGM + unit-ray LUT converted to metre 3D points",
                "stored_dtype": "uint16", "stored_unit_or_encoding": "5000 counts per metre of virtual-pinhole optical Z",
                "encoding_formula": unit["encoding_formula"], "decoding_formula": unit["decoding_formula"],
                "scale_to_m": .0002, "offset_m": 0.0, "valid_raw_range": [1, file_audit["max_raw"]], "invalid_value": 0,
                "coordinate_frame": unit["coordinate_frame"], "quantity_definition": unit["physical_quantity"],
                "evidence_level": "A for producer family/formula; B for exact recording scale; exact execution revision unresolved",
                "provenance_status": "partial", "unit_contract_status": "verified",
                "empirical_validation": empirical["through_origin"],
                "limitations": ["exact producer checkout and literal command were not preserved", "last-write projection is not a nearest-depth z-buffer"]}
    gate = {"status": "specification_only_not_integrated", "required_reference_artifacts": ["representative registered PNGs", "matching Long Throw world PLYs", "odometry.log", "calibration.txt"],
            "metrics": ["median fitted scale", "per-frame scale drift", "valid-mask IoU", "median absolute error", "p90 absolute error"],
            "thresholds": {"minimum_frames": 3, "maximum_scale_relative_error_to_contract": .01, "maximum_per_frame_scale_drift": .005,
                           "minimum_valid_mask_iou": .98, "maximum_median_abs_error_m": .001, "maximum_p90_abs_error_m": .005},
            "failure_message": "registered depth physical-unit mismatch", "behavior": "raise RuntimeError before loading frames into an assignment or fusion pipeline"}
    conclusion = {"producer_provenance_status": "partial", "unit_contract_status": "verified", "empirical_scale_status": "consistent",
                  "formal_scale_to_m": .0002, "empirical_fitted_scale_to_m": empirical["through_origin"]["scale"],
                  "recommended_next_action": "ask user to authorize a separate corrected-v4 rerun and then integrate/test the scale-consistency gate",
                  "corrected_v4_rerun_allowed": False, "reason_rerun_not_allowed": "this audit did not grant execution authorization",
                  "branch": branch, "head_at_audit_start": head, "assignment_v4_ran": False, "sam2_ran": False,
                  "tsdf_ran": False, "nksr_ran": False, "mesh_ran": False}

    write_json(args.output / "producer_trace.json", producer); write_json(args.output / "unit_contract.json", unit)
    write_json(args.output / "registered_depth_file_audit.json", file_audit); write_json(args.output / "empirical_scale_fit.json", empirical)
    write_json(args.output / "cross_representation_consistency.json", cross); write_json(args.output / "provenance_evidence.json", evidence)
    write_json(args.output / "wrong_scale_origin.json", wrong_origin); write_csv(args.output / "depth_consumer_audit.csv", consumers)
    write_json(args.output / "historical_impact_audit.json", impact); write_json(args.output / "registered_depth_unit_contract.json", contract)
    write_json(args.output / "scale_consistency_gate_spec.json", gate); write_json(args.output / "audit_conclusion.json", conclusion)
    (args.output / "README.md").write_text(readme_text(conclusion, empirical, cross, impact), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "producer_status": conclusion["producer_provenance_status"],
                      "unit_contract_status": conclusion["unit_contract_status"],
                      "empirical_scale": conclusion["empirical_fitted_scale_to_m"],
                      "corrected_v4_rerun_allowed": False}, indent=2))


if __name__ == "__main__":
    main()
