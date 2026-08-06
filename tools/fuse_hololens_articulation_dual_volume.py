#!/usr/bin/env python3
"""Fuse HoloLens closed/interaction/open observations into two canonical volumes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import yaml


def load_odometry(path: Path) -> np.ndarray:
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    return np.stack([np.asarray([[float(v) for v in row.split()] for row in lines[i + 1:i + 5]]) for i in range(0, len(lines), 5)])


def load_phase(path: Path) -> list[int]:
    indices = [int(x) for x in json.loads(path.read_text())["source_indices"]]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"Non-forward phase: {path}")
    return indices


def sampled(count: int, stride: int) -> list[int]:
    result = list(range(0, count, stride))
    if result[-1] != count - 1:
        result.append(count - 1)
    return result


def rgb_support(rgb: np.ndarray, close_radius: int, erosion_radius: int) -> np.ndarray:
    mask = np.any(rgb != 0, axis=2).astype(np.uint8)
    if close_radius:
        k = np.ones((2 * close_radius + 1,) * 2, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if erosion_radius:
        k = np.ones((2 * erosion_radius + 1,) * 2, np.uint8)
        mask = cv2.erode(mask, k)
    return mask.astype(bool)


def project(points: np.ndarray, pose: np.ndarray, k: np.ndarray, shape):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    u = np.rint(k[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + k[1, 2]).astype(int)
    h, w = shape
    inside = (z > 0.2) & (z < 4.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, inside


def depth_world(depth, pose, k, mask):
    yy, xx = np.indices(depth.shape)
    valid = mask & np.isfinite(depth) & (depth > 0.2) & (depth < 4.0)
    z = depth[valid]
    camera = np.stack(((xx[valid] - k[0, 2]) * z / k[0, 0], (yy[valid] - k[1, 2]) * z / k[1, 1], z), axis=1)
    return camera @ pose[:3, :3].T + pose[:3, 3]


def front_basis(points, axis):
    centered = points - np.median(points, axis=0)
    planar = centered - np.outer(centered @ axis, axis)
    values, vectors = np.linalg.eigh(planar.T @ planar / len(planar))
    u = vectors[:, np.argmax(values)]
    u -= axis * np.dot(u, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v / np.linalg.norm(v)


def coordinates(points, origin, basis):
    return (points - origin) @ basis


def cloud(points, colors, normals):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    if len(normals) == len(points):
        result.normals = o3d.utility.Vector3dVector(normals)
    return result


def finalize(parts, voxel, normal_radius, threshold):
    result = cloud(np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts]), np.concatenate([x[2] for x in parts]))
    result = result.voxel_down_sample(voxel)
    result, _ = result.remove_statistical_outlier(
        nb_neighbors=threshold["outlier_nb_neighbors"],
        std_ratio=threshold["outlier_std_ratio"],
    )
    result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=80))
    result.normalize_normals()
    try:
        result.orient_normals_consistent_tangent_plane(30)
    except RuntimeError:
        pass
    return result


def overlay(rgb, u, v, drawer, unknown, static, title):
    labels = np.zeros_like(rgb)
    labels[v[static], u[static]] = (200, 110, 30)
    labels[v[drawer], u[drawer]] = (20, 40, 245)
    labels[v[unknown], u[unknown]] = (20, 220, 220)
    occupied = drawer | unknown | static
    pix = np.zeros(rgb.shape[:2], bool)
    pix[v[occupied], u[occupied]] = True
    result = rgb.copy()
    mixed = cv2.addWeighted(rgb, 0.45, labels, 0.55, 0)
    result[pix] = mixed[pix]
    cv2.rectangle(result, (0, 0), (result.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(result, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    inp, threshold, sampling = cfg["inputs"], cfg["thresholds"], cfg["sampling"]
    raw = Path(inp["raw_root"])
    output = Path(inp["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    phases = {
        "closed": load_phase(Path(inp["closed_manifest"])),
        "interaction": load_phase(Path(inp["interaction_manifest"])),
        "open": load_phase(Path(inp["open_manifest"])),
    }
    poses = load_odometry(raw / "pinhole_projection/odometry.log")
    timestamps = [x.split()[0] for x in (raw / "pinhole_projection/depth.txt").read_text().splitlines() if x.strip()]
    rgb_paths = [raw / "pinhole_projection" / x.split(maxsplit=1)[1].replace("\\", "/") for x in (raw / "pinhole_projection/rgb.txt").read_text().splitlines() if x.strip()]
    fx, fy, cx, cy = np.loadtxt(raw / "pinhole_projection/calibration.txt").reshape(-1)[:4]
    k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    axis = np.load(inp["axis_path"]).astype(float)
    axis /= np.linalg.norm(axis)
    q = np.load(inp["q_path"]).astype(float)
    travel = float(q.max())
    labels = np.load(inp["moving_labels_path"])["a"]
    depths = sorted(Path(inp["interaction_depth_dir"]).glob("*.npy"))
    interaction = phases["interaction"]
    if not (len(interaction) == len(q) == len(labels) == len(depths)):
        raise ValueError("Interaction input counts disagree")
    fixed_poses = np.load(inp["interaction_pose_path"])
    pose_difference = float(np.max(np.abs(fixed_poses - poses[np.asarray(interaction)])))
    if pose_difference > threshold["pose_equality_tolerance"]:
        raise ValueError(f"Fixed interaction poses differ from source odometry: {pose_difference}")

    front_parts = []
    for local, source in enumerate(interaction):
        world = depth_world(np.load(depths[local]), poses[source], k, labels[local] == 2)
        if len(world) < threshold["minimum_seed_points"]:
            raise RuntimeError(f"Too few moving points: source {source}")
        front_parts.append(world - q[local] * axis)
    front = np.concatenate(front_parts)
    origin = np.median(front, axis=0)
    basis_u, basis_v = front_basis(front, axis)
    basis = np.stack((axis, basis_u, basis_v), axis=1)
    fc = coordinates(front, origin, basis)
    lo_q, hi_q = threshold["front_cross_quantiles"]
    margin = threshold["front_cross_margin_m"]
    u_lo, u_hi = np.quantile(fc[:, 1], (lo_q, hi_q)) + (-margin, margin)
    v_lo, v_hi = np.quantile(fc[:, 2], (lo_q, hi_q)) + (-margin, margin)
    drawer_lo = np.asarray([-threshold["drawer_depth_m"], u_lo, v_lo])
    drawer_hi = np.asarray([threshold["drawer_front_margin_m"], u_hi, v_hi])

    hand_dir = Path(inp["corrected_hand_dir"])
    template_candidates, template_records = [], []
    for local, source in enumerate(interaction):
        if q[local] < threshold["template_min_q_fraction"] * travel:
            continue
        pcd = o3d.io.read_point_cloud(str(raw / "Depth Long Throw" / f"{timestamps[source]}.ply"))
        points, colors, normals = map(np.asarray, (pcd.points, pcd.colors, pcd.normals))
        rgb = cv2.imread(str(rgb_paths[source]))
        support = rgb_support(rgb, threshold["rgb_close_radius_pixels"], threshold["rgb_erosion_radius_pixels"])
        u, v, inside = project(points, poses[source], k, support.shape)
        valid = inside.copy()
        valid[inside] &= support[v[inside], u[inside]]
        hand = np.load(hand_dir / f"{source}.npy").squeeze().astype(bool)
        valid[inside] &= ~hand[v[inside], u[inside]]
        proposal_seed = np.zeros(len(points), bool)
        proposal_seed[inside] = labels[local, v[inside], u[inside]] == 2
        canonical = points - q[local] * axis
        cc = coordinates(canonical, origin, basis)
        in_box = np.all((cc >= drawer_lo) & (cc <= drawer_hi), axis=1)
        selected = valid & in_box
        if selected.sum() >= threshold["minimum_template_points_per_frame"]:
            count = int(selected.sum())
            template_candidates.append(
                (
                    canonical[selected], colors[selected], normals[selected],
                    np.full(count, local, np.int32), np.full(count, q[local]),
                    proposal_seed[selected],
                )
            )
        template_records.append({
            "original_frame_id": source, "q_m": float(q[local]),
            "candidate_points": int(selected.sum()),
            "proposal_seed_points": int((selected & proposal_seed).sum()),
        })
    if not template_candidates:
        raise RuntimeError("No drawer body template observations")
    candidate_points = np.concatenate([part[0] for part in template_candidates])
    candidate_colors = np.concatenate([part[1] for part in template_candidates])
    candidate_normals = np.concatenate([part[2] for part in template_candidates])
    candidate_frames = np.concatenate([part[3] for part in template_candidates])
    candidate_q = np.concatenate([part[4] for part in template_candidates])
    candidate_seed = np.concatenate([part[5] for part in template_candidates])
    keys = np.floor(
        candidate_points / threshold["template_consistency_voxel_size_m"]
    ).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    frame_pairs = np.unique(np.stack((inverse, candidate_frames), axis=1), axis=0)
    supporting_frames = np.bincount(frame_pairs[:, 0], minlength=voxel_count)
    q_min = np.full(voxel_count, np.inf)
    q_max = np.full(voxel_count, -np.inf)
    np.minimum.at(q_min, inverse, candidate_q)
    np.maximum.at(q_max, inverse, candidate_q)
    seed_observations = np.bincount(
        inverse, weights=candidate_seed.astype(np.int32), minlength=voxel_count
    )
    keep_voxel = (
        (supporting_frames >= threshold["template_consistency_min_frames"])
        & ((q_max - q_min) >= threshold["template_consistency_min_q_span_fraction"] * travel)
    )
    consistent = keep_voxel[inverse]
    if not consistent.any():
        raise RuntimeError("No canonical-consistent drawer template voxels")
    template = finalize(
        [(candidate_points[consistent], candidate_colors[consistent], candidate_normals[consistent])],
        threshold["template_voxel_size_m"], threshold["normal_radius_m"], threshold,
    )
    template_consistency = {
        "candidate_observations": int(len(candidate_points)),
        "candidate_voxels": voxel_count,
        "accepted_voxels": int(keep_voxel.sum()),
        "accepted_observations": int(consistent.sum()),
        "accepted_voxels_with_proposal_support": int((keep_voxel & (seed_observations > 0)).sum()),
    }
    if len(template.points) < threshold["minimum_template_points_total"]:
        raise RuntimeError("Drawer template is too small")
    template_tree = cKDTree(np.asarray(template.points))
    template_coord = coordinates(np.asarray(template.points), origin, basis)
    template_q_lo, template_q_hi = threshold["template_bounds_quantiles"]
    template_margin = np.asarray(threshold["template_bounds_margin_m"])
    drawer_lo = np.maximum(
        drawer_lo, np.quantile(template_coord, template_q_lo, axis=0) - template_margin
    )
    drawer_hi = np.minimum(
        drawer_hi, np.quantile(template_coord, template_q_hi, axis=0) + template_margin
    )
    o3d.io.write_point_cloud(str(output / "drawer_template_seed_points.ply"), template)

    closed_mask_dir = Path(inp["closed_cabinet_mask_dir"])
    cabinet_frame_bounds = []
    for local in sampled(len(phases["closed"]), sampling["closed_stride"]):
        source = phases["closed"][local]
        pcd = o3d.io.read_point_cloud(str(raw / "Depth Long Throw" / f"{timestamps[source]}.ply"))
        points = np.asarray(pcd.points)
        rgb = cv2.imread(str(rgb_paths[source]))
        support = rgb_support(rgb, threshold["rgb_close_radius_pixels"], threshold["rgb_erosion_radius_pixels"])
        mask = np.load(closed_mask_dir / f"{local:06d}.npy").squeeze().astype(bool)
        u, v, inside = project(points, poses[source], k, support.shape)
        keep = inside.copy()
        keep[inside] &= support[v[inside], u[inside]] & mask[v[inside], u[inside]]
        if keep.sum() >= threshold["minimum_cabinet_points_per_frame"]:
            frame_coord = coordinates(points[keep], origin, basis)
            cabinet_frame_bounds.append(
                (
                    np.quantile(frame_coord, threshold["cabinet_lower_quantile"], axis=0),
                    np.quantile(frame_coord, threshold["cabinet_upper_quantile"], axis=0),
                )
            )
    if len(cabinet_frame_bounds) < threshold["minimum_cabinet_bound_frames"]:
        raise RuntimeError(
            f"Only {len(cabinet_frame_bounds)} frames support robust cabinet bounds; "
            f"need {threshold['minimum_cabinet_bound_frames']}"
        )
    # Robust crop from per-frame bounds prevents one anomalous mask dominating.
    cabinet_lo = np.median([bounds[0] for bounds in cabinet_frame_bounds], axis=0)
    cabinet_hi = np.median([bounds[1] for bounds in cabinet_frame_bounds], axis=0)
    cabinet_lo -= np.asarray(threshold["cabinet_margin_m"])
    cabinet_hi += np.asarray(threshold["cabinet_margin_m"])
    cross_span = drawer_hi[1:] - drawer_lo[1:]
    cross_margin = threshold["cabinet_cross_margin_ratio"] * cross_span
    cabinet_lo[1:] = np.maximum(cabinet_lo[1:], drawer_lo[1:] - cross_margin)
    cabinet_hi[1:] = np.minimum(cabinet_hi[1:], drawer_hi[1:] + cross_margin)


    selected_frames = []
    for phase, sources in phases.items():
        for local in sampled(len(sources), sampling[f"{phase}_stride"]):
            state = 0.0 if phase == "closed" else travel if phase == "open" else float(q[local])
            selected_frames.append((phase, local, sources[local], state))
    visual = set()
    for phase in phases:
        phase_frames = [x for x in selected_frames if x[0] == phase]
        picks = np.linspace(0, len(phase_frames) - 1, sampling["visual_frames_per_phase"]).round().astype(int)
        visual.update((phase_frames[i][0], phase_frames[i][1]) for i in picks)

    static_parts, drawer_parts, records, tiles = [], [], [], []
    for order, (phase, local, source, state) in enumerate(selected_frames):
        pcd = o3d.io.read_point_cloud(str(raw / "Depth Long Throw" / f"{timestamps[source]}.ply"))
        points, colors, normals = map(np.asarray, (pcd.points, pcd.colors, pcd.normals))
        rgb = cv2.imread(str(rgb_paths[source]))
        support = rgb_support(rgb, threshold["rgb_close_radius_pixels"], threshold["rgb_erosion_radius_pixels"])
        u, v, inside = project(points, poses[source], k, support.shape)
        valid = inside.copy()
        valid[inside] &= support[v[inside], u[inside]]
        exact = np.zeros(len(points), bool)
        if phase == "closed":
            cabinet_mask = np.load(closed_mask_dir / f"{local:06d}.npy").squeeze().astype(bool)
            valid[inside] &= cabinet_mask[v[inside], u[inside]]
        elif phase == "interaction":
            hand = np.load(hand_dir / f"{source}.npy").squeeze().astype(bool)
            valid[inside] &= ~hand[v[inside], u[inside]]
            exact[inside] = labels[local, v[inside], u[inside]] == 2
        canonical = points - state * axis
        canonical_coord = coordinates(canonical, origin, basis)
        world_coord = coordinates(points, origin, basis)
        inside_drawer = np.all((canonical_coord >= drawer_lo) & (canonical_coord <= drawer_hi), axis=1)
        inside_cabinet = np.all((world_coord >= cabinet_lo) & (world_coord <= cabinet_hi), axis=1)
        distance = np.full(len(points), np.inf)
        query = valid & inside_drawer
        if query.any():
            distance[query] = template_tree.query(canonical[query], workers=-1)[0]
        drawer = valid & inside_drawer & (
            (distance <= threshold["drawer_growth_distance_m"])
            | (exact & (distance <= threshold["exact_seed_growth_distance_m"]))
        )
        unknown = valid & inside_drawer & (~drawer) & (distance <= threshold["drawer_unknown_distance_m"])
        static = valid & inside_cabinet & (~drawer) & (~unknown)
        if drawer.any():
            drawer_parts.append((canonical[drawer], colors[drawer], normals[drawer]))
        if static.any():
            static_parts.append((points[static], colors[static], normals[static]))
        records.append({"phase": phase, "phase_local_index": local, "original_frame_id": source, "q_m": state, "valid_points": int(valid.sum()), "drawer_points": int(drawer.sum()), "static_points": int(static.sum()), "unknown_points": int(unknown.sum()), "exact_moving_seed_points": int((valid & exact).sum())})
        if (phase, local) in visual:
            tiles.append(overlay(rgb, u, v, drawer, unknown, static, f"{phase} source {source}: red moving / blue static / yellow unknown"))
        print(f"[{order + 1:03d}/{len(selected_frames):03d}] {phase} src={source} q={state:.4f} drawer={drawer.sum()} static={static.sum()} unknown={unknown.sum()}", flush=True)

    drawer_closed = finalize(drawer_parts, threshold["voxel_size_m"], threshold["normal_radius_m"], threshold)
    static = finalize(static_parts, threshold["voxel_size_m"], threshold["normal_radius_m"], threshold)
    drawer_open = o3d.geometry.PointCloud(drawer_closed)
    drawer_open.translate(axis * travel)
    combined = (static + drawer_open).voxel_down_sample(threshold["voxel_size_m"])
    output_paths = {"drawer_closed": output / "drawer_canonical_closed_points.ply", "drawer_open": output / "drawer_canonical_open_points.ply", "static": output / "cabinet_static_points.ply", "combined_open": output / "combined_open_points.ply"}
    o3d.io.write_point_cloud(str(output_paths["drawer_closed"]), drawer_closed)
    o3d.io.write_point_cloud(str(output_paths["drawer_open"]), drawer_open)
    o3d.io.write_point_cloud(str(output_paths["static"]), static)
    o3d.io.write_point_cloud(str(output_paths["combined_open"]), combined)
    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start:start + 3]
        row += [np.zeros_like(tiles[0])] * (3 - len(row))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(output / "tri_state_masks_contact_sheet.jpg"), np.concatenate(rows, axis=0))
    report = {
        "output_kind": "extended_non_official_hololens_articulation_aware_dual_volume",
        "official_baseline_modified": False,
        "camera_policy": "recorded HoloLens T_world_camera fixed; interaction equality verified",
        "moving_proposal_policy": "repaired moving labels are geometric seeds only; no proposal ID is an ownership label",
        "motion": {"joint_type": "prismatic", "opening_axis_world": axis.tolist(), "travel_m": travel, "canonical_state": "closed"},
        "frame_sampling": {"selected_count": len(selected_frames), "records": records},
        "geometry_bounds_drawer_axis_u_v_m": {"origin_world": origin.tolist(), "basis_u_world": basis_u.tolist(), "basis_v_world": basis_v.tolist(), "drawer_lower": drawer_lo.tolist(), "drawer_upper": drawer_hi.tolist(), "cabinet_lower": cabinet_lo.tolist(), "cabinet_upper": cabinet_hi.tolist()},
        "template": {"records": template_records, "consistency": template_consistency, "points": len(template.points)},
        "points": {"static": len(static.points), "drawer_closed": len(drawer_closed.points), "drawer_open": len(drawer_open.points), "combined_open": len(combined.points)},
        "pose_max_abs_difference": pose_difference,
        "outputs": {key: str(path.resolve()) for key, path in output_paths.items()},
    }
    (output / "selected_frames.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "fusion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "points": report["points"], "travel_m": travel}, indent=2))


if __name__ == "__main__":
    main()
