import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


ROOT = Path("/workspace_whz")
sys.path.insert(0, str(ROOT / "tools"))
import build_sam3d_clean_prismatic_qpos1 as clean  # noqa: E402
import fit_sam3d_constrained_plane_to_rgbd as c  # noqa: E402
import fit_sam3d_plane_to_rgbd_plane as pfit  # noqa: E402


OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_base_twoplane_to_rgbd_plane_fit"
PREV = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_plane_to_rgbd_plane_fit"
OBJECT_POSE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_object_pose_aligned/object_pose_aligned_report.json"
CAPTURE = "20260622_081031_636398Z"
COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
BASE_MASK = ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png"
DRAWER_MASK = ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png"
BASE_RAW = ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb"
BASE_REF = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth/base_rgbd_mask_surface.glb"
RNG = np.random.default_rng(20260720)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    return c.unit(v)


def pair_frame(primary_normal, secondary_normal):
    primary = unit(primary_normal)
    secondary = np.asarray(secondary_normal, dtype=np.float64).reshape(3)
    secondary = secondary - primary * float(secondary @ primary)
    if np.linalg.norm(secondary) < 1e-8:
        return None
    secondary = unit(secondary)
    third = unit(np.cross(primary, secondary))
    frame = np.column_stack([secondary, third, primary])
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1.0
    return frame


def sorted_extent(plane):
    ext = np.asarray(plane["extent"], dtype=np.float64).reshape(2)
    return np.sort(np.maximum(ext, 1e-6))


def plane_scale(source_plane, target_plane):
    ratios = sorted_extent(target_plane) / np.maximum(sorted_extent(source_plane), 1e-6)
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
    if len(ratios) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(ratios))))


def initial_translation(rotation, scale, source_planes, target_planes):
    offsets = []
    for source_plane, target_plane in zip(source_planes, target_planes):
        src_center = np.asarray(source_plane["center"], dtype=np.float64)
        tgt_center = np.asarray(target_plane["center"], dtype=np.float64)
        offsets.append(tgt_center - float(scale) * (src_center @ rotation.T))
    return np.mean(np.asarray(offsets), axis=0)


def transform_points(points, rotation, scale, translation):
    return c.transform_points(points, rotation, scale, translation)


def secondary_plane_penalty(candidate, rotation, scale, translation):
    source_plane = candidate["source_secondary_plane"]
    target_plane = candidate["target_secondary_plane"]
    source_sign = float(candidate["source_secondary_sign"])
    target_sign = float(candidate["target_secondary_sign"])
    source_normal = unit(np.asarray(source_plane["normal"], dtype=np.float64) * source_sign)
    target_normal = unit(np.asarray(target_plane["normal"], dtype=np.float64) * target_sign)
    source_normal_cam = unit(rotation @ source_normal)
    normal_absdot = float(abs(source_normal_cam @ target_normal))
    source_center_cam = transform_points(
        np.asarray(source_plane["center"], dtype=np.float64).reshape(1, 3),
        rotation,
        scale,
        translation,
    )[0]
    target_center = np.asarray(target_plane["center"], dtype=np.float64)
    delta = source_center_cam - target_center
    normal_offset = float(abs(delta @ target_normal))
    tangent_offset = float(np.linalg.norm(delta - target_normal * float(delta @ target_normal)))
    extent_ratio = sorted_extent(target_plane) / np.maximum(sorted_extent(source_plane) * float(scale), 1e-6)
    extent_log_err = float(np.linalg.norm(np.log(np.clip(extent_ratio, 1e-4, 1e4))))
    penalty = (
        2.25 * (1.0 - normal_absdot)
        + 1.15 * min(normal_offset / 0.040, 3.0)
        + 0.25 * min(tangent_offset / 0.160, 3.0)
        + 0.25 * min(extent_log_err, 4.0)
    )
    return float(penalty), {
        "secondary_plane_normal_absdot_after_transform": normal_absdot,
        "secondary_plane_center_normal_offset_m": normal_offset,
        "secondary_plane_center_tangent_offset_m": tangent_offset,
        "secondary_plane_extent_log_err": extent_log_err,
        "secondary_plane_penalty": float(penalty),
        "secondary_plane_normal_camera": source_normal_cam.tolist(),
        "target_secondary_plane_normal_camera": target_normal.tolist(),
    }


def score_candidate(src_eval, tgt_eval, candidate, rotation, scale, translation, k, mask, depth_m):
    score, metrics = pfit.score_pose(
        src_eval,
        tgt_eval,
        rotation,
        scale,
        translation,
        k,
        mask,
        depth_m,
        candidate["source_primary_plane"],
        candidate["source_primary_sign"],
        candidate["target_primary_plane"],
    )
    penalty, secondary = secondary_plane_penalty(candidate, rotation, scale, translation)
    score = float(score + penalty)
    metrics = {**metrics, **secondary, "score_before_secondary_plane_penalty": float(metrics.get("score", score))}
    metrics["score"] = score
    return score, metrics


def load_joint():
    report = read_json(OBJECT_POSE_REPORT)
    joint = report["joint_axis_kept_fixed"]
    axis = unit(joint["axis_camera_closed_to_open"])
    q = float(joint["best_residual_q_m"])
    return report, axis, q


def export_scene(path, named_meshes):
    scene = trimesh.Scene()
    for name, mesh in named_meshes:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def fit_base_twoplane(k, depth_m, mask, color_bgr, drawer_plane_normal, previous_base_report):
    raw_mesh = c.load_mesh(BASE_RAW)
    target_mesh = c.load_mesh(BASE_REF)
    raw_planes = previous_base_report["raw_planes"]
    target_planes = previous_base_report["target_rgbd_planes"]
    primary_priors = previous_base_report["top_exports"][:8]
    raw_plane_by_index = {int(plane["index"]): plane for plane in raw_planes}
    target_plane_by_index = {int(plane["index"]): plane for plane in target_planes}

    raw_sample, raw_normals = c.sample_mesh_points(raw_mesh, 56000)
    target_points_all = c.voxel_downsample(
        np.asarray(target_mesh.vertices, dtype=np.float64),
        voxel=0.0045,
        max_points=15000,
    )
    print(
        f"[base-twoplane] reused raw_planes={len(raw_planes)} target_planes={len(target_planes)} "
        f"primary_priors={len(primary_priors)}",
        flush=True,
    )
    if len(raw_planes) < 2 or len(target_planes) < 2:
        raise RuntimeError("Need at least two source and target planes")

    src_fit = raw_sample
    if len(src_fit) > 6200:
        src_fit = src_fit[RNG.choice(len(src_fit), size=6200, replace=False)]
    src_eval = raw_sample
    if len(src_eval) > 17000:
        src_eval = src_eval[RNG.choice(len(src_eval), size=17000, replace=False)]
    src_coarse = src_eval
    if len(src_coarse) > 3200:
        src_coarse = src_coarse[RNG.choice(len(src_coarse), size=3200, replace=False)]

    tgt_fit = target_points_all
    if len(tgt_fit) > 4500:
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), size=4500, replace=False)]
    tgt_eval = target_points_all
    if len(tgt_eval) > 9000:
        tgt_eval = tgt_eval[RNG.choice(len(tgt_eval), size=9000, replace=False)]
    tgt_coarse = tgt_eval
    if len(tgt_coarse) > 2400:
        tgt_coarse = tgt_coarse[RNG.choice(len(tgt_coarse), size=2400, replace=False)]

    source_planes = raw_planes[:6]
    target_plane_candidates = target_planes[:5]
    coarse_candidates = []
    candidate_id = 0
    for prior in primary_priors:
        target_primary = target_plane_by_index.get(int(prior["target_plane_index"]))
        source_primary = raw_plane_by_index.get(int(prior["source_plane_index"]))
        if target_primary is None or source_primary is None:
            continue
        fixed_source_primary_sign = float(prior.get("source_normal_sign", 1.0))
        tpn0 = unit(target_primary["normal"])
        spn0 = unit(source_primary["normal"])
        for target_secondary in target_plane_candidates:
            if int(target_secondary["index"]) == int(target_primary["index"]):
                continue
            tsn0 = unit(target_secondary["normal"])
            if abs(float(tpn0 @ tsn0)) > 0.72:
                continue
            for source_secondary in source_planes:
                if int(source_secondary["index"]) == int(source_primary["index"]):
                    continue
                ssn0 = unit(source_secondary["normal"])
                if abs(float(spn0 @ ssn0)) > 0.72:
                    continue
                for target_primary_sign, target_secondary_sign, source_secondary_sign in itertools.product(
                    (-1.0, 1.0), repeat=3
                ):
                    source_primary_sign = fixed_source_primary_sign
                    target_frame = pair_frame(tpn0 * target_primary_sign, tsn0 * target_secondary_sign)
                    source_frame = pair_frame(spn0 * source_primary_sign, ssn0 * source_secondary_sign)
                    if target_frame is None or source_frame is None:
                        continue
                    rotation = target_frame @ source_frame.T
                    if np.linalg.det(rotation) < 0:
                        continue
                    s1 = plane_scale(source_primary, target_primary)
                    s2 = plane_scale(source_secondary, target_secondary)
                    s3 = math.sqrt(max(s1 * s2, 1e-12))
                    s4 = c.initial_scale(src_fit, tgt_fit, rotation)
                    scale_seeds = sorted({round(float(s), 8) for s in (s1, s2, s3, s4) if np.isfinite(s) and s > 1e-7})
                    for scale0 in scale_seeds:
                        candidate_id += 1
                        candidate = {
                            "id": int(candidate_id),
                            "rotation": rotation,
                            "scale_init": float(scale0),
                            "translation_init": initial_translation(
                                rotation,
                                scale0,
                                [source_primary, source_secondary],
                                [target_primary, target_secondary],
                            ),
                            "source_primary_plane": source_primary,
                            "source_secondary_plane": source_secondary,
                            "target_primary_plane": target_primary,
                            "target_secondary_plane": target_secondary,
                            "source_primary_sign": float(source_primary_sign),
                            "source_secondary_sign": float(source_secondary_sign),
                            "target_primary_sign": float(target_primary_sign),
                            "target_secondary_sign": float(target_secondary_sign),
                        }
                        score, metrics = score_candidate(
                            src_coarse,
                            tgt_coarse,
                            candidate,
                            rotation,
                            scale0,
                            candidate["translation_init"],
                            k,
                            mask,
                            depth_m,
                        )
                        primary_normal_cam = unit(rotation @ (spn0 * source_primary_sign))
                        pair_penalty = 1.35 * (1.0 - abs(float(primary_normal_cam @ drawer_plane_normal)))
                        candidate.update(
                            {
                                "coarse_score": float(score),
                                "coarse_score_with_drawer_pair": float(score + pair_penalty),
                                "coarse_metrics": metrics,
                                "drawer_pair_penalty": float(pair_penalty),
                                "primary_normal_camera": primary_normal_cam,
                            }
                        )
                        coarse_candidates.append(candidate)
    coarse_candidates.sort(key=lambda item: item["coarse_score_with_drawer_pair"])
    keep_n = min(140, len(coarse_candidates))
    print(f"[base-twoplane] coarse_candidates={len(coarse_candidates)} refining={keep_n}", flush=True)

    full_candidates = []
    for item in coarse_candidates[:keep_n]:
        scale, translation, history = pfit.refine_fixed_rotation_from_init(
            src_fit,
            tgt_fit,
            item["rotation"],
            item["scale_init"],
            item["translation_init"],
            iterations=8,
        )
        score, metrics = score_candidate(
            src_eval,
            tgt_eval,
            item,
            item["rotation"],
            scale,
            translation,
            k,
            mask,
            depth_m,
        )
        primary_normal_cam = unit(item["rotation"] @ (unit(item["source_primary_plane"]["normal"]) * item["source_primary_sign"]))
        drawer_pair_penalty = 1.35 * (1.0 - abs(float(primary_normal_cam @ drawer_plane_normal)))
        full_candidates.append(
            {
                **{k2: v for k2, v in item.items() if k2 not in {"coarse_metrics"}},
                "score": float(score),
                "selection_score": float(score + drawer_pair_penalty),
                "drawer_pair_penalty": float(drawer_pair_penalty),
                "metrics": metrics,
                "scale": float(scale),
                "translation": translation,
                "refinement_history": history,
                "primary_normal_camera": primary_normal_cam.tolist(),
            }
        )
    full_candidates.sort(key=lambda item: item["selection_score"])
    if not full_candidates:
        raise RuntimeError("No two-plane base candidates")

    base_dir = OUT / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for rank, candidate in enumerate(full_candidates[:10], start=1):
        mesh = raw_mesh.copy()
        mesh.vertices = transform_points(
            np.asarray(raw_mesh.vertices, dtype=np.float64),
            candidate["rotation"],
            candidate["scale"],
            candidate["translation"],
        )
        glb = base_dir / f"base_twoplane_rank{rank:02d}.glb"
        mesh.export(glb)
        pts, _ = c.sample_mesh_points(mesh, 38000)
        overlay = base_dir / f"base_twoplane_rank{rank:02d}_overlay.png"
        c.draw_overlay(color_bgr, k, {"base": pts}, {"base": mask}, overlay)
        exports.append(
            {
                "rank": int(rank),
                "glb": str(glb),
                "overlay": str(overlay),
                "selection_score": float(candidate["selection_score"]),
                "score": float(candidate["score"]),
                "drawer_pair_penalty": float(candidate["drawer_pair_penalty"]),
                "scale_uniform": float(candidate["scale"]),
                "rotation_matrix_source_to_camera": candidate["rotation"].tolist(),
                "translation_camera_m": np.asarray(candidate["translation"], dtype=float).tolist(),
                "source_primary_plane_index": int(candidate["source_primary_plane"]["index"]),
                "source_secondary_plane_index": int(candidate["source_secondary_plane"]["index"]),
                "target_primary_plane_index": int(candidate["target_primary_plane"]["index"]),
                "target_secondary_plane_index": int(candidate["target_secondary_plane"]["index"]),
                "source_primary_sign": float(candidate["source_primary_sign"]),
                "source_secondary_sign": float(candidate["source_secondary_sign"]),
                "target_primary_sign": float(candidate["target_primary_sign"]),
                "target_secondary_sign": float(candidate["target_secondary_sign"]),
                "primary_normal_camera": candidate["primary_normal_camera"],
                "secondary_normal_camera": candidate["metrics"]["secondary_plane_normal_camera"],
                "metrics": candidate["metrics"],
                "refinement_last": candidate["refinement_history"][-1] if candidate["refinement_history"] else None,
            }
        )
    best_mesh = c.load_mesh(exports[0]["glb"])
    samples, _ = c.sample_mesh_points(best_mesh, 42000)
    return {
        "mesh": best_mesh,
        "samples": samples,
        "exports": exports,
        "raw_planes": [c.plane_json(pl) for pl in raw_planes],
        "target_planes": [c.plane_json(pl) for pl in target_planes],
        "candidate_count": int(len(coarse_candidates)),
        "refined_candidate_count": int(len(full_candidates)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev_report = read_json(PREV / "sam3d_plane_to_rgbd_plane_fit_report.json")
    drawer_rank = int(prev_report["parts"]["drawer"]["selected_rank"])
    drawer_export = prev_report["parts"]["drawer"]["top_exports"][drawer_rank - 1]
    drawer_plane_normal = unit(drawer_export["source_plane_normal_camera"])

    meta = read_json(META_PATH)
    k = np.asarray(meta["PVCamera"]["k"], dtype=np.float64)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)
    depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(DEPTH_PATH)
    depth_m = depth.astype(np.float32) / 1000.0
    base_mask = c.load_mask(BASE_MASK)
    drawer_mask = c.load_mask(DRAWER_MASK)
    masks = {"base": base_mask, "drawer": drawer_mask}

    object_pose_report, axis, q = load_joint()
    translation_closed_to_open = axis * q
    translation_open_to_closed = -translation_closed_to_open

    base = fit_base_twoplane(k, depth_m, base_mask, color, drawer_plane_normal, prev_report["parts"]["base"])
    drawer_open = c.load_mesh(drawer_export["glb"])
    drawer_closed = drawer_open.copy()
    drawer_closed.vertices = np.asarray(drawer_closed.vertices, dtype=np.float64) + translation_open_to_closed

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    open_scene = OUT / "cabinet_drawer_sam3d_base_twoplane_open.glb"
    closed_scene = OUT / "cabinet_drawer_sam3d_base_twoplane_closed.glb"
    open_closed_scene = OUT / "cabinet_drawer_sam3d_base_twoplane_open_closed_overlay.glb"
    animated_glb = OUT / "cabinet_drawer_sam3d_base_twoplane_prismatic_animated.glb"
    urdf = OUT / "cabinet_drawer_sam3d_base_twoplane.urdf"

    base["mesh"].export(base_path)
    drawer_open.export(drawer_open_path)
    drawer_closed.export(drawer_closed_path)
    export_scene(open_scene, [("base", base["mesh"]), ("drawer_open", drawer_open)])
    export_scene(closed_scene, [("base", base["mesh"]), ("drawer_closed", drawer_closed)])
    export_scene(open_closed_scene, [("base", base["mesh"]), ("drawer_open", drawer_open), ("drawer_closed", drawer_closed)])
    clean.build_animated_glb(animated_glb, base["mesh"], drawer_closed, axis, q)
    clean.write_urdf(urdf, axis, q)

    drawer_samples, _ = c.sample_mesh_points(drawer_open, 32000)
    drawer_closed_samples, _ = c.sample_mesh_points(drawer_closed, 32000)
    c.draw_overlay(
        color,
        k,
        {"base": base["samples"], "drawer": drawer_samples},
        masks,
        OUT / "combined_qpos1_base_twoplane_overlay.png",
    )
    c.draw_overlay(
        color,
        k,
        {"base": base["samples"], "drawer": drawer_samples, "drawer_closed": drawer_closed_samples},
        masks,
        OUT / "combined_open_closed_base_twoplane_overlay.png",
    )

    joint = {
        "type": "prismatic",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "axis_camera_closed_to_open": axis.astype(float).tolist(),
        "qpos_closed_m": 0.0,
        "qpos_open_m": float(q),
        "displacement_m": float(q),
        "translation_open_to_closed_camera_m": translation_open_to_closed.astype(float).tolist(),
        "translation_closed_to_open_camera_m": translation_closed_to_open.astype(float).tolist(),
        "source": str(OBJECT_POSE_REPORT),
    }
    base_export = base["exports"][0]
    base_primary = unit(base_export["primary_normal_camera"])
    base_secondary = unit(base_export["secondary_normal_camera"])
    report = {
        "method": (
            "Base two-plane lock after plane-to-plane fit. Drawer is kept from the previous nearly-correct "
            "single-plane result. Base rotation is constrained by two non-parallel RGB-D target planes matched "
            "to two non-parallel raw SAM3D planes, so the in-plane roll ambiguity of a single plane is removed."
        ),
        "frame": joint["frame"],
        "camera_axes": joint["camera_axes"],
        "constraints": {
            "uses_sfm": False,
            "uses_raw_camera_pose": False,
            "uses_closed_partmask": False,
            "uses_foundationpose": False,
            "uses_raw_sam3d_normal_vs_motion_axis": False,
            "uses_target_plane_axis_preselection": False,
            "updates_rotation_during_icp": False,
            "base_rotation_source": "two source SAM3D plane normals matched to two target RGB-D plane normals",
            "drawer_source": str(drawer_export["glb"]),
            "refined_parameters": ["uniform_scale", "camera_frame_translation"],
        },
        "joint": joint,
        "q_evidence": {
            "object_pose_aligned_report": str(OBJECT_POSE_REPORT),
            "best_residual_q_m": float(q),
            "q_sweep_top3": object_pose_report.get("q_sweep_top15", [])[:3],
        },
        "pair_sanity": {
            "base_primary_drawer_plane_absdot": float(abs(base_primary @ drawer_plane_normal)),
            "base_primary_secondary_absdot": float(abs(base_primary @ base_secondary)),
            "base_selected_score": float(base_export["score"]),
            "base_selected_selection_score": float(base_export["selection_score"]),
        },
        "parts": {
            "base": {
                "raw_mesh": str(BASE_RAW),
                "reference_surface": str(BASE_REF),
                "candidate_count": base["candidate_count"],
                "refined_candidate_count": base["refined_candidate_count"],
                "raw_planes": base["raw_planes"],
                "target_rgbd_planes": base["target_planes"],
                "top_exports": base["exports"],
            },
            "drawer": {
                "source": "previous_plane_to_rgbd_plane_fit",
                "previous_report": str(PREV / "sam3d_plane_to_rgbd_plane_fit_report.json"),
                "selected_rank": drawer_rank,
                "selected_export": drawer_export,
            },
        },
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_glb": str(open_scene),
            "closed_glb": str(closed_scene),
            "open_closed_overlay_glb": str(open_closed_scene),
            "animated_prismatic_glb": str(animated_glb),
            "urdf": str(urdf),
            "joint_json": str(OUT / "joint.json"),
            "qpos1_projection_overlay": str(OUT / "combined_qpos1_base_twoplane_overlay.png"),
            "open_closed_projection_overlay": str(OUT / "combined_open_closed_base_twoplane_overlay.png"),
            "report": str(OUT / "sam3d_base_twoplane_to_rgbd_plane_fit_report.json"),
        },
    }
    (OUT / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    report_path = OUT / "sam3d_base_twoplane_to_rgbd_plane_fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "open_glb": str(open_scene),
                "animated_prismatic_glb": str(animated_glb),
                "overlay": str(OUT / "combined_qpos1_base_twoplane_overlay.png"),
                "pair_sanity": report["pair_sanity"],
                "base_score": base_export["score"],
                "base_metrics": base_export["metrics"],
                "drawer_source_rank": drawer_rank,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
