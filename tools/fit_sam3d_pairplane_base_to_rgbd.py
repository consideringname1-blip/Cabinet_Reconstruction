import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
sys.path.insert(0, str(ROOT / "tools"))
import fit_sam3d_constrained_plane_to_rgbd as c  # noqa: E402


OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_pairplane_base_fit"
PREV = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_constrained_plane_fit"
RNG = np.random.default_rng(20260709)


def pair_frame(front_normal, side_normal):
    front = c.unit(front_normal)
    side = np.asarray(side_normal, dtype=np.float64).reshape(3)
    side = side - front * float(side @ front)
    side = c.unit(side)
    second = c.unit(np.cross(front, side))
    frame = np.column_stack([side, second, front])
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1.0
    return frame


def choose_target_planes(target_planes, axis):
    enriched = []
    for plane in target_planes:
        n = c.unit(plane["normal"])
        axis_absdot = float(abs(n @ axis))
        enriched.append((plane, axis_absdot, int(plane["sample_inliers"]), float(plane["approx_bbox_area"])))
    front = [x for x in enriched if x[1] > 0.70]
    side = [x for x in enriched if x[1] < 0.35]
    front.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    side.sort(key=lambda x: (x[2], x[3]), reverse=True)
    if not front:
        front = sorted(enriched, key=lambda x: x[1], reverse=True)
    if not side:
        side = sorted(enriched, key=lambda x: x[1])
    return [x[0] for x in front[:2]], [x[0] for x in side[:2]]


def plane_summary(plane, axis):
    n = c.unit(plane["normal"])
    return {
        "index": int(plane["index"]),
        "sample_inliers": int(plane["sample_inliers"]),
        "sample_fraction": float(plane["sample_fraction"]),
        "approx_bbox_area": float(plane["approx_bbox_area"]),
        "absdot_axis": float(abs(n @ axis)),
        "normal": n.tolist(),
        "center": np.asarray(plane["center"], dtype=float).tolist(),
    }


def fit_base_pairplane(k, depth_m, color, axis):
    raw_mesh = c.load_mesh(c.RAW_MESHES["base"])
    target_mesh = c.load_mesh(c.REF_SURFACES["base"])
    mask = c.load_mask(c.MASKS["base"])

    print("[base-pair] sampling raw mesh", flush=True)
    raw_sample, raw_normals = c.sample_mesh_points(raw_mesh, 36000)
    raw_planes = c.extract_planes(
        raw_sample,
        raw_normals,
        max_planes=6,
        threshold=0.014,
        min_inliers=520,
        trials=420,
        normal_sim=0.86,
    )
    target_points = c.voxel_downsample(np.asarray(target_mesh.vertices, dtype=np.float64), voxel=0.0045, max_points=13000)
    target_planes = c.extract_planes(target_points, None, max_planes=5, threshold=0.014, min_inliers=240, trials=460)
    target_fronts, target_sides = choose_target_planes(target_planes, axis)
    print(
        f"[base-pair] raw_planes={len(raw_planes)} target_planes={len(target_planes)} "
        f"fronts={[int(p['index']) for p in target_fronts]} sides={[int(p['index']) for p in target_sides]}",
        flush=True,
    )

    src_fit = raw_sample
    if len(src_fit) > 3600:
        src_fit = src_fit[RNG.choice(len(src_fit), size=3600, replace=False)]
    src_eval = raw_sample
    if len(src_eval) > 11000:
        src_eval = src_eval[RNG.choice(len(src_eval), size=11000, replace=False)]
    tgt_fit = target_points
    if len(tgt_fit) > 2800:
        tgt_fit = tgt_fit[RNG.choice(len(tgt_fit), size=2800, replace=False)]
    tgt_eval = target_points
    if len(tgt_eval) > 7200:
        tgt_eval = tgt_eval[RNG.choice(len(tgt_eval), size=7200, replace=False)]

    candidates = []
    cid = 0
    raw_fronts = raw_planes[:5]
    raw_sides = raw_planes[:5]
    for target_front in target_fronts:
        tf_normal = c.unit(target_front["normal"])
        if float(tf_normal @ axis) < 0:
            tf_normal *= -1.0
        for target_side in target_sides:
            ts_normal = c.unit(target_side["normal"])
            if abs(float(ts_normal @ tf_normal)) > 0.55:
                continue
            target_frame = pair_frame(tf_normal, ts_normal)
            for source_front in raw_fronts:
                for source_side in raw_sides:
                    if source_side["index"] == source_front["index"]:
                        continue
                    sf_normal0 = c.unit(source_front["normal"])
                    ss_normal0 = c.unit(source_side["normal"])
                    if abs(float(sf_normal0 @ ss_normal0)) > 0.55:
                        continue
                    for sf_sign in (-1.0, 1.0):
                        for ss_sign in (-1.0, 1.0):
                            sf_normal = sf_normal0 * sf_sign
                            ss_normal = ss_normal0 * ss_sign
                            try:
                                source_frame = pair_frame(sf_normal, ss_normal)
                            except ValueError:
                                continue
                            rotation = target_frame @ source_frame.T
                            if np.linalg.det(rotation) < 0:
                                continue
                            scale0 = c.initial_scale(src_fit, tgt_fit, rotation)
                            scale, translation, history = c.refine_fixed_rotation(
                                src_fit, tgt_fit, rotation, scale0, iterations=6
                            )
                            transformed_front = rotation @ sf_normal
                            score, metrics = c.score_pose(
                                src_eval,
                                tgt_eval,
                                rotation,
                                scale,
                                translation,
                                k,
                                mask,
                                depth_m,
                                transformed_front,
                                axis,
                            )
                            source_front_center = c.transform_points(
                                np.asarray(source_front["center"]).reshape(1, 3), rotation, scale, translation
                            )[0]
                            source_side_center = c.transform_points(
                                np.asarray(source_side["center"]).reshape(1, 3), rotation, scale, translation
                            )[0]
                            target_front_center = np.asarray(target_front["center"], dtype=np.float64)
                            target_side_center = np.asarray(target_side["center"], dtype=np.float64)
                            front_offset = float((source_front_center - target_front_center) @ tf_normal)
                            side_offset = float((source_side_center - target_side_center) @ c.unit(target_frame[:, 0]))
                            score += min(abs(front_offset) / 0.035, 2.5)
                            score += 0.75 * min(abs(side_offset) / 0.055, 2.0)
                            cid += 1
                            candidates.append(
                                {
                                    "id": cid,
                                    "score": float(score),
                                    "rotation": rotation,
                                    "scale": float(scale),
                                    "translation": translation,
                                    "metrics": metrics,
                                    "target_front_plane_index": int(target_front["index"]),
                                    "target_side_plane_index": int(target_side["index"]),
                                    "source_front_plane_index": int(source_front["index"]),
                                    "source_side_plane_index": int(source_side["index"]),
                                    "source_front_sign": float(sf_sign),
                                    "source_side_sign": float(ss_sign),
                                    "source_front_normal_camera": transformed_front.tolist(),
                                    "target_front_normal_camera": tf_normal.tolist(),
                                    "front_center_normal_offset_m": front_offset,
                                    "side_center_normal_offset_m": side_offset,
                                    "refinement_iterations": int(len(history)),
                                }
                            )
        print(f"[base-pair] target_front={int(target_front['index'])} candidates={len(candidates)}", flush=True)

    candidates.sort(key=lambda item: item["score"])
    base_dir = OUT / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for rank, candidate in enumerate(candidates[:6], start=1):
        mesh = raw_mesh.copy()
        mesh.vertices = c.transform_points(np.asarray(raw_mesh.vertices), candidate["rotation"], candidate["scale"], candidate["translation"])
        glb = base_dir / f"base_pairplane_rank{rank:02d}.glb"
        mesh.export(glb)
        pts, _ = c.sample_mesh_points(mesh, 30000)
        overlay = base_dir / f"base_pairplane_rank{rank:02d}_overlay.png"
        c.draw_overlay(color, k, {"base": pts}, {"base": mask}, overlay)
        exports.append(
            {
                "rank": rank,
                "glb": str(glb),
                "overlay": str(overlay),
                "score": float(candidate["score"]),
                "scale_uniform": float(candidate["scale"]),
                "rotation_matrix_source_to_camera": candidate["rotation"].tolist(),
                "translation_camera_m": np.asarray(candidate["translation"], dtype=float).tolist(),
                "target_front_plane_index": candidate["target_front_plane_index"],
                "target_side_plane_index": candidate["target_side_plane_index"],
                "source_front_plane_index": candidate["source_front_plane_index"],
                "source_side_plane_index": candidate["source_side_plane_index"],
                "source_front_normal_camera": candidate["source_front_normal_camera"],
                "target_front_normal_camera": candidate["target_front_normal_camera"],
                "front_center_normal_offset_m": float(candidate["front_center_normal_offset_m"]),
                "side_center_normal_offset_m": float(candidate["side_center_normal_offset_m"]),
                "metrics": candidate["metrics"],
            }
        )
    best_mesh = c.load_mesh(exports[0]["glb"])
    samples, _ = c.sample_mesh_points(best_mesh, 36000)
    return {
        "mesh": best_mesh,
        "samples": samples,
        "exports": exports,
        "raw_planes": [c.plane_json(p) for p in raw_planes],
        "target_planes": [c.plane_json(p) for p in target_planes],
        "target_front_planes": [plane_summary(p, axis) for p in target_fronts],
        "target_side_planes": [plane_summary(p, axis) for p in target_sides],
        "candidate_count": len(candidates),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = c.read_json(c.META_PATH)
    k = np.asarray(meta["PVCamera"]["k"], dtype=np.float64)
    color = cv2.imread(str(c.COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(c.COLOR_PATH)
    depth = cv2.imread(str(c.DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(c.DEPTH_PATH)
    depth_m = depth.astype(np.float32) / 1000.0
    joint_doc = c.read_json(c.JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    axis = c.unit(joint["axis_camera_closed_to_open"])
    displacement = float(joint["displacement_m"])
    open_to_closed = np.asarray(joint["translation_open_to_closed_camera_m"], dtype=np.float64)

    base = fit_base_pairplane(k, depth_m, color, axis)
    drawer_open = c.load_mesh(PREV / "drawer_open_reference.glb")
    drawer_closed = drawer_open.copy()
    drawer_closed.vertices = np.asarray(drawer_closed.vertices, dtype=np.float64) + open_to_closed
    drawer_samples, _ = c.sample_mesh_points(drawer_open, 30000)

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    base["mesh"].export(base_path)
    drawer_open.export(drawer_open_path)
    drawer_closed.export(drawer_closed_path)
    c.export_scene(OUT / "cabinet_drawer_sam3d_pairplane_base_open.glb", [("base", base["mesh"]), ("drawer_open", drawer_open)])
    c.export_scene(OUT / "cabinet_drawer_sam3d_pairplane_base_closed.glb", [("base", base["mesh"]), ("drawer_closed", drawer_closed)])
    c.export_scene(
        OUT / "cabinet_drawer_sam3d_pairplane_base_open_closed_overlay.glb",
        [("base", base["mesh"]), ("drawer_open", drawer_open), ("drawer_closed", drawer_closed)],
    )
    axis_origin = np.median(np.asarray(drawer_closed.vertices), axis=0)
    c.add_axis_line(axis_origin, axis, max(displacement, 0.18)).export(OUT / "axis_line_closed_to_open.glb")
    urdf_path = OUT / "cabinet_drawer_sam3d_pairplane_base.urdf"
    c.write_urdf(urdf_path, "base.glb", "drawer_closed_link.glb", axis, displacement)

    masks = {name: c.load_mask(path) for name, path in c.MASKS.items()}
    c.draw_overlay(
        color,
        k,
        {"base": base["samples"], "drawer": drawer_samples},
        masks,
        OUT / "combined_qpos1_pairplane_base_overlay.png",
    )

    prev_report = c.read_json(PREV / "sam3d_constrained_plane_fit_report.json")
    base_n = c.unit(base["exports"][0]["source_front_normal_camera"])
    drawer_n = c.unit(prev_report["parts"]["drawer"]["top_exports"][0]["source_plane_normal_camera"])
    report = {
        "method": (
            "Second-pass base fit using two RGB-D planes: a front plane aligned to the validated prismatic "
            "axis plus a non-parallel large RGB-D plane to lock roll. Drawer is reused from the first "
            "constrained plane fit."
        ),
        "frame": "qpos1 OpenCV/PV camera frame",
        "joint": {
            "axis_camera_closed_to_open": axis.tolist(),
            "displacement_m": displacement,
            "translation_open_to_closed_camera_m": open_to_closed.tolist(),
            "source_json": str(c.JOINT_JSON),
        },
        "constraints": {
            "uses_closed_partmask": False,
            "updates_rotation_during_icp": False,
            "base_rotation_source": "front RGB-D plane + side/top RGB-D plane paired with two raw SAM3D planes",
            "drawer_source": str(PREV / "drawer_open_reference.glb"),
        },
        "pair_sanity": {
            "base_front_normal_absdot_axis": float(abs(base_n @ axis)),
            "drawer_front_normal_absdot_axis": float(abs(drawer_n @ axis)),
            "base_drawer_front_normal_absdot": float(abs(base_n @ drawer_n)),
        },
        "parts": {
            "base": {
                "candidate_count": base["candidate_count"],
                "target_front_planes": base["target_front_planes"],
                "target_side_planes": base["target_side_planes"],
                "raw_planes": base["raw_planes"],
                "target_rgbd_planes": base["target_planes"],
                "top_exports": base["exports"],
            },
            "drawer": prev_report["parts"]["drawer"],
        },
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_glb": str(OUT / "cabinet_drawer_sam3d_pairplane_base_open.glb"),
            "closed_glb": str(OUT / "cabinet_drawer_sam3d_pairplane_base_closed.glb"),
            "open_closed_overlay_glb": str(OUT / "cabinet_drawer_sam3d_pairplane_base_open_closed_overlay.glb"),
            "axis_line_glb": str(OUT / "axis_line_closed_to_open.glb"),
            "urdf": str(urdf_path),
            "qpos1_projection_overlay": str(OUT / "combined_qpos1_pairplane_base_overlay.png"),
        },
    }
    report_path = OUT / "sam3d_pairplane_base_fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "open_glb": report["outputs"]["open_glb"],
                "overlay": report["outputs"]["qpos1_projection_overlay"],
                "pair_sanity": report["pair_sanity"],
                "base_score": report["parts"]["base"]["top_exports"][0]["score"],
                "base_metrics": {
                    k2: report["parts"]["base"]["top_exports"][0]["metrics"][k2]
                    for k2 in [
                        "projected_mask_iou",
                        "target_coverage",
                        "leakage",
                        "depth_abs_median_m",
                        "target_to_source_chamfer_median_m",
                    ]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
