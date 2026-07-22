import json
from pathlib import Path

import numpy as np

import build_rgbd_real_pointcloud_animation as glb
import estimate_artgs_storage_joint_0004 as est


OUT = est.OUT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    return est.unit(v)


def transform_vector_world_to_camera(v, state):
    matrix = est.world_to_camera_cv(state)
    return unit(matrix[:3, :3] @ unit(v))


def transform_point_world_to_camera(p, state):
    return est.transform_points(np.asarray(p, dtype=np.float64).reshape(1, 3), est.world_to_camera_cv(state))[0]


def angle_metrics(pred_axis, gt_axis):
    pred = unit(pred_axis)
    gt = unit(gt_axis)
    dot = float(np.clip(np.dot(pred, gt), -1.0, 1.0))
    dot_abs = float(abs(dot))
    return {
        "dot_signed": dot,
        "dot_abs": dot_abs,
        "signed_angle_deg": float(np.degrees(np.arccos(dot))),
        "unsigned_angle_deg": float(np.degrees(np.arccos(dot_abs))),
    }


def point_to_line_distance(point, line_origin, line_dir):
    point = np.asarray(point, dtype=np.float64)
    line_origin = np.asarray(line_origin, dtype=np.float64)
    line_dir = unit(line_dir)
    return float(np.linalg.norm(np.cross(point - line_origin, line_dir)))


def closest_points_between_lines(o1, d1, o2, d2):
    o1 = np.asarray(o1, dtype=np.float64)
    o2 = np.asarray(o2, dtype=np.float64)
    d1 = unit(d1)
    d2 = unit(d2)
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    w0 = o1 - o2
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-4:
        p1 = o1
        p2 = o2 + d2 * np.dot(d2, o1 - o2)
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        p1 = o1 + s * d1
        p2 = o2 + t * d2
    return p1, p2, float(np.linalg.norm(p1 - p2))


def compare_joint(label, joint, gt):
    pred_axis = unit(joint["axis_closed_to_open_world"])
    pred_origin = np.asarray(joint["origin_world"], dtype=np.float64)
    pred_q = float(joint["displacement_m"])
    pred_translation = np.asarray(joint["translation_closed_to_open_world"], dtype=np.float64)

    gt_axis = unit(gt["axis_closed_to_open_world"])
    gt_origin = np.asarray(gt["origin_world"], dtype=np.float64)
    gt_q = float(gt["displacement_m"])
    gt_translation = gt_axis * gt_q

    p_pred, p_gt, line_sep = closest_points_between_lines(pred_origin, pred_axis, gt_origin, gt_axis)

    return {
        "label": label,
        "world": {
            "pred_axis_closed_to_open": pred_axis.astype(float).tolist(),
            "gt_axis_closed_to_open": gt_axis.astype(float).tolist(),
            "pred_origin": pred_origin.astype(float).tolist(),
            "gt_origin": gt_origin.astype(float).tolist(),
            "pred_translation_closed_to_open": pred_translation.astype(float).tolist(),
            "gt_translation_closed_to_open": gt_translation.astype(float).tolist(),
        },
        "end_camera_opencv": {
            "pred_axis_closed_to_open": transform_vector_world_to_camera(pred_axis, "end").astype(float).tolist(),
            "gt_axis_closed_to_open": transform_vector_world_to_camera(gt_axis, "end").astype(float).tolist(),
            "pred_origin": transform_point_world_to_camera(pred_origin, "end").astype(float).tolist(),
            "gt_origin": transform_point_world_to_camera(gt_origin, "end").astype(float).tolist(),
        },
        "start_camera_opencv": {
            "pred_axis_closed_to_open": transform_vector_world_to_camera(pred_axis, "start").astype(float).tolist(),
            "gt_axis_closed_to_open": transform_vector_world_to_camera(gt_axis, "start").astype(float).tolist(),
            "pred_origin": transform_point_world_to_camera(pred_origin, "start").astype(float).tolist(),
            "gt_origin": transform_point_world_to_camera(gt_origin, "start").astype(float).tolist(),
        },
        "errors": {
            "axis_angle": angle_metrics(pred_axis, gt_axis),
            "displacement_error_m": float(pred_q - gt_q),
            "abs_displacement_error_m": float(abs(pred_q - gt_q)),
            "translation_vector_error_m": float(np.linalg.norm(pred_translation - gt_translation)),
            "pred_origin_to_gt_axis_perpendicular_m": point_to_line_distance(pred_origin, gt_origin, gt_axis),
            "gt_origin_to_pred_axis_perpendicular_m": point_to_line_distance(gt_origin, pred_origin, pred_axis),
            "closest_axis_line_separation_m": line_sep,
            "closest_point_on_pred_axis": p_pred.astype(float).tolist(),
            "closest_point_on_gt_axis": p_gt.astype(float).tolist(),
            "origin_note": "For a prismatic joint, axis direction and translation are physically identifiable; the reported origin is a chosen joint-frame convention, so perpendicular line distance is more meaningful than raw origin-coordinate difference.",
        },
    }


def add_axis_line(builder, name, origin, axis, length, color):
    origin = np.asarray(origin, dtype=np.float32)
    axis = unit(axis).astype(np.float32)
    positions = np.stack([origin - axis * length * 0.5, origin + axis * length * 0.5], axis=0)
    colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 4), (2, 1))
    mesh = builder.add_mesh(name, positions, colors, glb.LINES)
    return builder.add_node(name, mesh)


def add_origin_marker(builder, name, origin, color):
    points = np.asarray(origin, dtype=np.float32).reshape(1, 3)
    colors = np.asarray(color, dtype=np.uint8).reshape(1, 4)
    vertices, vertex_colors, indices = glb.make_splats(points, colors, half_size=0.018)
    mesh = builder.add_mesh(name, vertices, vertex_colors, glb.TRIANGLES, indices)
    return builder.add_node(name, mesh)


def build_axis_glb(path, gt, joints):
    builder = glb.GlbBuilder("artgs-joint-reference-axis-comparison")
    line_len = max(0.55, gt["displacement_m"] * 1.7)
    add_axis_line(builder, "gt_axis_closed_to_open_cyan", gt["origin_world"], gt["axis_closed_to_open_world"], line_len, [20, 230, 255, 255])
    add_origin_marker(builder, "gt_origin_cyan", gt["origin_world"], [20, 230, 255, 255])
    colors = {
        "visible_end_base_only": [255, 210, 20, 255],
        "gt_static_mesh_reference": [255, 80, 210, 255],
    }
    for label, joint in joints.items():
        add_axis_line(builder, f"{label}_pred_axis", joint["origin_world"], joint["axis_closed_to_open_world"], line_len, colors[label])
        add_origin_marker(builder, f"{label}_pred_origin", joint["origin_world"], colors[label])
    builder.write(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trans = read_json(est.DATASET / "gt/trans.json")
    gt_axis = unit(trans["trans_info"]["axis"]["d"])
    gt_origin = np.asarray(trans["trans_info"]["axis"]["o"], dtype=np.float64)
    gt_q = float(trans["trans_info"]["translate"]["r"] - trans["trans_info"]["translate"]["l"])
    gt = {
        "axis_closed_to_open_world": gt_axis.astype(float).tolist(),
        "origin_world": gt_origin.astype(float).tolist(),
        "displacement_m": gt_q,
        "translation_closed_to_open_world": (gt_axis * gt_q).astype(float).tolist(),
        "source": str(est.DATASET / "gt/trans.json"),
    }
    joints = {
        "visible_end_base_only": read_json(OUT / "joint_visible_end_base_only.json")["joint"],
        "gt_static_mesh_reference": read_json(OUT / "joint_gt_static_mesh_reference.json")["joint"],
    }
    comparisons = {label: compare_joint(label, joint, gt) for label, joint in joints.items()}
    axis_glb = OUT / "joint_axis_reference_comparison.glb"
    build_axis_glb(axis_glb, gt, joints)
    report = {
        "method": "Compare predicted prismatic joints against the SAPIEN/ArtGS reference joint in gt/trans.json.",
        "dataset": str(est.DATASET),
        "end_view": est.END_VIEW,
        "start_view": est.START_VIEW,
        "gt_reference": gt,
        "predictions": joints,
        "comparisons": comparisons,
        "outputs": {
            "axis_reference_comparison_glb": str(axis_glb),
        },
    }
    report_path = OUT / "joint_reference_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["outputs"], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
