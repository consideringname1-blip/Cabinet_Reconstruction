import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

import build_rgbd_real_pointcloud_animation as anim


ROOT = Path("/workspace_whz")
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_clean_prismatic_qpos1"
FIT_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_to_rgbd_surface_fit/sam3d_to_rgbd_surface_fit_report.json"
OBJECT_POSE_REPORT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_fixed_camera_object_pose_aligned/object_pose_aligned_report.json"
CAPTURE = "20260622_081031_636398Z"
COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
RNG = np.random.default_rng(20260717)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        mesh = trimesh.util.concatenate([g.copy() for g in loaded.geometry.values()])
    mesh.process(validate=False)
    return mesh


def translate_mesh(mesh, translation):
    out = mesh.copy()
    out.vertices = np.asarray(out.vertices, dtype=np.float64) + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return out


def export_scene(path, items):
    scene = trimesh.Scene()
    for name, mesh in items.items():
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    keep = points[:, 2] > 1e-5
    p = points[keep]
    uv[keep, 0] = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]
    uv[keep, 1] = k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
    return uv, keep


def sample_mesh(mesh, count):
    if len(mesh.faces):
        pts, _ = trimesh.sample.sample_surface(mesh, min(count, max(count // 3, len(mesh.faces) * 2)))
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(pts) > count:
        pts = pts[RNG.choice(len(pts), size=count, replace=False)]
    return pts


def draw_overlay(color, k, samples, masks, path):
    img = color.copy()
    colors = {
        "base": (60, 220, 80),
        "drawer": (40, 70, 245),
        "drawer_closed": (20, 165, 255),
    }
    for name, points in samples.items():
        uv, keep = project(points, k)
        h, w = img.shape[:2]
        inb = keep & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 80000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, colors.get(name, (255, 255, 255)), -1, cv2.LINE_AA)
    for name, mask_path in masks.items():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colors.get(name, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def mesh_colors(mesh, fallback_rgba):
    fallback = np.asarray(fallback_rgba, dtype=np.uint8).reshape(1, 4)
    try:
        visual = mesh.visual.to_color()
        colors = np.asarray(visual.vertex_colors, dtype=np.uint8)
    except Exception:
        colors = np.zeros((0, 4), dtype=np.uint8)
    if colors.shape[0] != len(mesh.vertices):
        colors = np.tile(fallback, (len(mesh.vertices), 1))
    if colors.shape[1] == 3:
        colors = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])
    colors[:, 3] = np.maximum(colors[:, 3], 220)
    return colors.astype(np.uint8)


def add_trimesh(builder, name, mesh, fallback_rgba):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
    colors = mesh_colors(mesh, fallback_rgba)
    return builder.add_mesh(name, vertices, colors, anim.TRIANGLES, faces)


def build_animated_glb(path, base, drawer_closed, axis, q):
    builder = anim.GlbBuilder("sam3d-clean-prismatic-qpos1")
    base_mesh = add_trimesh(builder, "base_sam3d_qpos1_fitted", base, [220, 220, 215, 255])
    drawer_mesh = add_trimesh(builder, "drawer_sam3d_closed_link", drawer_closed, [235, 80, 65, 255])
    axis_pos, axis_col = anim.make_axis_line(np.asarray(drawer_closed.vertices), axis * float(q))
    axis_mesh = builder.add_mesh("axis_line_closed_to_open", axis_pos, axis_col, anim.LINES)
    builder.add_node("base_static", base_mesh)
    drawer_node = builder.add_node("drawer_closed_link_animated", drawer_mesh)
    builder.add_node("axis_line_closed_to_open", axis_mesh)
    builder.add_translation_animation(
        "drawer_closed_to_open_clean_prismatic",
        drawer_node,
        np.asarray(axis * float(q), dtype=np.float32),
    )
    builder.write(path)


def write_urdf(path, axis, q):
    axis = unit(axis)
    text = f"""<?xml version="1.0"?>
<robot name="cabinet_drawer_sam3d_clean_prismatic">
  <link name="base_link">
    <visual name="base_visual">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="base.glb" scale="1 1 1"/></geometry>
    </visual>
  </link>
  <link name="drawer_link">
    <visual name="drawer_visual">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="drawer_closed_link.glb" scale="1 1 1"/></geometry>
    </visual>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="base_link"/>
    <child link="drawer_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="{axis[0]:.10f} {axis[1]:.10f} {axis[2]:.10f}"/>
    <limit lower="0" upper="{float(q):.10g}" effort="1" velocity="0.25"/>
  </joint>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def load_clean_joint():
    report = read_json(OBJECT_POSE_REPORT)
    joint = report["joint_axis_kept_fixed"]
    axis = unit(joint["axis_camera_closed_to_open"])
    q = float(joint["best_residual_q_m"])
    return report, axis, q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fit = read_json(FIT_REPORT)
    object_pose_report, axis, q = load_clean_joint()
    open_to_closed = -axis * q
    closed_to_open = axis * q

    base_source = Path(fit["parts"]["base"]["top_exports"][0]["glb"])
    drawer_source = Path(fit["parts"]["drawer"]["top_exports"][0]["glb"])
    base = load_mesh(base_source)
    drawer_open = load_mesh(drawer_source)
    drawer_closed = translate_mesh(drawer_open, open_to_closed)

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    open_scene = OUT / "cabinet_drawer_sam3d_clean_open.glb"
    closed_scene = OUT / "cabinet_drawer_sam3d_clean_closed.glb"
    overlay_scene = OUT / "cabinet_drawer_sam3d_clean_open_closed_overlay.glb"
    animated_glb = OUT / "cabinet_drawer_sam3d_clean_prismatic_animated.glb"
    urdf = OUT / "cabinet_drawer_sam3d_clean_prismatic.urdf"

    base.export(base_path)
    drawer_open.export(drawer_open_path)
    drawer_closed.export(drawer_closed_path)
    export_scene(open_scene, {"base": base, "drawer_open": drawer_open})
    export_scene(closed_scene, {"base": base, "drawer_closed": drawer_closed})
    export_scene(overlay_scene, {"base": base, "drawer_open": drawer_open, "drawer_closed": drawer_closed})
    build_animated_glb(animated_glb, base, drawer_closed, axis, q)
    write_urdf(urdf, axis, q)

    meta = read_json(META_PATH)
    k = np.asarray(meta["PVCamera"]["k"], dtype=np.float64)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)
    base_pts = sample_mesh(base, 70000)
    drawer_pts = sample_mesh(drawer_open, 45000)
    drawer_closed_pts = sample_mesh(drawer_closed, 45000)
    qpos1_open_overlay = OUT / "qpos1_sam3d_clean_open_overlay.png"
    qpos1_open_closed_overlay = OUT / "qpos1_sam3d_clean_open_closed_overlay.png"
    draw_overlay(color, k, {"base": base_pts, "drawer": drawer_pts}, MASKS, qpos1_open_overlay)
    draw_overlay(
        color,
        k,
        {"base": base_pts, "drawer": drawer_pts, "drawer_closed": drawer_closed_pts},
        MASKS,
        qpos1_open_closed_overlay,
    )

    joint = {
        "type": "prismatic",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "axis_camera_closed_to_open": axis.astype(float).tolist(),
        "qpos_closed_m": 0.0,
        "qpos_open_m": q,
        "displacement_m": q,
        "translation_open_to_closed_camera_m": open_to_closed.astype(float).tolist(),
        "translation_closed_to_open_camera_m": closed_to_open.astype(float).tolist(),
        "source": {
            "axis": str(OBJECT_POSE_REPORT),
            "q": str(OBJECT_POSE_REPORT),
            "sam3d_fit": str(FIT_REPORT),
        },
        "sfm_used": False,
        "raw_camera_pose_used": False,
        "closed_partmask_used": False,
    }
    report = {
        "method": (
            "Clean SAM3D articulated export: qpos1 RGB-D visible surfaces ground the complete SAM3D base/drawer mesh "
            "poses; fixed-camera object-pose alignment supplies only the prismatic displacement q. No SfM, raw camera "
            "pose, qpos0 part mask, FoundationPose, or plane-frame enumeration is used."
        ),
        "frame": joint["frame"],
        "camera_axes": joint["camera_axes"],
        "parts": {
            "base": {
                "selected_source": "sam3d_to_rgbd_surface_fit_base_rank01",
                "source_glb": str(base_source),
                "rgbd_fit_metrics": fit["parts"]["base"]["top_exports"][0]["metrics"],
            },
            "drawer": {
                "selected_source": "sam3d_to_rgbd_surface_fit_drawer_rank01",
                "source_glb": str(drawer_source),
                "rgbd_fit_metrics": fit["parts"]["drawer"]["top_exports"][0]["metrics"],
            },
        },
        "joint": joint,
        "q_evidence": {
            "object_pose_aligned_report": str(OBJECT_POSE_REPORT),
            "best_residual_q_m": q,
            "q_sweep_top3": object_pose_report.get("q_sweep_top15", [])[:3],
            "note": "q is diagnostic until the base ICP overlay is visually accepted; it does not come from SfM.",
        },
        "alternate_visual_candidate": {
            "base_rank02_drawer_rank01_open_glb": str(
                ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_to_rgbd_surface_fit/cabinet_drawer_sam3d_fitted_to_rgbd_surface_base02_drawer01.glb"
            ),
            "reason": "Kept as visual fallback because base rank02 was previously worth inspecting, but primary clean export uses rank01 by metric score.",
        },
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_scene_glb": str(open_scene),
            "closed_scene_glb": str(closed_scene),
            "open_closed_overlay_glb": str(overlay_scene),
            "animated_prismatic_glb": str(animated_glb),
            "urdf": str(urdf),
            "joint_json": str(OUT / "joint.json"),
            "qpos1_open_overlay": str(qpos1_open_overlay),
            "qpos1_open_closed_overlay": str(qpos1_open_closed_overlay),
            "report": str(OUT / "sam3d_clean_prismatic_report.json"),
        },
    }
    (OUT / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    (OUT / "sam3d_clean_prismatic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), **report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
