import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path("/workspace_whz")
CAPTURE = "20260622_081031_636398Z"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_parametric_prismatic_proxy"
SURFACE_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_mask_surface_groundtruth"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
SURFACES = {
    "base": SURFACE_DIR / "base_rgbd_mask_surface.glb",
    "drawer": SURFACE_DIR / "drawer_rgbd_mask_surface.glb",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    meshes = [g.copy() for g in loaded.geometry.values()]
    return trimesh.util.concatenate(meshes)


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    keep = points[:, 2] > 1e-5
    p = points[keep]
    uv[keep, 0] = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]
    uv[keep, 1] = k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
    return uv, keep


def drawer_frame(drawer_points, axis):
    n = unit(axis)
    center = np.median(drawer_points, axis=0)
    tangent = drawer_points - center
    tangent = tangent - np.outer(tangent @ n, n)
    _, _, vh = np.linalg.svd(tangent, full_matrices=False)
    u = unit(vh[0] - n * float(vh[0] @ n))
    v = unit(np.cross(n, u))
    if float(v @ np.array([0.0, 1.0, 0.0])) < 0:
        u *= -1.0
        v *= -1.0
    frame = np.column_stack([u, v, n])
    if np.linalg.det(frame) < 0:
        v *= -1.0
        frame = np.column_stack([u, v, n])
    return u, v, n


def coords(points, axes):
    u, v, n = axes
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack([points @ u, points @ v, points @ n])


def world_from_box_coords(vertices, axes):
    u, v, n = axes
    c = np.asarray(vertices, dtype=np.float64)
    return c[:, [0]] * u.reshape(1, 3) + c[:, [1]] * v.reshape(1, 3) + c[:, [2]] * n.reshape(1, 3)


def make_box(umin, umax, vmin, vmax, qmin, qmax, axes, rgba):
    if umax <= umin or vmax <= vmin or qmax <= qmin:
        return None
    local = np.array(
        [
            [umin, vmin, qmin],
            [umax, vmin, qmin],
            [umax, vmax, qmin],
            [umin, vmax, qmin],
            [umin, vmin, qmax],
            [umax, vmin, qmax],
            [umax, vmax, qmax],
            [umin, vmax, qmax],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=world_from_box_coords(local, axes), faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (8, 1))
    return mesh


def combine(meshes):
    meshes = [m for m in meshes if m is not None and len(m.vertices)]
    if not meshes:
        raise ValueError("empty mesh list")
    return trimesh.util.concatenate(meshes)


def extent(vals, lo=2, hi=98, pad=0.0):
    a, b = np.percentile(vals, [lo, hi])
    return float(a - pad), float(b + pad)


def write_urdf(path, axis, displacement):
    axis = unit(axis)
    text = f"""<?xml version="1.0"?>
<robot name="cabinet_drawer_rgbd_parametric_proxy">
  <link name="base_link">
    <visual name="base_visual"><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="base.glb" scale="1 1 1"/></geometry></visual>
  </link>
  <link name="drawer_link">
    <visual name="drawer_visual"><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="drawer_closed_link.glb" scale="1 1 1"/></geometry></visual>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="base_link"/>
    <child link="drawer_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="{axis[0]:.10f} {axis[1]:.10f} {axis[2]:.10f}"/>
    <limit lower="0" upper="{float(displacement):.10g}" effort="1" velocity="0.25"/>
  </joint>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def export_scene(path, items):
    scene = trimesh.Scene()
    for name, mesh in items:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def draw_overlay(color, k, samples, masks, path):
    img = color.copy()
    colors = {"base": (60, 220, 80), "drawer": (40, 70, 245), "drawer_closed": (30, 150, 255)}
    for name, points in samples.items():
        uv, keep = project(points, k)
        h, w = img.shape[:2]
        inb = keep & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pix = np.round(uv[inb]).astype(np.int32)
        step = max(1, len(pix) // 70000)
        for x, y in pix[::step]:
            cv2.circle(img, (int(x), int(y)), 1, colors.get(name, (255, 255, 255)), -1)
    for name, mask_path in masks.items():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colors.get(name, (255, 255, 255)), 2)
    cv2.imwrite(str(path), img)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint_doc = read_json(JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    axis = unit(joint["axis_camera_closed_to_open"])
    displacement = float(joint["displacement_m"])
    open_to_closed = np.asarray(joint["translation_open_to_closed_camera_m"], dtype=np.float64)
    meta = read_json(META_PATH)
    k = np.asarray(meta["PVCamera"]["k"], dtype=np.float64)
    color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(COLOR_PATH)

    base_surface = load_mesh(SURFACES["base"])
    drawer_surface = load_mesh(SURFACES["drawer"])
    base_points = np.asarray(base_surface.vertices, dtype=np.float64)
    drawer_points = np.asarray(drawer_surface.vertices, dtype=np.float64)
    axes = drawer_frame(drawer_points, axis)
    base_c = coords(base_points, axes)
    drawer_c = coords(drawer_points, axes)

    du0, du1 = extent(drawer_c[:, 0], 2, 98, pad=0.010)
    dv0, dv1 = extent(drawer_c[:, 1], 2, 98, pad=0.010)
    dq0_obs, dq1 = extent(drawer_c[:, 2], 5, 95, pad=0.004)
    drawer_depth = dq1 - dq0_obs
    if drawer_depth < max(0.10, 0.35 * displacement):
        dq0 = dq1 - max(0.16, 0.70 * displacement)
    else:
        dq0 = dq0_obs
    drawer_open = make_box(du0, du1, dv0, dv1, dq0, dq1, axes, [235, 85, 70, 230])
    drawer_closed = drawer_open.copy()
    drawer_closed.vertices = np.asarray(drawer_closed.vertices, dtype=np.float64) + open_to_closed.reshape(1, 3)

    closed_drawer_c = coords(np.asarray(drawer_closed.vertices), axes)
    all_u = np.r_[base_c[:, 0], closed_drawer_c[:, 0]]
    all_v = np.r_[base_c[:, 1], closed_drawer_c[:, 1]]
    ou0, ou1 = extent(all_u, 1, 99, pad=0.030)
    ov0, ov1 = extent(all_v, 1, 99, pad=0.030)
    iu0, iu1 = float(closed_drawer_c[:, 0].min() - 0.018), float(closed_drawer_c[:, 0].max() + 0.018)
    iv0, iv1 = float(closed_drawer_c[:, 1].min() - 0.018), float(closed_drawer_c[:, 1].max() + 0.018)
    base_front_q = float(dq1 - displacement)
    bq0_obs, bq1_obs = extent(base_c[:, 2], 4, 96, pad=0.006)
    bq1 = max(base_front_q, bq1_obs)
    bq0 = min(bq0_obs, bq1 - max(0.20, 0.85 * displacement))
    if bq1 - bq0 < 0.16:
        bq0 = bq1 - 0.16

    base_boxes = [
        make_box(ou0, iu0, ov0, ov1, bq0, bq1, axes, [230, 230, 225, 220]),
        make_box(iu1, ou1, ov0, ov1, bq0, bq1, axes, [230, 230, 225, 220]),
        make_box(iu0, iu1, ov0, iv0, bq0, bq1, axes, [230, 230, 225, 220]),
        make_box(iu0, iu1, iv1, ov1, bq0, bq1, axes, [230, 230, 225, 220]),
        make_box(ou0, ou1, ov0, ov1, bq0, bq0 + min(0.030, 0.25 * (bq1 - bq0)), axes, [210, 210, 205, 210]),
    ]
    base_proxy = combine(base_boxes)

    base_path = OUT / "base.glb"
    drawer_open_path = OUT / "drawer_open_reference.glb"
    drawer_closed_path = OUT / "drawer_closed_link.glb"
    base_proxy.export(base_path)
    drawer_open.export(drawer_open_path)
    drawer_closed.export(drawer_closed_path)
    export_scene(OUT / "cabinet_drawer_rgbd_parametric_open.glb", [("base", base_proxy), ("drawer_open", drawer_open)])
    export_scene(OUT / "cabinet_drawer_rgbd_parametric_closed.glb", [("base", base_proxy), ("drawer_closed", drawer_closed)])
    export_scene(
        OUT / "cabinet_drawer_rgbd_parametric_open_closed_overlay.glb",
        [("base", base_proxy), ("drawer_open", drawer_open), ("drawer_closed", drawer_closed)],
    )
    write_urdf(OUT / "cabinet_drawer_rgbd_parametric.urdf", axis, displacement)

    base_sample, _ = trimesh.sample.sample_surface(base_proxy, 60000)
    drawer_open_sample, _ = trimesh.sample.sample_surface(drawer_open, 30000)
    drawer_closed_sample, _ = trimesh.sample.sample_surface(drawer_closed, 30000)
    draw_overlay(
        color,
        k,
        {"base": base_sample, "drawer": drawer_open_sample},
        MASKS,
        OUT / "qpos1_rgbd_parametric_open_overlay.png",
    )
    draw_overlay(
        color,
        k,
        {"base": base_sample, "drawer": drawer_open_sample, "drawer_closed": drawer_closed_sample},
        MASKS,
        OUT / "qpos1_rgbd_parametric_open_closed_overlay.png",
    )

    report = {
        "method": "RGB-D grounded parametric proxy. This intentionally does not use SAM3D pose or raw mesh planes; it extrudes a cabinet frame and drawer box from the validated qpos1 RGB-D mask surfaces and selected prismatic joint.",
        "status": "diagnostic_baseline_not_sam3d_shape_completion",
        "frame": "qpos1 OpenCV/PV camera frame",
        "joint": {
            "axis_camera_closed_to_open": axis.tolist(),
            "displacement_m": displacement,
            "translation_open_to_closed_camera_m": open_to_closed.tolist(),
            "source_json": str(JOINT_JSON),
        },
        "cabinet_frame_axes": {
            "u": axes[0].tolist(),
            "v": axes[1].tolist(),
            "normal_axis": axes[2].tolist(),
        },
        "box_extents_uvq": {
            "drawer_open": {"u": [du0, du1], "v": [dv0, dv1], "q": [dq0, dq1], "observed_q": [dq0_obs, dq1]},
            "base_outer": {"u": [ou0, ou1], "v": [ov0, ov1], "q": [bq0, bq1]},
            "base_inner_opening": {"u": [iu0, iu1], "v": [iv0, iv1], "q_front": base_front_q},
        },
        "outputs": {
            "base_glb": str(base_path),
            "drawer_open_reference_glb": str(drawer_open_path),
            "drawer_closed_link_glb": str(drawer_closed_path),
            "open_glb": str(OUT / "cabinet_drawer_rgbd_parametric_open.glb"),
            "closed_glb": str(OUT / "cabinet_drawer_rgbd_parametric_closed.glb"),
            "open_closed_overlay_glb": str(OUT / "cabinet_drawer_rgbd_parametric_open_closed_overlay.glb"),
            "urdf": str(OUT / "cabinet_drawer_rgbd_parametric.urdf"),
            "qpos1_open_overlay": str(OUT / "qpos1_rgbd_parametric_open_overlay.png"),
            "qpos1_open_closed_overlay": str(OUT / "qpos1_rgbd_parametric_open_closed_overlay.png"),
        },
        "rejected_branches": [
            str(ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_constrained_plane_fit"),
            str(ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/sam3d_pairplane_base_fit"),
        ],
    }
    report_path = OUT / "rgbd_parametric_proxy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "open_glb": report["outputs"]["open_glb"],
                "closed_glb": report["outputs"]["closed_glb"],
                "urdf": report["outputs"]["urdf"],
                "overlay": report["outputs"]["qpos1_open_overlay"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
