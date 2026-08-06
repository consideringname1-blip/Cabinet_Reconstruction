from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw


INPUT_DIR = Path("/workspace_whz/data/upload/2026-07-27-175228/pinhole_projection")
MASK_DIR = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_tight_gt_motion_masks_round2/mid_conf090_res014_vote1_r1/masks")
OUT = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_axis_visualization")
OUT.mkdir(parents=True, exist_ok=True)

ZMAX = 8.5

AXES = {
    "GT_smoke_prismatic": {
        "axis": np.array([0.9052935990511087, -0.13540812326246998, 0.4026265511260107], float),
        "color": (255, 80, 80),
        "source": "coarse_gt_camera smoke",
    },
    "native_tight_prismatic": {
        "axis": np.array([0.9375063634931158, -0.04294623956819298, 0.3453077452315038], float),
        "color": (80, 255, 80),
        "source": "tight mask + native official core",
    },
    "official_tight_prismatic": {
        "axis": np.array([0.10732415246850162, -0.6619809175406564, 0.7417970012806391], float),
        "color": (80, 140, 255),
        "source": "tight mask + official real wrapper",
    },
    "official_tight_revolute": {
        "axis": np.array([0.04947209697775119, 0.8244142483744125, 0.5638205908778755], float),
        "color": (255, 220, 40),
        "source": "official real wrapper revolute candidate",
    },
}
for item in AXES.values():
    item["axis"] = item["axis"] / np.linalg.norm(item["axis"])


def load_list(name: str):
    paths = []
    for line in (INPUT_DIR / name).read_text().splitlines():
        if line.strip():
            rel = line.split()[1].replace(chr(92), "/")
            paths.append(INPUT_DIR / rel)
    return paths


def load_odometry(n=32):
    lines = (INPUT_DIR / "odometry.log").read_text().splitlines()
    transforms = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if i + 4 >= len(lines):
            break
        transforms.append(np.array([[float(x) for x in lines[i + j].split()] for j in range(1, 5)], float))
        i += 5
    return transforms[:n]


RGB_PATHS = load_list("rgb.txt")[:32]
DEPTH_PATHS = load_list("depth.txt")[:32]
TS = load_odometry(32)

T0 = TS[0]
R0 = T0[:3, :3]
t0 = T0[:3, 3]
WORLD_TO_CAM0 = np.eye(4)
WORLD_TO_CAM0[:3, :3] = R0.T
WORLD_TO_CAM0[:3, 3] = -R0.T @ t0

fx, fy, cx, cy = [float(x) for x in (INPUT_DIR / "calibration.txt").read_text().split()[:4]]


def backproject(depth_mm, rgb, transform, mask=None, stride=6, zmax=ZMAX, color_mode="rgb"):
    h, w = depth_mm.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_mm[yy, xx].astype(np.float32) / 1000.0
    valid = (z > 0.2) & (z < zmax)
    if mask is not None:
        valid &= mask[yy, xx]
    if not np.any(valid):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    u = xx[valid].astype(np.float32)
    v = yy[valid].astype(np.float32)
    z = z[valid]
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    pc = (transform[:3, :3] @ pc.T).T + transform[:3, 3]
    cols = rgb[yy[valid], xx[valid], :3].astype(np.uint8)
    if color_mode == "soft":
        cols = np.clip(cols.astype(np.float32) * 0.45 + 135, 0, 255).astype(np.uint8)
    return pc, cols


def build_clouds():
    scene_pts, scene_cols, moving_pts, moving_cols, centroids = [], [], [], [], []
    for idx, (rgb_path, depth_path, transform) in enumerate(zip(RGB_PATHS, DEPTH_PATHS, TS)):
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.array(Image.open(depth_path))
        if idx % 2 == 0:
            p, c = backproject(depth, rgb, transform, None, stride=8, zmax=ZMAX, color_mode="soft")
            scene_pts.append(p)
            scene_cols.append(c)
        mask_path = MASK_DIR / f"{idx:06d}.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            p, _ = backproject(depth, rgb, transform, mask, stride=1, zmax=ZMAX, color_mode="rgb")
            if len(p):
                centroids.append(p.mean(axis=0))
                if len(p) > 900:
                    keep = np.random.default_rng(idx).choice(len(p), 900, replace=False)
                    p = p[keep]
                moving_pts.append(p)
                moving_cols.append(np.tile(np.array([[255, 80, 30]], dtype=np.uint8), (len(p), 1)))
    return (
        np.vstack(scene_pts) if scene_pts else np.zeros((0, 3)),
        np.vstack(scene_cols) if scene_cols else np.zeros((0, 3), dtype=np.uint8),
        np.vstack(moving_pts) if moving_pts else np.zeros((0, 3)),
        np.vstack(moving_cols) if moving_cols else np.zeros((0, 3), dtype=np.uint8),
        np.vstack(centroids) if centroids else np.zeros((0, 3)),
    )


def basis_from_dir(direction):
    d = direction / np.linalg.norm(direction)
    tmp = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


class MeshBuilder:
    def __init__(self):
        self.vertices = []
        self.colors = []
        self.faces = []

    def add_vertex(self, point, color):
        self.vertices.append([float(point[0]), float(point[1]), float(point[2])])
        self.colors.append(color)
        return len(self.vertices) - 1

    def add_axis(self, origin, direction, color, length=0.75, radius=0.012, head_len=0.11, head_radius=0.035, segments=20):
        d = direction / np.linalg.norm(direction)
        u, v = basis_from_dir(d)
        p0 = origin - 0.5 * length * d
        p1 = origin + 0.5 * length * d
        ring0, ring1 = [], []
        for k in range(segments):
            theta = 2 * math.pi * k / segments
            off = radius * (math.cos(theta) * u + math.sin(theta) * v)
            ring0.append(self.add_vertex(p0 + off, color))
            ring1.append(self.add_vertex(p1 + off, color))
        for k in range(segments):
            self.faces.append([ring0[k], ring0[(k + 1) % segments], ring1[(k + 1) % segments], ring1[k]])
        tip = self.add_vertex(p1 + head_len * d, color)
        base = []
        for k in range(segments):
            theta = 2 * math.pi * k / segments
            off = head_radius * (math.cos(theta) * u + math.sin(theta) * v)
            base.append(self.add_vertex(p1 + off, color))
        for k in range(segments):
            self.faces.append([base[k], base[(k + 1) % segments], tip])


def write_ply(path, points, colors, faces):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for face in faces:
            f.write(str(len(face)) + " " + " ".join(str(int(x)) for x in face) + "\n")


def add_axis_meshes(origin, axis_map):
    mesh = MeshBuilder()
    for name, direction in axis_map.items():
        mesh.add_axis(origin, direction, AXES[name]["color"])
    for direction, color in [
        (np.array([1.0, 0, 0]), (200, 0, 0)),
        (np.array([0, 1.0, 0]), (0, 180, 0)),
        (np.array([0, 0, 1.0]), (0, 0, 200)),
    ]:
        mesh.add_axis(origin, direction, color, length=0.28, radius=0.006, head_len=0.04, head_radius=0.016, segments=12)
    return mesh


def make_preview(path, cloud, moving, origin, axis_map, title):
    width, height = 1800, 1200
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    panels = [("XY/top-ish", (0, 1)), ("XZ/front-ish", (0, 2)), ("YZ/side-ish", (1, 2))]
    panel_w = width // 3
    panel_h = height - 210
    pts = cloud
    if len(pts) > 30000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 30000, replace=False)]
    for panel_idx, (panel_name, axes_2d) in enumerate(panels):
        x0 = panel_idx * panel_w
        draw.rectangle([x0 + 12, 60, x0 + panel_w - 12, 60 + panel_h], outline=(180, 180, 180), width=2)
        draw.text((x0 + 24, 68), panel_name, fill=(0, 0, 0))
        combo = np.vstack([
            cloud[:, axes_2d] if len(cloud) else np.zeros((0, 2)),
            moving[:, axes_2d] if len(moving) else np.zeros((0, 2)),
        ])
        axis_pts = []
        for direction in axis_map.values():
            axis_pts += [(origin - 0.47 * direction)[list(axes_2d)], (origin + 0.58 * direction)[list(axes_2d)]]
        if axis_pts:
            combo = np.vstack([combo, np.vstack(axis_pts)])
        mn = np.percentile(combo, 1, axis=0)
        mx = np.percentile(combo, 99, axis=0)
        if axis_pts:
            ap = np.vstack(axis_pts)
            mn = np.minimum(mn, ap.min(axis=0))
            mx = np.maximum(mx, ap.max(axis=0))
        span = np.maximum(mx - mn, 1e-3)
        scale = min((panel_w - 70) / span[0], (panel_h - 80) / span[1])

        def transform(q):
            xy = (q[list(axes_2d)] - mn) * scale
            return (int(x0 + 35 + xy[0]), int(60 + panel_h - 35 - xy[1]))

        step = max(1, len(pts) // 12000)
        for q in pts[::step]:
            x, y = transform(q)
            if x0 + 12 <= x <= x0 + panel_w - 12 and 60 <= y <= 60 + panel_h:
                draw.point((x, y), fill=(155, 155, 155))
        mv = moving
        if len(mv) > 12000:
            mv = mv[np.random.default_rng(1).choice(len(mv), 12000, replace=False)]
        for q in mv:
            x, y = transform(q)
            if x0 + 12 <= x <= x0 + panel_w - 12 and 60 <= y <= 60 + panel_h:
                draw.point((x, y), fill=(255, 70, 20))
        for name, direction in axis_map.items():
            color = AXES[name]["color"]
            p0 = transform(origin - 0.45 * direction)
            p1 = transform(origin + 0.55 * direction)
            draw.line([p0, p1], fill=color, width=5)
            vx, vy = p1[0] - p0[0], p1[1] - p0[1]
            norm = (vx * vx + vy * vy) ** 0.5 or 1
            ux, uy = vx / norm, vy / norm
            px, py = -uy, ux
            head_len, head_w = 18, 8
            draw.polygon(
                [
                    p1,
                    (int(p1[0] - head_len * ux + head_w * px), int(p1[1] - head_len * uy + head_w * py)),
                    (int(p1[0] - head_len * ux - head_w * px), int(p1[1] - head_len * uy - head_w * py)),
                ],
                fill=color,
            )
    draw.text((24, 20), title, fill=(0, 0, 0))
    y = height - 130
    x = 42
    draw.text((x, y - 35), f"orange = tightened moving-mask 3D points; grey = GT-depth scene cloud; zmax={ZMAX}m", fill=(0, 0, 0))
    for k, (name, item) in enumerate(AXES.items()):
        color = item["color"]
        yy = y + 26 * (k // 2)
        xx = x + 620 * (k % 2)
        draw.rectangle([xx, yy, xx + 22, yy + 14], fill=color)
        draw.text((xx + 32, yy - 2), name, fill=(0, 0, 0))
    img.save(path)


def angle(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def main():
    scene_pts, scene_cols, moving_pts, moving_cols, centroids = build_clouds()
    origin = centroids.mean(axis=0)
    disp = centroids[-1] - centroids[0] if len(centroids) > 1 else np.array([1.0, 0, 0])
    if np.linalg.norm(disp) < 1e-6:
        disp = np.array([1.0, 0, 0])

    signed_world_axes = {}
    for name, item in AXES.items():
        direction = item["axis"].copy()
        if np.dot(direction, disp) < 0:
            direction = -direction
        signed_world_axes[name] = direction
    mesh = add_axis_meshes(origin, signed_world_axes)
    offset = len(scene_pts) + len(moving_pts)
    points = np.vstack([scene_pts, moving_pts, np.array(mesh.vertices, float)])
    colors = np.vstack([scene_cols, moving_cols, np.array(mesh.colors, dtype=np.uint8)])
    write_ply(OUT / "axis_compare_gt_world_z8m.ply", points, colors, [[offset + i for i in face] for face in mesh.faces])
    make_preview(OUT / "axis_compare_gt_world_z8m_preview.png", scene_pts, moving_pts, origin, signed_world_axes, "Motion-axis candidates in HoloLens GT world coordinates")

    scene_cam = (WORLD_TO_CAM0[:3, :3] @ scene_pts.T).T + WORLD_TO_CAM0[:3, 3] if len(scene_pts) else scene_pts
    moving_cam = (WORLD_TO_CAM0[:3, :3] @ moving_pts.T).T + WORLD_TO_CAM0[:3, 3] if len(moving_pts) else moving_pts
    origin_cam = WORLD_TO_CAM0[:3, :3] @ origin + WORLD_TO_CAM0[:3, 3]
    disp_cam = WORLD_TO_CAM0[:3, :3] @ disp
    cam_axes = {}
    for name, item in AXES.items():
        direction = item["axis"].copy() if name.startswith("official") else WORLD_TO_CAM0[:3, :3] @ signed_world_axes[name]
        direction = direction / np.linalg.norm(direction)
        if np.dot(direction, disp_cam) < 0:
            direction = -direction
        cam_axes[name] = direction
    mesh = add_axis_meshes(origin_cam, cam_axes)
    offset = len(scene_cam) + len(moving_cam)
    points = np.vstack([scene_cam, moving_cam, np.array(mesh.vertices, float)])
    colors = np.vstack([scene_cols, moving_cols, np.array(mesh.colors, dtype=np.uint8)])
    write_ply(OUT / "axis_compare_first_camera0_z8m.ply", points, colors, [[offset + i for i in face] for face in mesh.faces])
    make_preview(OUT / "axis_compare_first_camera0_z8m_preview.png", scene_cam, moving_cam, origin_cam, cam_axes, "Motion-axis candidates in first-camera coordinates; official axes raw")

    names = list(AXES)
    angle_table = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            angle_table[f"{name_a} vs {name_b}"] = angle(signed_world_axes[name_a], signed_world_axes[name_b])

    legend = {
        "zmax_m": ZMAX,
        "origin_world_dynamic_centroid": origin.tolist(),
        "motion_displacement_world_first_to_last_dynamic_centroid": disp.tolist(),
        "n_scene_points": int(len(scene_pts)),
        "n_moving_points": int(len(moving_pts)),
        "files": {
            "gt_world_ply": str(OUT / "axis_compare_gt_world_z8m.ply"),
            "gt_world_preview": str(OUT / "axis_compare_gt_world_z8m_preview.png"),
            "first_camera0_ply": str(OUT / "axis_compare_first_camera0_z8m.ply"),
            "first_camera0_preview": str(OUT / "axis_compare_first_camera0_z8m_preview.png"),
        },
        "colors_rgb": {name: list(AXES[name]["color"]) for name in AXES},
        "signed_axes_world_raw_comparison": {name: signed_world_axes[name].tolist() for name in AXES},
        "signed_axes_first_camera0_comparison": {name: cam_axes[name].tolist() for name in AXES},
        "angle_table_world_raw_deg_abs_axis_sign_invariant": angle_table,
        "notes": [
            "Prismatic axis sign is ambiguous; arrows were flipped to roughly agree with first-to-last dynamic centroid displacement.",
            "GT_smoke_prismatic and native_tight_prismatic are treated as HoloLens/GT-world directions.",
            "official_tight_prismatic/revolute are drawn raw in GT-world comparison; first_camera0 additionally transforms GT-world axes into first camera frame while keeping official axes raw.",
            "All candidates are drawn through the tightened moving-mask 3D centroid for direction comparison; prismatic joint_pos is not meaningful.",
        ],
    }
    (OUT / "axis_visualization_legend_z8m.json").write_text(json.dumps(legend, indent=2))
    print(json.dumps(legend, indent=2))


if __name__ == "__main__":
    main()
