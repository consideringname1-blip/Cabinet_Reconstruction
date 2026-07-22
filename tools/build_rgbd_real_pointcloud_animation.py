import json
import struct
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/workspace_whz")
CAPTURE = "20260622_081031_636398Z"
OUT = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_real_pointcloud_animation"
COLOR_PATH = ROOT / f"data/upload/larm_captures/{CAPTURE}/color.png"
DEPTH_PATH = ROOT / f"data/output/hololens2/{CAPTURE}_align_depth.png"
META_PATH = ROOT / f"data/upload/{CAPTURE}_meta.json"
JOINT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_synthesis_axis_estimate/selected_axis1d_joint.json"
MASKS = {
    "base": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "drawer": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
}
RNG = np.random.default_rng(20260712)

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
POINTS = 0
LINES = 1
TRIANGLES = 4
FLOAT = 5126
UNSIGNED_BYTE = 5121
UNSIGNED_INT = 5125


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def load_inputs():
    color_bgr = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(COLOR_PATH)
    depth_raw = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(DEPTH_PATH)
    depth_m = depth_raw.astype(np.float32) / 1000.0
    k = np.asarray(read_json(META_PATH)["PVCamera"]["k"], dtype=np.float64)
    return color_bgr, depth_m, k


def unproject_mask_points(name, max_points):
    color_bgr, depth_m, k = load_inputs()
    mask = cv2.imread(str(MASKS[name]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(MASKS[name])
    valid = (mask > 127) & (depth_m > 0.2) & (depth_m < 4.0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    ys, xs = np.nonzero(valid)
    if len(xs) > max_points:
        idx = RNG.choice(len(xs), size=max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    points = np.column_stack([x, y, z]).astype(np.float32)
    rgb = color_bgr[ys, xs][:, ::-1]
    rgba = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)])
    return points, rgba.astype(np.uint8), {"input_pixels": int(valid.sum()), "exported_points": int(len(points))}


def pad4_bytes(data, byte=0):
    rem = len(data) % 4
    if rem:
        data += bytes([byte]) * (4 - rem)
    return data


class GlbBuilder:
    def __init__(self, generator):
        self.doc = {
            "asset": {"version": "2.0", "generator": generator},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "buffers": [{"byteLength": 0}],
            "bufferViews": [],
            "accessors": [],
            "animations": [],
            "materials": [
                {
                    "name": "vertex_color_material",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0,
                    },
                }
            ],
        }
        self.bin = bytearray()

    def add_buffer_view(self, data, target=None):
        pad = (-len(self.bin)) % 4
        if pad:
            self.bin.extend(b"\x00" * pad)
        offset = len(self.bin)
        self.bin.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = int(target)
        self.buffer_views_append(view)
        return len(self.doc["bufferViews"]) - 1

    def buffer_views_append(self, view):
        self.doc["bufferViews"].append(view)

    def add_accessor(self, array, component_type, accessor_type, target=None, normalized=False, include_minmax=True):
        arr = np.asarray(array)
        if component_type == FLOAT:
            data = arr.astype("<f4", copy=False).tobytes()
        elif component_type == UNSIGNED_BYTE:
            data = arr.astype(np.uint8, copy=False).tobytes()
        elif component_type == UNSIGNED_INT:
            data = arr.astype("<u4", copy=False).tobytes()
        else:
            raise ValueError(component_type)
        view_idx = self.add_buffer_view(data, target=target)
        accessor = {
            "bufferView": view_idx,
            "byteOffset": 0,
            "componentType": int(component_type),
            "count": int(len(arr)),
            "type": accessor_type,
        }
        if normalized:
            accessor["normalized"] = True
        if include_minmax and len(arr) and component_type in (FLOAT, UNSIGNED_INT):
            if accessor_type == "SCALAR":
                accessor["min"] = [float(np.min(arr))]
                accessor["max"] = [float(np.max(arr))]
            else:
                accessor["min"] = np.min(arr, axis=0).astype(float).tolist()
                accessor["max"] = np.max(arr, axis=0).astype(float).tolist()
        self.doc["accessors"].append(accessor)
        return len(self.doc["accessors"]) - 1

    def add_mesh(self, name, positions, colors, mode, indices=None):
        pos_acc = self.add_accessor(positions, FLOAT, "VEC3", target=ARRAY_BUFFER)
        color_acc = self.add_accessor(colors, UNSIGNED_BYTE, "VEC4", target=ARRAY_BUFFER, normalized=True, include_minmax=False)
        primitive = {
            "attributes": {"POSITION": pos_acc, "COLOR_0": color_acc},
            "mode": int(mode),
            "material": 0,
        }
        if indices is not None:
            primitive["indices"] = self.add_accessor(indices, UNSIGNED_INT, "SCALAR", target=ELEMENT_ARRAY_BUFFER)
        self.doc["meshes"].append({"name": name, "primitives": [primitive]})
        return len(self.doc["meshes"]) - 1

    def add_node(self, name, mesh_idx):
        self.doc["nodes"].append({"name": name, "mesh": int(mesh_idx)})
        node_idx = len(self.doc["nodes"]) - 1
        self.doc["scenes"][0]["nodes"].append(node_idx)
        return node_idx

    def add_translation_animation(self, name, node_idx, translation, duration=2.0):
        times = np.array([0.0, duration], dtype=np.float32)
        values = np.asarray([[0.0, 0.0, 0.0], translation], dtype=np.float32)
        time_acc = self.add_accessor(times, FLOAT, "SCALAR", include_minmax=True)
        value_acc = self.add_accessor(values, FLOAT, "VEC3", include_minmax=False)
        self.doc["animations"].append(
            {
                "name": name,
                "samplers": [{"input": time_acc, "output": value_acc, "interpolation": "LINEAR"}],
                "channels": [{"sampler": 0, "target": {"node": int(node_idx), "path": "translation"}}],
            }
        )

    def write(self, path):
        bin_data = pad4_bytes(bytes(self.bin), 0)
        self.doc["buffers"][0]["byteLength"] = len(bin_data)
        json_data = json.dumps(self.doc, separators=(",", ":")).encode("utf-8")
        json_data = pad4_bytes(json_data, 0x20)
        total_len = 12 + 8 + len(json_data) + 8 + len(bin_data)
        with path.open("wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total_len))
            f.write(struct.pack("<II", len(json_data), 0x4E4F534A))
            f.write(json_data)
            f.write(struct.pack("<II", len(bin_data), 0x004E4942))
            f.write(bin_data)


def make_axis_line(drawer_closed_points, translation_closed_to_open):
    origin = np.median(drawer_closed_points, axis=0).astype(np.float32)
    end = (origin + np.asarray(translation_closed_to_open, dtype=np.float32)).astype(np.float32)
    positions = np.stack([origin, end], axis=0)
    colors = np.array([[255, 210, 20, 255], [255, 210, 20, 255]], dtype=np.uint8)
    return positions, colors


def make_splats(points, colors, half_size=0.0022):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    x = np.array([half_size, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, half_size, 0.0], dtype=np.float32)
    offsets = np.stack([-x - y, x - y, x + y, -x + y], axis=0)
    vertices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    vertex_colors = np.repeat(colors, 4, axis=0)
    base = (np.arange(len(points), dtype=np.uint32) * 4).reshape(-1, 1)
    faces = np.hstack(
        [
            np.column_stack([base[:, 0] + 0, base[:, 0] + 1, base[:, 0] + 2]),
            np.column_stack([base[:, 0] + 0, base[:, 0] + 2, base[:, 0] + 3]),
        ]
    ).reshape(-1, 3)
    return vertices.astype(np.float32), vertex_colors.astype(np.uint8), faces.reshape(-1).astype(np.uint32)


def build_glb(path, mode_name, base_points, base_colors, drawer_closed_points, drawer_colors, translation_closed_to_open):
    builder = GlbBuilder(f"rgbd-real-pointcloud-animation-{mode_name}")
    if mode_name == "points":
        base_mesh = builder.add_mesh("base_real_rgbd_points", base_points, base_colors, POINTS)
        drawer_mesh = builder.add_mesh("drawer_closed_real_rgbd_points", drawer_closed_points, drawer_colors, POINTS)
    elif mode_name == "splats":
        b_v, b_c, b_i = make_splats(base_points, base_colors)
        d_v, d_c, d_i = make_splats(drawer_closed_points, drawer_colors)
        base_mesh = builder.add_mesh("base_real_rgbd_splats", b_v, b_c, TRIANGLES, b_i)
        drawer_mesh = builder.add_mesh("drawer_closed_real_rgbd_splats", d_v, d_c, TRIANGLES, d_i)
    else:
        raise ValueError(mode_name)
    axis_pos, axis_col = make_axis_line(drawer_closed_points, translation_closed_to_open)
    axis_mesh = builder.add_mesh("axis_line_closed_to_open", axis_pos, axis_col, LINES)
    builder.add_node("base_points_static", base_mesh)
    drawer_node = builder.add_node("drawer_closed_points_animated", drawer_mesh)
    builder.add_node("axis_line_closed_to_open", axis_mesh)
    builder.add_translation_animation(
        "drawer_closed_to_open_camera_frame_direct",
        drawer_node,
        np.asarray(translation_closed_to_open, dtype=np.float32),
    )
    builder.write(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint_doc = read_json(JOINT_JSON)
    joint = joint_doc.get("joint", joint_doc)
    translation_open_to_closed = np.asarray(joint["translation_open_to_closed_camera_m"], dtype=np.float32)
    translation_closed_to_open = np.asarray(joint["translation_closed_to_open_camera_m"], dtype=np.float32)
    axis = unit(joint["axis_camera_closed_to_open"])

    base_points, base_colors, base_stats = unproject_mask_points("base", max_points=70000)
    drawer_open_points, drawer_colors, drawer_stats = unproject_mask_points("drawer", max_points=50000)
    drawer_closed_points = drawer_open_points + translation_open_to_closed.reshape(1, 3)

    points_path = OUT / "cabinet_drawer_rgbd_real_points_animated.glb"
    splats_path = OUT / "cabinet_drawer_rgbd_real_splats_animated.glb"
    build_glb(points_path, "points", base_points, base_colors, drawer_closed_points, drawer_colors, translation_closed_to_open)
    # Smaller visible splat version for viewers that render GLB POINTS too tiny.
    if len(base_points) > 22000:
        b_idx = RNG.choice(len(base_points), size=22000, replace=False)
        base_splat_points = base_points[b_idx]
        base_splat_colors = base_colors[b_idx]
    else:
        base_splat_points = base_points
        base_splat_colors = base_colors
    if len(drawer_closed_points) > 16000:
        d_idx = RNG.choice(len(drawer_closed_points), size=16000, replace=False)
        drawer_splat_points = drawer_closed_points[d_idx]
        drawer_splat_colors = drawer_colors[d_idx]
    else:
        drawer_splat_points = drawer_closed_points
        drawer_splat_colors = drawer_colors
    build_glb(
        splats_path,
        "splats",
        base_splat_points,
        base_splat_colors,
        drawer_splat_points,
        drawer_splat_colors,
        translation_closed_to_open,
    )

    report = {
        "method": "Animated real RGB-D point cloud. Base is static qpos1 RGB-D mask points. Drawer geometry is the qpos1 drawer RGB-D points shifted to closed by translation_open_to_closed, then animated directly in glTF node translation to translation_closed_to_open. No SAM3D, proxy cage, Blender, or 6DoF pose is used.",
        "frame": "qpos1 OpenCV/PV camera frame",
        "camera_axes": {"+X": "image right", "+Y": "image down", "+Z": "forward/deeper"},
        "joint": {
            "axis_camera_closed_to_open": axis.tolist(),
            "translation_open_to_closed_camera_m": translation_open_to_closed.astype(float).tolist(),
            "translation_closed_to_open_camera_m": translation_closed_to_open.astype(float).tolist(),
            "displacement_m": float(joint["displacement_m"]),
            "source_json": str(JOINT_JSON),
        },
        "parts": {
            "base": {**base_stats, "mask": str(MASKS["base"])},
            "drawer": {**drawer_stats, "mask": str(MASKS["drawer"])},
            "splat_downsample": {
                "base_points": int(len(base_splat_points)),
                "drawer_points": int(len(drawer_splat_points)),
                "half_size_m": 0.0022,
            },
        },
        "animation": {
            "name": "drawer_closed_to_open_camera_frame_direct",
            "duration_s": 2.0,
            "target_node": "drawer_closed_points_animated",
            "start": "closed, drawer points shifted by translation_open_to_closed_camera_m",
            "end": "open qpos1 drawer points",
        },
        "outputs": {
            "animated_points_glb": str(points_path),
            "animated_splats_glb": str(splats_path),
        },
    }
    report_path = OUT / "rgbd_real_pointcloud_animation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "animated_points_glb": str(points_path),
                "animated_splats_glb": str(splats_path),
                "base_points": int(len(base_points)),
                "drawer_points": int(len(drawer_open_points)),
                "translation_closed_to_open": translation_closed_to_open.astype(float).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
