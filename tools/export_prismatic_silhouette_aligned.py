import json
import math
from pathlib import Path

import bpy
import numpy as np


ROOT = Path("/workspace_whz")
OUT_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_silhouette_aligned"
BASE_GLB = ROOT / "data/output/sam3d-objects/meshes/qpos1_base_sam3mask_masked_rgb_sam3d_raw.glb"
DRAWER_GLB = ROOT / "data/output/sam3d-objects/meshes/qpos1_door_sam3mask_masked_rgb_sam3d_raw.glb"
PROJ_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_projection_aligned/projection_fit_targets.json"
DRAWER_FIT_JSON = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_silhouette_fit/drawer_silhouette_fit.json"

MAX_BASE_FACES = 150_000
MAX_DRAWER_FACES = 100_000


def blender_to_sam3d(v: np.ndarray) -> np.ndarray:
    # Blender's glTF importer maps raw glTF coordinates as [x, -z, y].
    return np.array([v[0], v[2], -v[1]], dtype=np.float64)


def camera_to_blender(v: np.ndarray) -> np.ndarray:
    # Keep the deployment files in the same visual convention as earlier exports:
    # Blender X = camera X, Blender Y = camera depth Z, Blender Z = camera image Y.
    return np.array([v[0], v[2], v[1]], dtype=np.float64)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def imported_meshes() -> list[bpy.types.Object]:
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def import_single_mesh(path: Path, name: str) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = f"{name}_mesh"
    return obj


def raw_vertices_in_sam3d_frame(obj: bpy.types.Object) -> np.ndarray:
    mw = obj.matrix_world.copy()
    verts = np.empty((len(obj.data.vertices), 3), dtype=np.float64)
    for i, vert in enumerate(obj.data.vertices):
        p = mw @ vert.co
        verts[i] = blender_to_sam3d(np.array([p.x, p.y, p.z], dtype=np.float64))
    return verts


def set_vertices_from_camera(obj: bpy.types.Object, cam_vertices: np.ndarray) -> None:
    obj.matrix_world.identity()
    for vert, p_cam in zip(obj.data.vertices, cam_vertices):
        p = camera_to_blender(p_cam)
        vert.co = (float(p[0]), float(p[1]), float(p[2]))
    obj.data.update()


def set_vertices_child_frame(obj: bpy.types.Object, cam_vertices: np.ndarray, origin_cam: np.ndarray) -> None:
    obj.matrix_world.identity()
    rel = cam_vertices - origin_cam.reshape(1, 3)
    for vert, p_cam in zip(obj.data.vertices, rel):
        p = camera_to_blender(p_cam)
        vert.co = (float(p[0]), float(p[1]), float(p[2]))
    obj.data.update()


def ensure_vertex_color_material(obj: bpy.types.Object, name: str) -> None:
    obj.data.materials.clear()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        vc = nodes.new(type="ShaderNodeVertexColor")
        vc.layer_name = "Color"
        mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.72
    obj.data.materials.append(mat)
    if obj.data.color_attributes:
        color = obj.data.color_attributes.get("Color") or obj.data.color_attributes[0]
        obj.data.color_attributes.active = color
        obj.data.color_attributes.render_color_index = list(obj.data.color_attributes).index(color)


def decimate(obj: bpy.types.Object, max_faces: int) -> dict:
    before = len(obj.data.polygons)
    if before <= max_faces:
        return {"before_faces": before, "after_faces": before, "ratio": 1.0}
    ratio = max_faces / float(before)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mod = obj.modifiers.new("deployment_decimate", "DECIMATE")
    mod.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=mod.name)
    after = len(obj.data.polygons)
    obj.select_set(False)
    return {"before_faces": before, "after_faces": after, "ratio": ratio}


def export_fbx(path: Path, objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[-1]
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        use_selection=True,
        object_types={"EMPTY", "MESH"},
        apply_unit_scale=True,
        bake_space_transform=False,
        add_leaf_bones=False,
        path_mode="AUTO",
        use_custom_props=True,
        axis_forward="-Z",
        axis_up="Y",
        colors_type="SRGB",
        prioritize_active_color=True,
    )


def export_obj(path: Path, obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=False,
        export_normals=True,
        export_colors=True,
        path_mode="RELATIVE",
    )


def write_urdf(path: Path, joint: dict) -> None:
    axis = joint["axis_camera"]
    origin = joint["origin_camera_m"]
    limit = joint["open_distance_m"]
    text = f"""<?xml version="1.0"?>
<robot name="cabinet_drawer_prismatic_silhouette">
  <link name="base">
    <visual><geometry><mesh filename="base_silhouette_vertexcolor.obj"/></geometry></visual>
    <collision><geometry><mesh filename="base_silhouette_vertexcolor.obj"/></geometry></collision>
  </link>
  <link name="drawer">
    <visual><geometry><mesh filename="drawer_closed_child_silhouette_vertexcolor.obj"/></geometry></visual>
    <collision><geometry><mesh filename="drawer_closed_child_silhouette_vertexcolor.obj"/></geometry></collision>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="base"/>
    <child link="drawer"/>
    <origin xyz="{origin[0]:.9f} {origin[1]:.9f} {origin[2]:.9f}" rpy="0 0 0"/>
    <axis xyz="{axis[0]:.9f} {axis[1]:.9f} {axis[2]:.9f}"/>
    <limit lower="0" upper="{limit:.9f}" effort="20" velocity="0.5"/>
  </joint>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_scene()

    projection = json.loads(PROJ_JSON.read_text(encoding="utf-8"))
    drawer_fit = json.loads(DRAWER_FIT_JSON.read_text(encoding="utf-8"))
    joint = projection["prismatic_guess"]

    base_fit = json.loads((ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_projection_aligned/projection_prismatic_manifest.json").read_text(encoding="utf-8"))["projection_fit"]["base"]
    base_center = np.array(base_fit["center_cam_m"], dtype=np.float64)
    base_scale = float(base_fit["scale"])

    drawer_best = drawer_fit["best"]
    drawer_center = np.array(drawer_fit["target_center_3d"], dtype=np.float64)
    drawer_source_center = np.array(drawer_fit["source_center"], dtype=np.float64)
    drawer_scale = float(drawer_best["scale"])
    drawer_R = np.array(drawer_best["R"], dtype=np.float64)

    origin_cam = np.array(joint["origin_camera_m"], dtype=np.float64)
    axis_cam = np.array(joint["axis_camera"], dtype=np.float64)
    axis_cam = axis_cam / np.linalg.norm(axis_cam)
    open_distance = float(joint["open_distance_m"])

    base_obj = import_single_mesh(BASE_GLB, "base_silhouette_aligned")
    base_raw = raw_vertices_in_sam3d_frame(base_obj)
    base_source_center = (base_raw.min(axis=0) + base_raw.max(axis=0)) * 0.5
    base_cam = base_center.reshape(1, 3) + base_scale * (base_raw - base_source_center.reshape(1, 3))
    set_vertices_from_camera(base_obj, base_cam)
    ensure_vertex_color_material(base_obj, "base_vertex_color")
    base_dec = decimate(base_obj, MAX_BASE_FACES)

    drawer_open = import_single_mesh(DRAWER_GLB, "drawer_open_silhouette_aligned")
    drawer_raw = raw_vertices_in_sam3d_frame(drawer_open)
    drawer_open_cam = drawer_center.reshape(1, 3) + drawer_scale * ((drawer_raw - drawer_source_center.reshape(1, 3)) @ drawer_R.T)
    set_vertices_from_camera(drawer_open, drawer_open_cam)
    ensure_vertex_color_material(drawer_open, "drawer_vertex_color")
    drawer_open_dec = decimate(drawer_open, MAX_DRAWER_FACES)

    drawer_closed = drawer_open.copy()
    drawer_closed.data = drawer_open.data.copy()
    drawer_closed.name = "drawer_closed_child_silhouette_aligned"
    drawer_closed.data.name = "drawer_closed_child_silhouette_mesh"
    bpy.context.collection.objects.link(drawer_closed)
    drawer_closed_cam = drawer_open_cam - axis_cam.reshape(1, 3) * open_distance
    set_vertices_child_frame(drawer_closed, drawer_closed_cam, origin_cam)
    ensure_vertex_color_material(drawer_closed, "drawer_vertex_color_child")

    pivot = bpy.data.objects.new("drawer_slide_pivot", None)
    pivot.empty_display_type = "ARROWS"
    pivot.empty_display_size = 0.08
    pivot.location = tuple(camera_to_blender(origin_cam))
    pivot["joint_type"] = "prismatic"
    pivot["axis_camera"] = [float(x) for x in axis_cam]
    pivot["axis_blender"] = [float(x) for x in camera_to_blender(axis_cam)]
    pivot["open_distance_m"] = open_distance
    bpy.context.collection.objects.link(pivot)
    drawer_closed.parent = pivot

    # Keep the open-reference drawer separate from the closed hierarchy export.
    base_obj["source"] = str(BASE_GLB)
    drawer_open["source"] = str(DRAWER_GLB)
    drawer_closed["source"] = str(DRAWER_GLB)

    export_obj(OUT_DIR / "base_silhouette_vertexcolor.obj", base_obj)
    export_obj(OUT_DIR / "drawer_open_silhouette_vertexcolor.obj", drawer_open)
    export_obj(OUT_DIR / "drawer_closed_child_silhouette_vertexcolor.obj", drawer_closed)
    export_fbx(OUT_DIR / "cabinet_drawer_prismatic_silhouette_open_reference.fbx", [base_obj, drawer_open])
    export_fbx(OUT_DIR / "cabinet_drawer_prismatic_silhouette_closed_hierarchy.fbx", [base_obj, pivot, drawer_closed])
    export_fbx(OUT_DIR / "cabinet_base_prismatic_silhouette.fbx", [base_obj])
    export_fbx(OUT_DIR / "drawer_closed_child_prismatic_silhouette.fbx", [pivot, drawer_closed])
    write_urdf(OUT_DIR / "cabinet_drawer_prismatic_silhouette.urdf", {
        "axis_camera": axis_cam.tolist(),
        "origin_camera_m": origin_cam.tolist(),
        "open_distance_m": open_distance,
    })

    manifest = {
        "method": "SAM3D raw GLB vertex-color meshes, drawer orientation fitted against the qpos1 SAM mask silhouette, prismatic drawer motion. Shape is preserved with uniform scale only.",
        "coordinate_note": "FBX/OBJ vertices use Blender visual coordinates [camera_x, camera_z_depth, camera_y_image]. URDF stores the camera-frame joint for reference.",
        "base_fit": {
            "source": str(BASE_GLB),
            "scale": base_scale,
            "source_center_sam3d": base_source_center.tolist(),
            "center_camera_m": base_center.tolist(),
        },
        "drawer_silhouette_fit": {
            "source": str(DRAWER_GLB),
            "scale": drawer_scale,
            "source_center_sam3d": drawer_source_center.tolist(),
            "center_camera_m": drawer_center.tolist(),
            "R_sam3d_to_camera": drawer_R.tolist(),
            "score": drawer_best["score"],
            "iou": drawer_best["iou"],
            "projected_bbox_xyxy": drawer_best["proj_bbox"],
        },
        "joint": {
            "type": "prismatic",
            "origin_camera_m": origin_cam.tolist(),
            "axis_camera": axis_cam.tolist(),
            "axis_blender": camera_to_blender(axis_cam).tolist(),
            "open_distance_m": open_distance,
        },
        "decimate": {"base": base_dec, "drawer": drawer_open_dec},
        "outputs": {
            "fbx_closed_hierarchy": str(OUT_DIR / "cabinet_drawer_prismatic_silhouette_closed_hierarchy.fbx"),
            "fbx_open_reference": str(OUT_DIR / "cabinet_drawer_prismatic_silhouette_open_reference.fbx"),
            "fbx_base": str(OUT_DIR / "cabinet_base_prismatic_silhouette.fbx"),
            "fbx_drawer": str(OUT_DIR / "drawer_closed_child_prismatic_silhouette.fbx"),
            "urdf": str(OUT_DIR / "cabinet_drawer_prismatic_silhouette.urdf"),
            "base_obj": str(OUT_DIR / "base_silhouette_vertexcolor.obj"),
            "drawer_closed_obj": str(OUT_DIR / "drawer_closed_child_silhouette_vertexcolor.obj"),
            "drawer_open_obj": str(OUT_DIR / "drawer_open_silhouette_vertexcolor.obj"),
        },
    }
    (OUT_DIR / "silhouette_prismatic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
