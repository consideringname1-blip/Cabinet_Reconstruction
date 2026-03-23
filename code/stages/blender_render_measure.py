from __future__ import annotations

import sys
from math import radians
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.node_groups,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_obj(mesh_path: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    try:
        bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="NEGATIVE_X", up_axis="Z")
    except Exception:
        bpy.ops.import_scene.obj(filepath=str(mesh_path), axis_forward="-X", axis_up="Z")

    imported = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
    if not imported:
        imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError("No mesh object was imported into Blender")
    return imported


def import_ply(pointcloud_path: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    try:
        bpy.ops.wm.ply_import(filepath=str(pointcloud_path), forward_axis="NEGATIVE_X", up_axis="Y")
    except Exception:
        bpy.ops.import_mesh.ply(filepath=str(pointcloud_path))

    imported = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
    if not imported:
        imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError("No point cloud object was imported into Blender")
    return imported


def build_emission_material(name: str, color_rgb: tuple[float, float, float], alpha: float = 1.0) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color_rgb, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    if alpha < 0.999:
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        mix = nodes.new(type="ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 1.0 - alpha
        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(emission.outputs["Emission"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Surface"])
        if hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
        if hasattr(material, "shadow_method"):
            material.shadow_method = "NONE"
    else:
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

    return material


def assign_material(objects: list[bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in objects:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def apply_group_transform(
    objects: list[bpy.types.Object],
    location_xyz: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    uniform_scale: float,
) -> None:
    rotation = Euler(tuple(radians(v) for v in rotation_xyz_deg), "XYZ").to_matrix().to_4x4()
    scale = Matrix.Diagonal((uniform_scale, uniform_scale, uniform_scale, 1.0))
    translation = Matrix.Translation(Vector(location_xyz))
    transform = translation @ rotation @ scale

    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def collect_world_vertices(objects: list[bpy.types.Object], evaluated: bool = False) -> list[Vector]:
    vertices: list[Vector] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if evaluated:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            try:
                vertices.extend(eval_obj.matrix_world @ v.co for v in mesh.vertices)
            finally:
                eval_obj.to_mesh_clear()
        else:
            vertices.extend(obj.matrix_world @ v.co for v in obj.data.vertices)
    if not vertices:
        raise RuntimeError("Imported object has no vertices")
    return vertices


def setup_front_camera(center: Vector, width: float, height: float) -> None:
    camera_data = bpy.data.cameras.new(name="FrontCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(width, height) * 1.15

    camera = bpy.data.objects.new("FrontCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)

    distance = max(width, height) * 2.5 + 1.0
    camera.location = Vector((center.x, center.y - distance, center.z))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera


def setup_preview_camera(center: Vector, spans: Vector) -> None:
    camera_data = bpy.data.cameras.new(name="PreviewCamera")
    camera_data.type = "PERSP"
    camera_data.lens = 55.0

    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)

    span = max(spans.x, spans.y, spans.z, 0.15)
    offset = Vector((0.75 * span, -2.4 * span, 0.95 * span))
    camera.location = center + offset
    look_target = center + Vector((0.0, 0.2 * span, 0.05 * span))
    direction = look_target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera


def configure_scene(render_path: Path, transparent: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(render_path)

    world = bpy.data.worlds.new(name="RenderWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 1.0


def add_point_instances(objects: list[bpy.types.Object], radius: float, material: bpy.types.Material) -> None:
    for obj in objects:
        modifier = obj.modifiers.new(name="PointPreview", type="NODES")
        node_group = bpy.data.node_groups.new(name=f"{obj.name}_PointPreview", type="GeometryNodeTree")
        modifier.node_group = node_group

        try:
            node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        except Exception:
            node_group.inputs.new("NodeSocketGeometry", "Geometry")
            node_group.outputs.new("NodeSocketGeometry", "Geometry")

        nodes = node_group.nodes
        links = node_group.links
        nodes.clear()

        group_in = nodes.new(type="NodeGroupInput")
        group_out = nodes.new(type="NodeGroupOutput")
        mesh_to_points = nodes.new(type="GeometryNodeMeshToPoints")
        ico_sphere = nodes.new(type="GeometryNodeMeshIcoSphere")
        instance_on_points = nodes.new(type="GeometryNodeInstanceOnPoints")
        realize_instances = nodes.new(type="GeometryNodeRealizeInstances")
        set_material = nodes.new(type="GeometryNodeSetMaterial")

        mesh_to_points.mode = "VERTICES"
        mesh_to_points.inputs["Radius"].default_value = radius
        ico_sphere.inputs["Radius"].default_value = radius
        ico_sphere.inputs["Subdivisions"].default_value = 1
        set_material.inputs["Material"].default_value = material

        links.new(group_in.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
        links.new(mesh_to_points.outputs["Points"], instance_on_points.inputs["Points"])
        links.new(ico_sphere.outputs["Mesh"], instance_on_points.inputs["Instance"])
        links.new(instance_on_points.outputs["Instances"], realize_instances.inputs["Geometry"])
        links.new(realize_instances.outputs["Geometry"], set_material.inputs["Geometry"])
        links.new(set_material.outputs["Geometry"], group_out.inputs["Geometry"])


def render_front_model(
    mesh_path: Path,
    render_path: Path,
    location: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    uniform_scale: float,
) -> None:
    objects = import_obj(mesh_path)
    assign_material(objects, build_emission_material("AlignedWhite", (0.92, 0.92, 0.92), alpha=0.65))
    apply_group_transform(objects, location, rotation_deg, uniform_scale)

    vertices = collect_world_vertices(objects, evaluated=False)
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    zs = [v.z for v in vertices]
    min_corner = Vector((min(xs), min(ys), min(zs)))
    max_corner = Vector((max(xs), max(ys), max(zs)))
    center = (min_corner + max_corner) / 2.0

    width = max_corner.x - min_corner.x
    height = max_corner.z - min_corner.z
    setup_front_camera(center=center, width=width, height=height)
    configure_scene(render_path, transparent=True)
    bpy.ops.render.render(write_still=True)


def render_overlay_preview(
    mesh_path: Path,
    pointcloud_path: Path,
    render_path: Path,
    location: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    uniform_scale: float,
) -> None:
    pointcloud_objects = import_ply(pointcloud_path)
    point_vertices = collect_world_vertices(pointcloud_objects, evaluated=False)

    point_material = build_emission_material("PointCloudDark", (0.08, 0.08, 0.08), alpha=1.0)
    add_point_instances(pointcloud_objects, radius=0.0020, material=point_material)

    model_objects = import_obj(mesh_path)
    model_material = build_emission_material("AlignedWhite", (0.93, 0.93, 0.93), alpha=0.58)
    assign_material(model_objects, model_material)
    apply_group_transform(model_objects, location, rotation_deg, uniform_scale)
    model_vertices = collect_world_vertices(model_objects, evaluated=False)

    all_vertices = point_vertices + model_vertices
    xs = [v.x for v in all_vertices]
    ys = [v.y for v in all_vertices]
    zs = [v.z for v in all_vertices]
    min_corner = Vector((min(xs), min(ys), min(zs)))
    max_corner = Vector((max(xs), max(ys), max(zs)))
    center = (min_corner + max_corner) / 2.0
    spans = max_corner - min_corner

    setup_preview_camera(center=center, spans=spans)
    configure_scene(render_path, transparent=False)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    mode = "front_model"
    if argv and argv[0] in {"front_model", "overlay_preview"}:
        mode = argv[0]
        argv = argv[1:]

    if mode == "front_model":
        if len(argv) != 9:
            print(
                "Usage: blender --background --python code/stages/blender_render_measure.py -- "
                "front_model <mesh.obj> <render.png> <tx> <ty> <tz> <rx> <ry> <rz> <scale>",
                file=sys.stderr,
            )
            return 2

        mesh_path = ensure_file(Path(argv[0]).expanduser().resolve(), "OBJ mesh")
        render_path = Path(argv[1]).expanduser().resolve()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        location = tuple(float(v) for v in argv[2:5])
        rotation_deg = tuple(float(v) for v in argv[5:8])
        uniform_scale = float(argv[8])

        try:
            clean_scene()
            render_front_model(mesh_path, render_path, location, rotation_deg, uniform_scale)
            return 0
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if len(argv) != 10:
        print(
            "Usage: blender --background --python code/stages/blender_render_measure.py -- "
            "overlay_preview <mesh.obj> <pointcloud.ply> <render.png> <tx> <ty> <tz> <rx> <ry> <rz> <scale>",
            file=sys.stderr,
        )
        return 2

    mesh_path = ensure_file(Path(argv[0]).expanduser().resolve(), "OBJ mesh")
    pointcloud_path = ensure_file(Path(argv[1]).expanduser().resolve(), "PLY pointcloud")
    render_path = Path(argv[2]).expanduser().resolve()
    render_path.parent.mkdir(parents=True, exist_ok=True)
    location = tuple(float(v) for v in argv[3:6])
    rotation_deg = tuple(float(v) for v in argv[6:9])
    uniform_scale = float(argv[9])

    try:
        clean_scene()
        render_overlay_preview(mesh_path, pointcloud_path, render_path, location, rotation_deg, uniform_scale)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
