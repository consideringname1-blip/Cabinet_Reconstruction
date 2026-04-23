from __future__ import annotations

import sys
from math import radians
from pathlib import Path

_BOOTSTRAP_ROOTS = (
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
)
for _bootstrap_root in _BOOTSTRAP_ROOTS:
    _bootstrap_root_str = str(_bootstrap_root)
    if _bootstrap_root_str not in sys.path:
        sys.path.insert(0, _bootstrap_root_str)

import _bootstrap
import bpy
from blender_common import clean_scene, ensure_file
from mathutils import Euler, Matrix, Vector


def import_obj(mesh_path: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="NEGATIVE_X", up_axis="Z")
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


def import_optional_ply(pointcloud_path: Path) -> list[bpy.types.Object]:
    try:
        return import_ply(pointcloud_path)
    except Exception:
        return []


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


def build_surface_material(
    name: str,
    color_rgb: tuple[float, float, float],
    alpha: float = 1.0,
    roughness: float = 0.42,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color_rgb, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.35
    elif "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = 0.35
    if "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = alpha
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND" if alpha < 0.999 else "OPAQUE"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "NONE" if alpha < 0.999 else "OPAQUE"
    if hasattr(material, "use_backface_culling"):
        material.use_backface_culling = False
    return material


def assign_material(objects: list[bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in objects:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

def set_object_color(objects: list[bpy.types.Object], rgba: tuple[float, float, float, float]) -> None:
    for obj in objects:
        obj.color = rgba


def set_display_wireframe(objects: list[bpy.types.Object], thickness: float = 0.0008) -> None:
    for obj in objects:
        modifier = obj.modifiers.new(name="PreviewWire", type="WIREFRAME")
        modifier.thickness = thickness
        modifier.use_replace = False

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
    # Match Blender's default Front view: look along +Y with Z-up in the image.
    camera.rotation_euler = Euler((radians(90.0), 0.0, 0.0), "XYZ")
    bpy.context.scene.camera = camera


def setup_camera_from_pv_intrinsics(
    width_px: int,
    height_px: int,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> None:
    camera_data = bpy.data.cameras.new(name="PreviewCamera")
    camera_data.type = "PERSP"
    camera_data.clip_start = 0.01
    camera_data.clip_end = 50.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.sensor_height = 36.0 * float(height_px) / max(float(width_px), 1.0)
    camera_data.lens = float(fx_px) * camera_data.sensor_width / max(float(width_px), 1.0)

    # Match a pinhole camera looking along +Z in Unity camera-local space.
    camera_data.shift_x = (float(width_px) * 0.5 - float(cx_px)) / max(float(width_px), 1.0)
    camera_data.shift_y = (float(cy_px) - float(height_px) * 0.5) / max(float(width_px), 1.0)

    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((0.0, 0.0, 0.0))
    camera.rotation_euler = Euler((radians(90.0), 0.0, 0.0), "XYZ")
    bpy.context.scene.camera = camera


def configure_scene(render_path: Path, transparent: bool, width_px: int, height_px: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = int(width_px)
    scene.render.resolution_y = int(height_px)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(render_path)

    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'OBJECT'
    scene.display.shading.show_object_outline = True
    scene.display.shading.show_backface_culling = False
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = True

    world = bpy.data.worlds.get("RenderWorld")
    if world is None:
        world = bpy.data.worlds.new(name="RenderWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 1.0


def configure_compare_scene(render_path: Path, transparent: bool, width_px: int, height_px: int) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = int(width_px)
    scene.render.resolution_y = int(height_px)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(render_path)

    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 32
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "gtao_factor"):
            scene.eevee.gtao_factor = 1.2

    world = bpy.data.worlds.get("RenderWorld")
    if world is None:
        world = bpy.data.worlds.new(name="RenderWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.955, 0.965, 0.985, 1.0)
        bg.inputs[1].default_value = 0.9


def add_sun_light(name: str, location: tuple[float, float, float], rotation_deg: tuple[float, float, float], energy: float) -> None:
    light_data = bpy.data.lights.new(name=name, type="SUN")
    light_data.energy = energy
    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = Vector(location)
    light.rotation_euler = Euler(tuple(radians(v) for v in rotation_deg), "XYZ")


def setup_preview_lights(center: Vector, span: float) -> None:
    add_sun_light(
        name="KeySun",
        location=(center.x + 0.8 * span, center.y - 1.6 * span, center.z + 1.8 * span),
        rotation_deg=(48.0, 0.0, 28.0),
        energy=2.4,
    )
    add_sun_light(
        name="FillSun",
        location=(center.x - 1.4 * span, center.y + 0.8 * span, center.z + 0.9 * span),
        rotation_deg=(62.0, 0.0, -115.0),
        energy=1.1,
    )


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
    front_material = build_emission_material("AlignedBlueFront", (0.25, 0.55, 1.00), alpha=0.55)
    assign_material(objects, front_material)
    set_object_color(objects, (0.25, 0.55, 1.00, 0.55))
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
    configure_scene(render_path, transparent=True, width_px=1280, height_px=960)
    bpy.ops.render.render(write_still=True)


def render_overlay_preview(
    mesh_path: Path,
    pointcloud_path: Path,
    icp_pointcloud_path: Path,
    render_path: Path,
    location: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    uniform_scale: float,
    width_px: int,
    height_px: int,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> None:
    pointcloud_objects = import_optional_ply(pointcloud_path)
    point_vertices = collect_world_vertices(pointcloud_objects, evaluated=False) if pointcloud_objects else []

    point_material = build_emission_material("PointCloudOrange", (1.00, 0.28, 0.10), alpha=1.0)
    if pointcloud_objects:
        add_point_instances(pointcloud_objects, radius=0.0035, material=point_material)
        set_object_color(pointcloud_objects, (1.00, 0.28, 0.10, 1.0))

    icp_pointcloud_objects = import_optional_ply(icp_pointcloud_path)
    icp_point_vertices = collect_world_vertices(icp_pointcloud_objects, evaluated=False) if icp_pointcloud_objects else []
    icp_point_material = build_emission_material("PointCloudGreen", (0.18, 0.80, 0.30), alpha=1.0)
    if icp_pointcloud_objects:
        add_point_instances(icp_pointcloud_objects, radius=0.0039, material=icp_point_material)
        set_object_color(icp_pointcloud_objects, (0.18, 0.80, 0.30, 1.0))

    model_objects = import_obj(mesh_path)
    model_material = build_emission_material("AlignedBlue", (0.25, 0.55, 1.00), alpha=0.42)
    assign_material(model_objects, model_material)
    set_object_color(model_objects, (0.25, 0.55, 1.00, 0.42))
    apply_group_transform(model_objects, location, rotation_deg, uniform_scale)
    model_vertices = collect_world_vertices(model_objects, evaluated=False)

    setup_camera_from_pv_intrinsics(width_px, height_px, fx_px, fy_px, cx_px, cy_px)
    configure_scene(render_path, transparent=False, width_px=width_px, height_px=height_px)
    bpy.ops.render.render(write_still=True)


def render_model_compare_preview(
    mesh_path: Path,
    render_path: Path,
    aligned_location: tuple[float, float, float],
    aligned_rotation_deg: tuple[float, float, float],
    aligned_scale: float,
    reference_location: tuple[float, float, float],
    reference_rotation_deg: tuple[float, float, float],
    reference_scale: float,
    width_px: int,
    height_px: int,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> None:
    aligned_objects = import_obj(mesh_path)
    aligned_material = build_surface_material("AlignedBlue", (0.16, 0.42, 0.94), alpha=0.50, roughness=0.36)
    assign_material(aligned_objects, aligned_material)
    set_object_color(aligned_objects, (0.16, 0.42, 0.94, 0.50))
    apply_group_transform(aligned_objects, aligned_location, aligned_rotation_deg, aligned_scale)

    reference_objects = import_obj(mesh_path)
    reference_material = build_surface_material("ReferenceOrange", (0.96, 0.48, 0.10), alpha=0.50, roughness=0.48)
    assign_material(reference_objects, reference_material)
    set_object_color(reference_objects, (0.96, 0.48, 0.10, 0.50))
    set_display_wireframe(reference_objects, thickness=0.0012)
    apply_group_transform(reference_objects, reference_location, reference_rotation_deg, reference_scale)

    all_vertices = collect_world_vertices(aligned_objects, evaluated=True) + collect_world_vertices(reference_objects, evaluated=True)
    xs = [v.x for v in all_vertices]
    ys = [v.y for v in all_vertices]
    zs = [v.z for v in all_vertices]
    min_corner = Vector((min(xs), min(ys), min(zs)))
    max_corner = Vector((max(xs), max(ys), max(zs)))
    center = (min_corner + max_corner) / 2.0
    spans = max_corner - min_corner

    setup_camera_from_pv_intrinsics(width_px, height_px, fx_px, fy_px, cx_px, cy_px)
    setup_preview_lights(center=center, span=max(spans.x, spans.y, spans.z, 0.15))
    configure_compare_scene(render_path, transparent=False, width_px=width_px, height_px=height_px)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    mode = "front_model"
    if argv and argv[0] in {"front_model", "overlay_preview", "model_compare_preview"}:
        mode = argv[0]
        argv = argv[1:]

    if mode == "front_model":
        if len(argv) != 9:
            print(
                "Usage: blender --background --python code/stages/hololens3d_reconstruction/blender_render_measure.py -- "
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

    if mode == "model_compare_preview":
        if len(argv) != 22:
            print(
                "Usage: blender --background --python code/stages/hololens3d_reconstruction/blender_render_measure.py -- "
                "model_compare_preview <mesh.obj> <render.png> "
                "<aligned_tx> <aligned_ty> <aligned_tz> <aligned_rx> <aligned_ry> <aligned_rz> <aligned_scale> "
                "<ref_tx> <ref_ty> <ref_tz> <ref_rx> <ref_ry> <ref_rz> <ref_scale> "
                "<width> <height> <fx> <fy> <cx> <cy>",
                file=sys.stderr,
            )
            return 2

        mesh_path = ensure_file(Path(argv[0]).expanduser().resolve(), "OBJ mesh")
        render_path = Path(argv[1]).expanduser().resolve()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_location = tuple(float(v) for v in argv[2:5])
        aligned_rotation_deg = tuple(float(v) for v in argv[5:8])
        aligned_scale = float(argv[8])
        reference_location = tuple(float(v) for v in argv[9:12])
        reference_rotation_deg = tuple(float(v) for v in argv[12:15])
        reference_scale = float(argv[15])
        width_px = int(float(argv[16]))
        height_px = int(float(argv[17]))
        fx_px = float(argv[18])
        fy_px = float(argv[19])
        cx_px = float(argv[20])
        cy_px = float(argv[21])

        try:
            clean_scene()
            render_model_compare_preview(
                mesh_path,
                render_path,
                aligned_location,
                aligned_rotation_deg,
                aligned_scale,
                reference_location,
                reference_rotation_deg,
                reference_scale,
                width_px,
                height_px,
                fx_px,
                fy_px,
                cx_px,
                cy_px,
            )
            return 0
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if len(argv) != 17:
        print(
            "Usage: blender --background --python code/stages/hololens3d_reconstruction/blender_render_measure.py -- "
            "overlay_preview <mesh.obj> <pointcloud.ply> <icp_pointcloud.ply> <render.png> "
            "<tx> <ty> <tz> <rx> <ry> <rz> <scale> <width> <height> <fx> <fy> <cx> <cy>",
            file=sys.stderr,
        )
        return 2

    mesh_path = ensure_file(Path(argv[0]).expanduser().resolve(), "OBJ mesh")
    pointcloud_path = ensure_file(Path(argv[1]).expanduser().resolve(), "PLY pointcloud")
    icp_pointcloud_path = ensure_file(Path(argv[2]).expanduser().resolve(), "ICP pointcloud")
    render_path = Path(argv[3]).expanduser().resolve()
    render_path.parent.mkdir(parents=True, exist_ok=True)
    location = tuple(float(v) for v in argv[4:7])
    rotation_deg = tuple(float(v) for v in argv[7:10])
    uniform_scale = float(argv[10])
    width_px = int(float(argv[11]))
    height_px = int(float(argv[12]))
    fx_px = float(argv[13])
    fy_px = float(argv[14])
    cx_px = float(argv[15])
    cy_px = float(argv[16])

    try:
        clean_scene()
        render_overlay_preview(
            mesh_path,
            pointcloud_path,
            icp_pointcloud_path,
            render_path,
            location,
            rotation_deg,
            uniform_scale,
            width_px,
            height_px,
            fx_px,
            fy_px,
            cx_px,
            cy_px,
        )
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
