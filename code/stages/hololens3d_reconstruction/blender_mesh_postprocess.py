from __future__ import annotations

from pathlib import Path
from typing import Any

import bmesh
import bpy


def ratio_label(ratio: float) -> str:
    return f"{int(round(float(ratio) * 100)):02d}pct"


def live_mesh_objects(objects: list) -> list:
    live = []
    for obj in objects:
        if obj is None or getattr(obj, "type", None) != "MESH":
            continue
        if obj.name not in bpy.data.objects:
            continue
        live.append(obj)
    return live


def count_mesh_objects(objects: list) -> tuple[int, int]:
    vertices = 0
    faces = 0
    for obj in live_mesh_objects(objects):
        vertices += len(obj.data.vertices)
        faces += len(obj.data.polygons)
    return vertices, faces


def count_mesh(obj) -> tuple[int, int]:
    return len(obj.data.vertices), len(obj.data.polygons)


def select_objects(objects: list, *, active=None) -> list:
    meshes = live_mesh_objects(objects)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if active is None and meshes:
        active = meshes[0]
    if active is not None and active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = active
    return meshes


def duplicate_mesh_objects(objects: list, *, suffix: str) -> list:
    duplicates = []
    for obj in live_mesh_objects(objects):
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        duplicate.animation_data_clear()
        duplicate.name = f"{obj.name}{suffix}"
        duplicate.data.name = f"{duplicate.name}_mesh"
        duplicate.matrix_world = obj.matrix_world.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicates.append(duplicate)
    return duplicates


def clean_mesh_geometry(obj) -> None:
    if obj is None or obj.type != "MESH" or obj.name not in bpy.data.objects:
        return
    select_objects([obj], active=obj)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        try:
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
        except Exception:
            pass
        try:
            bpy.ops.mesh.delete_loose()
        except Exception:
            pass
        try:
            bpy.ops.mesh.normals_make_consistent(inside=False)
        except Exception:
            pass
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass


def delete_empty_mesh_objects(objects: list) -> list:
    kept = []
    for obj in live_mesh_objects(objects):
        if len(obj.data.polygons) <= 0 or len(obj.data.vertices) <= 0:
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            kept.append(obj)
    return kept


def _separate_loose_parts(obj) -> list:
    if obj is None or obj.type != "MESH" or obj.name not in bpy.data.objects:
        return []
    select_objects([obj], active=obj)
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    parts = [part for part in bpy.context.selected_objects if part.type == "MESH"]
    return parts or [obj]


def clean_connected_components(
    objects: list,
    *,
    enabled: bool,
    min_face_ratio: float,
    min_faces: int,
) -> tuple[list, dict[str, Any]]:
    meshes = live_mesh_objects(objects)
    original_vertices, original_faces = count_mesh_objects(meshes)
    if not enabled or not meshes:
        return meshes, {
            "enabled": bool(enabled),
            "component_count": len(meshes),
            "kept_component_count": len(meshes),
            "removed_component_count": 0,
            "original_vertices": int(original_vertices),
            "original_faces": int(original_faces),
            "clean_vertices": int(original_vertices),
            "clean_faces": int(original_faces),
            "removed_faces": 0,
        }

    parts = []
    for obj in list(meshes):
        if obj.name not in bpy.data.objects:
            continue
        clean_mesh_geometry(obj)
        parts.extend(_separate_loose_parts(obj))
    parts = delete_empty_mesh_objects(parts)
    if not parts:
        return [], {
            "enabled": True,
            "component_count": 0,
            "kept_component_count": 0,
            "removed_component_count": 0,
            "original_vertices": int(original_vertices),
            "original_faces": int(original_faces),
            "clean_vertices": 0,
            "clean_faces": 0,
            "removed_faces": int(original_faces),
        }

    component_infos = sorted(
        [(obj, len(obj.data.polygons), len(obj.data.vertices)) for obj in parts],
        key=lambda item: item[1],
        reverse=True,
    )
    largest_face_count = int(component_infos[0][1])
    threshold = max(int(min_faces), int(round(largest_face_count * float(min_face_ratio))))
    keep = []
    remove = []
    for index, (obj, face_count, _vertex_count) in enumerate(component_infos):
        if index == 0 or int(face_count) >= threshold:
            keep.append(obj)
        else:
            remove.append(obj)

    removed_faces = sum(len(obj.data.polygons) for obj in remove if obj.name in bpy.data.objects)
    for obj in remove:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    keep = delete_empty_mesh_objects(keep)
    clean_vertices, clean_faces = count_mesh_objects(keep)
    return keep, {
        "enabled": True,
        "component_count": int(len(component_infos)),
        "kept_component_count": int(len(keep)),
        "removed_component_count": int(len(remove)),
        "component_face_counts": [int(info[1]) for info in component_infos[:12]],
        "min_face_ratio": float(min_face_ratio),
        "min_faces": int(min_faces),
        "face_threshold": int(threshold),
        "original_vertices": int(original_vertices),
        "original_faces": int(original_faces),
        "clean_vertices": int(clean_vertices),
        "clean_faces": int(clean_faces),
        "removed_faces": int(removed_faces),
    }


def _has_image_texture_material(obj) -> bool:
    for slot in obj.material_slots:
        material = slot.material
        if material_has_image_texture(material):
            return True
    return False


def material_has_image_texture(material) -> bool:
    if material is None or not getattr(material, "use_nodes", False):
        return False
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeTexImage" and getattr(node, "image", None) is not None:
            return True
    return False


def first_color_attribute(obj):
    color_attributes = getattr(obj.data, "color_attributes", None)
    if color_attributes and len(color_attributes) > 0:
        active = getattr(color_attributes, "active_color", None) or getattr(color_attributes, "active", None)
        return active or color_attributes[0]
    vertex_colors = getattr(obj.data, "vertex_colors", None)
    if vertex_colors and len(vertex_colors) > 0:
        return vertex_colors[0]
    return None


def first_color_attribute_name(obj) -> str | None:
    attr = first_color_attribute(obj)
    return getattr(attr, "name", None) if attr is not None else None


def ensure_source_materials(objects: list) -> None:
    for obj in live_mesh_objects(objects):
        if _has_image_texture_material(obj):
            continue
        attr_name = first_color_attribute_name(obj)
        if not attr_name:
            continue
        material = bpy.data.materials.new(f"{obj.name}_vertex_color_source")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        attr = nodes.new(type="ShaderNodeAttribute")
        attr.attribute_name = attr_name
        if bsdf is not None and "Color" in attr.outputs and "Base Color" in bsdf.inputs:
            material.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(material)


def _color_tuple(value) -> tuple[float, float, float, float]:
    values = list(value)
    while len(values) < 4:
        values.append(1.0)
    return tuple(float(v) for v in values[:4])


def _polygon_color_from_attribute(obj, polygon, attr) -> tuple[float, float, float, float] | None:
    if attr is None:
        return None
    colors = []
    domain = str(getattr(attr, "domain", "CORNER"))
    if domain == "POINT":
        for vertex_index in polygon.vertices:
            try:
                colors.append(_color_tuple(attr.data[vertex_index].color))
            except Exception:
                return None
    else:
        for loop_index in polygon.loop_indices:
            try:
                colors.append(_color_tuple(attr.data[loop_index].color))
            except Exception:
                return None
    if not colors:
        return None
    count = float(len(colors))
    return tuple(sum(color[i] for color in colors) / count for i in range(4))


def _principled_base_color(material) -> tuple[float, float, float, float] | None:
    if material is None:
        return None
    color = getattr(material, "diffuse_color", None)
    if color is not None:
        return _color_tuple(color)
    if not getattr(material, "use_nodes", False):
        return None
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        return None
    base_input = bsdf.inputs.get("Base Color")
    if base_input is not None and hasattr(base_input, "default_value"):
        return _color_tuple(base_input.default_value)
    return None


def _polygon_material_color(obj, polygon) -> tuple[float, float, float, float] | None:
    material = None
    if 0 <= polygon.material_index < len(obj.material_slots):
        material = obj.material_slots[polygon.material_index].material
    if material_has_image_texture(material):
        return None
    return _principled_base_color(material)


def _is_missing_black_color(
    color: tuple[float, float, float, float],
    *,
    rgb_threshold: float,
    alpha_threshold: float,
) -> bool:
    rgb = color[:3]
    alpha = color[3]
    return alpha <= float(alpha_threshold) or max(rgb) <= float(rgb_threshold)


def _black_face_candidates(
    obj,
    *,
    rgb_threshold: float,
    alpha_threshold: float,
) -> list[int]:
    attr = first_color_attribute(obj)
    candidates = []
    for polygon in obj.data.polygons:
        color = _polygon_color_from_attribute(obj, polygon, attr)
        if color is None:
            color = _polygon_material_color(obj, polygon)
        if color is None:
            continue
        if _is_missing_black_color(color, rgb_threshold=rgb_threshold, alpha_threshold=alpha_threshold):
            candidates.append(polygon.index)
    return candidates


def _delete_faces(obj, face_indices: set[int]) -> None:
    if not face_indices:
        return
    select_objects([obj], active=obj)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        faces = [face for face in bm.faces if face.index in face_indices]
        if faces:
            bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")
            bm.to_mesh(mesh)
            mesh.update()
    finally:
        bm.free()
    clean_mesh_geometry(obj)


def _average_color(colors: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not colors:
        return (1.0, 1.0, 1.0, 1.0)
    count = float(len(colors))
    rgb = [sum(color[i] for color in colors) / count for i in range(3)]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _ensure_corner_color_attribute(obj):
    attr = first_color_attribute(obj)
    if attr is not None:
        return attr
    color_attributes = getattr(obj.data, "color_attributes", None)
    if color_attributes is None:
        return None
    attr = color_attributes.new(name="Color", type="BYTE_COLOR", domain="CORNER")
    for polygon in obj.data.polygons:
        color = _polygon_material_color(obj, polygon) or (1.0, 1.0, 1.0, 1.0)
        for loop_index in polygon.loop_indices:
            attr.data[loop_index].color = color
    try:
        color_attributes.active_color = attr
    except Exception:
        pass
    return attr


def _set_polygon_color(obj, polygon, attr, color: tuple[float, float, float, float]) -> None:
    if attr is None:
        return
    color = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    domain = str(getattr(attr, "domain", "CORNER"))
    if domain == "POINT":
        for vertex_index in polygon.vertices:
            attr.data[vertex_index].color = color
    else:
        for loop_index in polygon.loop_indices:
            attr.data[loop_index].color = color


def repair_black_or_transparent_faces(
    objects: list,
    *,
    enabled: bool,
    rgb_threshold: float,
    alpha_threshold: float,
    max_repair_ratio: float,
    max_iterations: int = 12,
) -> tuple[list, dict[str, Any]]:
    meshes = live_mesh_objects(objects)
    total_faces = sum(len(obj.data.polygons) for obj in meshes)
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "rgb_threshold": float(rgb_threshold),
        "alpha_threshold": float(alpha_threshold),
        "max_repair_ratio": float(max_repair_ratio),
        "objects_checked": int(len(meshes)),
        "faces_checked": int(total_faces),
        "candidate_faces": 0,
        "repaired_faces": 0,
        "fallback_faces": 0,
        "skipped_faces": 0,
    }
    if not enabled:
        return meshes, stats

    for obj in meshes:
        if obj.name not in bpy.data.objects or len(obj.data.polygons) <= 0:
            continue
        attr = _ensure_corner_color_attribute(obj)
        if attr is None:
            continue

        polygons = list(obj.data.polygons)
        polygon_colors: dict[int, tuple[float, float, float, float]] = {}
        missing: set[int] = set()
        for polygon in polygons:
            color = _polygon_color_from_attribute(obj, polygon, attr)
            if color is None:
                color = _polygon_material_color(obj, polygon)
            if color is None:
                continue
            if _is_missing_black_color(color, rgb_threshold=rgb_threshold, alpha_threshold=alpha_threshold):
                missing.add(int(polygon.index))
            else:
                polygon_colors[int(polygon.index)] = color

        stats["candidate_faces"] += int(len(missing))
        if not missing:
            continue
        repair_ratio = float(len(missing)) / float(max(len(polygons), 1))
        if repair_ratio > float(max_repair_ratio):
            stats["skipped_faces"] += int(len(missing))
            continue

        global_color = _average_color(list(polygon_colors.values()))
        polygons_by_vertex: dict[int, list[int]] = {}
        for polygon in polygons:
            for vertex_index in polygon.vertices:
                polygons_by_vertex.setdefault(int(vertex_index), []).append(int(polygon.index))

        repaired: dict[int, tuple[float, float, float, float]] = {}
        remaining = set(missing)
        for _iteration in range(max(1, int(max_iterations))):
            progress = False
            for face_index in list(remaining):
                polygon = polygons[face_index]
                neighbor_colors = []
                for vertex_index in polygon.vertices:
                    for neighbor_index in polygons_by_vertex.get(int(vertex_index), []):
                        if neighbor_index == face_index or neighbor_index in remaining:
                            continue
                        color = polygon_colors.get(neighbor_index) or repaired.get(neighbor_index)
                        if color is not None:
                            neighbor_colors.append(color)
                if not neighbor_colors:
                    continue
                color = _average_color(neighbor_colors)
                repaired[face_index] = color
                polygon_colors[face_index] = color
                remaining.remove(face_index)
                progress = True
            if not remaining or not progress:
                break

        for face_index in list(remaining):
            repaired[face_index] = global_color
            polygon_colors[face_index] = global_color
            stats["fallback_faces"] += 1
            remaining.remove(face_index)

        for face_index, color in repaired.items():
            _set_polygon_color(obj, polygons[face_index], attr, color)
        obj.data.update()
        stats["repaired_faces"] += int(len(repaired))

    return meshes, stats


def remove_black_or_transparent_faces(
    objects: list,
    *,
    enabled: bool,
    rgb_threshold: float,
    alpha_threshold: float,
    max_remove_ratio: float,
) -> tuple[list, dict[str, Any]]:
    meshes = live_mesh_objects(objects)
    total_faces = sum(len(obj.data.polygons) for obj in meshes)
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "rgb_threshold": float(rgb_threshold),
        "alpha_threshold": float(alpha_threshold),
        "max_remove_ratio": float(max_remove_ratio),
        "objects_checked": int(len(meshes)),
        "faces_checked": int(total_faces),
        "candidate_faces": 0,
        "removed_faces": 0,
        "skipped_faces": 0,
    }
    if not enabled:
        return meshes, stats

    for obj in list(meshes):
        if obj.name not in bpy.data.objects or len(obj.data.polygons) <= 0:
            continue
        candidates = _black_face_candidates(
            obj,
            rgb_threshold=float(rgb_threshold),
            alpha_threshold=float(alpha_threshold),
        )
        stats["candidate_faces"] += int(len(candidates))
        if not candidates:
            continue
        object_face_count = len(obj.data.polygons)
        candidate_ratio = float(len(candidates)) / float(max(object_face_count, 1))
        if candidate_ratio > float(max_remove_ratio):
            stats["skipped_faces"] += int(len(candidates))
            continue
        _delete_faces(obj, set(candidates))
        stats["removed_faces"] += int(len(candidates))

    return delete_empty_mesh_objects(meshes), stats


def smart_unwrap_objects(objects: list, *, island_margin: float = 0.03) -> None:
    for obj in live_mesh_objects(objects):
        if obj.name not in bpy.data.objects or len(obj.data.polygons) <= 0:
            continue
        select_objects([obj], active=obj)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(
                angle_limit=1.15192,
                island_margin=float(island_margin),
                area_weight=0.0,
            )
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass


def apply_decimate_to_objects(
    objects: list,
    *,
    ratio: float,
    modifier_prefix: str,
    enabled: bool = True,
) -> tuple[list, dict[str, Any]]:
    meshes = live_mesh_objects(objects)
    before_vertices, before_faces = count_mesh_objects(meshes)
    ratio = float(ratio)
    if ratio <= 0.0:
        raise ValueError("Decimate ratio must be positive")

    if not enabled:
        return meshes, {
            "enabled": False,
            "ratio": float(ratio),
            "before_vertices": int(before_vertices),
            "before_faces": int(before_faces),
            "vertices": int(before_vertices),
            "faces": int(before_faces),
        }

    for obj in meshes:
        if obj.name not in bpy.data.objects or len(obj.data.polygons) <= 0:
            continue
        clean_mesh_geometry(obj)
        if ratio < 0.999:
            select_objects([obj], active=obj)
            modifier = obj.modifiers.new(name=f"{modifier_prefix}_{ratio_label(ratio)}", type="DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = ratio
            if hasattr(modifier, "use_collapse_triangulate"):
                modifier.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            clean_mesh_geometry(obj)

    meshes = delete_empty_mesh_objects(meshes)
    after_vertices, after_faces = count_mesh_objects(meshes)
    return meshes, {
        "enabled": True,
        "ratio": float(ratio),
        "before_vertices": int(before_vertices),
        "before_faces": int(before_faces),
        "vertices": int(after_vertices),
        "faces": int(after_faces),
    }


def _safe_texture_token(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return safe.strip("._") or "mesh"


def _make_vertex_color_bake_material(name: str, attr_name: str, image):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    try:
        color_node = nodes.new(type="ShaderNodeVertexColor")
        color_node.layer_name = attr_name
        color_output = color_node.outputs.get("Color")
    except Exception:
        color_node = nodes.new(type="ShaderNodeAttribute")
        color_node.attribute_name = attr_name
        color_output = color_node.outputs.get("Color")
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.image = image
    nodes.active = image_node
    if bsdf is not None and color_output is not None:
        material.node_tree.links.new(color_output, bsdf.inputs["Base Color"])
    return material


def _make_image_texture_material(name: str, image):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.image = image
    if bsdf is not None:
        material.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
    nodes.active = image_node
    return material


def bake_vertex_color_textures(
    objects: list,
    *,
    output_dir: Path,
    texture_stem: str,
    texture_size: int,
    margin_px: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "enabled": True,
        "texture_size": int(texture_size),
        "margin_px": int(margin_px),
        "textures": [],
        "objects_baked": 0,
        "objects_skipped": 0,
    }

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.render.bake.use_selected_to_active = False
    bpy.context.scene.render.bake.margin = int(margin_px)
    bpy.context.scene.render.bake.use_clear = True

    meshes = live_mesh_objects(objects)
    multiple = len(meshes) > 1
    for index, obj in enumerate(meshes):
        attr = first_color_attribute(obj)
        attr_name = getattr(attr, "name", None) if attr is not None else None
        if not attr_name:
            stats["objects_skipped"] += 1
            continue

        token = _safe_texture_token(obj.name if multiple else texture_stem)
        texture_name = f"{_safe_texture_token(texture_stem)}_{token}_color.png" if multiple else f"{_safe_texture_token(texture_stem)}_color.png"
        texture_path = output_dir / texture_name
        image = bpy.data.images.new(
            name=Path(texture_name).stem,
            width=int(texture_size),
            height=int(texture_size),
            alpha=False,
            float_buffer=False,
        )

        bake_material = _make_vertex_color_bake_material(f"{token}_vertex_color_bake", attr_name, image)
        obj.data.materials.clear()
        obj.data.materials.append(bake_material)
        select_objects([obj], active=obj)
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})

        image.filepath_raw = str(texture_path)
        image.file_format = "PNG"
        image.save()

        image_material = _make_image_texture_material(f"{token}_texture", image)
        obj.data.materials.clear()
        obj.data.materials.append(image_material)
        stats["textures"].append(str(texture_path))
        stats["objects_baked"] += 1

    return stats


def bake_vertex_color_sources_to_targets(
    source_objects: list,
    target_objects: list,
    *,
    output_dir: Path,
    texture_stem: str,
    texture_size: int,
    margin_px: int,
    cage_extrusion: float = 0.04,
    max_ray_distance: float = 0.12,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = live_mesh_objects(source_objects)
    targets = live_mesh_objects(target_objects)
    if not sources:
        raise RuntimeError("No high-resolution source mesh objects available for texture baking")
    if not targets:
        raise RuntimeError("No low-resolution target mesh objects available for texture baking")

    ensure_source_materials(sources)
    stats: dict[str, Any] = {
        "enabled": True,
        "mode": "selected_to_active_high_to_low",
        "texture_size": int(texture_size),
        "margin_px": int(margin_px),
        "cage_extrusion": float(cage_extrusion),
        "max_ray_distance": float(max_ray_distance),
        "source_objects": int(len(sources)),
        "target_objects": int(len(targets)),
        "textures": [],
        "objects_baked": 0,
    }

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = float(cage_extrusion)
    scene.render.bake.max_ray_distance = float(max_ray_distance)
    scene.render.bake.margin = int(margin_px)
    scene.render.bake.use_clear = True

    multiple = len(targets) > 1
    for index, target in enumerate(targets):
        token = _safe_texture_token(target.name if multiple else texture_stem)
        texture_name = (
            f"{_safe_texture_token(texture_stem)}_{token}_color.png"
            if multiple
            else f"{_safe_texture_token(texture_stem)}_color.png"
        )
        texture_path = output_dir / texture_name
        image = bpy.data.images.new(
            name=Path(texture_name).stem,
            width=int(texture_size),
            height=int(texture_size),
            alpha=False,
            float_buffer=False,
        )
        try:
            image.colorspace_settings.name = "sRGB"
        except Exception:
            pass

        image_material = _make_image_texture_material(f"{token}_texture", image)
        target.data.materials.clear()
        target.data.materials.append(image_material)

        bpy.ops.object.select_all(action="DESELECT")
        for source in sources:
            source.select_set(True)
        target.select_set(True)
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})

        image.filepath_raw = str(texture_path)
        image.file_format = "PNG"
        image.save()

        stats["textures"].append(str(texture_path))
        stats["objects_baked"] += 1

    return stats


def make_placeholder_texture(texture_path: Path) -> None:
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new(name=texture_path.stem, width=1, height=1, alpha=False, float_buffer=False)
    image.pixels[0:4] = (1.0, 1.0, 1.0, 1.0)
    image.filepath_raw = str(texture_path)
    image.file_format = "PNG"
    image.save()
