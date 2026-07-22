from pathlib import Path

import bpy


ROOT = Path("/workspace_whz")
OUT_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_existing_fitted_visible"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_obj(path: Path, name: str) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=str(path))
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


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.74
    return mat


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


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
    )


def main() -> None:
    clear_scene()
    base_mat = material("base_debug_warm_white", (0.72, 0.72, 0.68, 1.0))
    drawer_mat = material("drawer_debug_front", (0.94, 0.88, 0.78, 1.0))

    base = import_obj(OUT_DIR / "base_open_visible.obj", "base_open_visible")
    drawer_open = import_obj(OUT_DIR / "drawer_open_visible.obj", "drawer_open_visible")
    assign_material(base, base_mat)
    assign_material(drawer_open, drawer_mat)
    export_fbx(OUT_DIR / "cabinet_drawer_existing_fitted_open_visible.fbx", [base, drawer_open])

    clear_scene()
    base_mat = material("base_debug_warm_white", (0.72, 0.72, 0.68, 1.0))
    drawer_mat = material("drawer_debug_front", (0.94, 0.88, 0.78, 1.0))
    base = import_obj(OUT_DIR / "base_open_visible.obj", "base")
    drawer_child = import_obj(OUT_DIR / "drawer_closed_child_prismatic.obj", "drawer_closed_child")
    assign_material(base, base_mat)
    assign_material(drawer_child, drawer_mat)
    pivot = bpy.data.objects.new("drawer_slide_pivot", None)
    pivot.empty_display_type = "ARROWS"
    pivot.empty_display_size = 0.08
    pivot.location = (-0.012905526704909687, 0.02878868314194126, 1.2823)
    pivot["joint_type"] = "prismatic"
    pivot["axis_camera"] = [-0.7416754196065102, 0.28029929067613685, -0.6093848370266212]
    pivot["open_distance_m"] = 0.2581291664024917
    bpy.context.collection.objects.link(pivot)
    drawer_child.parent = pivot
    export_fbx(OUT_DIR / "cabinet_drawer_existing_fitted_closed_hierarchy.fbx", [base, pivot, drawer_child])


if __name__ == "__main__":
    main()
