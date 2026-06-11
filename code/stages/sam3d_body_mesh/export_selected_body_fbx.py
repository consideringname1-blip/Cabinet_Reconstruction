from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def clean_scene() -> None:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def import_obj(path: str):
    before = {obj.name for obj in bpy.context.scene.objects}
    if hasattr(bpy.ops.wm, 'obj_import'):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)
    return [obj for obj in bpy.context.scene.objects if obj.name not in before]


def configure_material(color, alpha: float):
    mat = bpy.data.materials.new('sam3d_body_selected_black')
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    mat.use_screen_refraction = True
    rgba = (float(color[0]), float(color[1]), float(color[2]), float(alpha))
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf is not None:
        if 'Base Color' in bsdf.inputs:
            bsdf.inputs['Base Color'].default_value = rgba
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = float(alpha)
    mat.diffuse_color = rgba
    return mat


def decimate_meshes(meshes, ratio: float) -> None:
    ratio = max(0.01, min(1.0, float(ratio)))
    if ratio >= 0.999:
        return
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        mod = obj.modifiers.new('sam3d_body_decimate', 'DECIMATE')
        mod.ratio = ratio
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass
        obj.select_set(False)


def main(argv: list[str]) -> int:
    args = argv[argv.index('--') + 1:] if '--' in argv else argv[1:]
    if len(args) != 1:
        print('Usage: blender --background --python export_selected_body_fbx.py -- <config.json>', file=sys.stderr)
        return 2
    cfg = json.loads(Path(args[0]).read_text(encoding='utf-8'))
    clean_scene()
    imported = import_obj(cfg['obj_path'])
    meshes = [obj for obj in imported if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError('selected body OBJ contains no mesh')
    mat = configure_material(cfg.get('material_color') or [0.0, 0.0, 0.0], float(cfg.get('material_alpha', 0.5)))
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    decimate_meshes(meshes, float(cfg.get('decimate_ratio', 0.125)))
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    Path(cfg['fbx_path']).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=cfg['fbx_path'],
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=False,
        object_types={'MESH'},
        path_mode='AUTO',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
