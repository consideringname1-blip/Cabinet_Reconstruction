import bpy
import sys
import os
import shutil
import traceback

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def fix_mtl(mtl_path, tex_name):
    print(f"修正MTL: {mtl_path} 贴图名: {tex_name}")
    try:
        lines = []
        with open(mtl_path, 'r') as f:
            for line in f:
                if line.strip().startswith("map_Kd"):
                    lines.append(f"map_Kd {tex_name}\n")
                else:
                    lines.append(line)
        with open(mtl_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"修正MTL文件失败: {e}")
        traceback.print_exc()

def ensure_tex_in_obj_dir(obj_path, tex_path):
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.basename(tex_path)
    dst_tex = os.path.join(obj_dir, tex_name)
    if not os.path.exists(dst_tex):
        try:
            shutil.copy(tex_path, dst_tex)
            print(f"贴图已复制到: {dst_tex}")
        except Exception as e:
            print(f"贴图复制失败: {e}")
            traceback.print_exc()
    else:
        print(f"贴图已存在于: {dst_tex}")
    return dst_tex

def main():
    try:
        argv = sys.argv
        argv = argv[argv.index('--')+1:] if '--' in argv else []
        if len(argv) < 3:
            print("Usage: blender --background --python convert_obj_to_fbx.py -- input.obj input.png output.fbx")
            sys.exit(1)
        obj_path, tex_path, fbx_path = argv[:3]
        mtl_path = os.path.splitext(obj_path)[0]+".mtl"
        tex_name = os.path.basename(tex_path)
        print(f"OBJ路径: {obj_path}, 存在: {os.path.exists(obj_path)}")
        print(f"MTL路径: {mtl_path}, 存在: {os.path.exists(mtl_path)}")
        print(f"PNG路径: {tex_path}, 存在: {os.path.exists(tex_path)}")
        print(f"目标FBX路径: {fbx_path}")

        tex_in_obj_dir = ensure_tex_in_obj_dir(obj_path, tex_path)
        fix_mtl(mtl_path, tex_name)

        clean_scene()
        print("开始导入OBJ...")
        bpy.ops.import_scene.obj(filepath=obj_path, use_image_search=True)
        print("OBJ导入完成。")

        # ★★★ 你的真实缩放系数 ★★★
        # ==== ★ 在这里插入模型缩放 ★ ====
        scale_factor = 0.1   # TODO: 由服务器 depth 计算
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.scale = (scale_factor, scale_factor, scale_factor)
        print(f"模型缩放完成: factor = {scale_factor}")


        print("开始导出FBX...")
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            embed_textures=True,
            path_mode='COPY',
            axis_forward='-Z',
            axis_up='Y',
            bake_space_transform=True,
        )
        print(f"FBX导出完成: {fbx_path}, 存在: {os.path.exists(fbx_path)}")
    except Exception as e:
        print(f"Blender FBX转换流程异常: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
