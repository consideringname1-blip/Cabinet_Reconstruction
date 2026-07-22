import sys
from pathlib import Path

import bpy
import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: blender -b --python check_blender_glb_bounds.py -- mesh.glb")
    path = Path(sys.argv[-1])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.gltf(filepath=str(path))
    pts = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world.copy()
        for vert in obj.data.vertices:
            p = mw @ vert.co
            pts.append((p.x, p.y, p.z))
    arr = np.asarray(pts, dtype=np.float64)
    print("path", path)
    print("verts", len(arr))
    print("min", arr.min(axis=0).tolist())
    print("max", arr.max(axis=0).tolist())
    print("mean", arr.mean(axis=0).tolist())
    print("aabb_center", ((arr.min(axis=0) + arr.max(axis=0)) * 0.5).tolist())


if __name__ == "__main__":
    main()
