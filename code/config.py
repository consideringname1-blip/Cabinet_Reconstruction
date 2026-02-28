# config.py
from pathlib import Path

# ========================= 项目根目录 =========================
# config.py 位于 /workspace/code
BASE_DIR = Path(__file__).resolve().parent       # /workspace/code
PROJECT_ROOT = BASE_DIR.parent                   # /workspace

# ========================= 主要目录 =========================
CODE_ROOT = PROJECT_ROOT / "code"                # /workspace/code
RECON_ROOT = CODE_ROOT / "reconstruction"        # /workspace/code/reconstruction
DATA_ROOT = PROJECT_ROOT / "data"                # /workspace/data
MODELS_ROOT = PROJECT_ROOT / "models"            # /workspace/models

# ===== 每个 conda 环境对应的 python 解释器 =====
IMESH_PY    = "/opt/miniconda/envs/imesh/bin/python"      # InstantMesh 用
SERVER_PY    = "/opt/miniconda/envs/server/bin/python"    # Flask / API 用（只给你参考）

# ========================= InstantMesh 路径 =========================
INSTANTMESH_DIR = RECON_ROOT / "InstantMesh"
INSTANTMESH_CONFIG = INSTANTMESH_DIR / "configs" / "instant-mesh-large.yaml"
INSTANTMESH_RUN_PY = INSTANTMESH_DIR / "run.py"

# ========================= 上传目录 =========================
UPLOAD_FOLDER = DATA_ROOT / "upload"             # /workspace/data/upload

# ========================= 输出目录（未来多个模型共用） =========================
OUTPUT_ROOT = DATA_ROOT / "output"               # /workspace/data/output

# InstantMesh 输出
INSTANTMESH_OUTPUT = OUTPUT_ROOT / "instant-mesh-large"
INSTANTMESH_OUTPUT_MESHES = INSTANTMESH_OUTPUT / "meshes"
INSTANTMESH_OUTPUT_IMAGES = INSTANTMESH_OUTPUT / "images"
INSTANTMESH_OUTPUT_VIDEOS = INSTANTMESH_OUTPUT / "videos"

# Blender 输出
BLENDER_OUTPUT_ROOT = OUTPUT_ROOT / "blender"
BLENDER_FBX_DIR = BLENDER_OUTPUT_ROOT / "fbx"

# SAM3 输出（预留）
SAM2_OUTPUT_ROOT = OUTPUT_ROOT / "sam3"

# ========================= Blender 可执行路径 =========================
BLENDER_BIN = "/usr/bin/blender"
CONVERT_SCRIPT = CODE_ROOT / "convert_obj_to_fbx.py"

# ========================= 文件映射（接口用） =========================
FOLDER_MAP = {
    "meshes": INSTANTMESH_OUTPUT_MESHES,
    "images": INSTANTMESH_OUTPUT_IMAGES,
    "videos": INSTANTMESH_OUTPUT_VIDEOS,
    "fbx": BLENDER_FBX_DIR,
}

# ========================= 自动创建目录 =========================
for p in [
    UPLOAD_FOLDER,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    INSTANTMESH_OUTPUT_VIDEOS,
    BLENDER_FBX_DIR,
    SAM2_OUTPUT_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)
