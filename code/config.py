# config.py
from pathlib import Path

# ========================= 从Hololens2下载数据时设备的ip(使用前修改) =========================
HOLOLENS2_HOST = "10.40.1.132"

# ========================= 现在是flask客户端(模型生成客户端) =========================
IS_RUN_FLASK_SERVER = True

# ========================= 项目根目录 =========================
# config.py 位于 /workspace/code
BASE_DIR = Path(__file__).resolve().parent       # /workspace/code
PROJECT_ROOT = BASE_DIR.parent                   # /workspace

# ========================= 主要目录 =========================
CODE_ROOT = PROJECT_ROOT / "code"                # /workspace/code
RECON_ROOT = CODE_ROOT / "reconstruction"        # /workspace/code/reconstruction
DATA_ROOT = PROJECT_ROOT / "data"                # /workspace/data
ENV_CONFIG_ROOT = DATA_ROOT / "config"           # /workspace/data/config
MODELS_ROOT = PROJECT_ROOT / "models"            # /workspace/models
HOLOLENS_ROOT = CODE_ROOT / "Hololens2"          # /workspace/code/Hololens2

# ===== 每个 conda 环境对应的 python 解释器 =====
IMESH_PY = "/opt/miniconda/envs/imesh/bin/python"      # InstantMesh 用
SERVER_PY  = "/opt/miniconda/envs/server/bin/python"    # Flask / API 用
HOLOLENS2_PY = SERVER_PY    # hololens2 用，和server用一个 （还未实装）

# ========================= 入口flask程序 =========================
FLASK_SERVER = CODE_ROOT / "generateModel.py"

# ========================= InstantMesh 路径 =========================
INSTANTMESH_DIR = RECON_ROOT / "InstantMesh"
INSTANTMESH_CONFIG = INSTANTMESH_DIR / "configs" / "instant-mesh-large.yaml"
INSTANTMESH_RUN_PY = INSTANTMESH_DIR / "run.py"

# ========================= Hololens2 路径 =========================
HOLOLENS2_DOWNLOAD_DIR = HOLOLENS_ROOT / "DownloadHololens2CameraCalibration"
CALIBRATION_DIR = HOLOLENS2_DOWNLOAD_DIR / "hl2ss_calib"
HOLOLENS2_DOWNLOAD = HOLOLENS2_DOWNLOAD_DIR / "download_calibration_all.py"

HOLOLENS2_CONVERT_DIR = HOLOLENS_ROOT / "DepthConvertToRGB"
HOLOLENS2_CONVERT = HOLOLENS2_CONVERT_DIR / "align_pv_depth.py"

# ========================= 上传目录 =========================
UPLOAD_FOLDER = DATA_ROOT / "upload"             # /workspace/data/upload

# ========================= 输出目录（未来多个模型共用） =========================
OUTPUT_ROOT = DATA_ROOT / "output"               # /workspace/data/output

# Hololens2 输出
HOLOLENS2_OUTPUT_DEPTH_IMAGES = OUTPUT_ROOT / "hololens2"

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
    ENV_CONFIG_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    INSTANTMESH_OUTPUT_VIDEOS,
    BLENDER_FBX_DIR,
    SAM2_OUTPUT_ROOT,
    HOLOLENS2_OUTPUT_DEPTH_IMAGES,
]:
    p.mkdir(parents=True, exist_ok=True)
