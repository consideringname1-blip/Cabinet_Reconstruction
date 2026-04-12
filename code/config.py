from pathlib import Path


# Runtime switches
HOLOLENS2_HOST = "10.40.1.132"
IS_RUN_FLASK_SERVER = True

# Depth / alignment tuning
# Fraction cropped inward from the SAM3 mask periphery before depth->pointcloud
# conversion. Set to 0.0 to disable.
ICP_DEPTH_BORDER_CROP_RATIO = 0.03
# Reject candidate poses that leave the reconstructed model upside-down in
# camera-local Unity space.
ICP_IGNORE_INVERTED_SOLUTIONS = True
# When enabled, ICP and coarse search only use the nearest model surface along
# the camera ray and ignore occluded model geometry behind it.
ICP_IGNORE_OCCLUDED_MODEL_POINTS = True


# Project roots
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CODE_ROOT = PROJECT_ROOT / "code"
STAGES_ROOT = CODE_ROOT / "stages"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
RECON_ROOT = CODE_ROOT / "reconstruction"
HOLOLENS_ROOT = CODE_ROOT / "Hololens2"

DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_FOLDER = DATA_ROOT / "upload"
OUTPUT_ROOT = DATA_ROOT / "output"
ENV_CONFIG_ROOT = DATA_ROOT / "config"
DATABASE_ROOT = DATA_ROOT / "database"
MODELS_ROOT = PROJECT_ROOT / "models"


# Python runtimes
IMESH_PY = "/opt/miniconda/envs/imesh/bin/python"
SERVER_PY = "/opt/miniconda/envs/server/bin/python"
SAM3_PY = "/opt/miniconda/envs/sam3/bin/python"
HOLOLENS2_PY = SERVER_PY


# App entrypoints
SERVER_API_RUN = CODE_ROOT / "server_api.py"
FLASK_SERVER = SERVER_API_RUN


# HoloLens tools
HOLOLENS2_DOWNLOAD_DIR = HOLOLENS_ROOT / "DownloadHololens2CameraCalibration"
CALIBRATION_DIR = HOLOLENS2_DOWNLOAD_DIR / "hl2ss_calib"
HOLOLENS2_DOWNLOAD_RUN = HOLOLENS2_DOWNLOAD_DIR / "download_calibration_all.py"

HOLOLENS2_CONVERT_DIR = HOLOLENS_ROOT / "DepthConvertToRGB"
HOLOLENS2_CONVERT_RUN = HOLOLENS2_CONVERT_DIR / "align_pv_depth.py"


# Reconstruction integrations
INSTANTMESH_DIR = RECON_ROOT / "InstantMesh"
INSTANTMESH_CONFIG = INSTANTMESH_DIR / "configs" / "instant-mesh-large.yaml"
INSTANTMESH_RUN_PY = INSTANTMESH_DIR / "run.py"

SAM3_ROOT = RECON_ROOT / "sam3"
SAM3_DIR = SAM3_ROOT / "sam3"
SAM3_BEP = SAM3_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"


# Stage scripts
SAM3_BOX_MASK_RUN = STAGES_ROOT / "run_sam3_boxmask_from_json.py"
INSTANTMESH_STAGE_RUN = STAGES_ROOT / "run_instantmesh_from_json.py"
DEPTHPOINTCLOUD_STAGE_RUN = STAGES_ROOT / "run_depthpointcloud_from_json.py"
MODELSCALE_STAGE_RUN = STAGES_ROOT / "run_model_scale_from_json.py"
ICPALIGNMENT_STAGE_RUN = STAGES_ROOT / "run_object_icp_alignment_from_json.py"
POSE_STAGE_RUN = STAGES_ROOT / "run_pose_from_json.py"
BLENDER_STAGE_RUN = STAGES_ROOT / "run_blender_from_json.py"
CONVERT_SCRIPT = STAGES_ROOT / "convert_obj_to_fbx.py"

INSTANTMESH_STAGE_PY = SERVER_PY
DEPTHPOINTCLOUD_STAGE_PY = SERVER_PY
MODELSCALE_STAGE_PY = SERVER_PY
ICPALIGNMENT_STAGE_PY = SERVER_PY
POSE_STAGE_PY = SERVER_PY
BLENDER_STAGE_PY = SERVER_PY


# Storage
DATABASE_PATH = DATABASE_ROOT / "tasks.db"

HOLOLENS2_OUTPUT_DEPTH_IMAGES = OUTPUT_ROOT / "hololens2"

INSTANTMESH_OUTPUT = OUTPUT_ROOT / "instant-mesh-large"
INSTANTMESH_INPUT_ROOT = OUTPUT_ROOT / "instantmesh-input"
INSTANTMESH_OUTPUT_MESHES = INSTANTMESH_OUTPUT / "meshes"
INSTANTMESH_OUTPUT_IMAGES = INSTANTMESH_OUTPUT / "images"
INSTANTMESH_OUTPUT_VIDEOS = INSTANTMESH_OUTPUT / "videos"

SAM3_OUTPUT_ROOT = OUTPUT_ROOT / "sam3"
OBJECT_ALIGNMENT_OUTPUT_ROOT = OUTPUT_ROOT / "object_alignment"

BLENDER_OUTPUT_ROOT = OUTPUT_ROOT / "blender"
BLENDER_FBX_DIR = BLENDER_OUTPUT_ROOT / "fbx"
BLENDER_BIN = "/usr/local/bin/blender"


# HTTP file serving
FOLDER_MAP = {
    "meshes": INSTANTMESH_OUTPUT_MESHES,
    "images": INSTANTMESH_OUTPUT_IMAGES,
    "videos": INSTANTMESH_OUTPUT_VIDEOS,
    "fbx": BLENDER_FBX_DIR,
}


# Bootstrap directories
for path in [
    UPLOAD_FOLDER,
    ENV_CONFIG_ROOT,
    DATABASE_ROOT,
    INSTANTMESH_INPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    INSTANTMESH_OUTPUT_VIDEOS,
    SAM3_OUTPUT_ROOT,
    OBJECT_ALIGNMENT_OUTPUT_ROOT,
    BLENDER_FBX_DIR,
    HOLOLENS2_OUTPUT_DEPTH_IMAGES,
]:
    path.mkdir(parents=True, exist_ok=True)
