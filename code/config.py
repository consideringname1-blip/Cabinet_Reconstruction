import os
from pathlib import Path


def _resolve_icp_mode() -> str:
    raw = os.environ.get("ICP_MODE")
    if raw is not None:
        value = str(raw).strip().lower()
        if value in {"camera_refine", "off"}:
            return value
        raise ValueError("ICP_MODE must be one of: camera_refine / off")
    return "camera_refine"


# Runtime switches
HOLOLENS2_HOST = "10.40.1.132"
IS_RUN_FLASK_SERVER = True

# Depth / alignment tuning
# Fraction cropped inward from the SAM3 mask periphery before depth->pointcloud
# conversion. Set to 0.0 to disable.
ICP_DEPTH_BORDER_CROP_RATIO = 0.03
# camera_refine = camera-view local rotation+translation+scale adjustment.
# off = measured-distance placement without ICP. The runtime only keeps one
# active ICP path plus the skip-ICP path.
# Default now uses camera_refine.
ICP_MODE = _resolve_icp_mode()
ICP_ENABLE = ICP_MODE != "off"
# Maximum number of points written to the exported depth point cloud PLY.
# Set to 0 or None to keep all valid depth points.
DEPTHPOINTCLOUD_MAX_EXPORT_POINTS = 6000
# Enable preview/front-view PNG renders produced by the alignment pipeline.
ENABLE_ALIGNMENT_RENDER_OUTPUTS = True
# Enable InstantMesh circular-view MP4 generation.
ENABLE_INSTANTMESH_VIDEO_OUTPUT = True
# Reject candidate poses that leave the reconstructed model upside-down in
# camera-local Unity space.
ICP_IGNORE_INVERTED_SOLUTIONS = True
# When enabled, ICP and coarse search only use the nearest model surface along
# the camera ray and ignore occluded model geometry behind it.
ICP_IGNORE_OCCLUDED_MODEL_POINTS = True
ICP_TARGET_FRONT_MAX_POINTS = 3600
ICP_ALIGNMENT_MODEL_MAX_POINTS = 2800
ICP_COARSE_VISIBLE_MAX_POINTS = 2600
ICP_MEDIUM_VISIBLE_MAX_POINTS = 3200
ICP_FINE_VISIBLE_MAX_POINTS = 3600
ICP_LOCAL_REFINE_VISIBLE_MAX_POINTS = 3600
ICP_FINAL_VISIBLE_MAX_POINTS = 3600
ICP_ACCELERATION_DEVICE = "auto"
ICP_COARSE_SCALE_EVAL_KEEP = 3
ICP_FINAL_SCALE_CANDIDATE_KEEP = 4
ICP_LOCAL_REFINE_CANDIDATE_KEEP = 96
ICP_FINAL_ITERATIONS = 12
ICP_LOCAL_REFINE_ITERATIONS = 6
ICP_COARSE_CANDIDATE_KEEP = 3
ICP_AXIS_SEED_RETAIN_TOPK = 3
ICP_MEDIUM_RETAIN_TOPK = 2
ICP_CAMERA_REFINE_MAX_ROTATION_DELTA_DEG = 18.0
ICP_CAMERA_REFINE_SCALE_DELTA_RATIO = 0.12
ICP_CAMERA_REFINE_SEED_KEEP = 9
# Penalize rotations that drift too far from the InstantMesh identity
# orientation. 180-degree flips receive the full weight; smaller rotations are
# scaled quadratically by angle / pi.
ICP_INITIAL_ROTATION_PENALTY_WEIGHT = 0.005


# Project roots
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CODE_ROOT = PROJECT_ROOT / "code"
STAGES_ROOT = CODE_ROOT / "stages"
HOLOLENS3D_RECON_STAGE_ROOT = STAGES_ROOT / "hololens3d_reconstruction"
ARUCO_STAGE_ROOT = STAGES_ROOT / "hololens_aruco_reference"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
RECON_ROOT = CODE_ROOT / "reconstruction"
HOLOLENS_ROOT = CODE_ROOT / "Hololens2"

DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_FOLDER = DATA_ROOT / "upload"
OUTPUT_ROOT = DATA_ROOT / "output"
ENV_CONFIG_ROOT = DATA_ROOT / "config"
DATABASE_ROOT = DATA_ROOT / "database"
ARUCO_DATA_ROOT = DATA_ROOT / "aruco"
ARUCO_TEMPLATE_PATH = ARUCO_DATA_ROOT / "aruco_template.json"
ARUCO_RAW_ROOT = ARUCO_DATA_ROOT / "raw"
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
SAM3_BOX_MASK_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_sam3_boxmask_from_json.py"
ARUCO_DETECT_STAGE_RUN = ARUCO_STAGE_ROOT / "run_aruco_detect_from_json.py"
INSTANTMESH_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_instantmesh_from_json.py"
DEPTHPOINTCLOUD_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_depthpointcloud_from_json.py"
MODELSCALE_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_model_scale_from_json.py"
ICPALIGNMENT_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_object_icp_alignment_from_json.py"
POSE_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_pose_from_json.py"
ARUCO_SYNC_STAGE_RUN = ARUCO_STAGE_ROOT / "run_aruco_sync_from_json.py"
BLENDER_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_blender_from_json.py"
CONVERT_SCRIPT = HOLOLENS3D_RECON_STAGE_ROOT / "convert_obj_to_fbx.py"

ARUCO_STAGE_PY = SERVER_PY
ARUCO_DETECT_STAGE_PY = ARUCO_STAGE_PY
ARUCO_SYNC_STAGE_PY = ARUCO_STAGE_PY
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
    ARUCO_DATA_ROOT,
    ARUCO_RAW_ROOT,
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
