import os
from pathlib import Path


# Runtime switches
HOLOLENS2_HOST = "10.40.1.132"
IS_RUN_FLASK_SERVER = True
# Model generation backend for the shared model-generation stage slot.
# Use "sam3d_objects" or "instantmesh".
MODEL_GENERATION_BACKEND = "instantmesh"
MODEL_SERVICE_PREWARM_ENABLE = os.environ.get("MODEL_SERVICE_PREWARM_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
SHIGURE_HISTORY_RECORDING_ENABLE = os.environ.get("SHIGURE_HISTORY_RECORDING_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}

# Depth camera and ArUco server defaults
AHAT_SENSOR_NAME = "AHAT"
AHAT_MIN_DEPTH_MM = 200
AHAT_MAX_RELIABLE_DEPTH_MM = 1000
AHAT_MIN_USABLE_DEPTH_PIXELS = 4096
AHAT_MAX_UPLOAD_PNG_BYTES = 450000
AHAT_ENABLE_UPLOAD_GUARD = False
ARUCO_ROI_PADDING_RATIO = 0.18
ARUCO_ROI_PADDING_MIN_PX = 24
ARUCO_ANCHOR_MARKER_ID = 1
ARUCO_SYNC_MARKER_REGISTRY_ON_START = False
# Legacy compatibility knob; task_worker serializes InstantMesh to one
# persistent service because the underlying model process is GPU-heavy.
INSTANTMESH_MAX_WORKERS = int(os.environ.get("INSTANTMESH_MAX_WORKERS", "1"))
_instantmesh_gpu_ids = []
for gpu_id in os.environ.get("INSTANTMESH_GPU_IDS", "0,1,2").split(","):
    gpu_id = gpu_id.strip()
    if gpu_id and gpu_id not in _instantmesh_gpu_ids:
        _instantmesh_gpu_ids.append(gpu_id)
INSTANTMESH_GPU_IDS = tuple(_instantmesh_gpu_ids)

# Project roots
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CODE_ROOT = PROJECT_ROOT / "code"
STAGES_ROOT = CODE_ROOT / "stages"
HOLOLENS3D_RECON_STAGE_ROOT = STAGES_ROOT / "hololens3d_reconstruction"
ARUCO_STAGE_ROOT = STAGES_ROOT / "hololens_aruco_reference"
SHIGURE_HISTORY_STAGE_ROOT = STAGES_ROOT / "shigure_history"
HISTORY_PLACEMENT_STAGE_ROOT = STAGES_ROOT / "history_placement_restoration"
TAKEN_OBJECT_STAGE_ROOT = STAGES_ROOT / "taken_object_detection"
SAM3D_BODY_STAGE_ROOT = STAGES_ROOT / "sam3d_body_mesh"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
RECON_ROOT = CODE_ROOT / "reconstruction"
HOLOLENS_ROOT = CODE_ROOT / "Hololens2"

DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_FOLDER = DATA_ROOT / "upload"
OUTPUT_ROOT = DATA_ROOT / "output"
ENV_CONFIG_ROOT = DATA_ROOT / "config"
DATABASE_ROOT = DATA_ROOT / "database"
WORKER_SOCKET_ROOT = DATA_ROOT / "worker_sockets"
SHIGURE_HISTORY_CACHE_ROOT = DATA_ROOT / "shigure_history_cache"
ARUCO_DATA_ROOT = DATA_ROOT / "aruco"
ARUCO_REFERENCE_ROOT = ARUCO_DATA_ROOT / "reference"
ARUCO_RUNTIME_ROOT = ARUCO_DATA_ROOT / "runtime"
SHIGURE_MARKER_HISTORY_ROOT = ARUCO_DATA_ROOT / "shigure_marker_history"
SHIGURE_MARKER_HISTORY_PATH = SHIGURE_MARKER_HISTORY_ROOT / "latest_marker_6d_pose.json"
ARUCO_TEMPLATE_PATH = ARUCO_REFERENCE_ROOT / "aruco.json"
ARUCO_REFERENCE_MARKER_IMAGE_PATH = ARUCO_REFERENCE_ROOT / "ar_marker_7x7_1.png"
ARUCO_RAW_ROOT = ARUCO_RUNTIME_ROOT
MODELS_ROOT = PROJECT_ROOT / "models"


# Python runtimes
IMESH_PY = "/opt/miniconda/envs/imesh/bin/python"
SERVER_PY = "/opt/miniconda/envs/server/bin/python"
SAM3_PY = "/opt/miniconda/envs/sam3/bin/python"
SAM3D_OBJECTS_PY = "/opt/miniconda/envs/sam3d-objects-cu118/bin/python"
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

SAM3D_OBJECTS_ROOT = RECON_ROOT / "sam-3d-objects"
SAM3D_OBJECTS_CONFIG = SAM3D_OBJECTS_ROOT / "checkpoints" / "hf" / "pipeline.yaml"
SAM3D_BODY_ROOT = RECON_ROOT / "sam3d-body"


# Stage scripts
SAM3_BOX_MASK_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_sam3_boxmask_from_json.py"
ARUCO_DETECT_STAGE_RUN = ARUCO_STAGE_ROOT / "run_aruco_detect_from_json.py"
INSTANTMESH_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_instantmesh_from_json.py"
SAM3D_OBJECTS_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_sam3d_objects_from_json.py"
DEPTHPOINTCLOUD_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_depthpointcloud_from_json.py"
MODELSCALE_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_model_scale_from_json.py"
OBJECT_ALIGNMENT_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_object_alignment_from_json.py"
ICPALIGNMENT_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_object_icp_alignment_from_json.py"  # legacy wrapper
FOUNDATIONPOSE_ALIGNMENT_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_foundationpose_alignment_worker.py"
POSE_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_pose_from_json.py"
ARUCO_SYNC_STAGE_RUN = ARUCO_STAGE_ROOT / "run_aruco_sync_from_json.py"
RUNTIME_MESH_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_runtime_mesh_from_json.py"
RUNTIME_MESH_BAKE_SCRIPT = HOLOLENS3D_RECON_STAGE_ROOT / "bake_runtime_mesh.py"
SAM3D_OBJECTS_POSTPROCESS_SCRIPT = HOLOLENS3D_RECON_STAGE_ROOT / "postprocess_sam3d_glb.py"
BLENDER_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_blender_from_json.py"
MODEL_BOUNDS_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_model_bounds_from_json.py"
SHIGURE_HISTORY_RECORDER_RUN = SHIGURE_HISTORY_STAGE_ROOT / "run_shigure_history_recorder.py"
HISTORY_PLACEMENT_RESTORATION_STAGE_RUN = HISTORY_PLACEMENT_STAGE_ROOT / "run_history_placement_restoration_from_json.py"
TAKEN_OBJECT_DETECTION_STAGE_RUN = TAKEN_OBJECT_STAGE_ROOT / "run_taken_object_detection_from_json.py"
SAM3D_BODY_MESH_STAGE_RUN = SAM3D_BODY_STAGE_ROOT / "run_sam3d_body_mesh_from_json.py"
SAM3D_BODY_FBX_EXPORT_SCRIPT = SAM3D_BODY_STAGE_ROOT / "export_selected_body_fbx.py"
CONVERT_SCRIPT = HOLOLENS3D_RECON_STAGE_ROOT / "convert_obj_to_fbx.py"

ARUCO_STAGE_PY = SERVER_PY
ARUCO_DETECT_STAGE_PY = ARUCO_STAGE_PY
ARUCO_SYNC_STAGE_PY = ARUCO_STAGE_PY
INSTANTMESH_STAGE_PY = SERVER_PY
SAM3D_OBJECTS_STAGE_PY = SERVER_PY
DEPTHPOINTCLOUD_STAGE_PY = SERVER_PY
MODELSCALE_STAGE_PY = SERVER_PY
OBJECT_ALIGNMENT_STAGE_PY = SERVER_PY
ICPALIGNMENT_STAGE_PY = SERVER_PY  # legacy wrapper
FOUNDATIONPOSE_ALIGNMENT_PY = "/opt/miniconda/envs/foundationpose/bin/python"
POSE_STAGE_PY = SERVER_PY
RUNTIME_MESH_STAGE_PY = SERVER_PY
BLENDER_STAGE_PY = SERVER_PY
MODEL_BOUNDS_STAGE_PY = SERVER_PY
SHIGURE_HISTORY_RECORDER_STAGE_PY = os.environ.get("SHIGURE_HISTORY_RECORDER_PY", "/usr/bin/python3")
HISTORY_PLACEMENT_RESTORATION_STAGE_PY = SERVER_PY
TAKEN_OBJECT_DETECTION_STAGE_PY = SERVER_PY
SAM3D_BODY_MESH_STAGE_PY = os.environ.get("SAM3D_BODY_PY", SERVER_PY)


# Storage
DATABASE_PATH = DATABASE_ROOT / "tasks.db"

HOLOLENS2_OUTPUT_DEPTH_IMAGES = OUTPUT_ROOT / "hololens2"

INSTANTMESH_OUTPUT = OUTPUT_ROOT / "instant-mesh-large"
INSTANTMESH_INPUT_ROOT = OUTPUT_ROOT / "instantmesh-input"
INSTANTMESH_OUTPUT_MESHES = INSTANTMESH_OUTPUT / "meshes"
INSTANTMESH_OUTPUT_IMAGES = INSTANTMESH_OUTPUT / "images"
INSTANTMESH_OUTPUT_VIDEOS = INSTANTMESH_OUTPUT / "videos"

SAM3_OUTPUT_ROOT = OUTPUT_ROOT / "sam3"
SAM3D_OBJECTS_OUTPUT = OUTPUT_ROOT / "sam3d-objects"
SAM3D_OBJECTS_OUTPUT_MESHES = SAM3D_OBJECTS_OUTPUT / "meshes"
OBJECT_ALIGNMENT_OUTPUT_ROOT = OUTPUT_ROOT / "object_alignment"
HISTORY_PLACEMENT_OUTPUT_ROOT = OUTPUT_ROOT / "history_placement_restoration"
TAKEN_OBJECT_OUTPUT_ROOT = OUTPUT_ROOT / "taken_object_detection"
SAM3D_BODY_OUTPUT_ROOT = OUTPUT_ROOT / "sam3d_body"
SAM3D_BODY_OUTPUT_MESHES = SAM3D_BODY_OUTPUT_ROOT / "meshes"
SAM3D_BODY_OUTPUT_FBX = SAM3D_BODY_OUTPUT_ROOT / "fbx"

# Persistent model worker idle release windows.
SAM3MASK_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("SAM3MASK_WORKER_IDLE_TIMEOUT_SEC", "300"))
INSTANTMESH_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("INSTANTMESH_WORKER_IDLE_TIMEOUT_SEC", "300"))
FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC = int(
    os.environ.get("FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC", "300")
)

BLENDER_OUTPUT_ROOT = OUTPUT_ROOT / "blender"
BLENDER_FBX_DIR = BLENDER_OUTPUT_ROOT / "fbx"
BLENDER_BIN = "/usr/local/bin/blender"

RUNTIME_MESH_OUTPUT_ROOT = OUTPUT_ROOT / "runtime_mesh"


# HTTP file serving
FOLDER_MAP = {
    "meshes": INSTANTMESH_OUTPUT_MESHES,
    "images": INSTANTMESH_OUTPUT_IMAGES,
    "videos": INSTANTMESH_OUTPUT_VIDEOS,
    "sam3d_object_meshes": SAM3D_OBJECTS_OUTPUT_MESHES,
    "runtime_meshes": RUNTIME_MESH_OUTPUT_ROOT,
    "fbx": BLENDER_FBX_DIR,
    "history_placement_restoration": HISTORY_PLACEMENT_OUTPUT_ROOT,
    "sam3d_body_meshes": SAM3D_BODY_OUTPUT_MESHES,
    "sam3d_body_fbx": SAM3D_BODY_OUTPUT_FBX,
}


# Bootstrap directories
for path in [
    UPLOAD_FOLDER,
    ENV_CONFIG_ROOT,
    DATABASE_ROOT,
    ARUCO_DATA_ROOT,
    ARUCO_REFERENCE_ROOT,
    ARUCO_RUNTIME_ROOT,
    SHIGURE_MARKER_HISTORY_ROOT,
    ARUCO_RAW_ROOT,
    INSTANTMESH_INPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    INSTANTMESH_OUTPUT_VIDEOS,
    SAM3_OUTPUT_ROOT,
    SAM3D_OBJECTS_OUTPUT_MESHES,
    OBJECT_ALIGNMENT_OUTPUT_ROOT,
    HISTORY_PLACEMENT_OUTPUT_ROOT,
    TAKEN_OBJECT_OUTPUT_ROOT,
    SAM3D_BODY_OUTPUT_MESHES,
    SAM3D_BODY_OUTPUT_FBX,
    RUNTIME_MESH_OUTPUT_ROOT,
    BLENDER_FBX_DIR,
    HOLOLENS2_OUTPUT_DEPTH_IMAGES,
    SHIGURE_HISTORY_CACHE_ROOT,
]:
    path.mkdir(parents=True, exist_ok=True)
