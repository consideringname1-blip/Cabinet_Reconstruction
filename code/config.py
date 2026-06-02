import os
from pathlib import Path


# Runtime switches
HOLOLENS2_HOST = "10.40.1.132"
IS_RUN_FLASK_SERVER = True
# Model generation backend for the shared model-generation stage slot.
# Use "sam3d_objects" or "instantmesh".
MODEL_GENERATION_BACKEND = "instantmesh"

# Depth / alignment tuning
# Fraction cropped inward from the SAM3 mask periphery before depth->pointcloud
# conversion. Set to 0.0 to disable.
ICP_DEPTH_BORDER_CROP_RATIO = 0.03
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
# foundationpose = model-based FoundationPose registration from RGB-D + mask.
# camera_refine = existing camera-view ICP rotation+translation+scale refinement.
# off = measured-distance placement without ICP.
OBJECT_ALIGNMENT_MODE = "foundationpose"
ICP_MODE = OBJECT_ALIGNMENT_MODE  # legacy field consumed by older debug/output code
ICP_ENABLE = OBJECT_ALIGNMENT_MODE == "camera_refine"
# Maximum number of points written to the exported depth point cloud PLY.
# Set to 0 or None to keep all valid depth points.
DEPTHPOINTCLOUD_MAX_EXPORT_POINTS = 6000
# Enable preview/front-view PNG renders produced by the alignment pipeline.
# Disabled by default so ICP_MODE=off stays fast.
ENABLE_ALIGNMENT_RENDER_OUTPUTS = False
# Enable InstantMesh circular-view MP4 generation.
ENABLE_INSTANTMESH_VIDEO_OUTPUT = True
# Maximum number of InstantMesh stage workers. If INSTANTMESH_GPU_IDS is set,
# workers are capped to the number of listed GPU ids.
INSTANTMESH_MAX_WORKERS = int(os.environ.get("INSTANTMESH_MAX_WORKERS", "3"))
INSTANTMESH_GPU_IDS = tuple(
    gpu_id.strip()
    for gpu_id in os.environ.get("INSTANTMESH_GPU_IDS", "").split(",")
    if gpu_id.strip()
)
# Remove tiny disconnected mesh islands immediately after InstantMesh export.
INSTANTMESH_CLEAN_ENABLE = True
INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO = 0.01
INSTANTMESH_CLEAN_COMPONENT_MIN_FACES = 32
# SAM3D generation does not pass simplification/step/format limits; these are runtime/compat knobs, not quality caps.
SAM3D_OBJECTS_SEED = int(os.environ.get("SAM3D_OBJECTS_SEED", "42"))
SAM3D_OBJECTS_ATTN_BACKEND = os.environ.get("SAM3D_OBJECTS_ATTN_BACKEND", "sdpa")
# Parameters for ICP_MODE=off bbox-front-surface placement.
# The current values were fitted against the latest five camera_refine results:
# avg position delta ~= 2.8 cm, max ~= 5.9 cm.
ICP_BBOX_SURFACE_RAY_SOURCE = "all_points"
ICP_BBOX_SURFACE_DISTANCE_MODE = "mean_depth"
ICP_BBOX_SURFACE_LATERAL_MODE = "centroid_xy"
ICP_BBOX_SURFACE_THICKNESS_FACTOR = 0.20
# Camera rotation components used when ICP_MODE=off computes the final runtime
# pose. Yaw-only keeps the model upright while preserving the camera's
# horizontal facing direction; enabling pitch/roll restores more of the
# original camera tilt.
SKIP_ICP_POSE_USE_CAMERA_YAW = True
SKIP_ICP_POSE_USE_CAMERA_PITCH = False
SKIP_ICP_POSE_USE_CAMERA_ROLL = False
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
WORKER_SOCKET_ROOT = DATA_ROOT / "worker_sockets"
ARUCO_DATA_ROOT = DATA_ROOT / "aruco"
ARUCO_REFERENCE_ROOT = ARUCO_DATA_ROOT / "reference"
ARUCO_RUNTIME_ROOT = ARUCO_DATA_ROOT / "runtime"
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
FOUNDATIONPOSE_EST_REFINE_ITER = int(os.environ.get("FOUNDATIONPOSE_EST_REFINE_ITER", "5"))
FOUNDATIONPOSE_INITIAL_SEARCH_ENABLE = os.environ.get("FOUNDATIONPOSE_INITIAL_SEARCH_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
FOUNDATIONPOSE_INITIAL_SCALE_FACTORS = tuple(
    float(value.strip())
    for value in os.environ.get("FOUNDATIONPOSE_INITIAL_SCALE_FACTORS", "0.90,0.95,1.00,1.05,1.10").split(",")
    if value.strip()
)
POSE_STAGE_PY = SERVER_PY
RUNTIME_MESH_STAGE_PY = SERVER_PY
BLENDER_STAGE_PY = SERVER_PY
MODEL_BOUNDS_STAGE_PY = SERVER_PY


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

# Persistent model worker idle release windows.
SAM3MASK_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("SAM3MASK_WORKER_IDLE_TIMEOUT_SEC", "300"))
FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC = int(
    os.environ.get("FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC", "300")
)

BLENDER_OUTPUT_ROOT = OUTPUT_ROOT / "blender"
BLENDER_FBX_DIR = BLENDER_OUTPUT_ROOT / "fbx"
BLENDER_BIN = "/usr/local/bin/blender"

RUNTIME_MESH_OUTPUT_ROOT = OUTPUT_ROOT / "runtime_mesh"
RUNTIME_MESH_DECIMATE_RATIO = 1.0 / 16.0
RUNTIME_MESH_TEXTURE_SIZE = 1024
RUNTIME_MESH_BAKE_MARGIN_PX = 64
RUNTIME_MESH_UV_ISLAND_MARGIN = 0.03

MODEL_FBX_DECIMATE_RATIO = float(os.environ.get("MODEL_FBX_DECIMATE_RATIO", str(1.0 / 16.0)))
MODEL_FBX_CLEAN_ENABLE = True
MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO = float(
    os.environ.get("MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO", str(INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO))
)
MODEL_FBX_CLEAN_COMPONENT_MIN_FACES = int(
    os.environ.get("MODEL_FBX_CLEAN_COMPONENT_MIN_FACES", str(INSTANTMESH_CLEAN_COMPONENT_MIN_FACES))
)

SAM3D_OBJECTS_DECIMATE_ENABLE = False
SAM3D_OBJECTS_DEFAULT_DECIMATE_RATIO = 1.0 / 16.0
SAM3D_OBJECTS_DEFAULT_TEXTURE_SIZE = 1024
SAM3D_OBJECTS_DEFAULT_BAKE_MARGIN_PX = 64
SAM3D_OBJECTS_POSTPROCESS_DECIMATE_RATIO = SAM3D_OBJECTS_DEFAULT_DECIMATE_RATIO
SAM3D_OBJECTS_FBX_DECIMATE_RATIO = SAM3D_OBJECTS_DEFAULT_DECIMATE_RATIO
SAM3D_OBJECTS_POSTPROCESS_TEXTURE_SIZE = int(
    os.environ.get(
        "SAM3D_OBJECTS_POSTPROCESS_TEXTURE_SIZE",
        str(SAM3D_OBJECTS_DEFAULT_TEXTURE_SIZE),
    )
)
SAM3D_OBJECTS_POSTPROCESS_BAKE_MARGIN_PX = int(
    os.environ.get(
        "SAM3D_OBJECTS_POSTPROCESS_BAKE_MARGIN_PX",
        str(SAM3D_OBJECTS_DEFAULT_BAKE_MARGIN_PX),
    )
)
SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN = float(
    os.environ.get("SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN", str(RUNTIME_MESH_UV_ISLAND_MARGIN))
)
SAM3D_OBJECTS_VOXEL_REMESH_ENABLE = False
SAM3D_OBJECTS_VOXEL_SIZE_RATIO = float(os.environ.get("SAM3D_OBJECTS_VOXEL_SIZE_RATIO", "0.008"))
SAM3D_OBJECTS_REMOVE_BLACK_FACES = False
SAM3D_OBJECTS_REPAIR_BLACK_FACES = True
SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD = float(
    os.environ.get("SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD", "0.035")
)
SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD = float(
    os.environ.get("SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD", "0.05")
)
SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO = float(
    os.environ.get("SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO", "0.45")
)


# HTTP file serving
FOLDER_MAP = {
    "meshes": INSTANTMESH_OUTPUT_MESHES,
    "images": INSTANTMESH_OUTPUT_IMAGES,
    "videos": INSTANTMESH_OUTPUT_VIDEOS,
    "sam3d_object_meshes": SAM3D_OBJECTS_OUTPUT_MESHES,
    "runtime_meshes": RUNTIME_MESH_OUTPUT_ROOT,
    "fbx": BLENDER_FBX_DIR,
}


# Bootstrap directories
for path in [
    UPLOAD_FOLDER,
    ENV_CONFIG_ROOT,
    DATABASE_ROOT,
    ARUCO_DATA_ROOT,
    ARUCO_REFERENCE_ROOT,
    ARUCO_RUNTIME_ROOT,
    ARUCO_RAW_ROOT,
    INSTANTMESH_INPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    INSTANTMESH_OUTPUT_VIDEOS,
    SAM3_OUTPUT_ROOT,
    SAM3D_OBJECTS_OUTPUT_MESHES,
    OBJECT_ALIGNMENT_OUTPUT_ROOT,
    RUNTIME_MESH_OUTPUT_ROOT,
    BLENDER_FBX_DIR,
    HOLOLENS2_OUTPUT_DEPTH_IMAGES,
]:
    path.mkdir(parents=True, exist_ok=True)
