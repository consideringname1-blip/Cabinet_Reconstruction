import os


# Runtime switches
HOLOLENS2_HOST = "10.40.1.132"
IS_RUN_FLASK_SERVER = True
# Model generation backend for the shared model-generation stage slot.
# Use "sam3d_objects" or "instantmesh".
MODEL_GENERATION_BACKEND = "instantmesh"
MODEL_SERVICE_PREWARM_ENABLE = os.environ.get("MODEL_SERVICE_PREWARM_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
SHIGURE_HISTORY_RECORDING_ENABLE = os.environ.get("SHIGURE_HISTORY_RECORDING_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
CONSOLE_OUTPUT_LOG_ENABLE = os.environ.get("CONSOLE_OUTPUT_LOG_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
TASK_DEBUG_OUTPUT_ENABLE = os.environ.get("TASK_DEBUG_OUTPUT_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
TASK_LOG_OUTPUT_ENABLE = os.environ.get("TASK_LOG_OUTPUT_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}

# Historical model reuse / DINOv2 identity matching
HISTORICAL_MODEL_REUSE_ENABLE = os.environ.get("HISTORICAL_MODEL_REUSE_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
FORCE_NEW_3D_MODEL = os.environ.get("FORCE_NEW_3D_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}
DINO_IDENTITY_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("DINO_IDENTITY_WORKER_IDLE_TIMEOUT_SEC", "300"))
DINO_IDENTITY_CANDIDATE_LIMIT = int(os.environ.get("DINO_IDENTITY_CANDIDATE_LIMIT", "500"))
DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD = float(os.environ.get("DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD", "0.20"))
DINO_IDENTITY_MATCH_SECOND_MARGIN = float(os.environ.get("DINO_IDENTITY_MATCH_SECOND_MARGIN", "0.05"))
DINO_IDENTITY_MATCH_REQUIRE_MARGIN = os.environ.get("DINO_IDENTITY_MATCH_REQUIRE_MARGIN", "1").strip().lower() in {"1", "true", "yes", "on"}

# Preview 3D box / pending spatial hint
PREVIEW_3D_BOX_DEPTH_EXPANSION_FACTOR = float(os.environ.get("PREVIEW_3D_BOX_DEPTH_EXPANSION_FACTOR", "2.0"))

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

# Persistent model/service settings
INSTANTMESH_MAX_WORKERS = int(os.environ.get("INSTANTMESH_MAX_WORKERS", "1"))
_instantmesh_gpu_ids = []
for gpu_id in os.environ.get("INSTANTMESH_GPU_IDS", "0,1,2").split(","):
    gpu_id = gpu_id.strip()
    if gpu_id and gpu_id not in _instantmesh_gpu_ids:
        _instantmesh_gpu_ids.append(gpu_id)
INSTANTMESH_GPU_IDS = tuple(_instantmesh_gpu_ids)

SAM3MASK_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("SAM3MASK_WORKER_IDLE_TIMEOUT_SEC", "300"))
INSTANTMESH_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("INSTANTMESH_WORKER_IDLE_TIMEOUT_SEC", "300"))
FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC = int(
    os.environ.get("FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC", "300")
)
