import os


# Runtime switches
HOLOLENS2_HOST = "10.40.1.132"
IS_RUN_FLASK_SERVER = True
# Model generation backend for the shared model-generation stage slot.
# Use "sam3d_objects" or "instantmesh".
MODEL_GENERATION_BACKEND = "instantmesh"
if MODEL_GENERATION_BACKEND not in {"instantmesh", "sam3d_objects"}:
    raise ValueError("MODEL_GENERATION_BACKEND must be 'instantmesh' or 'sam3d_objects'")
MODEL_SERVICE_PREWARM_ENABLE = os.environ.get("MODEL_SERVICE_PREWARM_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
SHIGURE_HISTORY_RECORDING_ENABLE = os.environ.get("SHIGURE_HISTORY_RECORDING_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
SHIGURE_DEBUG_CACHE_ENABLE = os.environ.get("SHIGURE_DEBUG_CACHE_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SHIGURE_DEBUG_CACHE_RETENTION_SECONDS = float(
    os.environ.get("SHIGURE_DEBUG_CACHE_RETENTION_SECONDS", "600")
)
if not 0.0 < SHIGURE_DEBUG_CACHE_RETENTION_SECONDS <= 600.0:
    raise ValueError("SHIGURE_DEBUG_CACHE_RETENTION_SECONDS must be in (0, 600]")
SHIGURE_DEBUG_CACHE_MAX_ENTRIES = int(
    os.environ.get("SHIGURE_DEBUG_CACHE_MAX_ENTRIES", "18000")
)
if SHIGURE_DEBUG_CACHE_MAX_ENTRIES <= 0:
    raise ValueError("SHIGURE_DEBUG_CACHE_MAX_ENTRIES must be positive")
# Model/live-pose delivery remains bounded. Shigure spatial boxes use an
# independent complete snapshot and deliberately do not use this limit.
MAX_REALTIME_MODEL_POSE_ITEMS = 5
REALTIME_TRACKING_EVENT_POLL_SEC = float(os.environ.get("REALTIME_TRACKING_EVENT_POLL_SEC", "0.25"))
REALTIME_TRACKING_FP_MIN_BBOX_IOU = float(os.environ.get("REALTIME_TRACKING_FP_MIN_BBOX_IOU", "0.20"))
REALTIME_TRACKING_FP_MAX_DEPTH_RESIDUAL_M = float(os.environ.get("REALTIME_TRACKING_FP_MAX_DEPTH_RESIDUAL_M", "0.20"))
CONSOLE_OUTPUT_LOG_ENABLE = os.environ.get("CONSOLE_OUTPUT_LOG_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
TASK_DEBUG_OUTPUT_ENABLE = os.environ.get("TASK_DEBUG_OUTPUT_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
TASK_LOG_OUTPUT_ENABLE = os.environ.get("TASK_LOG_OUTPUT_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}

# HoloLens historical-capture reuse / DINOv2 matching. This is only the
# database scan window used to find distinct objects; actual scoring shares
# the strict recent-five object cap below.
DINO_IDENTITY_WORKER_IDLE_TIMEOUT_SEC = int(os.environ.get("DINO_IDENTITY_WORKER_IDLE_TIMEOUT_SEC", "300"))
DINO_IDENTITY_CANDIDATE_LIMIT = int(os.environ.get("DINO_IDENTITY_CANDIDATE_LIMIT", "500"))
DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD = float(os.environ.get("DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD", "0.20"))
DINO_IDENTITY_MATCH_SECOND_MARGIN = float(os.environ.get("DINO_IDENTITY_MATCH_SECOND_MARGIN", "0.05"))
DINO_IDENTITY_MATCH_REQUIRE_MARGIN = os.environ.get("DINO_IDENTITY_MATCH_REQUIRE_MARGIN", "1").strip().lower() in {"1", "true", "yes", "on"}

# Every identity path keeps the same recent-five object policy. Changing
# transport capacity must never silently change identity candidate capacity.
# Shigure-local IDs are never persisted as object identity.
SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS = 5
SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD = float(
    os.environ.get("SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD", "0.20")
)
SHIGURE_IDENTITY_MATCH_SECOND_MARGIN = float(
    os.environ.get("SHIGURE_IDENTITY_MATCH_SECOND_MARGIN", "0.05")
)
SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN = os.environ.get(
    "SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SHIGURE_IDENTITY_GEOMETRY_WEIGHT = float(
    os.environ.get("SHIGURE_IDENTITY_GEOMETRY_WEIGHT", "0.0")
)
SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE = float(
    os.environ.get("SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE", "0.08")
)
SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY = float(
    os.environ.get("SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY", "0.03")
)
SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS = int(
    os.environ.get("SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS", "3")
)
SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS = int(
    os.environ.get("SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS", "30")
)
SHIGURE_EXAMPLE_STABLE_MASK_FRAMES = int(
    os.environ.get("SHIGURE_EXAMPLE_STABLE_MASK_FRAMES", "5")
)
SHIGURE_EXAMPLE_STABLE_BBOX_IOU = float(
    os.environ.get("SHIGURE_EXAMPLE_STABLE_BBOX_IOU", "0.82")
)
SHIGURE_EXAMPLE_STABLE_MASK_IOU = float(
    os.environ.get("SHIGURE_EXAMPLE_STABLE_MASK_IOU", "0.78")
)
SHIGURE_EXAMPLE_MAX_MASK_AREA_RATIO = float(
    os.environ.get("SHIGURE_EXAMPLE_MAX_MASK_AREA_RATIO", "1.25")
)
SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY = float(
    os.environ.get("SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY", "0.70")
)
SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD = float(
    os.environ.get("SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD", "0.15")
)
SHIGURE_EXAMPLE_DINO_SECOND_MARGIN = float(
    os.environ.get("SHIGURE_EXAMPLE_DINO_SECOND_MARGIN", "0.08")
)
SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS = float(
    os.environ.get("SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS", "1.5")
)
SHIGURE_SPATIAL_BOX_ACQUIRE_FRAMES = int(
    os.environ.get("SHIGURE_SPATIAL_BOX_ACQUIRE_FRAMES", "2")
)
SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES = int(
    os.environ.get("SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES", "5")
)
SHIGURE_SPATIAL_BOX_EMA_ALPHA = float(
    os.environ.get("SHIGURE_SPATIAL_BOX_EMA_ALPHA", "0.35")
)
SHIGURE_SPATIAL_BOX_CENTER_DEADBAND_M = float(
    os.environ.get("SHIGURE_SPATIAL_BOX_CENTER_DEADBAND_M", "0.008")
)
SHIGURE_SPATIAL_BOX_EXTENT_DEADBAND_M = float(
    os.environ.get("SHIGURE_SPATIAL_BOX_EXTENT_DEADBAND_M", "0.005")
)
SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS = int(
    os.environ.get("SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS", "10")
)
SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS = float(
    os.environ.get("SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS", "1.0")
)
SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M = float(os.environ.get("SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M", "0.50"))
SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE = float(os.environ.get("SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE", "1.0"))
SHIGURE_HOLO_SYNC_DINO_DISTANCE_THRESHOLD = float(os.environ.get("SHIGURE_HOLO_SYNC_DINO_DISTANCE_THRESHOLD", "0.35"))
SHIGURE_HOLO_SYNC_DINO_MARGIN = float(os.environ.get("SHIGURE_HOLO_SYNC_DINO_MARGIN", "0.03"))
SHIGURE_HOLO_SYNC_MAX_RECOVERY_ATTEMPTS = int(os.environ.get("SHIGURE_HOLO_SYNC_MAX_RECOVERY_ATTEMPTS", "30"))
if SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE < 0.0:
    raise ValueError("SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE must be non-negative")
if SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY < 0.0:
    raise ValueError("SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY must be non-negative")
if (
    SHIGURE_EXAMPLE_STABLE_MASK_FRAMES < 2
    or not 0.0 <= SHIGURE_EXAMPLE_STABLE_BBOX_IOU <= 1.0
    or not 0.0 <= SHIGURE_EXAMPLE_STABLE_MASK_IOU <= 1.0
    or SHIGURE_EXAMPLE_MAX_MASK_AREA_RATIO < 1.0
    or not 0.0 <= SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY <= 1.0
    or not 0.0 <= SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD <= 2.0
    or not 0.0 <= SHIGURE_EXAMPLE_DINO_SECOND_MARGIN <= 2.0
):
    raise ValueError("invalid strict Shigure example-admission settings")
if (
    not 0.0 <= SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS <= 60.0
    or SHIGURE_SPATIAL_BOX_ACQUIRE_FRAMES < 1
    or SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES < 1
    or not 0.0 < SHIGURE_SPATIAL_BOX_EMA_ALPHA <= 1.0
    or SHIGURE_SPATIAL_BOX_CENTER_DEADBAND_M < 0.0
    or SHIGURE_SPATIAL_BOX_EXTENT_DEADBAND_M < 0.0
):
    raise ValueError("invalid Shigure spatial-box stabilization settings")
if (
    SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS < 1
    or SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS < 0.0
):
    raise ValueError("invalid Shigure startup-recovery retry settings")
if (
    SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS < 1
    or SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS < SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS
):
    raise ValueError("invalid Shigure identity capture-window settings")
if SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M <= 0.0 or SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE < 0.0:
    raise ValueError("invalid Shigure HoloLens geometry-sync settings")
if (
    SHIGURE_HOLO_SYNC_DINO_DISTANCE_THRESHOLD < 0.0
    or SHIGURE_HOLO_SYNC_DINO_MARGIN < 0.0
    or SHIGURE_HOLO_SYNC_MAX_RECOVERY_ATTEMPTS < 1
):
    raise ValueError("invalid Shigure HoloLens DINO-sync settings")

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
ARUCO_SYNC_MARKER_REGISTRY_ON_START = os.environ.get(
    "ARUCO_SYNC_MARKER_REGISTRY_ON_START", "1"
).strip().lower() not in {"0", "false", "no", "off", ""}

# Persistent model/service settings
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
# Maximum elastic pool size. Runtime requests wait only for the first backend;
# additional backends join opportunistically when a GPU lease can fit them.
FOUNDATIONPOSE_POOL_SIZE = int(os.environ.get("FOUNDATIONPOSE_POOL_SIZE", "2"))
if not 1 <= FOUNDATIONPOSE_POOL_SIZE <= 4:
    raise ValueError("FOUNDATIONPOSE_POOL_SIZE must be between 1 and 4")

# GPU placement is fail-closed. A launch waits for an atomic expected-memory
# lease instead of falling back to a GPU that cannot fit the configured peak.
GPU_LEASE_WAIT_TIMEOUT_SEC = float(
    os.environ.get("GPU_LEASE_WAIT_TIMEOUT_SEC", "3600")
)
if GPU_LEASE_WAIT_TIMEOUT_SEC <= 0:
    raise ValueError("GPU_LEASE_WAIT_TIMEOUT_SEC must be positive")
