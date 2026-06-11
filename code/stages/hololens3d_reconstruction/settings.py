from __future__ import annotations

import os

# Depth / alignment tuning for the hololens3d_reconstruction stage.
# Directory conventions, script entrypoints, and Python runtimes stay in config.py.
ICP_DEPTH_BORDER_CROP_RATIO = 0.03

# foundationpose = model-based FoundationPose registration from RGB-D + mask.
# camera_refine = existing camera-view ICP rotation+translation+scale refinement.
# off = measured-distance placement without ICP.
OBJECT_ALIGNMENT_MODE = os.environ.get('OBJECT_ALIGNMENT_MODE', 'foundationpose').strip() or 'foundationpose'
ICP_MODE = OBJECT_ALIGNMENT_MODE
ICP_ENABLE = OBJECT_ALIGNMENT_MODE == 'camera_refine'

# Maximum number of points written to the exported depth point cloud PLY.
# Set to 0 or None to keep all valid depth points.
DEPTHPOINTCLOUD_MAX_EXPORT_POINTS = 6000

# Enable preview/front-view PNG renders produced by the alignment pipeline.
# Disabled by default so ICP_MODE=off stays fast.
ENABLE_ALIGNMENT_RENDER_OUTPUTS = False

# Enable InstantMesh circular-view MP4 generation.
ENABLE_INSTANTMESH_VIDEO_OUTPUT = True

# Remove tiny disconnected mesh islands immediately after InstantMesh export.
INSTANTMESH_CLEAN_ENABLE = True
INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO = 0.01
INSTANTMESH_CLEAN_COMPONENT_MIN_FACES = 32

# SAM3D generation does not pass simplification/step/format limits; these are runtime/compat knobs, not quality caps.
SAM3D_OBJECTS_SEED = int(os.environ.get('SAM3D_OBJECTS_SEED', '42'))
SAM3D_OBJECTS_ATTN_BACKEND = os.environ.get('SAM3D_OBJECTS_ATTN_BACKEND', 'sdpa')

# Parameters for ICP_MODE=off bbox-front-surface placement.
# The current values were fitted against the latest five camera_refine results:
# avg position delta ~= 2.8 cm, max ~= 5.9 cm.
ICP_BBOX_SURFACE_RAY_SOURCE = 'all_points'
ICP_BBOX_SURFACE_DISTANCE_MODE = 'mean_depth'
ICP_BBOX_SURFACE_LATERAL_MODE = 'centroid_xy'
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
ICP_ACCELERATION_DEVICE = 'auto'
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

FOUNDATIONPOSE_EST_REFINE_ITER = int(os.environ.get('FOUNDATIONPOSE_EST_REFINE_ITER', '5'))
FOUNDATIONPOSE_INITIAL_SEARCH_ENABLE = os.environ.get('FOUNDATIONPOSE_INITIAL_SEARCH_ENABLE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
FOUNDATIONPOSE_INITIAL_ROTATION_GRID_DEGREES = tuple(
    float(value.strip())
    for value in os.environ.get('FOUNDATIONPOSE_INITIAL_ROTATION_GRID_DEGREES', '0,-30,30,-60,60').split(',')
    if value.strip()
)
FOUNDATIONPOSE_INITIAL_ROTATION_MAX_DELTA_DEG = float(os.environ.get('FOUNDATIONPOSE_INITIAL_ROTATION_MAX_DELTA_DEG', '60'))

RUNTIME_MESH_DECIMATE_RATIO = 1.0 / 16.0
RUNTIME_MESH_TEXTURE_SIZE = 1024
RUNTIME_MESH_BAKE_MARGIN_PX = 64
RUNTIME_MESH_UV_ISLAND_MARGIN = 0.03

MODEL_FBX_DECIMATE_RATIO = float(os.environ.get('MODEL_FBX_DECIMATE_RATIO', str(1.0 / 16.0)))
MODEL_FBX_CLEAN_ENABLE = True
MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO = float(
    os.environ.get('MODEL_FBX_CLEAN_COMPONENT_MIN_FACE_RATIO', str(INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO))
)
MODEL_FBX_CLEAN_COMPONENT_MIN_FACES = int(
    os.environ.get('MODEL_FBX_CLEAN_COMPONENT_MIN_FACES', str(INSTANTMESH_CLEAN_COMPONENT_MIN_FACES))
)

SAM3D_OBJECTS_DECIMATE_ENABLE = False
SAM3D_OBJECTS_DEFAULT_DECIMATE_RATIO = 1.0 / 16.0
SAM3D_OBJECTS_DEFAULT_TEXTURE_SIZE = 1024
SAM3D_OBJECTS_DEFAULT_BAKE_MARGIN_PX = 64
SAM3D_OBJECTS_POSTPROCESS_DECIMATE_RATIO = SAM3D_OBJECTS_DEFAULT_DECIMATE_RATIO
SAM3D_OBJECTS_FBX_DECIMATE_RATIO = SAM3D_OBJECTS_DEFAULT_DECIMATE_RATIO
SAM3D_OBJECTS_POSTPROCESS_TEXTURE_SIZE = int(
    os.environ.get('SAM3D_OBJECTS_POSTPROCESS_TEXTURE_SIZE', str(SAM3D_OBJECTS_DEFAULT_TEXTURE_SIZE))
)
SAM3D_OBJECTS_POSTPROCESS_BAKE_MARGIN_PX = int(
    os.environ.get('SAM3D_OBJECTS_POSTPROCESS_BAKE_MARGIN_PX', str(SAM3D_OBJECTS_DEFAULT_BAKE_MARGIN_PX))
)
SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN = float(
    os.environ.get('SAM3D_OBJECTS_POSTPROCESS_UV_ISLAND_MARGIN', str(RUNTIME_MESH_UV_ISLAND_MARGIN))
)
SAM3D_OBJECTS_VOXEL_REMESH_ENABLE = False
SAM3D_OBJECTS_VOXEL_SIZE_RATIO = float(os.environ.get('SAM3D_OBJECTS_VOXEL_SIZE_RATIO', '0.008'))
SAM3D_OBJECTS_REMOVE_BLACK_FACES = False
SAM3D_OBJECTS_REPAIR_BLACK_FACES = True
SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD = float(os.environ.get('SAM3D_OBJECTS_BLACK_FACE_RGB_THRESHOLD', '0.035'))
SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD = float(os.environ.get('SAM3D_OBJECTS_BLACK_FACE_ALPHA_THRESHOLD', '0.05'))
SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO = float(os.environ.get('SAM3D_OBJECTS_BLACK_FACE_MAX_REMOVE_RATIO', '0.45'))
