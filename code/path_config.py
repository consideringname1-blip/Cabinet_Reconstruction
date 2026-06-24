from __future__ import annotations

import os
from pathlib import Path


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
MODELS_ROOT = PROJECT_ROOT / "models"

# Python runtimes
IMESH_PY = "/opt/miniconda/envs/imesh/bin/python"
SERVER_PY = "/opt/miniconda/envs/server/bin/python"
SAM3_PY = "/opt/miniconda/envs/sam3/bin/python"
SAM3D_OBJECTS_PY = "/opt/miniconda/envs/sam3d-objects-cu118/bin/python"
SAM3D_BODY_PY = "/opt/miniconda/envs/sam_3d_body/bin/python"
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
ICPALIGNMENT_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_object_icp_alignment_from_json.py"
FOUNDATIONPOSE_ALIGNMENT_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_foundationpose_alignment_worker.py"
POSE_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_pose_from_json.py"
ARUCO_SYNC_STAGE_RUN = ARUCO_STAGE_ROOT / "run_aruco_sync_from_json.py"
RUNTIME_MESH_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_runtime_mesh_from_json.py"
RUNTIME_MESH_BAKE_SCRIPT = HOLOLENS3D_RECON_STAGE_ROOT / "bake_runtime_mesh.py"
SAM3D_OBJECTS_POSTPROCESS_SCRIPT = HOLOLENS3D_RECON_STAGE_ROOT / "postprocess_sam3d_glb.py"
BLENDER_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_blender_from_json.py"
MODEL_BOUNDS_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_model_bounds_from_json.py"
DISPLAY_IDENTITY_STAGE_RUN = HOLOLENS3D_RECON_STAGE_ROOT / "run_display_identity_from_json.py"
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
ICPALIGNMENT_STAGE_PY = SERVER_PY
FOUNDATIONPOSE_ALIGNMENT_PY = "/opt/miniconda/envs/foundationpose/bin/python"
POSE_STAGE_PY = SERVER_PY
RUNTIME_MESH_STAGE_PY = SERVER_PY
BLENDER_STAGE_PY = SERVER_PY
MODEL_BOUNDS_STAGE_PY = SERVER_PY
DISPLAY_IDENTITY_STAGE_PY = SERVER_PY
SHIGURE_HISTORY_RECORDER_STAGE_PY = os.environ.get("SHIGURE_HISTORY_RECORDER_PY", "/usr/bin/python3")
HISTORY_PLACEMENT_RESTORATION_STAGE_PY = SERVER_PY
TAKEN_OBJECT_DETECTION_STAGE_PY = SERVER_PY
SAM3D_BODY_MESH_STAGE_PY = os.environ.get("SAM3D_BODY_PY", SAM3D_BODY_PY)

BLENDER_BIN = "/usr/local/bin/blender"
