from __future__ import annotations

import numpy as np


INTERNAL_COORDINATE_SYSTEM = "canonical_rh"

COORDINATE_SYSTEMS = {
    "canonical_rh": {
        "handedness": "right",
        "x": "right",
        "y": "up",
        "z": "backward",
        "forward": "-Z",
        "usage": "Server-internal reconstruction, alignment, and camera-local pose math.",
    },
    "opencv_camera": {
        "handedness": "right",
        "x": "right",
        "y": "down",
        "z": "forward",
        "forward": "+Z",
        "usage": "OpenCV/ArUco/FoundationPose/Shigurei camera-boundary math.",
    },
    "unity": {
        "handedness": "left",
        "x": "right",
        "y": "up",
        "z": "forward",
        "forward": "+Z",
        "usage": "Unity/HoloLens runtime payloads and world/object pose output.",
    },
    "windows_spatial": {
        "handedness": "right",
        "x": "right",
        "y": "up",
        "z": "backward",
        "forward": "-Z",
        "usage": "Raw HoloLens/hl2ss PV pose matrix boundary.",
    },
    "model_input": {
        "handedness": "right",
        "x": "backward",
        "y": "right",
        "z": "up",
        "forward": "-X",
        "usage": "Generated OBJ model vertices before runtime-axis baking.",
    },
    "unity_runtime_local": {
        "handedness": "right",
        "x": "generated OBJ X",
        "y": "generated OBJ Y",
        "z": "generated OBJ Z",
        "forward": "runtime-local axis remap",
        "usage": "RuntimeMesh/FBX local asset contract; pose stage remaps this basis for Unity.",
    },
    "blender_world": {
        "handedness": "right",
        "x": "right",
        "y": "forward",
        "z": "up",
        "forward": "+Y",
        "usage": "Blender debug render and OBJ/FBX conversion boundary.",
    },
    "pointcloud_export": {
        "handedness": "right",
        "x": "forward",
        "y": "up",
        "z": "left",
        "forward": "+X",
        "usage": "PLY coordinates chosen for Blender import compatibility.",
    },
}


# Canonical reconstruction camera basis:
# +X right, +Y up, -Z camera-forward. This is the server-internal basis.
OPENCV_CAMERA_TO_CANONICAL_RH_BASIS = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
CANONICAL_RH_TO_OPENCV_CAMERA_BASIS = OPENCV_CAMERA_TO_CANONICAL_RH_BASIS.copy()

# Unity runtime uses +Z forward at the boundary.
CANONICAL_RH_TO_UNITY_BASIS = np.diag([1.0, 1.0, -1.0]).astype(np.float64)
UNITY_TO_CANONICAL_RH_BASIS = CANONICAL_RH_TO_UNITY_BASIS.copy()

# Windows spatial poses use +Z backward relative to Unity's +Z forward.
WINDOWS_TO_UNITY_BASIS = CANONICAL_RH_TO_UNITY_BASIS.copy()

# OpenCV camera coordinates use +X right, +Y down, +Z forward.
OPENCV_CAMERA_TO_UNITY_CAMERA_BASIS = (
    CANONICAL_RH_TO_UNITY_BASIS @ OPENCV_CAMERA_TO_CANONICAL_RH_BASIS
)
UNITY_TO_OPENCV_CAMERA_BASIS = OPENCV_CAMERA_TO_UNITY_CAMERA_BASIS.copy()

# Exported PLY coordinates are chosen so that importing with forward=-X, up=+Y
# lands in Blender world as canonical [x, y, z] -> [x, -z, y].
POINTCLOUD_EXPORT_TO_CANONICAL_RH_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

CANONICAL_RH_TO_BLENDER_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
BLENDER_WORLD_TO_CANONICAL_RH = CANONICAL_RH_TO_BLENDER_WORLD.T

MODEL_INPUT_TO_CANONICAL_RH_BASIS = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
CANONICAL_RH_TO_MODEL_INPUT_BASIS = MODEL_INPUT_TO_CANONICAL_RH_BASIS.T

MODEL_INPUT_AXIS_CONTRACT = "model_input_minus_x_forward_z_up"
RUNTIME_AXIS_CONTRACT = "runtime_mesh_local_to_unity_pose"
SOURCE_AXIS_CONTRACT = MODEL_INPUT_AXIS_CONTRACT
MODEL_INPUT_TO_UNITY_RUNTIME_LOCAL = np.eye(3, dtype=np.float32)
RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# Blender import/export axis declarations.
ICP_OBJ_IMPORT_FORWARD_AXIS = "NEGATIVE_X"
ICP_OBJ_IMPORT_UP_AXIS = "Z"
FBX_CONVERT_OBJ_IMPORT_FORWARD_AXIS = "NEGATIVE_Z"
FBX_CONVERT_OBJ_IMPORT_UP_AXIS = "Y"
FBX_EXPORT_FORWARD_AXIS = "-Z"
FBX_EXPORT_UP_AXIS = "Y"

ICP_OBJ_IMPORT_TO_BLENDER_WORLD = CANONICAL_RH_TO_BLENDER_WORLD @ MODEL_INPUT_TO_CANONICAL_RH_BASIS

# Blender's default OBJ import orientation, forward=-Z and up=+Y.
FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# Blender world -> exported FBX local basis used by the runtime wrapper
# (axis_forward="-Z", axis_up="Y", bake_space_transform=True).
BLENDER_WORLD_TO_FBX_EXPORT_LOCAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

# RuntimeMesh preserves the generated model-input axes. The pose stage applies
# RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION so the final loaded model matches the
# pre-coordinate-refactor HoloLens orientation.
MODEL_INPUT_TO_FBX_RUNTIME_LOCAL = np.eye(3, dtype=np.float32)
FBX_RUNTIME_LOCAL_TO_MODEL_INPUT = MODEL_INPUT_TO_FBX_RUNTIME_LOCAL.T
FBX_RUNTIME_LOCAL_TO_UNITY_BASIS = np.eye(3, dtype=np.float32)
FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY = np.eye(3, dtype=np.float32)


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero-length quaternion")
    return q / n


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    normalized = u @ vh
    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vh
    return normalized.astype(np.float64)


def rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")

    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def quat_xyzw_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def convert_rotation_between_bases(rotation: np.ndarray, basis_change: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    basis_change = np.asarray(basis_change, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    if basis_change.shape != (3, 3):
        raise ValueError("basis change matrix must be 3x3")
    return orthonormalize_rotation(basis_change @ rotation @ basis_change.T)


def convert_translation_between_bases(
    translation: np.ndarray,
    basis_change: np.ndarray,
) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    basis_change = np.asarray(basis_change, dtype=np.float64)
    if basis_change.shape != (3, 3):
        raise ValueError("basis change matrix must be 3x3")
    return (basis_change @ translation.reshape(3, 1)).reshape(3).astype(np.float64)


def convert_points_between_bases(
    points: np.ndarray,
    basis_change: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64 if dtype is None else dtype)
    basis_change = np.asarray(basis_change, dtype=np.float64 if dtype is None else dtype)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if basis_change.shape != (3, 3):
        raise ValueError("basis change matrix must be 3x3")
    converted = points @ basis_change.T
    return converted.astype(points.dtype if dtype is None else dtype)


def convert_vector_between_bases(
    vector: np.ndarray,
    basis_change: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64 if dtype is None else dtype).reshape(3)
    basis_change = np.asarray(basis_change, dtype=np.float64 if dtype is None else dtype)
    if basis_change.shape != (3, 3):
        raise ValueError("basis change matrix must be 3x3")
    return (basis_change @ vector.reshape(3, 1)).reshape(3).astype(vector.dtype if dtype is None else dtype)


def convert_windows_pose_matrix_to_unity_pose_components(
    pose_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_matrix = np.asarray(pose_matrix, dtype=np.float64)
    if pose_matrix.shape != (4, 4):
        raise ValueError("pose matrix must be 4x4")

    rotation_windows = pose_matrix[:3, :3].astype(np.float64)
    translation_windows = pose_matrix[3, :3].astype(np.float64)
    rotation_unity = convert_rotation_between_bases(rotation_windows, WINDOWS_TO_UNITY_BASIS)
    translation_unity = convert_translation_between_bases(
        translation_windows,
        WINDOWS_TO_UNITY_BASIS,
    )
    quaternion_unity = rotation_matrix_to_quat_xyzw(rotation_unity)
    return translation_unity, rotation_unity, quaternion_unity


def convert_hololens_pv_pose_matrix_to_unity_pose_components(
    pose_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_matrix = np.asarray(pose_matrix, dtype=np.float64)
    if pose_matrix.shape != (4, 4):
        raise ValueError("pose matrix must be 4x4")

    rotation_windows = pose_matrix[:3, :3].astype(np.float64)
    translation_windows = pose_matrix[3, :3].astype(np.float64)

    # PV poses in this pipeline are stored in the hl2ss row-vector convention:
    # translation lives in the last row and p_local @ R is used. Downstream code
    # composes poses as column vectors, so transpose rotation while crossing the
    # Windows -> Unity boundary.
    rotation_unity = orthonormalize_rotation(
        WINDOWS_TO_UNITY_BASIS @ rotation_windows.T @ WINDOWS_TO_UNITY_BASIS
    )
    translation_unity = convert_translation_between_bases(
        translation_windows,
        WINDOWS_TO_UNITY_BASIS,
    )
    quaternion_unity = rotation_matrix_to_quat_xyzw(rotation_unity)
    return translation_unity, rotation_unity, quaternion_unity


def convert_opencv_camera_pose_to_unity_camera_pose(
    rotation_cv: np.ndarray,
    translation_cv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_cv = np.asarray(rotation_cv, dtype=np.float64)
    translation_cv = np.asarray(translation_cv, dtype=np.float64).reshape(3)
    rotation_unity = convert_rotation_between_bases(
        rotation_cv,
        OPENCV_CAMERA_TO_UNITY_CAMERA_BASIS,
    )
    translation_unity = convert_translation_between_bases(
        translation_cv,
        OPENCV_CAMERA_TO_UNITY_CAMERA_BASIS,
    )
    return rotation_unity.astype(np.float64), translation_unity.astype(np.float64)


def convert_opencv_camera_pose_to_canonical_rh_pose(
    rotation_cv: np.ndarray,
    translation_cv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_cv = np.asarray(rotation_cv, dtype=np.float64)
    translation_cv = np.asarray(translation_cv, dtype=np.float64).reshape(3)
    rotation_rh = convert_rotation_between_bases(
        rotation_cv,
        OPENCV_CAMERA_TO_CANONICAL_RH_BASIS,
    )
    translation_rh = convert_translation_between_bases(
        translation_cv,
        OPENCV_CAMERA_TO_CANONICAL_RH_BASIS,
    )
    return rotation_rh.astype(np.float64), translation_rh.astype(np.float64)


def convert_canonical_rh_pose_to_unity_pose(
    rotation_rh: np.ndarray,
    translation_rh: np.ndarray,
    *,
    convert_child_basis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_rh = np.asarray(rotation_rh, dtype=np.float64)
    translation_rh = np.asarray(translation_rh, dtype=np.float64).reshape(3)
    if convert_child_basis:
        rotation_unity = convert_rotation_between_bases(
            rotation_rh,
            CANONICAL_RH_TO_UNITY_BASIS,
        )
    else:
        rotation_unity = orthonormalize_rotation(CANONICAL_RH_TO_UNITY_BASIS @ rotation_rh)
    translation_unity = convert_translation_between_bases(
        translation_rh,
        CANONICAL_RH_TO_UNITY_BASIS,
    )
    return rotation_unity.astype(np.float64), translation_unity.astype(np.float64)


def pointcloud_export_to_canonical_rh(points: np.ndarray) -> np.ndarray:
    return convert_points_between_bases(
        points,
        POINTCLOUD_EXPORT_TO_CANONICAL_RH_BASIS,
        dtype=np.float32,
    )


def pointcloud_export_to_blender_world(points: np.ndarray) -> np.ndarray:
    points_canonical = pointcloud_export_to_canonical_rh(points)
    return canonical_rh_to_blender_world_points(points_canonical)


def obj_vertices_to_canonical_rh(points: np.ndarray) -> np.ndarray:
    return convert_points_between_bases(
        points,
        MODEL_INPUT_TO_CANONICAL_RH_BASIS,
        dtype=np.float32,
    )


def obj_vertices_to_blender_world(points: np.ndarray) -> np.ndarray:
    points_canonical = obj_vertices_to_canonical_rh(points)
    return canonical_rh_to_blender_world_points(points_canonical)


def model_pose_canonical_rh_to_unity_camera(
    rotation_canonical: np.ndarray,
    translation_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_canonical = np.asarray(rotation_canonical, dtype=np.float32)
    translation_canonical = np.asarray(translation_canonical, dtype=np.float32)
    basis = np.asarray(CANONICAL_RH_TO_UNITY_BASIS, dtype=np.float32)
    rotation_unity = basis @ rotation_canonical @ basis
    translation_unity = basis @ translation_canonical.reshape(3, 1)
    return rotation_unity.astype(np.float32), translation_unity.reshape(3).astype(np.float32)


def canonical_rh_to_blender_world_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return convert_points_between_bases(
        points,
        CANONICAL_RH_TO_BLENDER_WORLD,
        dtype=np.float32,
    )


def canonical_rh_to_blender_world_vector(vector: np.ndarray) -> np.ndarray:
    return convert_vector_between_bases(
        vector,
        CANONICAL_RH_TO_BLENDER_WORLD,
        dtype=np.float32,
    )


def blender_world_to_canonical_rh_vector(vector: np.ndarray) -> np.ndarray:
    return convert_vector_between_bases(
        vector,
        BLENDER_WORLD_TO_CANONICAL_RH,
        dtype=np.float32,
    )


def rotation_canonical_rh_to_blender_world(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return CANONICAL_RH_TO_BLENDER_WORLD @ rotation @ BLENDER_WORLD_TO_CANONICAL_RH


def rotation_blender_world_to_canonical_rh(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return BLENDER_WORLD_TO_CANONICAL_RH @ rotation @ CANONICAL_RH_TO_BLENDER_WORLD


def model_pose_canonical_rh_to_blender_world(
    rotation_canonical: np.ndarray,
    translation_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_blender = rotation_canonical_rh_to_blender_world(rotation_canonical)
    translation_blender = canonical_rh_to_blender_world_vector(translation_canonical)
    return rotation_blender.astype(np.float32), translation_blender.astype(np.float32)


def model_pose_blender_world_to_canonical_rh(
    rotation_blender: np.ndarray,
    translation_blender: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_canonical = rotation_blender_world_to_canonical_rh(rotation_blender)
    translation_canonical = blender_world_to_canonical_rh_vector(translation_blender)
    return rotation_canonical.astype(np.float32), translation_canonical.astype(np.float32)


def rotation_canonical_rh_to_blender_obj_import(
    rotation: np.ndarray,
    obj_import_to_blender_world: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    obj_import_to_blender_world = np.asarray(obj_import_to_blender_world, dtype=np.float32)
    return (
        CANONICAL_RH_TO_BLENDER_WORLD
        @ rotation
        @ MODEL_INPUT_TO_CANONICAL_RH_BASIS
        @ obj_import_to_blender_world.T
    ).astype(np.float32)


def rotation_canonical_rh_to_blender_default_obj_import(rotation: np.ndarray) -> np.ndarray:
    return rotation_canonical_rh_to_blender_obj_import(
        rotation,
        FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD,
    )


def transform_model_input_to_unity_runtime(values: list[float] | tuple[float, float, float] | np.ndarray) -> list[float]:
    x, y, z = [float(v) for v in values]
    return [x, y, z]


def runtime_axis_transform_info(stats: dict | None = None) -> dict:
    info = {
        "axis_contract": RUNTIME_AXIS_CONTRACT,
        "source_axis_contract": SOURCE_AXIS_CONTRACT,
        "axis_transform": "preserve_model_input_axes",
        "axis_transform_matrix": MODEL_INPUT_TO_UNITY_RUNTIME_LOCAL.astype(float).tolist(),
        "axis_transform_expression": "runtime_xyz = source_xyz",
        "axis_transform_determinant": 1.0,
        "face_winding_flipped_for_axis_transform": False,
    }
    if stats:
        info["axis_transform_stats"] = stats
    return info

