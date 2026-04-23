from __future__ import annotations

import numpy as np


# Windows spatial poses use +Z backward relative to Unity's +Z forward.
WINDOWS_TO_UNITY_BASIS = np.diag([1.0, 1.0, -1.0]).astype(np.float64)

# OpenCV camera coordinates use +X right, +Y down, +Z forward. For our
# Unity-facing camera-local calculations we keep +Z forward and flip Y upward.
OPENCV_CAMERA_TO_UNITY_CAMERA_BASIS = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


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


def convert_rotation_between_bases(rotation: np.ndarray, basis_change: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    basis_change = np.asarray(basis_change, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    if basis_change.shape != (3, 3):
        raise ValueError("basis change matrix must be 3x3")
    return orthonormalize_rotation(basis_change @ rotation @ basis_change)


def convert_translation_between_bases(
    translation: np.ndarray,
    basis_change: np.ndarray,
) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    basis_change = np.asarray(basis_change, dtype=np.float64)
    if basis_change.shape != (3, 3):
        raise ValueError("basis change matrix must be 3x3")
    return (basis_change @ translation.reshape(3, 1)).reshape(3).astype(np.float64)


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
