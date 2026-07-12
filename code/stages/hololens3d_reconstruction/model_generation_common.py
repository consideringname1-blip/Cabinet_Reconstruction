"""Shared model-output contract for reconstruction backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_layout import model_debug_dir, model_result_dir, model_worker_dir


BACKEND_INSTANTMESH = "instantmesh"
BACKEND_SAM3D_OBJECTS = "sam3d_objects"
MODEL_GENERATION_BACKENDS = frozenset({BACKEND_INSTANTMESH, BACKEND_SAM3D_OBJECTS})

MODEL_STAGE_INSTANTMESH = "InstantMesh"
MODEL_STAGE_SAM3D_OBJECTS = "SAM3DObjects"
MODEL_STAGE_RUNTIME_MESH = "RuntimeMesh"

_MODEL_STAGE_BY_BACKEND = {
    BACKEND_INSTANTMESH: MODEL_STAGE_INSTANTMESH,
    BACKEND_SAM3D_OBJECTS: MODEL_STAGE_SAM3D_OBJECTS,
}
_CANONICAL_ARTIFACT_FIELDS = frozenset({"backend", "artifact_root", "mesh", "mtl", "image", "video"})
_FORBIDDEN_ARTIFACT_FIELDS = frozenset(
    {
        "generator",
        "source_stage",
        "source_backend",
        "mesh_folder",
        "source_mesh_folder",
        "video_folder",
        "runtime_ready",
    }
)


@dataclass(frozen=True)
class ModelFileSource:
    source_stage: str
    backend: str
    payload: dict[str, Any]
    folder: str
    root: Path
    mesh: str
    mtl: str | None = None
    image: str | None = None
    video: str | None = None
    video_root: Path | None = None

    @property
    def mesh_path(self) -> Path:
        return self.root / self.mesh

    @property
    def mtl_path(self) -> Path | None:
        return self.root / self.mtl if self.mtl else None

    @property
    def image_path(self) -> Path | None:
        return self.root / self.image if self.image else None

    @property
    def video_path(self) -> Path | None:
        if not self.video or self.video_root is None:
            return None
        return self.video_root / self.video


def build_model_generation_payload(
    *,
    backend: str,
    mesh: str,
    mtl: str,
    image: str,
    artifact_root: str,
    video: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = _require_backend(backend)
    artifact_root = str(artifact_root or "").strip()
    if artifact_root not in {"model_worker", "model_result"}:
        raise ValueError("ModelGeneration.artifact_root must be model_worker or model_result")
    payload: dict[str, Any] = {
        "backend": backend,
        "artifact_root": artifact_root,
        "mesh": mesh,
        "mtl": mtl,
        "image": image,
    }
    if video is not None:
        payload["video"] = video
    if extra:
        forbidden = (_CANONICAL_ARTIFACT_FIELDS | _FORBIDDEN_ARTIFACT_FIELDS).intersection(extra)
        if forbidden:
            raise ValueError(f"ModelGeneration.extra contains reserved fields: {sorted(forbidden)}")
        payload.update(extra)
    return payload


def _as_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _require_backend(value: Any) -> str:
    backend = str(value or "").strip()
    if backend not in MODEL_GENERATION_BACKENDS:
        supported = ", ".join(sorted(MODEL_GENERATION_BACKENDS))
        raise ValueError(f"ModelGeneration.backend must be one of: {supported}")
    return backend


def _model_stage_for_backend(backend: str) -> str:
    return _MODEL_STAGE_BY_BACKEND[_require_backend(backend)]


def _reject_forbidden_fields(payload: dict[str, Any], label: str) -> None:
    found = _FORBIDDEN_ARTIFACT_FIELDS.intersection(payload)
    if found:
        raise ValueError(f"{label} contains unsupported fields: {sorted(found)}")


def _resolve_folder_root(
    source_stage: str,
    payload: dict[str, Any],
    *,
    task_timestamp: str | None = None,
) -> tuple[str, Path]:
    artifact_root = str(payload.get("artifact_root") or "").strip()
    if artifact_root == "model_worker":
        if not task_timestamp:
            raise ValueError(f"{source_stage}.artifact_root=model_worker requires task_timestamp")
        return artifact_root, model_worker_dir(task_timestamp)
    if artifact_root == "model_result":
        if not task_timestamp:
            raise ValueError(f"{source_stage}.artifact_root=model_result requires task_timestamp")
        return artifact_root, model_result_dir(task_timestamp)
    raise ValueError(
        f"{source_stage}.artifact_root must be model_worker or model_result; "
        "global output folders are no longer supported"
    )


def _source_from_payload(
    source_stage: str,
    payload: dict[str, Any],
    *,
    require_mtl_image: bool,
    task_timestamp: str | None = None,
) -> ModelFileSource:
    _reject_forbidden_fields(payload, source_stage)
    backend = _require_backend(payload.get("backend"))
    mesh_name = str(payload.get("mesh") or "").strip()
    if not mesh_name:
        raise ValueError(f"{source_stage}.mesh is missing")

    mtl_name = str(payload.get("mtl") or "").strip() or None
    image_name = str(payload.get("image") or "").strip() or None
    if require_mtl_image and (not mtl_name or not image_name):
        raise ValueError(f"{source_stage}.mesh / mtl / image is missing")

    artifact_root = str(payload.get("artifact_root") or "").strip()
    folder, root = _resolve_folder_root(source_stage, payload, task_timestamp=task_timestamp)

    video_name = str(payload.get("video") or "").strip() or None
    video_root = None
    if artifact_root == "model_worker" and video_name and task_timestamp:
        video_root = model_debug_dir(task_timestamp)
    elif artifact_root == "model_result" and video_name and task_timestamp:
        video_root = model_result_dir(task_timestamp)

    return ModelFileSource(
        source_stage=source_stage,
        backend=backend,
        payload=payload,
        folder=folder,
        root=root,
        mesh=mesh_name,
        mtl=mtl_name,
        image=image_name,
        video=video_name,
        video_root=video_root,
    )


def resolve_model_generation_source(
    task: dict[str, Any],
    *,
    require_mtl_image: bool = False,
) -> ModelFileSource:
    model_generation = _as_payload(task.get("ModelGeneration"))
    if not model_generation.get("mesh"):
        raise ValueError("ModelGeneration.mesh is missing")
    backend = _require_backend(model_generation.get("backend"))
    return _source_from_payload(
        _model_stage_for_backend(backend),
        model_generation,
        require_mtl_image=require_mtl_image,
        task_timestamp=str(task.get("task_timestamp") or "").strip() or None,
    )


def resolve_runtime_mesh_source(
    task: dict[str, Any],
    *,
    require_mtl_image: bool = True,
) -> ModelFileSource | None:
    runtime_mesh = _as_payload(task.get(MODEL_STAGE_RUNTIME_MESH))
    if not runtime_mesh.get("mesh"):
        return None
    return _source_from_payload(
        MODEL_STAGE_RUNTIME_MESH,
        runtime_mesh,
        require_mtl_image=require_mtl_image,
        task_timestamp=str(task.get("task_timestamp") or "").strip() or None,
    )
