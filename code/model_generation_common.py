from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_layout import FOLDER_MAP, INSTANTMESH_OUTPUT_VIDEOS, model_debug_dir, model_worker_dir


BACKEND_INSTANTMESH = "instantmesh"
BACKEND_SAM3D_OBJECTS = "sam3d_objects"

MODEL_STAGE_INSTANTMESH = "InstantMesh"
MODEL_STAGE_SAM3D_OBJECTS = "SAM3DObjects"
MODEL_STAGE_RUNTIME_MESH = "RuntimeMesh"

INSTANTMESH_MESH_FOLDER = "meshes"
INSTANTMESH_VIDEO_FOLDER = "videos"
SAM3D_OBJECTS_MESH_FOLDER = "sam3d_object_meshes"
RUNTIME_MESH_FOLDER = "runtime_meshes"

_DEFAULT_FOLDER_BY_STAGE = {
    MODEL_STAGE_INSTANTMESH: INSTANTMESH_MESH_FOLDER,
    MODEL_STAGE_SAM3D_OBJECTS: SAM3D_OBJECTS_MESH_FOLDER,
    MODEL_STAGE_RUNTIME_MESH: RUNTIME_MESH_FOLDER,
}
_DEFAULT_BACKEND_BY_STAGE = {
    MODEL_STAGE_INSTANTMESH: BACKEND_INSTANTMESH,
    MODEL_STAGE_SAM3D_OBJECTS: BACKEND_SAM3D_OBJECTS,
    MODEL_STAGE_RUNTIME_MESH: "runtime_mesh",
}


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
    video_folder: str | None = None
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
        if not self.video:
            return None
        video_root = self.video_root or INSTANTMESH_OUTPUT_VIDEOS
        return video_root / self.video


def build_model_generation_payload(
    *,
    backend: str,
    source_stage: str,
    mesh: str,
    mtl: str,
    image: str,
    mesh_folder: str,
    video: str | None = None,
    video_folder: str | None = None,
    runtime_ready: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": backend,
        "source_stage": source_stage,
        "mesh_folder": mesh_folder,
        "mesh": mesh,
        "mtl": mtl,
        "image": image,
    }
    if video is not None:
        payload["video"] = video
    if video_folder is not None:
        payload["video_folder"] = video_folder
    if runtime_ready is not None:
        payload["runtime_ready"] = bool(runtime_ready)
    if extra:
        payload.update(extra)
    return payload


def _as_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _infer_stage_from_model_generation(payload: dict[str, Any]) -> str:
    source_stage = str(payload.get("source_stage") or "").strip()
    if source_stage in {MODEL_STAGE_INSTANTMESH, MODEL_STAGE_SAM3D_OBJECTS}:
        return source_stage
    if not source_stage:
        raise ValueError("ModelGeneration.source_stage is missing")
    raise ValueError(f"Unsupported ModelGeneration.source_stage: {source_stage}")


def _default_folder(source_stage: str) -> str:
    return _DEFAULT_FOLDER_BY_STAGE.get(source_stage, INSTANTMESH_MESH_FOLDER)


def _default_backend(source_stage: str, payload: dict[str, Any]) -> str:
    backend = str(payload.get("backend") or payload.get("generator") or "").strip()
    if backend:
        return backend
    return _DEFAULT_BACKEND_BY_STAGE.get(source_stage, BACKEND_INSTANTMESH)


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

    folder = str(
        payload.get("mesh_folder")
        or payload.get("folder")
        or _default_folder(source_stage)
    )
    root = FOLDER_MAP.get(folder)
    if root is None:
        raise ValueError(f"No output folder is configured for {source_stage}: {folder}")
    return folder, root


def _source_from_payload(
    source_stage: str,
    payload: dict[str, Any],
    *,
    require_mtl_image: bool,
    task_timestamp: str | None = None,
) -> ModelFileSource:
    mesh_name = str(payload.get("mesh") or "").strip()
    if not mesh_name:
        raise ValueError(f"{source_stage}.mesh is missing")

    mtl_name = str(payload.get("mtl") or "").strip() or None
    image_name = str(payload.get("image") or "").strip() or None
    if require_mtl_image and (not mtl_name or not image_name):
        raise ValueError(f"{source_stage}.mesh / mtl / image is missing")

    folder, root = _resolve_folder_root(source_stage, payload, task_timestamp=task_timestamp)

    video_name = str(payload.get("video") or "").strip() or None
    video_folder = str(payload.get("video_folder") or INSTANTMESH_VIDEO_FOLDER).strip() or None
    video_root = FOLDER_MAP.get(video_folder) if video_folder else None
    if artifact_root == "model_worker" and video_name and task_timestamp and not payload.get("video_folder"):
        video_folder = None
        video_root = model_debug_dir(task_timestamp)

    return ModelFileSource(
        source_stage=source_stage,
        backend=_default_backend(source_stage, payload),
        payload=payload,
        folder=folder,
        root=root,
        mesh=mesh_name,
        mtl=mtl_name,
        image=image_name,
        video=video_name,
        video_folder=video_folder,
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
    return _source_from_payload(
        _infer_stage_from_model_generation(model_generation),
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


def resolve_runtime_or_generated_source(
    task: dict[str, Any],
    *,
    require_mtl_image: bool = True,
) -> ModelFileSource:
    runtime_source = resolve_runtime_mesh_source(task, require_mtl_image=require_mtl_image)
    if runtime_source is not None:
        return runtime_source
    return resolve_model_generation_source(task, require_mtl_image=require_mtl_image)


def resolve_model_source_from_stage(
    task: dict[str, Any],
    source_stage: str,
    source_mesh: str | None = None,
    *,
    require_mtl_image: bool = False,
) -> ModelFileSource:
    source_stage = str(source_stage or "").strip()
    if source_stage == MODEL_STAGE_RUNTIME_MESH:
        payload = dict(_as_payload(task.get(MODEL_STAGE_RUNTIME_MESH)))
    elif source_stage in {MODEL_STAGE_SAM3D_OBJECTS, MODEL_STAGE_INSTANTMESH}:
        payload = dict(_as_payload(task.get("ModelGeneration")))
        if not payload.get("mesh"):
            raise ValueError("ModelGeneration.mesh is missing")
        if _infer_stage_from_model_generation(payload) != source_stage:
            raise ValueError(f"ModelGeneration source stage is not {source_stage}")
    else:
        raise ValueError(f"Unsupported model source stage: {source_stage}")

    if source_mesh:
        payload["mesh"] = source_mesh
    return _source_from_payload(
        source_stage,
        payload,
        require_mtl_image=require_mtl_image,
        task_timestamp=str(task.get("task_timestamp") or "").strip() or None,
    )
