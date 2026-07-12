from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ServiceGpuBudget:
    service: str
    required_mib: int
    note: str


@dataclass(frozen=True)
class GpuInfo:
    index: str
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)


@dataclass(frozen=True)
class GpuPlacement:
    service: str
    required_mib: int
    headroom_mib: int
    gpu: GpuInfo | None
    reason: str

    @property
    def gpu_id(self) -> str | None:
        return self.gpu.index if self.gpu is not None else None

    @property
    def fits(self) -> bool:
        return self.gpu is not None and self.gpu.free_mib >= self.required_mib + self.headroom_mib


DEFAULT_BUDGETS_MIB: dict[str, ServiceGpuBudget] = {
    "instantmesh": ServiceGpuBudget(
        service="instantmesh",
        required_mib=18 * 1024,
        note="Conservative single-GPU InstantMesh inference budget; official demo can split across two GPUs to save memory.",
    ),
    "foundationpose": ServiceGpuBudget(
        service="foundationpose",
        required_mib=12 * 1024,
        note="Conservative budget for model-based FoundationPose registration plus rasterization/refinement buffers.",
    ),
    "sam3_image_mask": ServiceGpuBudget(
        service="sam3_image_mask",
        required_mib=12 * 1024,
        note="Conservative SAM3 image mask worker budget with encoder and decoder resident.",
    ),
    "sam3d_objects": ServiceGpuBudget(
        service="sam3d_objects",
        required_mib=24 * 1024,
        note="Conservative SAM3D object generation/postprocess budget.",
    ),
    "dinov2_identity": ServiceGpuBudget(
        service="dinov2_identity",
        required_mib=6 * 1024,
        note="DINOv2 ViT-L identity embedding worker with resident model weights and inference buffers.",
    ),
}


def _env_name(service: str) -> str:
    return "GPU_BUDGET_" + service.upper().replace("-", "_") + "_MIB"


def normalize_service_name(service: str) -> str:
    key = str(service).strip().lower()
    if key not in DEFAULT_BUDGETS_MIB:
        raise ValueError(f"unknown GPU service: {service}")
    return key


def service_budget(service: str) -> ServiceGpuBudget:
    key = normalize_service_name(service)
    default = DEFAULT_BUDGETS_MIB[key]
    required = int(os.environ.get(_env_name(key), str(default.required_mib)))
    return ServiceGpuBudget(service=key, required_mib=max(1, required), note=default.note)


def query_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(GpuInfo(index=parts[0], total_mib=int(parts[1]), used_mib=int(parts[2])))
        except Exception:
            continue
    return gpus


def _allowed_set(allowed_ids: Iterable[str] | None) -> set[str] | None:
    if allowed_ids is None:
        return None
    allowed = {str(item).strip() for item in allowed_ids if str(item).strip()}
    return allowed or None


def choose_gpu_for_service(
    service: str,
    *,
    allowed_ids: Iterable[str] | None = None,
    headroom_mib: int | None = None,
    gpu_snapshot: Iterable[GpuInfo] | None = None,
) -> GpuPlacement:
    budget = service_budget(service)
    try:
        headroom = int(os.environ.get("GPU_BUDGET_HEADROOM_MIB", str(headroom_mib or 2048)))
    except Exception:
        headroom = headroom_mib or 2048

    allowed = _allowed_set(allowed_ids)
    gpus = list(gpu_snapshot) if gpu_snapshot is not None else query_gpus()
    if allowed is not None:
        gpus = [gpu for gpu in gpus if gpu.index in allowed]
    if not gpus:
        return GpuPlacement(
            service=budget.service,
            required_mib=budget.required_mib,
            headroom_mib=headroom,
            gpu=None,
            reason="no GPU snapshot available",
        )

    required_with_headroom = budget.required_mib + max(0, headroom)
    fitting = [gpu for gpu in gpus if gpu.free_mib >= required_with_headroom]
    if fitting:
        gpu = max(fitting, key=lambda item: item.free_mib)
        return GpuPlacement(
            service=budget.service,
            required_mib=budget.required_mib,
            headroom_mib=headroom,
            gpu=gpu,
            reason=f"selected GPU {gpu.index} with {gpu.free_mib} MiB free",
        )

    gpu = max(gpus, key=lambda item: item.free_mib)
    return GpuPlacement(
        service=budget.service,
        required_mib=budget.required_mib,
        headroom_mib=headroom,
        gpu=gpu,
        reason=(
            f"best-effort GPU {gpu.index} has {gpu.free_mib} MiB free; "
            f"needs {required_with_headroom} MiB"
        ),
    )


def cuda_env_for_service(
    service: str,
    *,
    allowed_ids: Iterable[str] | None = None,
    headroom_mib: int | None = None,
) -> tuple[dict[str, str], GpuPlacement]:
    placement = choose_gpu_for_service(service, allowed_ids=allowed_ids, headroom_mib=headroom_mib)
    if placement.gpu_id is None:
        return {}, placement
    return {"CUDA_VISIBLE_DEVICES": placement.gpu_id}, placement


def budget_report() -> Mapping[str, dict[str, object]]:
    gpus = query_gpus()
    return {
        key: {
            "required_mib": service_budget(key).required_mib,
            "note": service_budget(key).note,
            "placement": choose_gpu_for_service(key, gpu_snapshot=gpus).__dict__,
        }
        for key in sorted(DEFAULT_BUDGETS_MIB)
    }
