from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceGpuBudget:
    service: str
    required_mib: int
    note: str


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
    if required <= 0:
        raise ValueError(f"{_env_name(key)} must be a positive MiB value")
    return ServiceGpuBudget(service=key, required_mib=required, note=default.note)
