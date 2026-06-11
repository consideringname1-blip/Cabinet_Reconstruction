from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import _bootstrap

from config import (
    SAM3_OUTPUT_ROOT,
    SAM3D_OBJECTS_OUTPUT_MESHES,
    SAM3D_OBJECTS_CONFIG,
    SAM3D_OBJECTS_POSTPROCESS_SCRIPT,
    SAM3D_OBJECTS_PY,
    SAM3D_OBJECTS_ROOT,
)
from settings import (
    SAM3D_OBJECTS_ATTN_BACKEND,
    SAM3D_OBJECTS_SEED,
)
from model_generation_common import (
    BACKEND_SAM3D_OBJECTS,
    MODEL_STAGE_SAM3D_OBJECTS,
    SAM3D_OBJECTS_MESH_FOLDER,
    build_model_generation_payload,
)
from object_alignment_common import resolve_blender_path
from stage_common import ensure_file, load_stage_task, resolve_python
from subprocess_stream import stream_command
from task_json import save_task_json


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = text.strip("._")
    return text or "sam3d_objects"


def require_sam3_artifacts(task: dict[str, Any]) -> tuple[Path, Path, str]:
    sam3_name = task.get("sam3Name") or {}
    color_name = sam3_name.get("color")
    mask_name = sam3_name.get("mask")
    if not color_name:
        raise ValueError("sam3Name.color is missing")
    if not mask_name:
        raise ValueError("sam3Name.mask is missing")

    color_path = ensure_file(SAM3_OUTPUT_ROOT / str(color_name), "SAM3 color image")
    mask_path = ensure_file(SAM3_OUTPUT_ROOT / str(mask_name), "SAM3 mask image")
    return color_path, mask_path, safe_name(Path(str(color_name)).stem)


def run_worker(
    color_path: Path,
    mask_path: Path,
    raw_glb_path: Path,
    config_path: Path,
    seed: int,
) -> None:
    from PIL import Image
    import numpy as np

    inference_mod = import_sam3d_objects_inference()
    inference = inference_mod.Inference(str(config_path), compile=False)

    image = np.array(Image.open(color_path).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    output = inference(image, mask, seed=seed)
    mesh = output.get("glb")
    if mesh is None:
        raise RuntimeError("SAM3D Objects did not return a mesh/glb output")

    raw_glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(raw_glb_path))


def import_sam3d_objects_inference():
    attn_backend = str(os.environ.get("SAM3D_OBJECTS_ATTN_BACKEND") or "").strip()
    if attn_backend:
        os.environ["ATTN_BACKEND"] = attn_backend
        os.environ["SPARSE_ATTN_BACKEND"] = attn_backend

    root = SAM3D_OBJECTS_ROOT.resolve()
    notebook = root / "notebook"
    for path in (notebook, root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    patched = False
    original_get_device_name = None
    if attn_backend:
        import torch

        original_get_device_name = torch.cuda.get_device_name

        def get_device_name_without_forced_flash(*args, **kwargs):
            name = str(original_get_device_name(*args, **kwargs))
            for token in ("A100", "H100", "H200"):
                name = name.replace(token, token.replace("100", "-100").replace("200", "-200"))
            return name

        torch.cuda.get_device_name = get_device_name_without_forced_flash
        patched = True

    try:
        import inference as inference_mod
    finally:
        if patched:
            import torch

            torch.cuda.get_device_name = original_get_device_name

    return inference_mod


def run_sam3d_generation(
    *,
    color_path: Path,
    mask_path: Path,
    raw_glb_path: Path,
    config_path: Path,
) -> None:
    env = os.environ.copy()
    env["SAM3D_OBJECTS_ATTN_BACKEND"] = str(SAM3D_OBJECTS_ATTN_BACKEND or "")
    if SAM3D_OBJECTS_ATTN_BACKEND:
        env["ATTN_BACKEND"] = str(SAM3D_OBJECTS_ATTN_BACKEND)
        env["SPARSE_ATTN_BACKEND"] = str(SAM3D_OBJECTS_ATTN_BACKEND)

    sam3d_python = resolve_python(SAM3D_OBJECTS_PY)
    sam3d_bin_path = Path(sam3d_python).resolve().parent
    sam3d_prefix = sam3d_bin_path.parent
    env["CONDA_PREFIX"] = str(sam3d_prefix)
    env.setdefault("CUDA_HOME", str(sam3d_prefix))
    env["PATH"] = str(sam3d_bin_path) + os.pathsep + env.get("PATH", "")

    try:
        stream_command(
            [
                sam3d_python,
                str(Path(__file__).resolve()),
                "--worker",
                str(color_path),
                str(mask_path),
                str(raw_glb_path),
                str(config_path),
                str(int(SAM3D_OBJECTS_SEED)),
            ],
            cwd=SAM3D_OBJECTS_ROOT,
            env=env,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def run_sam3d_postprocess(
    *,
    raw_glb_path: Path,
    obj_path: Path,
    mtl_path: Path,
    texture_path: Path,
    stats_path: Path,
) -> dict[str, Any]:
    blender_bin = resolve_blender_path()
    command = [
        str(blender_bin),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(SAM3D_OBJECTS_POSTPROCESS_SCRIPT),
        "--",
        str(raw_glb_path),
        str(obj_path),
        str(mtl_path),
        str(texture_path),
        str(stats_path),
    ]
    print("[DEBUG] running:", " ".join(command), flush=True)
    try:
        stream_command(command, check=True)
    except Exception as exc:
        raise RuntimeError(f"SAM3D Objects postprocess failed: {exc}") from exc

    ensure_file(stats_path, "SAM3D Objects postprocess stats")
    return json.loads(stats_path.read_text(encoding="utf-8"))


def run_sam3d_objects(json_path: Path, task: dict[str, Any]) -> None:
    color_path, mask_path, stem = require_sam3_artifacts(task)
    config_path = ensure_file(SAM3D_OBJECTS_CONFIG, "SAM3D Objects config")

    output_stem = f"{stem}_sam3d_processed"
    raw_glb_path = SAM3D_OBJECTS_OUTPUT_MESHES / f"{stem}_sam3d_raw.glb"
    obj_path = SAM3D_OBJECTS_OUTPUT_MESHES / f"{output_stem}.obj"
    mtl_path = SAM3D_OBJECTS_OUTPUT_MESHES / f"{output_stem}.mtl"
    texture_path = SAM3D_OBJECTS_OUTPUT_MESHES / f"{output_stem}.png"
    stats_path = SAM3D_OBJECTS_OUTPUT_MESHES / f"{output_stem}_postprocess.json"

    run_sam3d_generation(
        color_path=color_path,
        mask_path=mask_path,
        raw_glb_path=raw_glb_path,
        config_path=config_path,
    )
    ensure_file(raw_glb_path, "SAM3D Objects raw GLB")
    postprocess_info = run_sam3d_postprocess(
        raw_glb_path=raw_glb_path,
        obj_path=obj_path,
        mtl_path=mtl_path,
        texture_path=texture_path,
        stats_path=stats_path,
    )

    for output_path, label in (
        (raw_glb_path, "SAM3D Objects raw GLB"),
        (obj_path, "SAM3D Objects processed obj"),
        (mtl_path, "SAM3D Objects processed mtl"),
        (texture_path, "SAM3D Objects processed placeholder texture"),
    ):
        ensure_file(output_path, label)

    sam3d_payload = build_model_generation_payload(
        backend=BACKEND_SAM3D_OBJECTS,
        source_stage=MODEL_STAGE_SAM3D_OBJECTS,
        mesh=obj_path.name,
        mtl=mtl_path.name,
        image=texture_path.name,
        mesh_folder=SAM3D_OBJECTS_MESH_FOLDER,
        runtime_ready=False,
        extra={
            "raw_glb": raw_glb_path.name,
            "source_color": color_path.name,
            "source_mask": mask_path.name,
            "config": str(config_path),
            "seed": int(SAM3D_OBJECTS_SEED),
            "attn_backend": str(SAM3D_OBJECTS_ATTN_BACKEND or ""),
            "postprocess": postprocess_info,
        },
    )
    task["SAM3DObjects"] = sam3d_payload
    task["ModelGeneration"] = dict(sam3d_payload)
    save_task_json(json_path, task)
    print(f"[OK] sam3d_objects mesh: {obj_path}")


def main() -> int:
    try:
        if len(sys.argv) == 7 and sys.argv[1] == "--worker":
            run_worker(
                Path(sys.argv[2]),
                Path(sys.argv[3]),
                Path(sys.argv[4]),
                Path(sys.argv[5]),
                int(sys.argv[6]),
            )
            return 0

        json_path, task = load_stage_task(
            sys.argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_sam3d_objects_from_json.py <task_meta.json or filename>",
            stage_name="sam3d_objects",
        )
        run_sam3d_objects(json_path, task)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
