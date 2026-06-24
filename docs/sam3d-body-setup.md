# SAM 3D Body Setup

`code/reconstruction/sam3d-body` is tracked as a git submodule pointing to
`https://github.com/facebookresearch/sam-3d-body.git`.

Keep the submodule source clean. The upstream DINOv3 backbone currently calls
`torch.hub.load("facebookresearch/dinov3", ...)` without an explicit branch ref.
With PyTorch 2.5 this can still contact GitHub to resolve the default branch
even when `/root/.cache/torch/hub/facebookresearch_dinov3_main` already exists.
If GitHub returns a transient 504, SAM 3D Body startup may fail before it reaches
the cached DINOv3 code.

Do not patch the submodule just to add `:main`. Prefer keeping the upstream
submodule pointer clean and pre-warming the default caches on the deployment
machine.

Current path and cache locations:

- Hugging Face hub: `/root/.cache/huggingface/hub`
- SAM 3D Body checkpoint snapshot:
  `/root/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/11aaa346c7204874a1cbafe3d39a979080b2c55a`
- Repository symlink, resolved from `path_config.SAM3D_BODY_ROOT`:
  `code/reconstruction/sam3d-body/checkpoints/sam-3d-body-dinov3`
- Torch hub DINOv3 cache:
  `/root/.cache/torch/hub/facebookresearch_dinov3_main`
- Detectron2 ViTDet cache:
  `/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl`

The worker runs `code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py`
with `path_config.SAM3D_BODY_MESH_STAGE_PY`. By default this is the server
Python runtime; set `SAM3D_BODY_PY` only when the deployment uses a separate SAM
3D Body environment.

Useful cache pre-warm commands:

```bash
${SAM3D_BODY_PY:-/opt/miniconda/envs/server/bin/python} -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3')"
${SAM3D_BODY_PY:-/opt/miniconda/envs/server/bin/python} -c "import torch; torch.hub.load('facebookresearch/dinov3:main', 'dinov3_vith16plus', source='github', pretrained=False, drop_path=0.0)"
```

