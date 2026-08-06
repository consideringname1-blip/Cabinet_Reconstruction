# iTACO proposal moving-map minimal fix

This package is an independent adapter. It does not modify the official
`video2articulation` source tree and does not call the Stage 1/1.5 track code.

The three modes are:

- `official`: free proposal scalars and per-frame min-max, without sensor gate.
- `gate_only`: the same proposal representation and min-max, with RGB/depth/
  original-hand support in the Chamfer loss and outputs.
- `gate_no_minmax`: the same support plus sigmoid-bounded proposal logits and
  mean aggregation for any residual overlap. No per-frame min-max is used.

Run from `/workspace_whz` with the `video_articulation` environment:

```bash
CFG=tools/itaco_moving_map_fix/configs/hololens_validity_no_minmax.yaml
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
  envs/video_articulation/bin/python -m tools.itaco_moving_map_fix.cli run \
  --config "$CFG" --mode official
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
  envs/video_articulation/bin/python -m tools.itaco_moving_map_fix.cli run \
  --config "$CFG" --mode gate_only
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
  envs/video_articulation/bin/python -m tools.itaco_moving_map_fix.cli run \
  --config "$CFG" --mode gate_no_minmax
envs/video_articulation/bin/python -m tools.itaco_moving_map_fix.cli compare \
  --config "$CFG"
```

The evaluation annotation file is never imported by the optimizer. It exists
only for the post-run semantic audit.
