# iTACO track-motion stage 1

This package implements only:

```text
frame manifest -> reference selection -> validity -> explicit proposals
-> short 3D tracks/clusters -> static/moving/unknown
```

The configured `T_world_camera`, joint type, axis, origin (if revolute), and
per-frame `q_t` are fixed inputs. The package has no camera optimizer, joint
estimator/model selector, free-SE(3), TSDF, NKSR, or mesh entry point.

Run:

```bash
/workspace_whz/envs/video_articulation/bin/python -m tools.itaco_track_motion_v1.cli \
  --config tools/itaco_track_motion_v1/configs/hololens_2026-07-30-002840.yaml \
  --output data/output/itaco_track_motion_stage1/hololens_2026-07-30-002840_trial_001
```

The processing-view manifest may enumerate any view/order. The implementation
does not contain source frame ranges, AutoSeg UIDs, object positions, floor
heights, or semantic color rules. AutoSeg layer indices are retained only as
source provenance; persistent proposal IDs are inferred by mask association.

## Stage 1.5

Stage 1.5 keeps the stage-1 camera poses, known joint model/axis/state, and the
resolved stage-1 classifier mapping fixed. It adds only hand-mask recovery,
derived depth-quality provenance, short-gap track reassociation, independent
manual evaluation, and observability diagnostics. It has no joint/camera
optimizer or reconstruction entry point.

The configured hand re-detection command must be run first; its manifest and
command are hashed into the output environment manifest. Then run:

```bash
PYTHONPATH=/workspace_whz \
/workspace_whz/envs/video_articulation/bin/python \
  -m tools.itaco_track_motion_v1.cli_stage1_5 \
  --config tools/itaco_track_motion_v1/configs/hololens_2026-07-30-002840_stage1_5.yaml \
  --output data/output/itaco_track_motion_stage1_5/new_empty_trial
```

Manual annotations are loaded only after tracking and classification have
finished. A zero verified-error count in the gap audit is not treated as proof
unless the configured minimum number of accepted connections was actually
covered by annotations.
