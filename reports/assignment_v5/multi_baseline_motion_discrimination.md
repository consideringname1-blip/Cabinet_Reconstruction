# Assignment v5 multi-baseline motion discrimination

## Outcome

The frozen, control-first experiment completed once and failed the active
co-moving-box gate. It therefore stopped before the 116 ambiguity regions, did
not run region propagation or reconstruction, and remains
`ready_for_dual_tsdf=false`.

The useful result is narrower: multi-baseline evidence correctly separates the
drawer front and the documented floor controls, and short/medium box baselines
behave like a moving surface. Long box baselines hit unrelated valid depth while
both motion hypotheses are outside the calibrated plateau noise distribution.
The current long-baseline gate checks valid source-index coverage, not continued
finite-surface identity, so those observations contaminate the aggregate model
evidence. No parameter was changed and the control evaluation was not rerun.

## Frozen inputs and invariants

- Branch: `agent/region-assignment-v4`
- Experiment parent: `707634a6936c963c5d4cee5fea79adda70ac2c09`
- Registered-depth unit: verified `0.0002 m/count`
- Camera poses, prismatic axis, and per-frame `q_t`: frozen
- Source anchor: immutable and source-indexed
- Target AutoSeg proposal matching / surface hopping: absent
- Motion discrimination: forward and backward over the full ACTIVE_MOTION run
- Causal disocclusion: chronological forward only
- V4 formal labels: unchanged
- No SAM2, tracker, region propagation, TSDF, NKSR, Mesh, or Assignment v6

The classifier/noise configuration, source hashes, and noise-model hash were
written before the held-out control annotations were read. The run used no
post-gate threshold tuning and the output directory is non-overwriting.

## Plateau registered-depth noise floor

The noise model uses same-plateau, immutable source-index projections over 42
directed frame pairs. It does not use box results or control annotations.

| Surface category | Samples | Median | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| Interior | 8,555 | 1.68 mm | 2.77 mm | 3.23 mm | 4.64 mm |
| Depth/geometry edge | 183,296 | 2.90 mm | 6.69 mm | 8.65 mm | 14.40 mm |
| Combined | 191,851 | 2.81 mm | 6.57 mm | 8.52 mm | 14.23 mm |

The edge category is deliberately conservative and dominates the sample count;
this is visible in the saved histogram and is a limitation of this noise model,
not a threshold changed against the box controls.

## Residual versus discriminative displacement

`d_discriminative = |n·axis| |Δq|`. Values below are medians across accepted
observations from the five active-box controls.

| Baseline | Observations | d median (range) | Static median / p90 | Moving median / p90 | Common source coverage |
|---|---:|---:|---:|---:|---:|
| Short | 14 | 8.30 mm (5.63–13.46) | 13.36 / 17.02 mm | 2.77 / 4.00 mm | 0.940 |
| Medium | 12 | 29.80 mm (21.76–46.15) | 51.83 / 56.60 mm | 4.08 / 6.75 mm | 0.667 |
| Long | 75 | 209.10 mm (50.08–296.09) | 122.99 / 178.96 mm | 298.44 / 368.63 mm | 0.378 |

Short and medium baselines show the intended discrimination: the drawer model
stays near the plateau distribution while the static residual grows. At long
baselines both models are far outside that distribution. The large moving
residual is direct evidence that a valid measured target pixel is not sufficient
to prove that the original box surface is still observable.

For comparison, drawer-front controls remain stable under the moving model:
medium median residual is 3.33 mm versus 105.54 mm static, and long median is
4.19 mm versus 786.22 mm static. Documented floor controls show the symmetric
behavior: combined long observations have 3.97 mm static versus 17.78 mm moving.

## Control gate

| Check | Result |
|---|---|
| Drawer front | 4/4 `MOVING_LINK` — pass |
| Active co-moving box | 0/5 `MOVING_LINK`, 5 `UNKNOWN` — fail |
| Documented floor | 0/21 `MOVING_LINK` — pass |
| Plateau-only | 0/2 ownership — pass |
| V5 synthetic/regression tests | 52/52 — pass |
| V4 regression tests | 34/34 — pass |

All five box controls are `UNKNOWN` with reason
`both_models_incompatible_with_frozen_plateau_noise`. They were not mislabeled
world-static, but they do not satisfy the required ≥4/5 moving recall. The floor
result contains 12 `WORLD_STATIC` and 9 `UNKNOWN`, with no false moving labels.

## 116-region decision and remaining blocker

The hard gate failed, so the 116 regions were not run. No 116 ownership counts
exist for this experiment.

The remaining blocker is finite-surface observability at long baselines. The
current common-source-index coverage test only proves that both predicted rays
land on valid registered depth. It does not prove that either ray still observes
the immutable source surface. A future separately authorized experiment would
need a source-identity-aware long-baseline censoring rule frozen independently
of these now-observed controls. Reusing this control set to tune that rule would
not be a held-out evaluation.

`ready_for_dual_tsdf=false`.

## Commands and outputs

Tests:

```bash
/workspace_whz/envs/video_articulation/bin/python -m unittest discover \
  -s tools/itaco_ownership_v5/tests -p 'test_*.py' -v
/workspace_whz/envs/video_articulation/bin/python -m unittest discover \
  -s tools/itaco_region_assignment_v4/tests -p 'test_*.py' -v
git diff --check
```

Experiment:

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/audit_assignment_v5_multi_baseline.py \
  --config tools/itaco_moving_map_fix/configs/hololens_ownership_v5_multi_baseline.yaml
```

- Full local output:
  `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/multi_baseline_motion_discrimination_001`
- Tracked review bundle:
  `reports/assignment_v5/results/multi_baseline_motion_discrimination/`
- Main figures: plateau-noise histogram plus residual/discriminative-displacement
  plots for drawer front, active box, documented floor, and provisional static
  controls.
