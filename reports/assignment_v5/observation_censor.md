# Assignment v5 observation censor

## Outcome

The observation censor was implemented on `agent/region-assignment-v4` without
changing the frozen HoloLens camera poses, prismatic axis, `q_t`, registered
depth, source anchors, causal-disocclusion logic, or tangent/finite-edge gate.

The previously used controls passed as a regression:

| Regression group | Result |
|---|---:|
| Drawer front | 4/4 `MOVING_LINK` |
| Active co-moving box | 5/5 `MOVING_LINK`, 0 `WORLD_STATIC` |
| Documented floor | 0/21 `MOVING_LINK` (19 static, 2 unknown) |
| Plateau-only | 0/2 ownership |

These controls have already been examined in earlier experiments. This result
is explicitly a regression, not fresh held-out validation. Because the
regression passed, the implementation/config/noise hashes were frozen and the
116-region diagnostic ran. It produced 0 moving, 56 world-static, 60 unknown,
and 0 conflicting regions. Formal Assignment-v4 labels were not changed.

`ready_for_dual_tsdf=false`.

## Frozen noise criterion

The censor loaded the prior noise artifacts directly from the preserved
multi-baseline output. It did not refit plateau noise.

- Original noise-model hash:
  `d1c2a22cb4e26b05adafe1e08334d449d7b41e19f350ae5cecca9ae6e30f6d67`
- Statistic: per-model observation `mean_noise_surprisal`
- Frozen threshold source: `observation_mean_surprisal_p95`
- Quantile: p95
- Numerical threshold: `2.99549771060747`
- Compatible: mean surprisal at or below the frozen p95
- OOD: mean surprisal above the frozen p95

No residual-distance threshold was added for censoring, and the box result was
not used to change the noise threshold.

Each observation receives exactly one state:

| Static | Moving | Observation state | Aggregation behavior |
|---|---|---|---|
| compatible | OOD | `WORLD_STATIC_EVIDENCE` | retained |
| OOD | compatible | `MOVING_LINK_EVIDENCE` | retained |
| compatible | compatible | `AMBIGUOUS_COMPATIBLE` | neutral |
| OOD | OOD | `IDENTITY_LOST_CENSORED` | excluded completely |

Insufficient common source visibility is also censored before the noise test.
Censored observations have no aggregation likelihood and are excluded from
posterior, trend, effective chain length, and baseline evidence counts. The
implementation never selects the smaller residual when both models are OOD.

## Censor counts by regression group

Values are `IDENTITY_LOST_CENSORED / total` observations.

| Group | Short | Medium | Long |
|---|---:|---:|---:|
| Drawer front | 0/1 | 0/17 | 20/75 |
| Active box | 0/14 | 0/12 | 77/85 |
| Documented floor | 0/25 | 0/60 | 181/371 |
| Plateau-only | 0/0 | 0/0 | 0/0 |
| Provisional static | 0/0 | 0/10 | 27/59 |

The documented floor has no moving-evidence observations: short is 25/25
ambiguous, medium is 59 ambiguous plus one static-evidence observation, and long
contains 173 static evidence, 17 ambiguous, and 181 censored observations.

## Active box before and after censoring

All 26 short/medium observations are retained as moving evidence. Among 85 long
observations, 67 are both-OOD, 10 fail common-source visibility, and 8 remain
valid moving evidence.

| Source frame | Censored | Evidence before | Evidence after | Decision |
|---:|---:|---:|---:|---|
| 202 | 15 | -202.90 | 54.66 | moving |
| 203 | 15 | -207.36 | 48.50 | moving |
| 204 | 15 | -187.31 | 66.60 | moving |
| 205 | 17 | -175.20 | 37.90 | moving |
| 206 | 15 | -194.15 | 80.88 | moving |

Before censoring, unrelated long-baseline depth dominated every box posterior
in the wrong direction. After censoring, each source has 5–8 informative
observations, all `MOVING_LINK_EVIDENCE`, including medium support.

For censored long observations, the median static/moving residuals are
123.38/297.48 mm. The eight retained long moving observations instead have
105.39 mm static residual and 3.29 mm moving residual. Thus a large wrong-model
residual alone is not censored: the observation is retained when the other model
still lies in the frozen noise distribution.

As a reporting-only check, 81 box-long observations have at least one model
median residual at or above 100 mm; 76 are censored and five are retained moving
evidence. The 100 mm value is not used by the classifier.

## 116-region diagnostic

This diagnostic is not fresh held-out validation.

- Decisions: 0 moving, 56 world-static, 60 unknown, 0 conflicting
- Total observations: 1,857
- `IDENTITY_LOST_CENSORED`: 860
- `AMBIGUOUS_COMPATIBLE`: 322
- Static-evidence observations: 616
- Moving-evidence observations: 59
- Observations entering final per-region aggregation after excitation filtering:
  466

By baseline:

| Baseline | Total | Static evidence | Moving evidence | Ambiguous | Censored |
|---|---:|---:|---:|---:|---:|
| Short | 147 | 28 | 10 | 103 | 6 |
| Medium | 259 | 42 | 17 | 161 | 39 |
| Long | 1,451 | 546 | 32 | 58 | 815 |

`WORLD_STATIC` means only that the observed surface follows the frozen world
model. It does not mean cabinet membership. The 56 static diagnostics span
multiple proposal layers and do not resolve drawer-side versus cabinet-inner-
wall ownership by themselves.

## Validation and invariants

- Assignment-v5 tests: 63/63 passed
- Assignment-v4 regression tests: 34/34 passed
- Registered-depth scale gate: passed
- Source anchor remains immutable and source-indexed
- Forward/backward motion discrimination remains enabled
- Causal disocclusion remains chronological forward only
- Target proposal matching and surface hopping remain absent
- No SAM2, region propagation, TSDF, NKSR, Mesh, camera optimization, axis
  refinement, or `q_t` refinement ran

## Outputs and command

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/audit_assignment_v5_observation_censor.py \
  --config tools/itaco_moving_map_fix/configs/hololens_ownership_v5_observation_censor.yaml
```

- Full local output:
  `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/observation_censor_002`
- Tracked review bundle:
  `reports/assignment_v5/results/observation_censor/`
- Main figures:
  `control_censor_states_by_baseline.png`,
  `active_box_evidence_before_after.png`, and
  `active_box_observation_states.png`

## Remaining blocker

The censor fixes the identified unrelated-depth failure and passes the seen
regression, but there is still no fresh held-out control set or manual ground
truth for the 116 regions. The 56 world-static diagnostic regions must not be
treated as a cabinet mask or sent into fusion without independent evaluation.

`ready_for_dual_tsdf=false`.
