# Assignment v5 transition-event logic repair

Date: 2026-08-07
Branch: `agent/region-assignment-v4`
Exact parent: `70623ad22850231b0f13b6bc7b647c3d4f2f7d37`
Status: control hard gate failed; stopped before the 116-region diagnostic
`ready_for_dual_tsdf=false`

## Outcome

The real-data adapter was replaced with a directed, local transition path. It no longer uses Assignment-v4 `per_target` rows to choose observations, never uses unordered `frozenset` transition keys, and cannot search a target AutoSeg proposal to establish physical identity. Each hypothesis now projects immutable, indexed source samples into the target registered-depth image, reads measured depth at the same predicted pixels, unprojects those measurements, and evaluates paired residuals under either the world-static or frozen articulated transform.

The new real-data control gate did not pass. The four drawer-front controls were all recovered as `MOVING_LINK`, all 21 documented floor false positives produced zero `MOVING_LINK`, and both plateau controls created no ownership. However, all five active co-moving box controls remained `UNKNOWN`, so the required 80% true-positive gate failed and the program stopped before the 116 regions. No threshold was changed and the run was not retried.

| Control group | Count | MOVING_LINK | WORLD_STATIC | UNKNOWN | Gate result |
|---|---:|---:|---:|---:|---|
| Drawer front (195--198) | 4 | 4 | 0 | 0 | pass, 4/4 |
| Active co-moving box (202--206) | 5 | 0 | 0 | 5 | **fail, 0% < 80%** |
| Documented floor false positives | 21 | 0 | 0 | 21 | pass, 0 false moving |
| Plateau-only layer2 (211/213) | 2 | 0 | 0 | 2 | pass |
| Provisional v4 static (not GT) | 3 | 0 | 0 | 3 | diagnostic only |

The box controls are not lost: both models form valid local source-anchor chains. Their median moving-canonical residuals are 2.47, 2.76, 4.18, 2.52, and 6.73 mm, while the world-static medians are 17.70, 13.40, 16.22, 13.51, and 19.48 mm. Nevertheless, the static chain remains inside the frozen acceptance limits on these late-motion local intervals. The specification explicitly requires both-compatible cases to stay `UNKNOWN` (`motion_tangent_or_repeated_surface_ambiguity`) rather than selecting the lower residual by a score margin. This is the direct cause of the failed box gate.

The underlying observability issue is that frames 202--206 are near the end of motion and the allowed local intervals have small displacement. Within a maximum three-frame gap their `delta q` is comparable to or smaller than the registered-depth support band, so the static projection can remain compatible even though the moving residual is much lower. Resolving this requires new identity/discriminative evidence or an independently calibrated noise model, not post-gate threshold tuning.

## Repaired data flow

1. Raw depth timestamps and frozen `q_t` for frames 177--213 produce `CLOSED_PLATEAU`, `ACTIVE_MOTION`, `OPEN_PLATEAU`, or explicit `INACTIVE_INTERMEDIATE` states.
2. Directed local transitions are built directly from continuous raw `ACTIVE_MOTION` intervals with the frozen limits `0.005 <= |delta q| <= 0.060 m`, frame gap at most 3, and elapsed time at most 0.8 s. The run produced 66 accepted transitions and 39 rejected candidates.
3. A source proposal supplies only the initial finite mask. Its valid registered-depth samples receive immutable source indices.
4. `WORLD_STATIC` and `MOVING_LINK` predictions project those same indexed samples into each local target. Target depth is read and unprojected at the predicted pixels; target proposals are consulted only afterward for provenance.
5. Consecutive verified observations create symmetric world-static or articulated motion routes. Both-compatible chains remain unknown.
6. The boundary detector separately records RGB clipping, depth-footprint clipping, depth jumps, normal/plane jumps, invalid holes, and segmentation-only edges. Tangent motion can proceed only with an observed leading/trailing physical edge and moving edge support.
7. Causal reveal transforms the existing 3-of-4 trusted drawer-front silhouette and measures IoU, boundary displacement, local ray depth gap, mask departure, new registered-depth support, real frame/time gap, and future world residual. No accepted causal event occurred in the controls.
8. Evaluation annotations are loaded only after algorithm configuration and never enter classifier features.

## Control and test gate

- Assignment-v5 tests: 45/45 passed, including the 22 new transition-event tests.
- Assignment-v4 regression tests: 34/34 passed.
- Registered-depth provenance plus v3 projective regressions: 15/15 passed.
- Real registered-depth hard gate passed again at `0.0002 m/count` before frame loading.
- Control gate: failed only `active_comoving_box_at_least_80pct_moving_link`.
- Because the gate failed, `region propagation ran = false` and `116 diagnostic ran = false`.

## Required questions

1. **Did the implementation completely stop reusing v4 `per_target`?** Yes. Formal target selection comes only from raw-frame local transitions. V4 evidence is read only to locate the frozen source regions and preserve formal labels.
2. **Can it still search a target proposal as identity?** No. The projective identity function has no target-proposal argument. AutoSeg overlap is computed only after the anchor state is fixed and is marked post-hoc provenance.
3. **Is the `WORLD_STATIC` route implemented?** Yes. It is symmetric with the articulated route and does not imply cabinet membership. No control was released as `WORLD_STATIC` in this run because the moving model was also compatible wherever the static chain was positive.
4. **Are all causal fields really measured?** Yes. Transition timing/direction, trusted-front provenance, silhouette agreement, depth gap, boundary distance, departure, new support, and persistent world residual all come from real frames. No field is filled with a constant `true`, and world residual is not fixed to zero.
5. **What was the control-gate result?** Failed: the co-moving box true-positive fraction was 0/5; all other hard checks passed.
6. **Did floor false positives become zero?** Yes: 0/21 floor controls were `MOVING_LINK`.
7. **Is the drawer front still identifiable?** Yes: 4/4 front controls were `MOVING_LINK`.
8. **Is the co-moving box identifiable?** No: 0/5 were released; all five are both-compatible `UNKNOWN` despite substantially lower moving residuals.
9. **Did the 116-region diagnostic run?** No. The hard gate stopped execution first.
10. **What are the 116 resolved/unknown results?** Not applicable; no 116-region decisions were computed or written.
11. **Is `ready_for_dual_tsdf` false?** Yes. It remains explicitly false.

## Outputs and visual review

Local full output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/transition_event_logic_repair_001`

Tracked review bundle:

`reports/assignment_v5/results/transition_event_logic_repair/`

Selected files:

- [Control gate](results/transition_event_logic_repair/control_evaluation_report.json)
- [Control summary](results/transition_event_logic_repair/control_region_summary.csv)
- [Motion-state timeline](results/transition_event_logic_repair/motion_state_timeline.png)
- [Directed local-transition timeline](results/transition_event_logic_repair/local_transition_timeline.png)
- [Drawer-front controls](results/transition_event_logic_repair/control_contact_sheets/drawer_front.jpg)
- [Co-moving box controls](results/transition_event_logic_repair/control_contact_sheets/active_comoving_box.jpg)
- [Documented floor controls](results/transition_event_logic_repair/control_contact_sheets/documented_floor_false_positive.jpg)
- [Plateau-only controls](results/transition_event_logic_repair/control_contact_sheets/plateau_only.jpg)

The local output keeps all per-control source overlays, source-anchor transition panels, physical-boundary panels, causal-reveal panels, and JSON diagnostics. The tracked bundle is 3.1 MB and contains selected representatives rather than the complete image tree.

## Remaining blockers

- Late active-motion box observations do not separate the two models under the current physically frozen local-transition and registered-depth uncertainty contract.
- The real controls produced no positive measured causal reveal, so that independent world-static family remains unvalidated on this sequence.
- Provisional static controls are not manual ground truth and also remain both-compatible.
- The 116 regions remain untouched by this repair run; the historical `failed_001` 24-moving result and `corrected_002` 116-unknown result remain preserved.

The next valid step is to design and separately authorize a stronger source identity signal or a sensor-noise-calibrated likelihood that can distinguish the co-moving box without reviving floor false positives. This run does not justify Assignment v6 or reconstruction.
