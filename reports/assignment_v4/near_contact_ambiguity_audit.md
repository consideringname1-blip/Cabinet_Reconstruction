# Assignment v4 near-contact ambiguity audit

Date: 2026-08-07

Status: completed diagnostic audit; **not accepted for dual TSDF**.

## Scope and frozen boundary

This audit recomputes the fixed-pose, fixed-axis, fixed-`q_t` projective
evidence for all 131 unknown regions selected by the verified-depth Assignment
v4 near-contact/difficult-region report. It does not change ownership labels.
Camera poses, calibration, joint type, axis, `q_t`, moving map, hand masks, and
the `0.0002 m/count` registered-depth contract remain frozen. No TSDF, NKSR,
Mesh, region propagation, or joint optimization ran.

The term `near-contact` needs a qualification: this input set is selected by a
small static-versus-drawer evidence margin. The verified run had zero accepted
close static/drawer region pairs. The set therefore contains difficult regions
well outside a literal drawer-side/cabinet-wall contact zone.

## Command and output

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/audit_assignment_v4_near_contact_ambiguity.py \
  --config tools/itaco_moving_map_fix/configs/hololens_near_contact_ambiguity_audit.yaml
```

Primary output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/near_contact_ambiguity_audit_002`

An earlier complete run, `near_contact_ambiguity_audit_001`, is preserved. Its
global target-hit summary counted rows instead of hit points; the per-region
evidence was valid. `_002` corrects the aggregate accounting without overwriting
the earlier output.

## Explicit taxonomy

All thresholds are in YAML. `insufficient` has priority when the original reason
is `unknown_insufficient_evidence`, or either model has fewer than 80 total
testable samples. Among the remainder:

- `static_drawer_both_supported`: both median support values are at least 0.90
  and both median contradiction values are at most 0.10.
- `both_bad`: the region fails that indistinguishably-well-supported criterion.

`both_bad` is a taxonomy name, not a claim that both hypotheses have zero
support. The sole member is a borderline/moderate failure.

| Category | Regions | Fraction |
|---|---:|---:|
| static≈drawer≈1 (`static_drawer_both_supported`) | 116 | 88.55% |
| insufficient | 14 | 10.69% |
| both-bad | 1 | 0.76% |
| total | 131 | 100% |

The recomputed aggregate evidence matches the source Assignment v4 evidence
exactly: maximum absolute difference `0.0`.

## Spatial findings

The 116 both-supported regions span frames 187–213, q=0.0592–0.30023 m, and
layers 0, 3, 6, 10, 11, 12, 24, and 28. Their world-space centroid bounds are
`[-0.6624, -1.6352, -2.2659]` to `[0.5130, -1.0238, -1.0801]`. Visual review
shows broad cabinet-side planes, drawer surfaces, thin cabinet edges, and
floor/background patches—not one compact contact boundary.

The 14 insufficient regions span frames 181–212, q=0.00019–0.30023 m, and
layers 1, 4, 5, and 24. They include large early cabinet surfaces, later
drawer/interior patches, and lower floor-like patches. The exact source pairs
are:

`181/1, 184/1, 185/1, 200/5, 201/5, 202/5, 203/5, 204/24, 205/24, 208/24,
210/4, 210/24, 211/4, 212/4`.

The sole both-bad region is `frame_193_layer_9`, q=0.158384 m, at world centroid
`[-0.149906, -1.249495, -1.167744]`. It is a small horizontal drawer-interior or
bottom-like surface. Its aggregate evidence is not catastrophic: static support
0.76/contradiction 0.24 over 718 testable points, and drawer support
0.8196/contradiction 0.1804 over 3,117 testable points.

## q-dependent visibility

For the 116 both-supported regions, static observability increases with opening:
the early/middle/late q-bin medians are 0.302, 0.537, and 0.647. Drawer
observability remains 0.708, 0.720, and 0.647. Despite that changing visibility,
median support is essentially 1.0 and contradiction essentially 0.0 for both
models. This is an identifiability failure, not primarily missing depth.

The insufficient group has asymmetric or missing evidence. Median support is
zero in all broad q bins. Late-q drawer evidence becomes especially poor:
median observability 0.167 and contradiction 1.0. These regions must remain
unknown rather than being forced into either owner.

The both-bad region changes explanation with q. Static support is stronger at
early q, while drawer support is stronger later; later drawer contradiction is
about 0.206. This temporal swap is visible directly in its per-target grid.

## Predicted target surfaces

For every eligible target frame, the audit draws static and drawer predictions
side by side and records the AutoSeg target surfaces hit by supported, occluded,
contradictory, and unobservable samples.

For the both-supported group, static predictions predominantly receive support
on layer 28 and same-layer targets, while drawer predictions strongly receive
support on layer 6 and layer 28. Static predictions have only 275 contradictory
point-to-proposal memberships; drawer predictions have 23,307. Drawer
contradictions include 4,203 points with no AutoSeg proposal. Nevertheless,
the aggregate medians remain strongly supportive for both hypotheses.

This means the ambiguity is not simply both models hitting the identical target
UID. They can hit different nearby/overlapping surfaces and still satisfy the
registered-depth tolerance. AutoSeg layer IDs remain metadata, never hard
identity constraints.

For insufficient regions, static support is mostly on layer 24 in early frames,
while drawer support is almost absent (270 supported memberships in total).
Drawer contradictions often hit layer 6. For the sole both-bad region, static
predictions tend to hit layer 25 early, while drawer predictions increasingly
hit layer 9 later.

Target proposal masks may overlap, so these global hit totals are
prediction-to-proposal memberships and can double-count a projected point. The
per-region CSVs retain the underlying target-frame and surface breakdown.

## Interpretation and decision

The dominant failure is not `insufficient` evidence and not uniformly bad
geometry. It is that the present depth-aware projective test often supports both
the world-static and frozen-drawer hypotheses on nearby planar/occluded
surfaces. Because this occurs across unrelated spatial regions, resolving it by
loosening thresholds, assigning the remainder to static, or propagating the 116
regions would create systematic leakage.

The next ownership method needs stronger surface identity and negative evidence:
cross-frame surface continuity, explicit target-surface ownership constraints,
and a way to distinguish parallel/nearby planes under occlusion. Until that is
implemented and reviewed, all 131 regions remain unknown and
`ready_for_dual_tsdf=false`.

## Output inventory and validation

The final output contains 807 files (122,850,495 bytes). Each of the 131 region
directories contains:

- `source_region.jpg`
- `q_visibility.png` and `q_visibility.csv`
- `target_surfaces.jpg` and `target_surface_hits.csv`
- `overview.jpg`

Global outputs include category counts, 3D spatial distribution, category-wise
q visibility, target-surface matrices, category contact sheets, CSV/JSON tables,
centroid PLY, and an HTML index.

Validation:

- 131/131 source unknown regions audited.
- 131/131 regions have the six required per-region artifacts.
- Scale-consistency hard gate passed again.
- Maximum source-versus-recomputed evidence difference: 0.0.
- New audit tests plus Assignment v4 regressions: 22/22 passed.
- Python compilation and `git diff --check` passed.
- Original Assignment v4 output was not modified.
- No camera/axis/`q_t`/moving-map update and no TSDF/NKSR/Mesh run.
