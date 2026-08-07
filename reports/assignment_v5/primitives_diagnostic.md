# Assignment v5 physical-event primitives diagnostic

Date: 2026-08-07

## Scope

This is the deliberately limited first implementation step from the Assignment
v5 Ownership Logic Specification. It implements and tests only:

1. frozen-`q_t` motion-state decomposition;
2. ACTIVE_MOTION transition representation;
3. source-anchored finite-surface verification with no surface hopping;
4. local causal static-disocclusion checks;
5. independent physical evidence-family bookkeeping; and
6. generic prismatic/revolute `T(q)`.

It does **not** connect these primitives to AutoSeg, SAM2, real-data dense
ownership, object-scope filtering, geometry fusion, TSDF, NKSR, Mesh, GLB, or
URDF. It creates no new real-data ownership labels.

## Implemented semantics

- Every target observation is compared directly with the immutable source
  finite support. A target observation never becomes the next anchor. Nearby
  alternative geometry yields `IDENTITY_LOST`, not substitution.
- Only observations attached to an ACTIVE_MOTION transition are eligible to
  create motion ownership. Plateau compatibility can confirm geometry later,
  but cannot originate ownership.
- Causal static disocclusion requires the full local chain: verified moving
  occluder, known articulated motion, local depth relationship, silhouette
  boundary proximity, temporally adjacent reveal, and persistent world-static
  geometry. A large same-ray depth gap is rejected.
- Evidence is keyed by physical family. Multiple metrics from the same chain
  update one family record rather than creating extra votes.
- Independent moving and static positive evidence yields `CONFLICTING`.
  Insufficient evidence yields `UNKNOWN`. Trusted propagation alone cannot
  originate identity.
- World-static is intentionally not converted to cabinet-static; object scope
  remains a later independent stage.

## Configuration

All thresholds are explicit in
`tools/itaco_moving_map_fix/configs/hololens_ownership_v5_primitives.yaml`.
The diagnostic also hard-checks that post-primitive stages are prohibited and
that `ready_for_dual_tsdf` remains false.

## Validation

Command:

```bash
/workspace_whz/envs/video_articulation/bin/python -m unittest \
  tools.itaco_ownership_v5.tests.test_primitives -v
```

Result: **15/15 passed**. The first 12 tests directly cover the specification's
minimum conceptual cases. Three additional tests verify motion-state transition
construction and that trusted propagation cannot originate ownership.

Small diagnostic command:

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/run_assignment_v5_primitives_diagnostic.py \
  --config tools/itaco_moving_map_fix/configs/hololens_ownership_v5_primitives.yaml \
  --output /workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/primitives_diagnostic_001
```

Result: 15/15 tests passed; the synthetic sequence was decomposed into closed,
active, and open states, and only transitions with sufficient `|delta q|` that
traverse ACTIVE_MOTION were emitted.

## Output

- `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/primitives_diagnostic_001/primitive_diagnostic.json`

## Gate and next step

`ready_for_dual_tsdf=false`.

The next step, only after review, is a small read-only real-data diagnostic that
adapts these primitives to existing observations while retaining immutable
source anchors and evidence provenance. It must not yet add dense SAM2 ownership
or reconstruction. The present implementation is not evidence that the drawer
side / cabinet inner wall ambiguity is solved.
