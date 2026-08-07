# Assignment v5 ownership logic specification

This document records the hard invariants for the v5 transition-event motion-ownership diagnostic.

## Frozen scope

- Registered depth uses the verified `0.0002 m/count` contract and must pass its hard gate before any real-data evaluation.
- HoloLens poses, calibration, prismatic axis, `q_t`, joint type, moving map, hand masks, frame ranges, AutoSeg output, SAM2 output, and Assignment-v4 labels are frozen.
- The output labels are only `MOVING_LINK`, `WORLD_STATIC`, `UNKNOWN`, and `CONFLICTING`. `WORLD_STATIC` does not imply cabinet membership; `CONFLICTING` is treated as unknown for formal ownership.
- This stage does not run SAM2, feature/point trackers, optimization, dense propagation, TSDF, NKSR, Poisson, Mesh, GLB, URDF, or shape completion. `ready_for_dual_tsdf` remains `false`.

## Temporal evidence

- Motion states come from raw timestamps and frozen `q_t`. `INACTIVE_INTERMEDIATE` is explicit and cannot silently become a plateau.
- Ownership may originate only from directed, chronological, local intervals whose endpoints and intervening frames are observed `ACTIVE_MOTION`, with configured `delta q`, frame-gap, and elapsed-time limits.
- A transition is the ordered pair `(source_frame_id, target_frame_id)`; unordered sets are prohibited.
- Plateau-only comparisons cannot create ownership. Backward observations may confirm source identity, but causal reveal is chronological forward only.
- Assignment-v4 `per_target` rows are prohibited as target selection for this diagnostic.

## Finite identity

- The immutable source mask and registered depth define indexed source samples.
- Each hypothesis predicts each source sample into a target registered-depth image; measured depth is read at that exact predicted pixel, unprojected, and compared under the same source sample index.
- Target observations never become a new anchor.
- The formal identity engine cannot search, rank, or select target AutoSeg proposals, UIDs, or layers. Target proposal overlap is post-decision provenance only.
- The only per-hypothesis target states are `VERIFIED_SOURCE_ANCHOR`, `OCCLUDED_BY_TRUSTED_DRAWER`, `OCCLUDED_BY_OTHER`, `FREE_SPACE_CONTRADICTION`, `IDENTITY_LOST`, and `UNOBSERVABLE`.
- Positive motion evidence requires a consecutive source-anchored chain spanning configured frames, `q`, and elapsed time; counts alone are insufficient.

## Symmetric ownership routes

- `MOVING_LINK` originates from one `ARTICULATED_MOTION_TRANSITION` family.
- `WORLD_STATIC` originates independently from one `WORLD_STATIC_MOTION_TRANSITION` family; causal disocclusion is an additional independent world-static route, not a prerequisite.
- Correlated residual, overlap, run-length, and boundary measurements within one family do not receive multiple votes.
- If both static and moving models form verified source-anchor chains, the result is `UNKNOWN` with reason `motion_tangent_or_repeated_surface_ambiguity`.
- Independent moving and world-static families that disagree produce `CONFLICTING` and formal ownership remains unknown.

## Boundaries and causal reveal

- A segmentation boundary alone is not a physical boundary. The detector distinguishes RGB/depth footprint clipping, depth discontinuity, normal/plane discontinuity, invalid holes, and segmentation-only edges.
- Tangential moving evidence requires a measured leading or trailing physical edge and its continued support under the moving hypothesis; an observation-clipped or segmentation-only edge cannot lift ambiguity.
- Trusted drawer occlusion uses only the four existing front-seed SAM2 propagations at frames 195--198 with at least 3/4 agreement; payload seeds are not primary occluders and SAM2 is not rerun.
- Every causal field is measured from directed transitions, real timestamps, registered depth, transformed trusted silhouettes, and actual future residuals. Hard-coded positive booleans or a fixed zero world residual are prohibited.

## Evaluation gate

- Evaluation annotations never enter classifier logic.
- Before the 116-region diagnostic, all synthetic tests must pass; drawer-front controls must be 4/4 `MOVING_LINK`; active co-moving box controls must be at least 80% `MOVING_LINK` with zero `WORLD_STATIC`; documented floor controls must produce zero `MOVING_LINK`; and plateau-only controls must create no ownership.
- If any gate fails, execution stops before all 116 regions. The failed control result is preserved and reported without threshold tuning.
