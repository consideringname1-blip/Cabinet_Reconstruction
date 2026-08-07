# FunREC-inspired ownership Assignment v3

This directory publishes the formal Assignment v3 result. It is an
`extended_non_official_funrec_inspired_assignment_v3` experiment, not an
official FunREC reproduction.

The run uses frozen HoloLens camera poses, the verified prismatic axis and
monotonic joint state, spatially balanced periodic LoFTR tracks, explicit
AutoSeg proposals, and conservative static/moving/unknown evidence gates.

## Outcome

The formal run stopped safely during region voting:

- 882 tracks total;
- 8 moving, 6 static, and 868 unknown;
- 359 supported/excited tracks rejected by the 5 cm absolute residual gate;
- zero high-confidence drawer region seeds.

Consequently, formal SAM2 propagation, four-state ownership assignment, TSDF,
NKSR, and mesh generation did not run. `ready_for_dual_tsdf` is false.

See `RUN_REPORT.md`, `acceptance_report.json`, `failure_reasons.json`, and
`deterministic_rerun_report.json`. The two independent formal Phase B-D runs
produced identical hashes for tracks, labels, and region votes.

Large relative-only LK/LoFTR diagnostic propagation outputs are intentionally
not published here because they failed the absolute geometry gate.
