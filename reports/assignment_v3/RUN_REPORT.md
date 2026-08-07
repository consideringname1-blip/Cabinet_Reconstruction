# Assignment v3 stopped run report

This is a **FunREC-inspired extension**, not an official FunREC reproduction.

- Tracker: Kornia LoFTR 0.8.2, cached indoor checkpoint (no download)
- Frozen inputs: verified
- Tracks: 882 total; moving=8, static=6, unknown=868
- Absolute-residual rejections: 359
- High-confidence drawer region seeds: 0
- Stop stage: Phase D region voting
- SAM2 v3 propagation: not run
- Four-state assignment: not run
- TSDF/NKSR/Mesh: not run
- Independent repeat of Phase B-D: passed
- Ready for dual TSDF: **false**

The earlier relative-only LK/LoFTR outputs are preserved in explicitly named diagnostic directories and are not formal v3 results.
