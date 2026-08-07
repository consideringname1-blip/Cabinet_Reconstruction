# Depth-aware projective association diagnostic

This run keeps HoloLens poses, the prismatic axis, and q_t fixed. It evaluates
LoFTR candidates against both static and drawer projective-depth predictions.

- Tracks: 882
- Static: 21
- Moving: 12
- Unknown: 849
- High-absolute-residual tracks: 27
- Accepted projective attempts: 1531
- Rejected projective attempts: 10829
- Reliable moving baseline / required / current: 8 / 13 / 12
- Clearly increased: false
- Region voting/propagation: **not run**
- SAM2/TSDF/NKSR/Mesh: **not run**

Even if the numeric gate passes, this diagnostic never starts propagation; the
observation-error decomposition must be reviewed first.
