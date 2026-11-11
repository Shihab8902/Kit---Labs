# 0029 — Mixed precision (5.3)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.3
**Mode:** Build
**Goal:** Prove half-size math is safe and faster.
**Command(s):** 20 steps FP16 vs 20 steps FP32 (`--fp32` flag added)
**Result:** FP16 141k tok/s at 0.39 GB, FP32 105k tok/s at 0.45 GB. Val 7.785 vs 7.791, no NaN either way.
**Check:** pass — FP16 ~34% faster, same learning, less memory (local only, not committed)
**What we learned:** Safe math stays in full size (the scaler guards it); the rest runs at half size.
**Next:** 5.4 — Accumulation (same session).
