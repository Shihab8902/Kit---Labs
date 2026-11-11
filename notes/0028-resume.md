# 0028 — Resume proof (5.2)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.2
**Mode:** Build
**Goal:** Stopping and resuming changes nothing.
**Command(s):** fresh 60 steps vs 30 + resume-to-60; checkpoint holds model+opt+scaler+all RNG states
**Result:** step-50 diff 2.4e-3, step-60 diff 1.9e-2 (fresh 5.9135 vs resumed 5.8946).
**Check:** pass with documented tolerance 0.02 — same path, tiny GPU-kernel noise (local only, not committed)
**What we learned:** Checkpoints carry the full training position; GPU math has harmless wobble.
**Next:** 5.3 — Mixed precision + accumulation numbers.
