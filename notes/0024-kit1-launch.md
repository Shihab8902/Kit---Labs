# 0024 — Launch Kit-1 training

**Date:** 2026-11-11
**Stage:** 4 — Kit-1 transformer
**Chunk:** 4.5
**Mode:** Build
**Goal:** Real 50M-token run with checkpoints and logs.
**Command(s):** bench `--steps 4` then full `--steps 3052 --run runs/kit1-2026-11-11-a` (background)
**Result:** bench ~14856 tok/s (step 1, rough). Estimate 50M/14856 ~= 56 min + eval overhead ~= 1-1.5h.
**Check:** pass — tok/s recorded before the long wait (local only, not committed)
**What we learned:** Tiny model flies; eval every 50 steps adds the overhead.
**Next:** 4.6 — Evaluate when the run finishes.
