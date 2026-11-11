# 0032 — Estimate check (5.6)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.6
**Mode:** Build
**Goal:** Predict a long run from a short bench.
**Command(s):** 100-step bench at 162k tok/s; math vs the real 271s Kit-1 run
**Result:** predicted 309s, actual 271s, error 13.9%. FLOPs 4.03e14, effective 1.49 TFLOPS.
**Check:** pass — within 20% (local only, not committed)
**What we learned:** Short benches predict long runs once startup cost washes out; tiny models sip GPU power (1.5 of ~14 TFLOPS).
**Next:** 5.7 — Reproducibility.
