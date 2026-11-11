# 0033 — Reproducibility (5.7)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.7
**Mode:** Build
**Goal:** Same settings twice gives the same curve.
**Command(s):** two fresh 60-step runs (`runs/twin-a`, `runs/twin-b`)
**Result:** step 1/50/60 val losses identical to 4 decimals (8.3257 / 6.3694 / 5.9135 both).
**Check:** pass (local only, not committed)
**What we learned:** Fresh runs repeat exactly; only the stop/resume path adds ~0.02 wobble (noted in 0028). Seeds + saved RNG do their job.
**Next:** 6.1 — RMSNorm (Stage 6: modern architecture).
