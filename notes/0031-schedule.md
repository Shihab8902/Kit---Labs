# 0031 — Schedule + logging (5.5)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.5
**Mode:** Build
**Goal:** Learning rate follows the plan; everything is logged.
**Command(s):** LR now in `metrics.jsonl`; 100-step run shows the curve
**Result:** lr 0.00001 (step 1) -> 0.0005 (step 50) -> 0.001 peak (step 100). Console + JSONL both live.
**Check:** pass — warmup rises, cosine will fall after peak (local only, not committed)
**What we learned:** Start gentle (random weights break under big steps), then cruise, then settle.
**Next:** 5.6 — Estimate check (same session).
