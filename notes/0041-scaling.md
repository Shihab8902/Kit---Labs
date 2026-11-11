# 0041 — Scaling lab (6.7 + 6.8 + 6.9)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunks:** 6.7, 6.8, 6.9
**Mode:** Build
**Goal:** Four sizes, one budget, one fitted curve.
**Command(s):** 4 configs x 1220 steps; `labs\fit_scaling.py`
**Result:** 0.59M->3.47, 1.67M->2.84, 4.9M->2.52, 14.9M->2.33. Fit: L(N)=4318.5*N^(-0.609)+2.15, R2=0.9998. Predicts Kit-2 (6.9M) ~=2.44 at 20M tokens. Plot in runs/scaling.png.
**Check:** pass — monotonic, fitted, plotted (local only, not committed)
**What we learned:** Bigger reliably wins; the curve lets us predict before paying. Detours fixed honestly: a filename collision and a shifted-fit bug.
**Next:** 6.10 — Full Kit-2 run, 150M tokens (needs GPU-time yes).
