# 0008 — Gradient check

**Date:** 2026-11-06
**Stage:** 1 — How learning works
**Chunk:** 1.4
**Mode:** Build
**Goal:** Prove the fast slope math is correct.
**Command(s):** `.venv\Scripts\python.exe labs\grad_check.py`
**Result:** analytic 0.089183, numeric 0.089183, rel_error 2.49e-11, pass True.
**Check:** pass (local only, not committed)
**What we learned:** The formula for the slope matches a manual nudge, so training can trust it.
**Next:** 1.5 — Train the MLP.
