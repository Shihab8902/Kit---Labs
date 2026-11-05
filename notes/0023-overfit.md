# 0023 — Overfit one batch

**Date:** 2026-11-06
**Stage:** 4 — Kit-1 transformer
**Chunk:** 4.4
**Mode:** Build
**Goal:** Prove the model can memorize 8 fixed sentences.
**Command(s):** `.venv\Scripts\python.exe labs\overfit_check.py`
**Result:** 3e-3 LR, 800 steps: 4.61 -> 1.22 -> 0.02 -> 0.0017. Pass.
**Check:** pass (local only, not committed)
**What we learned:** First tries failed on settings, not the model: too-big steps bounced, steady steps memorized. Tuning the test is fair; tuning results is not.
**Next:** 4.5 — Launch Kit-1 (the big 1-2h run).
