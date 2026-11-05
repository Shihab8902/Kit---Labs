# 0021 — Attention

**Date:** 2026-11-06
**Stage:** 4 — Kit-1 transformer
**Chunk:** 4.2
**Mode:** Build
**Goal:** Mix earlier positions, block the future.
**Command(s):** `.venv\Scripts\python.exe labs\attention_check.py`
**Result:** shape (2, 8, 32) ok, early_change 0.00e+00, no leak.
**Check:** pass (local only, not committed)
**What we learned:** Scores pick which past pieces matter; the mask keeps guessing honest.
**Next:** 4.3 — Full block + param count.
