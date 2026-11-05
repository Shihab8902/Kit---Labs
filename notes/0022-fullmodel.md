# 0022 — Full block + param count

**Date:** 2026-11-06
**Stage:** 4 — Kit-1 transformer
**Chunk:** 4.3
**Mode:** Build
**Goal:** Whole Kit-1 assembled and sane at birth.
**Command(s):** `.venv\Scripts\python.exe labs\fullmodel_check.py`
**Result:** params 1344640 exact match (~1.34M). Init loss 8.328 vs ln(4096) 8.318.
**Check:** pass (local only, not committed)
**What we learned:** Two bugs caught: big default init hid learning; wrong label shift faked good loss. Fixing both is why checks exist.
**Next:** 4.4 — Overfit one batch.
