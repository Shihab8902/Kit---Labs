# 0025 — Kit-1 evaluation (4.6)

**Date:** 2026-11-11
**Stage:** 4 — Kit-1 transformer
**Chunk:** 4.6
**Mode:** Build
**Goal:** Honest numbers on unseen text.
**Command(s):** `.venv\Scripts\python.exe labs\sample_kit1.py`
**Result:** best val 2.275, test 2.299, perplexity 10.0. Train took 271s (~184k tok/s).
**Check:** pass — test within 0.03 of val, no wild gap (local only, not committed)
**What we learned:** The model is equally good on new text; it learned patterns, not just training stories.
**Next:** 4.7 — Samples (same run).
