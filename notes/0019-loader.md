# 0019 — Loader check

**Date:** 2026-11-06
**Stage:** 3 — Data
**Chunk:** 3.5
**Mode:** Build
**Goal:** Prove batches come out right.
**Command(s):** `.venv\Scripts\python.exe labs\loader_check.py`
**Result:** x (16, 256) ok, shift ok, val (4, 256) ok. Decoded reads as English story.
**Check:** pass (local only, not committed)
**What we learned:** The pipeline ends in readable batches; test file never opened in training.
**Next:** 4.1 — Embeddings.
