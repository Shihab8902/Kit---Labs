# 0007 — MLP forward pass

**Date:** 2026-11-06
**Stage:** 1 — How learning works
**Chunk:** 1.3
**Mode:** Build
**Goal:** Guess step with an 8-letter window, no learning yet.
**Command(s):** `.venv\Scripts\python.exe labs\mlp_forward.py`
**Result:** logits_shape (4, 65), shape_ok True.
**Check:** pass (local only, not committed)
**What we learned:** Embeddings turn IDs into vectors; one hidden layer mixes them into scores.
**Next:** 1.4 — Gradient check.
