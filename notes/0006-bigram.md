# 0006 — Bigram model

**Date:** 2026-11-06
**Stage:** 1 — How learning works
**Chunk:** 1.2
**Mode:** Build
**Goal:** Guess next char from last char only.
**Command(s):** download tinyshakespeare (~1.1 MB, D:); `.venv\Scripts\python.exe labs\bigram.py`
**Result:** vocab 65, bigram_loss 2.482 (below 3.0). Sample is broken Shakespeare, not random noise.
**Check:** pass (not committed, not pushed — waiting for user)
**What we learned:** Counting pairs gives a real predictor. Loss measures surprise.
**Next:** 1.3 — MLP forward pass.
