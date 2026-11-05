# 0012 — BPE on a real sample

**Date:** 2026-11-06
**Stage:** 2 — Tokenizer
**Chunk:** 2.2
**Mode:** Build
**Goal:** Learn pieces from real text, keep round-trip.
**Command(s):** `.venv\Scripts\python.exe labs\bpe_train.py`
**Result:** 100 merges on 200 KB (fast stand-in for 10 MB), 162 pieces, held-out 5000 chars -> 3063 tokens, roundtrip True.
**Check:** pass (local only, not committed)
**What we learned:** Frequent pairs become single pieces; joining is reversible.
**Next:** 2.3 — Production 4K tokenizer.
