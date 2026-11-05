# 0013 — Production 4K tokenizer

**Date:** 2026-11-06
**Stage:** 2 — Tokenizer
**Chunk:** 2.3
**Mode:** Build
**Goal:** Frozen 4K tokenizer with special tokens.
**Command(s):** `.venv\Scripts\python.exe labs\train_tokenizer.py`
**Result:** vocab 4096, sha 7a0a35cf7a06, specials single IDs True, 20-line roundtrip True.
**Check:** pass (local only, not committed)
**What we learned:** Byte-level BPE keeps every letter; specials act as single markers.
**Next:** 2.4 — Tokens-per-word comparison.
