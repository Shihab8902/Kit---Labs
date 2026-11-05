# 0005 — Character tokenizer

**Date:** 2026-11-06
**Stage:** 1 — How learning works
**Chunk:** 1.1
**Mode:** Build
**Goal:** Text <-> numbers with a tiny char tokenizer.
**Command(s):** `.venv\Scripts\python.exe labs\char_tokenizer.py`
**Result:** vocab_size 18, roundtrip_ok True on sample.
**Check:** pass
**What we learned:** A tokenizer is just a lookup table. Models read the IDs, not the letters.
**Next:** 1.2 — Bigram model.
