# 0014 — Tokens-per-word comparison

**Date:** 2026-11-06
**Stage:** 2 — Tokenizer
**Chunk:** 2.4
**Mode:** Build
**Goal:** Measure compression of BPE vs chars.
**Command(s):** `.venv\Scripts\python.exe labs\compare_tokens.py`
**Result:** 1000 lines, chars/word 5.282, BPE tokens/word 1.636, compression 3.23x.
**Check:** pass (local only, not committed)
**What we learned:** Pieces shorten sequences ~3x, so the model sees more words per context.
**Next:** 3.1 — Download TinyStories + manifest.
