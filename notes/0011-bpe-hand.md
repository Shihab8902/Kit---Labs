# 0011 — One BPE merge by hand

**Date:** 2026-11-06
**Stage:** 2 — Tokenizer
**Chunk:** 2.1
**Mode:** Build
**Goal:** Join the top pair once.
**Command(s):** `.venv\Scripts\python.exe labs\bpe_merge_hand.py`
**Result:** top pair ('l','o') count 5, merged to 'lo'.
**Check:** pass (local only, not committed)
**What we learned:** BPE grows pieces by joining what appears together most.
**Next:** 2.2 — BPE on a 10 MB sample.
