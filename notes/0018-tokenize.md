# 0018 — Tokenize to uint16 files

**Date:** 2026-11-06
**Stage:** 3 — Data
**Chunk:** 3.4
**Mode:** Build
**Goal:** Number-files on disk for fast training.
**Command(s):** `.venv\Scripts\python.exe labs\tokenize_data.py` (background)
**Result:** train 610741249 tokens (1.22 GB), val 2939105, test 3215828. sha train c457e39eb233, val 6f8570557381, test b0d1e6270b32.
**Check:** pass (local only, not committed)
**What we learned:** Full TinyStories is ~611M tokens; Kit-1 needs only the first ~50M.
**Next:** 3.5 — Loader check.
