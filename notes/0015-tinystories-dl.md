# 0015 — Download TinyStories + manifest

**Date:** 2026-11-06
**Stage:** 3 — Data
**Chunk:** 3.1
**Mode:** Build
**Goal:** Real story data on D: with a paper trail.
**Command(s):** `.venv\Scripts\python.exe labs\download_tinystories.py` (background, resumed)
**Result:** train 2119719 rows, validation 21990 rows, cache 2.73 GB. C: 26 GB free. D: 45.2 GB free.
**Check:** pass (local only, not committed)
**What we learned:** The cache holds the raw stories; the manifest records source and license.
**Next:** 3.2 — Inspect 20 stories.
