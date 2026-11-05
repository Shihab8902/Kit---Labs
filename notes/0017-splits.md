# 0017 — Split train/validation/test

**Date:** 2026-11-06
**Stage:** 3 — Data
**Chunk:** 3.3
**Mode:** Build
**Goal:** Separate piles with no story twice.
**Command(s):** `.venv\Scripts\python.exe labs\split_data.py`
**Result:** train 2119719, val 10995, test 10995. val/test overlap 0. train/held overlap 6601 (upstream duplicates). Full-train dedupe attempt aborted as too slow.
**Check:** PARTIAL — val/test disjoint yes; train vs held no (dataset's own duplicates). Recorded as debt, not hidden.
**What we learned:** Official splits leak; test scores will be slightly optimistic until we clean.
**Next:** 3.4 — Tokenize to uint16 files.
