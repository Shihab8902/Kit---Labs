# 0016 — Inspect 20 stories

**Date:** 2026-11-06
**Stage:** 3 — Data
**Chunk:** 3.2
**Mode:** Build
**Goal:** Read before trusting.
**Command(s):** `.venv\Scripts\python.exe labs\inspect_stories.py`
**Result:** 20 sampled, 0 empty/short, 0 email/phone-like. Rules in data/cleaning_rules.md.
**Check:** pass (local only, not committed)
**What we learned:** Sample is clean; rules are drop-empty, drop-short, dedupe, flag private-like.
**Next:** 3.3 — Split train/validation/test.
