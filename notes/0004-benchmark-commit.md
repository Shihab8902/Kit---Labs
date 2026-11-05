# 0004 — Benchmark + first commit

**Date:** 2026-11-06
**Stage:** 0 — Setup
**Chunk:** 0.4
**Mode:** Build
**Goal:** Measure GPU speed, save first commit.
**Command(s):** `.venv\Scripts\python.exe scripts\check_env.py`; `git init`; first commit
**Result:** CPU 20.17s, GPU 0.62s for 10x 4096 matmul. VRAM 6.0 GB total, 5.0 free. Commit 3e0119d.
**Check:** pass
**What we learned:** GPU is ~32x faster, so training belongs there. Git now tracks the lab.
**Next:** 1.1 — Character tokenizer.
