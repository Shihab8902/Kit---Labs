# 0020 — Embeddings

**Date:** 2026-11-06
**Stage:** 4 — Kit-1 transformer
**Chunk:** 4.1
**Mode:** Build
**Goal:** IDs become mixable vectors.
**Command(s):** `.venv\Scripts\python.exe labs\embed_check.py` (needs PYTHONPATH=Kit, now in env.ps1)
**Result:** out (4, 256, 128), shape ok.
**Check:** pass (local only, not committed)
**What we learned:** Each token gets a learned vector plus a position vector.
**Next:** 4.2 — Attention.
