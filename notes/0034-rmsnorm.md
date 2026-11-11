# 0034 — RMSNorm (6.1)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunk:** 6.1
**Mode:** Build
**Goal:** Cheaper steadier, same learning.
**Command(s):** `.venv\Scripts\python.exe labs\rms_check.py` (norm flag added to config, both versions kept)
**Result:** shape ok, 100 steps 8.3 -> 5.14. Params unchanged (same count as LayerNorm version).
**Check:** pass — loss still trains (local only, not committed)
**What we learned:** Rescaling by size alone is enough; the mean-shift was extra luggage.
**Next:** 6.2 — RoPE positions.
