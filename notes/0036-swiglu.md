# 0036 — SwiGLU mixer (6.3)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunk:** 6.3
**Mode:** Build
**Goal:** Gated mixer runs; Kit-2 size confirmed.
**Command(s):** `.venv\Scripts\python.exe labs\swiglu_check.py`
**Result:** small shape ok, 50 steps 8.3 -> 4.27. Kit-2 config counts 6,947,072 params.
**Check:** pass (local only, not committed)
**What we learned:** The gate learns faster early; Kit-2 sits at ~6.9M, on plan.
**Next:** 6.4 — 8K tokenizer + re-tokenize.
