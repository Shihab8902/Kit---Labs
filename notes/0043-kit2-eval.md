# 0043 — Kit-2 evaluation (6.11)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunk:** 6.11
**Mode:** Build
**Goal:** Prove Kit-2 beats Kit-1 fairly.
**Command(s):** `.venv\Scripts\python.exe labs\eval_kit2.py`
**Result:** kit2 test 1.9669, ppl 7.15, bits/byte 0.6846. kit1 test 2.2985, ppl 9.96, bits/byte 1.0441.
**Check:** pass — Kit-2 bits/byte wins on the same stories (local only, not committed)
**What we learned:** Modern parts + 3x tokens + bigger size = 35% less surprise per byte.
**Next:** 6.12 — Model card (same session).
