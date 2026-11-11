# 0040 — LR sweep (6.6)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunk:** 6.6
**Mode:** Build
**Goal:** Find the best step size at a fixed budget.
**Command(s):** modern build x3 peak LRs, 1220 steps each
**Result:** 5e-4 -> 3.1225, 1e-3 -> 2.8888, 2e-3 -> 2.7732. Winner: 2e-3.
**Check:** pass — full table, picked by val (local only, not committed)
**What we learned:** Bigger steps win at this size; small models tolerate bold LRs.
**Next:** 6.7 — Scaling lab, small sizes.
