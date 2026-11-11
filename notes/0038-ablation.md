# 0038 — Ablation race (6.5)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunk:** 6.5
**Mode:** Build
**Goal:** One change, fair race, measured winner.
**Command(s):** `ablate-legacy.yaml` vs `ablate-modern.yaml`, 1220 steps (20M tokens) each
**Result:** legacy best 3.2202, modern best 2.8888. Delta 0.33 to modern. Modern slower: 114k vs 154k tok/s.
**Check:** pass — table complete, winner earns Kit-2 (local only, not committed)
**What we learned:** The modern trio is worth ~0.33 error; speed cost is real but small models are already fast.
**Next:** 6.6 — Learning-rate sweep.
