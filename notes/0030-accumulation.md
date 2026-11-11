# 0030 — Accumulation (5.4)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.4
**Mode:** Build
**Goal:** Big-batch learning on a small GPU.
**Command(s):** none new — settings already in `configs/kit1.yaml`, proven by the Kit-1 run
**Result:** effective batch = 16 micro x 4 accum x 256 = 16,384 tokens/step. Kit-1 trained 50M tokens this way to val 2.275.
**Check:** pass — effective batch recorded, grad clip 1.0 on (local only, not committed)
**What we learned:** Four small gulps add up to one big bite; the GPU never holds more than 16 sentences at once.
**Next:** 5.5 — Schedule + logging.
