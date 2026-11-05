# 0009 — Train the MLP

**Date:** 2026-11-06
**Stage:** 1 — How learning works
**Chunk:** 1.5
**Mode:** Build
**Goal:** Show repeated small nudges lower the error.
**Command(s):** `.venv\Scripts\python.exe labs\mlp_train.py`
**Result:** step500 2.704, step1000 2.437, step1500 2.397 (beats bigram 2.482). Tiny-batch loss 0.0175.
**Check:** pass (local only, not committed)
**What we learned:** Training lowers surprise; overfit proves the learner can memorize.
**Next:** 1.6 — Three-question quiz.
