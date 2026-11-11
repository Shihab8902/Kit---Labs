# 0037 — 8K tokenizer + re-tokenize (6.4)

**Date:** 2026-11-11
**Stage:** 6 — Kit-2 modern architecture
**Chunk:** 6.4
**Mode:** Build
**Goal:** New family, new pieces, all stories re-numbered.
**Command(s):** `labs\train_tokenizer2.py` + `labs\retokenize_8k.py` (background)
**Result:** vocab 8192, sha 5989ca515237, specials single. train2 468M tok, val2 2.24M, test2 2.46M.
**Check:** pass (local only, not committed)
**What we learned:** Bigger pieces cover more words: 611M -> 468M tokens for the same stories.
**Next:** 6.5 — Ablation race.
