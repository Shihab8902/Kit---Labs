# 0001 — Check tools

**Date:** 2026-11-06
**Stage:** 0 — Setup
**Chunk:** 0.1
**Mode:** Build
**Goal:** Confirm py, git, GPU work.
**Command(s):** `py --version`; `git --version`; `nvidia-smi`
**Result:** Python 3.12.4; git 2.45.2; RTX 3050 6GB Laptop GPU, 6144 MiB, driver 592.82. C: 26.3 GB free. D: 55.3 GB free.
**Check:** pass
**What we learned:** Tools print correct versions, so setup can start. GPU is visible, so PyTorch CUDA can use it. Disk space is fine, so D: stays the home for everything.
**Next:** 0.2 — Create venv on D:.
