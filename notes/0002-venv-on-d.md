# 0002 — Create venv on D:

**Date:** 2026-11-06
**Stage:** 0 — Setup
**Chunk:** 0.2
**Mode:** Build
**Goal:** Private Python folder on D: with caches on D:.
**Command(s):** `py -3.12 -m venv .venv`; wrote `scripts/env.ps1`; `.venv\Scripts\python.exe -m pip install --upgrade pip`
**Result:** sys.prefix `D:\Web Projects\Kit\.venv`, Python 3.12.4, pip 26.3 in venv. C: 26.3 GB free. D: 55.3 GB free.
**Check:** pass
**What we learned:** Venv isolates Kit packages from C: Python. Env script points pip/hf/torch/temp to D:.
**Next:** 0.3 — Install PyTorch + base packages in venv.
