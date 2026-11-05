# 0003 — Install PyTorch + base packages

**Date:** 2026-11-06
**Stage:** 0 — Setup
**Chunk:** 0.3
**Mode:** Build
**Goal:** GPU torch + lab packages in venv, all on D:.
**Command(s):** `.venv\Scripts\python.exe -m pip install torch --index-url .../cu128`; then numpy pyyaml pytest ruff matplotlib tqdm datasets tokenizers fastapi uvicorn pydantic
**Result:** torch 2.11.0+cu128, CUDA 12.8, cuda available True, GPU RTX 3050 6GB Laptop GPU. C: 26.3 GB free. D: 48 GB free.
**Check:** pass
**What we learned:** Torch sees the GPU, so training can use it. C: unchanged, so caches stayed on D:.
**Next:** 0.4 — Benchmark + first commit.
