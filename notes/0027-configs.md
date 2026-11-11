# 0027 — Run configs (5.1)

**Date:** 2026-11-11
**Stage:** 5 — Training engineering
**Chunk:** 5.1
**Mode:** Build
**Goal:** Every run repeats from a settings file.
**Command(s):** created `configs/kit1.yaml`; trainer loads it, saves copy + env to run dir
**Result:** `runs/resume-a/config.yaml` matches `configs/kit1.yaml`. Run ID in folder name.
**Check:** pass (local only, not committed)
**What we learned:** Settings live in a file, not in code; reruns start from the same page.
**Next:** 5.2 — Resume proof (same session).
