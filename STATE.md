# Kit Lab — State

**Last updated:** 2026-11-06
**Current stage:** 1 — How learning works
**Current chunk:** 1.2 — Bigram model
**Status:** Chunk 1.1 done. Waiting for permission for 1.2.
**Mode:** Build (default)

---

## Next action

Explain chunk 0.1 in five short lines, then ask **“Start?”**

No command runs until the human says yes.

---

## Disk policy

Everything stays on D:.

| What | Where |
|---|---|
| Project | `D:\Web Projects\Kit` |
| Venv | `D:\Web Projects\Kit\.venv` |
| Caches | `D:\Web Projects\Kit\.cache\...` |
| Temp | `D:\Web Projects\Kit\.tmp` |
| Data | `D:\Web Projects\Kit\data` |
| Runs | `D:\Web Projects\Kit\runs` |

C: free space is about **26 GB**. Do not install globally or download to C:.

Before every install/download: state size and destination, then ask.

---

## Environment facts (verified 2026-11-06)

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.12.4 via `py` |
| Note | `python.exe` on PATH is a broken Store alias; use `py` outside the venv |
| PyTorch | Not installed |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 6 GB VRAM |
| Driver | 592.82 |
| CPU | Intel Core i5-13420H, 8 cores / 12 threads |
| RAM | 16 GB |
| Disk C: | About 26 GB free |
| Disk D: | About 55 GB free |
| Git | Installed |
| Repo | Not initialized |
| Project | `D:\Web Projects\Kit` |

---

## Decisions

| Date | Decision |
|---|---|
| 2026-11-06 | Text language model first; it teaches the core with the least compute |
| 2026-11-06 | PyTorch for real models; pure NumPy for the first lessons |
| 2026-11-06 | Local-first on the RTX 3050; cloud only for optional scale-up |
| 2026-11-06 | TinyStories for pretraining; Dolly 15K for SFT |
| 2026-11-06 | Work in micro-chunks; explain first, ask, then do one chunk |

---

## Open questions

- None blocking.
- Default mode is Build. Say “teach me” to switch to Learn mode.

---

## Recent notes

- `notes/0001-check-tools.md` — py 3.12.4, git 2.45.2, RTX 3050 6GB visible. C: 26.3 GB free. D: 55.3 GB free.
- `notes/0002-venv-on-d.md` — venv on D:, pip 26.3 in venv, caches on D:.
- `notes/0003-torch-install.md` — torch 2.11.0+cu128, CUDA True, D: 48 GB free.
- `notes/0004-benchmark-commit.md` — CPU 20.17s, GPU 0.62s, commit 3e0119d.
- `notes/0005-char-tokenizer.md` — vocab 18, roundtrip True.

---

## Blockers

None.

---

## Update rule

At the end of every chunk:

- Update current chunk and status.
- Record the next action.
- Note any skipped check.
- Keep it short.
