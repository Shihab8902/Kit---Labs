# Kit Lab — State

**Last updated:** 2026-11-11
**Current stage:** 6 — Kit-2 modern architecture
**Current chunk:** 6.1 — RMSNorm
**Status:** Stage 5 done! Chunks 5.1-5.7 all pass (local only, not committed). Waiting for permission for 6.1.
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
- `notes/0006-bigram.md` — bigram loss 2.482, local only.
- `notes/0007-mlp-forward.md` — shape (4, 65), local only.
- `notes/0008-grad-check.md` — rel_error 2.49e-11, local only.
- `notes/0009-mlp-train.md` — train 2.397 beats bigram, tiny 0.0175, local only.
- `notes/0010-stage1-quiz.md` — Q1/Q2/Q3 pass, local only.
- `notes/0011-bpe-hand.md` — merged 'lo', local only.
- `notes/0012-bpe-train.md` — 100 merges, roundtrip True, local only.
- `notes/0013-prod-tokenizer.md` — 4K vocab, specials ok, local only.
- `notes/0014-compare-tokens.md` — 1.636 tok/word, 3.23x, local only.
- `notes/0015-tinystories-dl.md` — 2119719 train rows, 2.73 GB, local only.
- `notes/0016-inspect-stories.md` — 20 clean, rules saved, local only.
- `notes/0017-splits.md` — val/test ok, train leak 6601 recorded, local only.
- `notes/0018-tokenize.md` — train 611M tok, val 2.9M, test 3.2M, local only.
- `notes/0019-loader.md` — shapes ok, readable English, local only.
- `notes/0020-embeddings.md` — (4, 256, 128) ok, local only.
- `notes/0021-attention.md` — no future leak, local only.
- `notes/0022-fullmodel.md` — 1.34M params, init loss 8.33, local only.
- `notes/0023-overfit.md` — 0.0017 on 8 sentences, local only.
- `notes/0024-kit1-launch.md` — ~14856 tok/s, 50M run launched, local only.
- `notes/0025-kit1-eval.md` — test 2.299, ppl 10.0, local only.
- `notes/0026-kit1-samples.md` — 3/3 stories read well, spec written, local only.
- `notes/0027-configs.md` — yaml config saved per run, local only.
- `notes/0028-resume.md` — resume diff 0.019, documented, local only.
- `notes/0029-mixed-precision.md` — FP16 34% faster, no NaN, local only.
- `notes/0030-accumulation.md` — effective batch 16384, local only.
- `notes/0031-schedule.md` — warmup curve logged, local only.
- `notes/0032-estimate.md` — prediction off by 13.9%, local only.
- `notes/0033-repro.md` — twin runs identical, local only.

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
