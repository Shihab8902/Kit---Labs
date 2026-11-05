# Kit

**A small language model built from scratch, one tiny chunk at a time.**

Kit is a learning lab, not a ChatGPT rival. The goal is a real model you can train, measure, run locally, and explain.

## Start here

1. `PLAN.md` — the chunk map.
2. `AGENTS.md` — the working rules.
3. `STATE.md` — where we are now.

## Current status

**Stage 0, chunk 0.1 — Check tools.**

Waiting for permission. Nothing is installed yet.

## How a chunk works

Before:

- **Concept:** one idea, in 1–3 short sentences.
- **Why:** why it matters.
- **Do:** the one action.
- **Check:** the proof.
- **Start?** I wait for your yes.

After:

- **Did**
- **Why it matters**
- **The number**
- **New word**

Then I stop. No next chunk without asking.

## Disk rule

**Everything on D:, nothing bloating C:.**

- Project: `D:\Web Projects\Kit`
- Venv: `D:\Web Projects\Kit\.venv`
- Caches: `D:\Web Projects\Kit\.cache`
- Temp: `D:\Web Projects\Kit\.tmp`
- Data: `D:\Web Projects\Kit\data`
- Runs: `D:\Web Projects\Kit\runs`

Before every install: size + destination + your approval.

## Ladder

| Model | Size | What it proves |
|---|---:|---|
| Kit-0 | ~4K | next-token prediction |
| Kit-0.5 | ~13K | gradients and backprop |
| Kit-1 | ~1.3M | a working transformer |
| Kit-2 | ~6.8M | modern architecture and ablations |
| Kit-3 | ~17M | chat and instruction following |
| Kit-4 | ~46M | local stretch goal |

## Useful phrases

- **“Go deeper”** — more detail on one idea.
- **“Teach me”** — Learn mode: you write the code.
- **“Quiz me”** — three explain-back questions.
- **“Review this”** — audit without changing.
- **“Just build it”** — Build mode.

## First principle

> Finish small before going big.
