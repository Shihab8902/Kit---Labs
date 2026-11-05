# AGENTS.md — Kit Lab Rules

## Mission

Build **Kit**, a small language model, in tiny chunks.

The human must understand each concept before the next chunk starts.

---

## Read first

1. `STATE.md`
2. The current stage in `PLAN.md`
3. The last note in `notes/` (if any)
4. This file

---

## The one rule

> **Explain first. Ask. Wait. Do one chunk. Explain after. Stop.**

If you cannot explain the chunk in plain English, do not run it.

---

## Chunk protocol

Before **every** chunk, send this short preflight:

```markdown
**Chunk 2.3 — Production tokenizer**
**Concept:** A tokenizer learns merge rules; vocab size is a trade-off.
**Why:** Kit-2 needs 8K tokens; this changes sequence length and embedding size.
**Do:** Train an 8K tokenizer on the train split and save its hash.
**Check:** Special tokens are single IDs; round-trip is 100%.
**Start?**
```

Then **wait**. Do not run anything until the human says yes.

- “Go” approves **one chunk only**.
- “Keep going” still means one chunk at a time, with a short report after each.
- If the human says “auto,” you may skip the pause, but still report after each chunk.

After the chunk, send this, under 80 words:

```markdown
**Did:** one sentence.
**Why it matters:** one sentence.
**Number:** one measured result.
**New word:** term = plain-English meaning.
```

Then stop and ask: **“Next chunk?”**

If the human says “go deeper,” expand one idea with one example. Do not dump a textbook.

---

## Small-text rule

- Maximum **3 short paragraphs** per chat message.
- One idea per paragraph; maximum 3 sentences.
- Prefer bullets and tables over prose.
- Define jargon in the same sentence.
- Define **every** technical word the first time it appears in a message, even repeat words like loss or neural net.
- If a sentence has two tech words, define both right there.
- Long details go in `notes/`, not in chat.
- No wall of text. Ever.

---

## Disk policy: D: only

The C: drive is small. Do not bloat it.

| What | Where |
|---|---|
| Project | `D:\Web Projects\Kit` |
| Venv | `D:\Web Projects\Kit\.venv` |
| Caches | `D:\Web Projects\Kit\.cache\...` |
| Temp | `D:\Web Projects\Kit\.tmp` |
| Data | `D:\Web Projects\Kit\data` |
| Runs | `D:\Web Projects\Kit\runs` |

`scripts/env.ps1` creates these folders, then sets:

```powershell
$env:HF_HOME = "D:\Web Projects\Kit\.cache\hf"
$env:HF_DATASETS_CACHE = "D:\Web Projects\Kit\.cache\hf\datasets"
$env:TORCH_HOME = "D:\Web Projects\Kit\.cache\torch"
$env:PIP_CACHE_DIR = "D:\Web Projects\Kit\.cache\pip"
$env:MPLCONFIGDIR = "D:\Web Projects\Kit\.cache\matplotlib"
$env:TMP = "D:\Web Projects\Kit\.tmp"
$env:TEMP = "D:\Web Projects\Kit\.tmp"
```

Rules:

1. **Never install globally.** Use `.\.venv\Scripts\python.exe -m pip`.
2. Before an install or download, state the approximate size and destination. Ask.
3. After it finishes, show C: and D: free space.
4. If a tool insists on C:, stop and ask.
5. Never download datasets or models to `C:\Users\...`, Downloads, or Desktop.
6. Check space with:

```powershell
Get-PSDrive C,D | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

---

## Modes

| Mode | Trigger | Behavior |
|---|---|---|
| **Build** | default | Agent writes code and runs the chunk. Human reviews. |
| **Learn** | “teach me,” “let me code” | Human writes; agent gives hints, not full answers. |
| **Review** | “review this” | Agent audits; makes no changes. |

Record the mode in `STATE.md` when it changes.

---

## Repo map

```text
PLAN.md        chunk map
STATE.md       current chunk, environment, blockers
AGENTS.md      these rules
specs/         spec sheet + model card
docs/          glossary
configs/       run configs
data/          raw and tokenized data (D:, gitignored)
src/kit/       model, data, train, eval, sample, serve
scripts/       env and setup helpers
labs/          NumPy lessons
tests/         pytest
runs/          checkpoints and logs (D:, gitignored)
notes/         lab notebook
tokenizer/     frozen tokenizer files
```

---

## Target commands

These are built as the chunks reach them. Do not pretend they exist early.

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\Activate.ps1
pytest -q

python -m kit.train --config configs/kit1.yaml
python -m kit.eval --checkpoint runs/.../best.pt --split val
python -m kit.sample --checkpoint runs/.../best.pt --prompt "Once upon a time"
python -m kit.spec --checkpoint runs/.../best.pt
python -m kit.bench --checkpoint runs/.../best.pt --device cuda
python -m kit.serve --checkpoint runs/.../best.pt --port 8000
```

---

## Work rules

1. **One change at a time.**
2. **No metric, no claim.**
3. **Never train on the test set.**
4. **Checkpoint before long runs.** Keep best and last.
5. **Record data source, license, hash, and split.**
6. **Do not delete** runs, data, or checkpoints without permission.
7. **Do not fabricate** numbers, benchmarks, or parameter counts.
8. **Kit is educational.** Never call it safe, AGI, or frontier.
9. **No public deployment** without auth, rate limits, and a warning.
10. **No secrets** in the repo.

Ask before:

- More than 30 minutes of GPU time in one run.
- More than $5 of cloud spend.
- A download over 5 GB.
- Deleting anything large.
- Changing a released tokenizer or architecture.

---

## Stage gates

Do not start the next stage until every chunk check passes.

If the human says skip:

- Record it in `STATE.md` as technical debt.
- Say what is missing.
- Never pretend the gate passed.

---

## Failure protocol

1. Stop.
2. Capture the error.
3. Explain the likely cause in one sentence.
4. Try one minimal fix.
5. Rerun the check.
6. Log it in `notes/troubleshooting.md`.
7. Ask before continuing.

Never hide a failure by changing the eval set or reporting training loss as validation loss.

---

## Done with a chunk

- [ ] The check passed.
- [ ] The number is recorded.
- [ ] The 4-line note is written.
- [ ] `STATE.md` points to the next chunk.
- [ ] The human knows what is next.

---

## First action

The project is at **Stage 0, chunk 0.1**. Nothing is installed.

Do not start until the human approves chunk 0.1.

Then:

1. Explain chunk 0.1 with the short preflight.
2. Ask “Start?”
3. Wait.
