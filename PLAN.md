# Kit — Small-Chunk Build Plan

> Kit is a small language model you build yourself.
> **One chunk at a time. One concept. One action. One check.**

---

## 1. How this works

Before every chunk, I tell you:

- **Chunk:** the number and name.
- **Concept:** the one idea in 1–3 short sentences.
- **Why:** why it matters for Kit.
- **Do:** exactly what I will run or change.
- **Check:** the number or behavior that proves it worked.
- **Start?** I wait for your yes.

Then I do **only that chunk**, give a 4-line note, and stop.

No long essays. No hidden steps. No starting the next chunk until you say yes.

---

## 2. Disk rule: everything on D:

**C: is small. Do not bloat it.**

| What | Where |
|---|---|
| Project | `D:\Web Projects\Kit` |
| Python environment | `D:\Web Projects\Kit\.venv` |
| Pip cache | `D:\Web Projects\Kit\.cache\pip` |
| Hugging Face cache | `D:\Web Projects\Kit\.cache\hf` |
| PyTorch cache | `D:\Web Projects\Kit\.cache\torch` |
| Temp files | `D:\Web Projects\Kit\.tmp` |
| Datasets | `D:\Web Projects\Kit\data` |
| Runs and checkpoints | `D:\Web Projects\Kit\runs` |

Before anything is installed, `scripts/env.ps1` will create these folders and set:

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

1. Never install packages globally. Use the Kit `.venv`.
2. Before an install or download, I tell you the approximate size and destination.
3. After it finishes, I show C: and D: free space.
4. If a tool insists on using C:, I stop and ask first.
5. All datasets, checkpoints, and model files stay on D:.

Check disk space:

```powershell
Get-PSDrive C,D | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

---

## 3. Hardware

| Part | What you have | What it means |
|---|---|---|
| GPU | RTX 3050 Laptop, 6 GB VRAM | Good for 1M–20M parameter models; 46M is tight |
| CPU | i5-13420H, 8 cores / 12 threads | Fine for data prep and tiny CPU models |
| RAM | 16 GB | Enough for this plan |
| Disk | C: about 26 GB free, D: about 55 GB free | Keep everything on D: |
| Python | 3.12.4 via `py` | Use `py` outside the venv; `python` inside it |

**Reality:** Kit will be a real small language model, not a ChatGPT rival. That is the point.

---

## 4. Model ladder

| Model | Size | Data | What it proves |
|---|---:|---|---|
| Kit-0 | ~4K | TinyShakespeare, characters | next-token prediction |
| Kit-0.5 | ~13K | TinyShakespeare, characters | gradients and backprop |
| Kit-1 | ~1.3M | TinyStories, ~50M tokens | a working transformer |
| Kit-2 | ~6.8M | TinyStories, ~150M tokens | modern architecture and ablations |
| Kit-3 | ~17M | ~350M tokens + SFT | chat and instruction following |
| Kit-4 | ~46M | ~800M tokens | the local stretch goal |

Times: Kit-1 about 1–2 hours; Kit-2 about 2–5 hours; Kit-3 about 8–12 hours; Kit-4 1–3 days.

---

## 5. Chunk map

Each row is one chunk. A stage is done when every check passes. Do not start the next stage until then.

### Stage 0 — Setup (about 1 hour)

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 0.1 | A tool check is cheaper than a broken install. | Check `py`, `git`, `nvidia-smi`. | Versions and GPU print. | 5m |
| 0.2 | A venv is a private Python folder. | Create `.venv` on D:; write `scripts/env.ps1`; set caches/temp; upgrade pip. | `sys.prefix` points into Kit; pip cache is on D:. | 10m |
| 0.3 | PyTorch is the math library; CUDA lets it use the GPU. | Install PyTorch + base packages inside `.venv`. | `torch.cuda.is_available()` is True. | 20–40m |
| 0.4 | A benchmark is a controlled speed test. | Write `check_env.py`; run it. | GPU matmul time recorded. | 15m |
| 0.5 | Git is a save history. | `git init`, `.gitignore`, first commit. | `git log` shows the commit. | 10m |

### Stage 1 — How learning works (pure NumPy)

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 1.1 | A token is a number for text. | Build a character tokenizer. | `decode(encode(text)) == text`. | 20m |
| 1.2 | A bigram guesses the next character from the last one. | Count pairs, sample, compute loss. | Loss below 3.0. | 30m |
| 1.3 | An MLP mixes several characters into a guess. | Build the forward pass. | Output shape is correct. | 20m |
| 1.4 | A gradient is a slope. | Compare a numerical slope with the analytic one. | Relative error below `1e-4`. | 30m |
| 1.5 | Training is repeated small corrections. | Train the MLP; overfit one batch. | Loss falls; tiny batch loss near 0. | 30–45m |
| 1.6 | If you can explain it, you own it. | Three-question quiz. | You answer all three. | 10m |

### Stage 2 — Tokenizer

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 2.1 | BPE merges the most frequent pair first. | Merge one pair by hand. | Merge table is correct. | 15m |
| 2.2 | A tokenizer learns from text. | Build BPE on a 10 MB sample. | Round-trip 100%. | 45m |
| 2.3 | Vocab size is a trade-off. | Train a 4K tokenizer on train-only text. | Special tokens are one ID; hash recorded. | 30m |
| 2.4 | Tokens per word measures compression. | Compare characters vs BPE. | Number recorded. | 20m |

### Stage 3 — Data

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 3.1 | Data is the real fuel. | Download TinyStories to D:; write manifest. | Size, license, hash recorded. | 20–40m |
| 3.2 | Bad data teaches bad patterns. | Inspect 20 stories; write cleaning rules. | Notes saved. | 20m |
| 3.3 | A test set is the final exam. | Split train/validation/test. | Splits are disjoint. | 20m |
| 3.4 | Tokens are stored as numbers. | Tokenize to `uint16` files; decode a batch. | Batch reads as English. | 30–45m |

### Stage 4 — Kit-1, the first transformer

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 4.1 | Embeddings turn token IDs into vectors. | Add token and position embeddings. | Shape test passes. | 20m |
| 4.2 | Attention lets a token look at earlier tokens. | Add Q/K/V and a causal mask. | No future token leaks. | 30m |
| 4.3 | A block is attention + MLP + leftovers. | Assemble the model; count params. | Init loss near `ln(vocab)`. | 30m |
| 4.4 | If it cannot overfit one batch, it is broken. | Train on one batch. | Loss below 0.1. | 30m |
| 4.5 | A real run needs a budget. | Estimate time; launch Kit-1. | Tokens/sec recorded. | 10m + run |
| 4.6 | The loss curve is the heartbeat. | Evaluate validation loss. | Validation loss fell. | 20m |
| 4.7 | Samples show what it learned. | Generate samples; write spec. | Readable samples + spec. | 30m |

### Stage 5 — Training engineering

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 5.1 | A config makes a run repeatable. | YAML config + run ID. | Config saved with run. | 20m |
| 5.2 | A checkpoint is a save file for training. | Save and resume. | Next loss matches. | 30m |
| 5.3 | Mixed precision is faster and smaller. | Add FP16 + `GradScaler`. | No NaN; speed recorded. | 30m |
| 5.4 | Gradient accumulation fakes a bigger batch. | Add accumulation + clipping. | Effective batch recorded. | 30m |
| 5.5 | A schedule changes the learning rate over time. | Warmup + cosine; log metrics. | Curve looks right. | 30m |
| 5.6 | Measure before a long run. | Benchmark and estimate time. | Estimate within 20%. | 20m |
| 5.7 | Reproducibility is a feature. | Rerun a short job. | Loss curves match closely. | 30m |

### Stage 6 — Kit-2 and real experiments

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 6.1 | RMSNorm is a simpler normalization. | Swap LayerNorm for RMSNorm. | Loss still trains. | 20m |
| 6.2 | RoPE encodes position by rotation. | Swap learned positions for RoPE. | Shape + sample test. | 30m |
| 6.3 | SwiGLU is a gated MLP. | Swap the MLP; count params. | Param count matches plan. | 30m |
| 6.4 | A new vocab means a new model family. | Train 8K tokenizer; re-tokenize. | Hash + token counts recorded. | 30m |
| 6.5 | An ablation changes one thing. | Legacy vs modern, 20M tokens each. | Comparison table. | 1–2h |
| 6.6 | Learning rate is the biggest knob. | Three short runs. | Best LR recorded. | 1–2h |
| 6.7 | Scaling laws predict bigger runs. | Train 0.5M and 1.5M at 20M tokens. | Loss points recorded. | 1h |
| 6.8 | More points make a better curve. | Train 5M and 15M at 20M tokens. | Loss points recorded. | 1–2h |
| 6.9 | Fit the curve, do not eyeball it. | Plot and fit `L(N)`. | Curve + fit reported. | 30m |
| 6.10 | The real run. | Launch Kit-2, 150M tokens. | Tokens/sec recorded. | 10m + run |
| 6.11 | Evaluate before celebrating. | Validation loss + samples. | Bits/byte beats Kit-1. | 20m |
| 6.12 | A model card is the receipt. | Write spec + card. | Complete and honest. | 30m |

### Stage 7 — Kit-3 and chat

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 7.1 | A base model is still autocomplete. | Launch Kit-3, ~350M tokens. | Tokens/sec recorded. | 15m + run |
| 7.2 | Evaluate the base before fine-tuning. | Validation + samples. | Base beats Kit-2 bits/byte. | 20m |
| 7.3 | A chat template is a contract. | Template + loss mask. | Round-trip exact. | 30m |
| 7.4 | SFT data teaches format. | Prepare Dolly 15K; inspect. | License + hashes recorded. | 30m |
| 7.5 | SFT is a short, low-LR run. | Launch SFT. | Format followed. | 10m + run |
| 7.6 | Fixed prompts make comparison fair. | 20-prompt evaluation. | Base vs SFT table. | 30m |
| 7.7 | DPO is optional. | 500 preference pairs. | Compared to SFT. | 1h |
| 7.8 | Document the limits. | Spec + model card. | Complete and honest. | 30m |

### Stage 8 — Running Kit

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 8.1 | Inference does not need gradients. | Add a no-grad generate function. | Output matches expectation. | 15m |
| 8.2 | The KV cache avoids recomputing the past. | Implement cache; compare logits. | Values match. | 30m |
| 8.3 | Sampling controls randomness. | Temperature, top-k, top-p CLI. | Same prompt gives controlled variety. | 20m |
| 8.4 | An API makes the model usable. | FastAPI + streaming. | Browser/curl gets tokens. | 30m |
| 8.5 | A web UI makes it demoable. | Minimal chat page. | You can chat locally. | 30m |
| 8.6 | Quantization shrinks the model. | int8 conversion. | Size/speed/quality measured. | 30m |
| 8.7 | Benchmarks need conditions. | TTFT, tok/s, VRAM, size. | Table recorded. | 20m |

### Stage 9 — Evaluation and release

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 9.1 | Loss is not the whole story. | Test loss, perplexity, bits/byte. | All three recorded. | 20m |
| 9.2 | Task evals measure usefulness. | 20–30 fixed prompts + rubric. | Scores + outputs saved. | 30m |
| 9.3 | Contamination inflates scores. | N-gram overlap check. | Overlap reported. | 20m |
| 9.4 | A release is reproducible. | Model card, spec, git tag. | Clean checkout command works. | 30m |

### Stage 10 — Scale-up (optional)

| # | Concept | Tiny action | Check | Time |
|---|---|---|---|---|
| 10.1 | Scaling is a budget. | Estimate FLOPs, hours, cost. | Range written. | 30m |
| 10.2 | Write the memo. | Go/no-go plan. | One-page memo. | 30m |

---

## 6. What gets measured

| Spec | How |
|---|---|
| Total parameters | Sum of all weights |
| Non-embedding parameters | Total minus token embeddings |
| Layers / heads / width / context / vocab | Model config |
| Tokens trained | Packed tokens × epochs |
| FLOPs | `6 × params × tokens` |
| GPU-hours | Wall-clock × GPUs |
| Validation/test loss | Cross-entropy, nats/token |
| Perplexity | `exp(loss)` |
| Bits per byte | `loss / ln(2) / bytes_per_token` |
| Decode speed | Tokens/sec, with hardware + settings |
| VRAM / RAM | Peak measured |
| File size | fp16 and int8 |
| KV cache | `2 × layers × d_model × context × bytes` |

A number without a method is a rumor.

---

## 7. Defaults

| Setting | Kit-1 | Kit-2 | Kit-3 |
|---|---:|---:|---:|
| Layers | 4 | 6 | 8 |
| Heads | 4 | 8 | 6 |
| Width | 128 | 256 | 384 |
| Context | 256 | 512 | 512 |
| Vocab | 4,096 | 8,192 | 8,192 |
| Learning rate | `1e-3` | `1e-3` | `5e-4` |
| Tokens | 50M | 150M | 350M |

Start here, then sweep one thing at a time.

---

## 8. If it breaks

| Problem | First fix |
|---|---|
| CUDA OOM | Halve batch or context; use gradient checkpointing |
| Loss flat | Check the label shift and the data |
| NaN loss | Lower the LR; keep gradient clipping |
| Slow data | Pre-tokenize; fewer workers; `uint16` |
| Resume mismatch | Save optimizer, scaler, step, and RNG |

Stop. Explain the cause in one sentence. Try one fix. Rerun the check. Ask before continuing.

---

## 9. Rules

1. One change at a time.
2. Never train on the test set.
3. Checkpoint before long runs.
4. No metric, no claim.
5. Record data licenses.
6. Keep old versions.
7. Ask before more than 30 minutes of GPU time, $5 of cloud spend, a 5 GB download, deleting a run, or a public deployment.
8. Kit is educational. Do not call it safe, AGI, or frontier.
9. Everything stays on D:.

---

## 10. When you want depth

Ask **“go deeper”** and I will explain one idea with one example. I will not dump a textbook.

Reference files:

- `docs/GLOSSARY.md` — plain-English terms.
- `specs/TEMPLATE.md` — the spec sheet and model card.
- `STATE.md` — current chunk.
- `AGENTS.md` — working rules.
