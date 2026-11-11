# Kit-2 v1 — Spec Sheet

**Status:** trained
**Date:** 2026-11-11
**Run:** `runs/kit2-2026-11-11-a` (local only)
**Checkpoint:** `runs/kit2-2026-11-11-a/checkpoints/best.pt`

## Size

| Spec | Value |
|---|---|
| Total params | 6,947,072 (~6.9M) |
| Non-embedding params | ~4.85M |
| Layers / heads / width | 6 / 8 / 256 |
| Context / vocab | 512 / 8,192 |
| Norm / pos / MLP | RMSNorm / RoPE / SwiGLU, no biases, tied embeddings |

## Training

| Field | Value |
|---|---|
| Data | TinyStories train, 150M tokens seen (468M available) |
| Steps | 4,580 x 32,768 tokens |
| Optimizer | AdamW, peak lr 2e-3 (swept), warmup 200, cosine to 1e-4 |
| Precision | FP16 mixed, grad clip 1.0 |
| Wall-clock | 3,865 s on RTX 3050 6GB (~38.8k tok/s) |
| Peak VRAM | 2.69 GB |
| FLOPs | ~6.3e15 (6xNxD) |

## Eval (same held-out stories)

| Model | Test loss | Perplexity | Bits/byte |
|---|---|---|---|
| Kit-2 | 1.9669 | 7.15 | 0.6846 |
| Kit-1 | 2.2985 | 9.96 | 1.0441 |

## Samples (temp 0.8, top-k 50)

Coherent multi-paragraph stories with dialogue and consistent characters; clearly above Kit-1.

## Limits

Same as Kit-1 plus: test overlap with train flatters scores slightly; no instruction following yet (Stage 7).
