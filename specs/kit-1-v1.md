# Kit-1 v1 — Spec Sheet

**Status:** trained
**Date:** 2026-11-11
**Run:** `runs/kit1-2026-11-11-a` (local only)
**Checkpoint:** `runs/kit1-2026-11-11-a/checkpoints/best.pt`

## Size

| Spec | Value |
|---|---|
| Total params | 1,344,640 |
| Non-embedding params | 820,352 |
| Layers / heads / width | 4 / 4 / 128 |
| Context / vocab | 256 / 4,096 |
| Tied embeddings | yes |

## Training

| Field | Value |
|---|---|
| Data | TinyStories train, first ~50M tokens seen |
| Steps | 3,052 × 16,384 tokens |
| Optimizer | AdamW, peak lr 1e-3, warmup 100, cosine to 1e-4 |
| Precision | FP16 mixed, grad clip 1.0 |
| Wall-clock | 271 s on RTX 3050 6GB (~184k tok/s) |
| FLOPs | ~4.0e14 (6×N×D) |

## Eval

| Split | Loss | Perplexity |
|---|---|---|
| Val (best) | 2.275 | 9.7 |
| Test | 2.299 | 10.0 |

## Samples (temp 0.8, top-k 50)

- "Once upon a time" → Lily story, coherent, simple sentences.
- "The little dog" → short dog story, on topic.
- "Lily was" → emotional mini-story, readable.

## Limits

- Tiny vocab/stories only; no facts, no reasoning, no long memory.
- Test overlap with train (6,601 shared stories) flatters scores slightly.
