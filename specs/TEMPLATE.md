# Kit Model Spec — `<model-name>`

> Copy this file to `specs/<model-name>.md` after a model is trained.
> Replace every `<...>` with a measured value. Do not invent numbers.
> If a field is unknown, write `not measured` and explain why.

**Status:** draft / trained / released
**Date:**
**Model version:**
**Git commit:**
**Checkpoint:**
**Checkpoint SHA-256:**
**Tokenizer:**
**Tokenizer SHA-256:**
**Code license:**
**Weights license:**

---

## 1. Identity

| Field | Value |
|---|---|
| Family | Kit |
| Architecture | Decoder-only Transformer |
| Intended use | Educational text generation and simple chat |
| Not intended for | Medical, legal, financial, safety-critical, or public production use |
| Language | English |
| Release tag | `kit-...` |

**One-sentence description:**

> `<What this model is, in plain English.>`

---

## 2. Size and architecture

| Spec | Value | How measured |
|---|---:|---|
| Total parameters | `<...>` | Sum of all weight tensors |
| Non-embedding parameters | `<...>` | Total minus token embeddings |
| Embedding parameters | `<...>` | `vocab × d_model` |
| Layers | `<...>` | Config |
| Attention heads | `<...>` | Config |
| `d_model` | `<...>` | Config |
| Head dimension | `<...>` | `d_model / heads` |
| Context length | `<...>` | Config |
| Vocabulary size | `<...>` | Tokenizer |
| Feed-forward type | `<GELU / SwiGLU>` | Code |
| Normalization | `<LayerNorm / RMSNorm>` | Code |
| Positional encoding | `<Learned / RoPE>` | Code |
| Tied embeddings | `<yes/no>` | Code |
| Dropout | `<...>` | Config |

**Parameter check:**

```text
N ≈ vocab × d_model + layers × 12 × d_model²
```

Show the arithmetic:

```text
N ≈ <...> + <...> = <...>
```

---

## 3. Training

| Field | Value |
|---|---|
| Dataset(s) | `<name>` |
| Dataset license(s) | `<...>` |
| Data manifest | `data/MANIFEST.md` |
| Tokenizer version | `<...>` |
| Unique training tokens | `<...>` |
| Tokens seen (including repeats) | `<...>` |
| Epochs | `<...>` |
| Sequence length | `<...>` |
| Micro-batch | `<...>` |
| Gradient accumulation | `<...>` |
| Effective batch (tokens) | `<...>` |
| Optimizer | AdamW |
| Learning rate | `<...>` |
| LR schedule | `<warmup + cosine>` |
| Weight decay | `<...>` |
| Gradient clip | `<...>` |
| Precision | `<FP16 / BF16>` |
| Hardware | `<GPU, count>` |
| Wall-clock | `<...>` |
| GPU-hours | `<...>` |
| Training FLOPs | `6 × N × D ≈ <...>` |
| Effective TFLOPS | `<...>` |
| Seed | `<...>` |
| Config path | `configs/<...>.yaml` |

**Compute check:**

```text
FLOPs ≈ 6 × <N> × <D> = <...>
Effective TFLOPS ≈ FLOPs / (wall-clock seconds × 10^12) = <...>
```

---

## 4. Evaluation

### 4.1 Loss and perplexity

| Split | Loss (nats/token) | Perplexity | Bits/byte |
|---|---:|---:|---:|
| Validation | `<...>` | `<...>` | `<...>` |
| Test | `<...>` | `<...>` | `<...>` |

Formulas:

```text
perplexity = exp(loss)
bits_per_byte = loss / ln(2) / bytes_per_token
```

`bytes_per_token` is measured on the evaluation text, not assumed.

### 4.2 Task evaluation

| Task | Method | Score | Notes |
|---|---|---:|---|
| Story coherence | 1–5 rubric, 10 prompts | `<...>` | `<...>` |
| Instruction following | 1–5 rubric, 20 prompts | `<...>` | `<...>` |
| Honesty / "I don't know" | Pass/fail, 10 prompts | `<...>` | `<...>` |
| Safety probes | Pass/fail, 10 prompts | `<...>` | `<...>` |

### 4.3 Contamination

| Check | Result |
|---|---|
| Test/train n-gram overlap | `<...>%` |
| Method | `<e.g., 13-gram overlap>` |
| Verdict | `<clean / flagged>` |

### 4.4 Samples

Include 3–5 fixed prompts and the exact sampling settings:

```text
temperature = <...>
top_k = <...>
top_p = <...>
repetition_penalty = <...>
max_new_tokens = <...>
```

**Prompt 1:** `<...>`

```text
<output>
```

**Prompt 2:** `<...>`

```text
<output>
```

**Prompt 3:** `<...>`

```text
<output>
```

---

## 5. Inference performance

| Setting | Device | Precision | TTFT | Decode tok/s | Peak memory | File size |
|---|---|---|---:|---:|---:|---:|
| Batch 1, 128-token prompt | `<...>` | FP16 | `<...>` | `<...>` | `<...>` | `<...>` |
| Batch 1, 128-token prompt | `<...>` | INT8 | `<...>` | `<...>` | `<...>` | `<...>` |
| CPU | `<...>` | FP32 | `<...>` | `<...>` | `<...>` | `<...>` |

**KV-cache size:**

```text
2 × layers × d_model × context × bytes = <...>
```

**Model file size:**

```text
params × bytes_per_param = <...>
```

---

## 6. Limitations

Be specific. Examples:

- `<Hallucinates facts; has no retrieval or tool use.>`
- `<Forgets earlier turns beyond the context window.>`
- `<Weak at math and code.>`
- `<May reproduce biases present in the training data.>`
- `<Can be prompted to produce unsafe text; not deployed publicly.>`
- `<English-only.>`

---

## 7. Safety and intended use

- **Intended:** local experimentation, education, demos, research on small models.
- **Not intended:** medical, legal, financial, safety-critical, or unsupervised use with the public.
- **Guardrails:** `<none / input length / content filter / rate limit>`.
- **Data privacy:** `<no personal data / describe>`.
- **Output labeling:** generated text should be labeled as AI-generated.

---

## 8. Reproduction

```powershell
# from a clean checkout
. .\scripts\env.ps1
.\.venv\Scripts\Activate.ps1
python -m kit.eval --checkpoint <path> --split test
python -m kit.sample --checkpoint <path> --prompt "Once upon a time"
```

Record:

- Git commit
- Config path
- Checkpoint hash
- Tokenizer hash
- Dataset manifest hash

---

## 9. Comparison to the previous Kit

| Metric | Previous | This model | Delta |
|---|---:|---:|---:|
| Parameters | `<...>` | `<...>` | `<...>` |
| Training tokens | `<...>` | `<...>` | `<...>` |
| Validation loss | `<...>` | `<...>` | `<...>` |
| Test loss | `<...>` | `<...>` | `<...>` |
| Decode tok/s | `<...>` | `<...>` | `<...>` |
| File size | `<...>` | `<...>` | `<...>` |

**What changed and why:**

> `<One paragraph. One architecture change, one data change, or one training change — not five.>`

**Did it work?**

> `<Yes/no, with the number and the decision.>`

---

## 10. Changelog

| Date | Change |
|---|---|
| `<...>` | `<...>` |
