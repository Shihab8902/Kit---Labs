# Kit Glossary

Plain-English definitions, in the order you will meet them. Add a term every time a new one appears.

| Term | Plain English | Why it matters |
|---|---|---|
| **Model** | A function with adjustable numbers inside. | It turns an input into an output; the numbers are learned. |
| **Parameter** | One of those adjustable numbers. | The count is the model's size; training changes these. |
| **Weight** | Another name for a parameter, usually a matrix entry. | Weights are what get saved in a checkpoint. |
| **Tensor** | A box of numbers with a shape. | GPUs do math on tensors, often in parallel. |
| **Token** | A small piece of text: a word, part of a word, or a character. | Models read tokens, not raw text. |
| **Tokenizer** | The translator between text and token IDs. | Its vocabulary and rules are frozen for a model family. |
| **Vocabulary** | The set of all tokens a tokenizer knows. | Bigger vocabulary = shorter sequences but a bigger embedding table. |
| **Token ID** | The integer for a token. | The model's input is a list of these integers. |
| **Embedding** | A learned vector for each token. | It turns an ID into numbers the model can mix. |
| **Logits** | Raw scores for every possible next token. | Softmax turns them into probabilities. |
| **Softmax** | A function that turns scores into probabilities that sum to 1. | It lets the model choose the next token. |
| **Loss** | A number measuring how surprised the model is by the correct answer. | Lower is better; training tries to reduce it. |
| **Cross-entropy** | The standard loss for next-token prediction. | It is the negative log-probability of the correct token. |
| **Perplexity** | `exp(loss)`. Roughly, how many equally likely options the model feels it is choosing among. | Lower is better, but only comparable across similar tokenizers. |
| **Bits per byte** | Loss converted to bits, divided by the number of bytes of text. | A fairer comparison across different tokenizers. |
| **Gradient** | The slope that says how changing a weight affects the loss. | It tells the optimizer which way to turn the knobs. |
| **Backpropagation** | The algorithm that computes gradients efficiently. | It is why training deep networks is possible. |
| **Optimizer** | The rule that updates weights using gradients. | AdamW is the standard starting choice. |
| **Learning rate** | How far the optimizer moves each weight per step. | Too high: divergence. Too low: slow or stuck. |
| **Epoch** | One full pass over the training data. | More epochs can overfit; tokens seen is the better budget. |
| **Batch** | A group of examples processed together. | Bigger batches are more stable but use more memory. |
| **Gradient accumulation** | Adding gradients from several small batches before one update. | It fakes a bigger batch when VRAM is limited. |
| **Overfitting** | Memorizing the training data instead of learning general patterns. | Training loss falls while validation loss rises. |
| **Validation set** | Data used to make tuning decisions. | It is not the final exam. |
| **Test set** | Data touched only at the end. | It estimates performance on unseen text. |
| **Contamination** | Test text accidentally appearing in training data. | It makes scores look better than they are. |
| **Attention** | A mechanism that lets each token mix information from earlier tokens. | It is the core of the transformer. |
| **Q / K / V** | Query, key, and value: the three projections used in attention. | Q asks what is relevant; K labels each token; V carries the content. |
| **Causal mask** | A rule that prevents a token from looking at future tokens. | It makes next-token prediction honest. |
| **Transformer** | The neural network architecture built from attention and MLP blocks. | It is the architecture behind Kit and most modern language models. |
| **Residual connection** | Adding a block's input to its output. | It helps gradients and information flow. |
| **LayerNorm / RMSNorm** | Normalization that stabilizes activations. | It makes training more reliable. |
| **RoPE** | Rotary positional encoding: encodes position by rotating vectors. | It helps the model reason about order and extend context. |
| **SwiGLU** | A gated feed-forward layer used in modern transformers. | It often improves quality at the same parameter budget. |
| **Context window** | The maximum number of tokens the model can see at once. | It limits memory and how much conversation it can hold. |
| **Inference** | Using a trained model to generate text. | No gradients, no weight updates. |
| **KV cache** | Saved keys and values from earlier tokens during generation. | It avoids recomputing attention and makes chat fast. |
| **Sampling** | Choosing the next token from the model's probabilities. | Temperature, top-k, and top-p control randomness. |
| **Temperature** | A divisor on logits before softmax. | Lower = more predictable; higher = more random. |
| **Top-k / top-p** | Keep only the k best tokens / the smallest set with probability p. | They cut off unlikely tokens and reduce nonsense. |
| **Quantization** | Storing weights in fewer bits (e.g., int8 instead of fp16). | Smaller and often faster, with some quality loss. |
| **Checkpoint** | A save file for training: weights, optimizer, step, and RNG. | It lets you resume exactly where you stopped. |
| **FLOPs** | Floating-point operations: a unit of compute. | Training compute is roughly `6 × params × tokens`. |
| **MFU** | Model FLOPs utilization: useful FLOPs divided by hardware peak. | It measures how efficiently the GPU is being used. |
| **VRAM** | The GPU's own memory. | If the model and training state do not fit, you get an OOM error. |
| **SFT** | Supervised fine-tuning: training on example answers. | It teaches format and instruction following. |
| **Chat template** | The exact text format for system, user, and assistant turns. | The model only behaves well if inference matches training. |
| **Loss masking** | Ignoring the loss on prompt tokens. | The model is graded only on its own answer. |
| **DPO** | Direct preference optimization: learning from chosen vs rejected answers. | A simpler alternative to full RLHF. |
| **Scaling law** | A rough math rule for how loss changes with size, data, or compute. | Labs use it to predict whether a big run is worth it. |
| **Distillation** | Training a small model on a bigger model's outputs. | It can transfer capability into a cheaper model. |
| **Ablation** | A controlled experiment that changes one thing. | It tells you whether a change actually helped. |
| **Spec sheet** | A measured summary of a model's size, training, quality, and speed. | It makes comparisons honest and reproducible. |
| **Model card** | A document describing intended use, limitations, and evaluation. | It is the standard way to release a model responsibly. |

---

## How to use this glossary

- When a new term appears, add it here in one line.
- Prefer the plain-English column. The technical name goes in the first column.
- If you cannot define a term without using the term itself, you do not understand it yet. Ask for a different analogy.
