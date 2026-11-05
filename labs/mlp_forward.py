"""Chunk 1.3 — MLP forward pass (guess step, no learning yet)."""
import numpy as np


def forward(ids, E, W1, b1, W2, b2):
    # ids: (batch, 8) -> E: (vocab, 32) -> h: (batch, 128) -> logits: (batch, vocab)
    x = E[ids].reshape(ids.shape[0], -1)
    h = np.tanh(x @ W1 + b1)
    logits = h @ W2 + b2
    return logits


def main():
    rng = np.random.default_rng(0)
    vocab, ctx, emb, hid, batch = 65, 8, 32, 128, 4
    E = rng.normal(0, 0.1, (vocab, emb))
    W1 = rng.normal(0, 0.1, (ctx * emb, hid))
    b1 = np.zeros(hid)
    W2 = rng.normal(0, 0.1, (hid, vocab))
    b2 = np.zeros(vocab)
    ids = rng.integers(0, vocab, (batch, ctx))
    logits = forward(ids, E, W1, b1, W2, b2)
    print(f"logits_shape={logits.shape} expected=({batch}, {vocab})")
    print(f"shape_ok={logits.shape == (batch, vocab)}")


if __name__ == "__main__":
    main()
