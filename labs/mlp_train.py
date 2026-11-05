"""Chunk 1.5 — train the MLP with plain SGD (tiny nudge rule)."""
import numpy as np


def forward(ids, E, W1, b1, W2, b2):
    x_emb = E[ids]
    x = x_emb.reshape(ids.shape[0], -1)
    h_pre = x @ W1 + b1
    h = np.tanh(h_pre)
    logits = h @ W2 + b2
    return x_emb, x, h, logits


def loss_and_grad(ids, tgt, E, W1, b1, W2, b2):
    B = len(ids)
    x_emb, x, h, logits = forward(ids, E, W1, b1, W2, b2)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    P = e / e.sum(axis=1, keepdims=True)
    loss = float(-np.log(P[np.arange(B), tgt] + 1e-12).mean())
    dlog = P.copy()
    dlog[np.arange(B), tgt] -= 1.0
    dlog /= B
    dW2 = h.T @ dlog
    db2 = dlog.sum(axis=0)
    dh = dlog @ W2.T
    dh_pre = dh * (1.0 - h ** 2)
    dW1 = x.T @ dh_pre
    db1 = dh_pre.sum(axis=0)
    dx = dh_pre @ W1.T
    dx_emb = dx.reshape(x_emb.shape)
    dE = np.zeros_like(E)
    np.add.at(dE, ids, dx_emb)
    return loss, (dE, dW1, db1, dW2, db2)


def batches(ids_all, tgt_all, bs, rng):
    idx = rng.permutation(len(ids_all))[:bs]
    return ids_all[idx], tgt_all[idx]


def main():
    with open("data/raw/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.array([stoi[ch] for ch in text], dtype=np.int64)
    ctx = 8
    xs = np.lib.stride_tricks.sliding_window_view(data, ctx + 1)
    ids_all = xs[:, :-1].copy()
    tgt_all = xs[:, -1].copy()
    n = int(len(ids_all) * 0.9)
    tr_ids, tr_tgt = ids_all[:n], tgt_all[:n]

    rng = np.random.default_rng(0)
    vocab, emb, hid = len(chars), 32, 128
    E = rng.normal(0, 0.1, (vocab, emb))
    W1 = rng.normal(0, 0.1, (ctx * emb, hid))
    b1 = np.zeros(hid)
    W2 = rng.normal(0, 0.1, (hid, vocab))
    b2 = np.zeros(vocab)
    params = [E, W1, b1, W2, b2]

    lr, bs, steps = 0.1, 512, 1500
    for s in range(steps):
        bi, bt = batches(tr_ids, tr_tgt, bs, rng)
        loss, grads = loss_and_grad(bi, bt, *params)
        for p, g in zip(params, grads):
            p -= lr * g
        if (s + 1) % 500 == 0:
            print(f"step={s + 1} train_loss={loss:.3f}")
    print(f"final_train_loss={loss:.3f} bigram=2.482 beat={loss < 2.482}")

    # overfit: one tiny fixed batch, prove the learner CAN memorize
    E2 = E.copy(); W1b = W1.copy(); b1b = b1.copy(); W2b = W2.copy(); b2b = b2.copy()
    p2 = [E2, W1b, b1b, W2b, b2b]
    oi, ot = tr_ids[:32].copy(), tr_tgt[:32].copy()
    for s in range(300):
        loss_o, grads = loss_and_grad(oi, ot, *p2)
        for p, g in zip(p2, grads):
            p -= 0.1 * g
    print(f"tiny_batch_loss={loss_o:.4f} near_zero={loss_o < 0.2}")


if __name__ == "__main__":
    main()
