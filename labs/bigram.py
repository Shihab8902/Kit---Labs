"""Chunk 1.2 — bigram: guess next char from the last char only."""
import numpy as np


def main():
    with open("data/raw/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([stoi[ch] for ch in text], dtype=np.int64)
    n = int(len(ids) * 0.9)
    train, test = ids[:n], ids[n:]
    v = len(chars)

    counts = np.ones((v, v), dtype=np.float64)  # +1 so no zero chance
    for a, b in zip(train[:-1], train[1:]):
        counts[a, b] += 1
    probs = counts / counts.sum(axis=1, keepdims=True)

    # loss = average surprise on held-out text (lower is better)
    p = probs[test[:-1], test[1:]]
    loss = float(-np.log(p).mean())
    print(f"vocab_size={v} train_chars={len(train)} test_chars={len(test)}")
    print(f"bigram_loss={loss:.3f}")

    # sample 300 chars starting from "T"
    rng = np.random.default_rng(0)
    cur = stoi.get("T", 0)
    out = [cur]
    for _ in range(300):
        cur = int(rng.choice(v, p=probs[cur]))
        out.append(cur)
    sample = "".join(chars[i] for i in out)
    print("---sample---")
    print(sample)


if __name__ == "__main__":
    main()
