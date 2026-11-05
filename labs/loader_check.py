"""Chunk 3.5 — loader check: right shape, readable English."""
import numpy as np
from tokenizers import Tokenizer


def get_batch(split, batch_size, block_size, rng):
    path = {"train": "data/train.bin", "val": "data/val.bin", "test": "data/test.bin"}[split]
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(data)
    ix = rng.integers(0, n - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def main():
    rng = np.random.default_rng(0)
    tok = Tokenizer.from_file("tokenizer/kit1-4k.json")
    x, y = get_batch("train", 16, 256, rng)
    print(f"x_shape={x.shape} y_shape={y.shape} ok={x.shape == (16, 256) and y.shape == (16, 256)}")
    print(f"shift_ok={bool((x[:, 1:] == y[:, :-1]).all())}")
    text = tok.decode(x[0].tolist())
    print(f"---decoded first 300 chars---\n{text[:300]}")
    xv, _ = get_batch("val", 4, 256, rng)
    print(f"val_shape={xv.shape} val_ok={xv.shape == (4, 256)}")


if __name__ == "__main__":
    main()
