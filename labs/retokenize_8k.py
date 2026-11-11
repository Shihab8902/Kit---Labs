"""Chunk 6.4b — re-tokenize everything with the frozen 8K pieces."""
import hashlib
import os
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def dump(texts, tok, path, batch=512):
    total = 0
    with open(path, "wb") as f:
        buf = []
        for i in range(0, len(texts), batch):
            chunk = [t + "<|endoftext|>" for t in texts[i:i + batch]]
            for enc in tok.encode_batch(chunk):
                buf.extend(enc.ids)
            if len(buf) >= 2_000_000:
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                buf = []
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
    return total


def main():
    tok = Tokenizer.from_file("tokenizer/kit2-8k.json")
    print(f"vocab={tok.get_vocab_size()}", flush=True)
    val_full = load_dataset("roneneldan/TinyStories", split="validation")
    n_half = len(val_full) // 2
    vv = [val_full[i]["text"] for i in range(n_half)]
    tt = [val_full[i]["text"] for i in range(n_half, len(val_full))]
    nv = dump(vv, tok, "data/val2.bin")
    nt = dump(tt, tok, "data/test2.bin")
    print(f"val2={nv} test2={nt}", flush=True)
    train = load_dataset("roneneldan/TinyStories", split="train")
    total = 0
    with open("data/train2.bin", "wb") as f:
        buf = []
        for i in range(0, len(train), 512):
            chunk = [train[j]["text"] + "<|endoftext|>" for j in range(i, min(i + 512, len(train)))]
            for enc in tok.encode_batch(chunk):
                buf.extend(enc.ids)
            if len(buf) >= 2_000_000:
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                buf = []
            if (i // 512 + 1) % 400 == 0:
                print(f"  train2: {i + 512}/{len(train)} tokens~{total}", flush=True)
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
    print(f"train2_tokens={total}", flush=True)
    for p in ["data/train2.bin", "data/val2.bin", "data/test2.bin"]:
        print(f"{p} bytes={os.path.getsize(p)} sha={sha12(p)}...", flush=True)


if __name__ == "__main__":
    main()
