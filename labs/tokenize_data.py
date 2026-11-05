"""Chunk 3.4 — tokenize splits to uint16 files on D:."""
import hashlib
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(split_texts, tok, eot, path, batch=512):
    total = 0
    with open(path, "wb") as f:
        buf = []
        for i in range(0, len(split_texts), batch):
            chunk = [t + "<|endoftext|>" for t in split_texts[i:i + batch]]
            encs = tok.encode_batch(chunk)
            for enc in encs:
                buf.extend(enc.ids)
            if len(buf) >= 2_000_000:
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                buf = []
            if (i // batch + 1) % 200 == 0:
                print(f"  {path}: {i + batch}/{len(split_texts)} stories", flush=True)
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
    return total


def main():
    tok = Tokenizer.from_file("tokenizer/kit1-4k.json")
    eot = tok.encode("<|endoftext|>").ids[0]
    print(f"eot_id={eot} vocab={tok.get_vocab_size()}", flush=True)

    val_full = load_dataset("roneneldan/TinyStories", split="validation")
    n_half = len(val_full) // 2
    val_texts = [val_full[i]["text"] for i in range(n_half)]
    test_texts = [val_full[i]["text"] for i in range(n_half, len(val_full))]
    print(f"val={len(val_texts)} test={len(test_texts)}", flush=True)
    nv = dump(val_texts, tok, eot, "data/val.bin")
    nt = dump(test_texts, tok, eot, "data/test.bin")
    print(f"val_tokens={nv} test_tokens={nt}", flush=True)

    train = load_dataset("roneneldan/TinyStories", split="train")
    print(f"train={len(train)} stories, streaming...", flush=True)
    total = 0
    with open("data/train.bin", "wb") as f:
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
                print(f"  train: {i + 512}/{len(train)} stories tokens~{total}", flush=True)
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
    print(f"train_tokens={total}", flush=True)
    for p in ["data/train.bin", "data/val.bin", "data/test.bin"]:
        import os
        print(f"{p} bytes={os.path.getsize(p)} sha={sha256_of(p)[:12]}...", flush=True)


if __name__ == "__main__":
    main()
