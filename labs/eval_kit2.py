"""Chunks 6.11/6.12 — Kit-2 test numbers vs Kit-1, plus first samples."""
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from datasets import load_dataset

from src.kit.model import GPT, GPTConfig


def test_loss(m, data, V, batches=20):
    rng = np.random.default_rng(1)
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(batches):
            ix = rng.integers(0, len(data) - 513, size=8)
            ids = torch.tensor(np.stack([data[i:i + 512] for i in ix]), dtype=torch.long, device="cuda")
            tot += F.cross_entropy(m(ids)[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1)).item() * 8 * 511
            n += 8 * 511
    return tot / n


def bytes_per_token(tok, texts):
    b, t = 0, 0
    for x in texts[:2000]:
        e = tok.encode(x + "<|endoftext|>")
        b += len((x + "<|endoftext|>").encode("utf-8"))
        t += len(e.ids)
    return b / t


@torch.no_grad()
def sample(m, tok, prompt, max_new=100, temp=0.8, top_k=50):
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device="cuda")
    eot = tok.encode("<|endoftext|>").ids[0]
    for _ in range(max_new):
        logits = m(ids[:, -512:])[0, -1] / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], 1)
        if nxt == eot:
            break
    return tok.decode(ids[0].tolist())


def main():
    val = load_dataset("roneneldan/TinyStories", split="validation")
    n_half = len(val) // 2
    val_txt = [val[i]["text"] for i in range(n_half)]
    test_txt = [val[i]["text"] for i in range(n_half, len(val))]

    # Kit-2
    m2 = GPT(GPTConfig(8192, 256, 6, 8, 512, norm="rmsnorm", pos="rope", mlp="swiglu")).cuda().eval()
    c2 = torch.load("runs/kit2-2026-11-11-a/checkpoints/best.pt", weights_only=False)
    m2.load_state_dict(c2["model"])
    t2 = Tokenizer.from_file("tokenizer/kit2-8k.json")
    d2 = np.memmap("data/test2.bin", dtype=np.uint16, mode="r")
    l2 = test_loss(m2, d2, 8192)
    b2 = bytes_per_token(t2, test_txt)
    print(f"kit2 test_loss={l2:.4f} ppl={np.exp(l2):.2f} bytes_per_tok={b2:.2f} bits_per_byte={l2 / 0.6931 / b2:.4f} best_val={c2['val_loss']:.4f}")

    # Kit-1 (same texts, its own pieces)
    m1 = GPT(GPTConfig()).cuda().eval()
    c1 = torch.load("runs/kit1-2026-11-11-a/checkpoints/best.pt", weights_only=False)
    m1.load_state_dict(c1["model"])
    t1 = Tokenizer.from_file("tokenizer/kit1-4k.json")
    d1 = np.memmap("data/test.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1)
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(20):
            ix = rng.integers(0, len(d1) - 257, size=8)
            ids = torch.tensor(np.stack([d1[i:i + 256] for i in ix]), dtype=torch.long, device="cuda")
            tot += F.cross_entropy(m1(ids)[:, :-1].reshape(-1, 4096), ids[:, 1:].reshape(-1)).item() * 8 * 255
            n += 8 * 255
    l1 = tot / n
    b1 = bytes_per_token(t1, test_txt)
    print(f"kit1 test_loss={l1:.4f} ppl={np.exp(l1):.2f} bytes_per_tok={b1:.2f} bits_per_byte={l1 / 0.6931 / b1:.4f}")

    for p in ["Once upon a time", "The little dog", "Lily was"]:
        print(f"--- {p}\n{sample(m2, t2, p)}")


if __name__ == "__main__":
    main()
