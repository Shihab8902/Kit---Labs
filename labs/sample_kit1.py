"""Chunks 4.6/4.7 — test numbers + first samples from Kit-1."""
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.kit.model import GPT, GPTConfig


def batch_loss(m, data, n_batches=20):
    rng = np.random.default_rng(1)
    tot, tok = 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            ix = rng.integers(0, len(data) - 257, size=8)
            ids = torch.tensor(np.stack([data[i:i + 256] for i in ix]), dtype=torch.long, device="cuda")
            l = F.cross_entropy(m(ids)[:, :-1].reshape(-1, 4096), ids[:, 1:].reshape(-1)).item()
            tot += l * 8 * 255
            tok += 8 * 255
    return tot / tok


@torch.no_grad()
def sample(m, tok, prompt, max_new=100, temp=0.8, top_k=50):
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device="cuda")
    eot = tok.encode("<|endoftext|>").ids[0]
    for _ in range(max_new):
        logits = m(ids[:, -256:])[0, -1] / temp
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], 1)
        if nxt == eot:
            break
    return tok.decode(ids[0].tolist())


def main():
    torch.manual_seed(0)
    m = GPT(GPTConfig()).cuda().eval()
    ckpt = torch.load("runs/kit1-2026-11-11-a/checkpoints/best.pt", weights_only=False)
    m.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_file("tokenizer/kit1-4k.json")
    test = np.memmap("data/test.bin", dtype=np.uint16, mode="r")
    tl = batch_loss(m, test)
    print(f"test_loss={tl:.3f} perplexity={np.exp(tl):.1f} best_val={ckpt['val_loss']:.3f}")
    for p in ["Once upon a time", "The little dog", "Lily was"]:
        print(f"--- prompt: {p}")
        print(sample(m, tok, p))


if __name__ == "__main__":
    main()
