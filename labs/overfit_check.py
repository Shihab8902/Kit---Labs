"""Chunk 4.4 — overfit 8 fixed sequences. Must reach loss < 0.1."""
import numpy as np
import torch
import torch.nn.functional as F

from src.kit.model import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ix = rng.integers(0, 50_000_000 - 257, size=8)
    ids = torch.tensor(np.stack([data[i:i + 256] for i in ix]), dtype=torch.long, device=device)
    tgt = ids[:, 1:]
    m = GPT(GPTConfig()).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for s in range(800):
        opt.zero_grad()
        loss = F.cross_entropy(m(ids)[:, :-1].reshape(-1, 4096), tgt.reshape(-1))
        loss.backward()
        opt.step()
        if (s + 1) % 100 == 0:
            print(f"step={s + 1} loss={loss.item():.4f}", flush=True)
    print(f"final_loss={loss.item():.4f} pass={loss.item() < 0.1}")


if __name__ == "__main__":
    main()
