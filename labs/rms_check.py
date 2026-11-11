"""Chunk 6.1 check — RMSNorm model runs and learns."""
import numpy as np
import torch
import torch.nn.functional as F

from src.kit.model import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = GPTConfig(norm="rmsnorm")
    m = GPT(cfg).to(device)
    ids = torch.randint(0, 4096, (2, 64), device=device)
    out = m(ids)
    print(f"shape_ok={tuple(out.shape) == (2, 64, 4096)} params={m.count_params()}")

    data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ix = rng.integers(0, 50_000_000 - 257, size=8)
    fix = torch.tensor(np.stack([data[i:i + 256] for i in ix]), dtype=torch.long, device=device)
    tgt = fix[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for s in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(m(fix)[:, :-1].reshape(-1, 4096), tgt.reshape(-1))
        loss.backward()
        opt.step()
    first = loss.item()
    print(f"loss_after_100={first:.3f} trains={first < 8.0}")


if __name__ == "__main__":
    main()
