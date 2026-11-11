"""Chunk 6.3 check — SwiGLU runs, Kit-2 size matches plan."""
import numpy as np
import torch
import torch.nn.functional as F

from src.kit.model import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = GPT(GPTConfig(norm="rmsnorm", pos="rope", mlp="swiglu")).to(device)
    ids = torch.randint(0, 4096, (2, 64), device=device)
    print(f"small_shape_ok={tuple(m(ids).shape) == (2, 64, 4096)}")

    data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ix = rng.integers(0, 50_000_000 - 257, size=8)
    fix = torch.tensor(np.stack([data[i:i + 256] for i in ix]), dtype=torch.long, device=device)
    tgt = fix[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for s in range(50):
        opt.zero_grad()
        loss = F.cross_entropy(m(fix)[:, :-1].reshape(-1, 4096), tgt.reshape(-1))
        loss.backward()
        opt.step()
    print(f"swiglu_50step_loss={loss.item():.3f} trains={loss.item() < 8.0}")

    k2 = GPT(GPTConfig(vocab_size=8192, d_model=256, n_layer=6, n_head=8, n_ctx=512,
                       norm="rmsnorm", pos="rope", mlp="swiglu"))
    n = k2.count_params()
    print(f"kit2_params={n} (~6.8M target, ok={6_000_000 < n < 7_500_000})")


if __name__ == "__main__":
    main()
