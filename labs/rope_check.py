"""Chunk 6.2 check — RoPE shape, no leak, tiny generation."""
import torch

from src.kit.model import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = GPTConfig(norm="rmsnorm", pos="rope")
    m = GPT(cfg).to(device).eval()
    ids = torch.randint(0, 4096, (2, 64), device=device)
    out = m(ids)
    print(f"shape_ok={tuple(out.shape) == (2, 64, 4096)}")

    # causality still holds with rotation
    x = torch.randn(2, 8, 32, device=device)
    from src.kit.model import MultiHeadAttention, GPTConfig as C
    attn = MultiHeadAttention(C(d_model=32, n_head=4, pos="rope")).to(device).eval()
    with torch.no_grad():
        o1 = attn(x)
        x2 = x.clone()
        x2[:, 4:] = torch.randn(2, 4, 32, device=device)
        o2 = attn(x2)
    print(f"no_leak={bool(((o2[:, :4] - o1[:, :4]).abs().max() < 1e-6))}")

    # tiny generation: 20 steps from a fixed start, must not crash
    cur = torch.tensor([[464]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(20):
            nxt = m(cur[:, -64:])[0, -1].argmax(-1).item()
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], 1)
    print(f"gen_ids={cur[0].tolist()[:8]}... gen_len_ok={cur.shape[1] == 21}")


if __name__ == "__main__":
    main()
