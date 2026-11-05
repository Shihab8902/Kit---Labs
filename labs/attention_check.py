"""Chunk 4.2 check — attention shape + no future leak."""
import torch

from src.kit.model import GPTConfig, SingleHeadAttention


def main():
    torch.manual_seed(0)
    cfg = GPTConfig(d_model=32, n_ctx=8)
    attn = SingleHeadAttention(cfg)
    x = torch.randn(2, 8, 32)
    out = attn(x)
    print(f"out_shape={tuple(out.shape)} expected=(2, 8, 32)")
    x2 = x.clone()
    x2[:, 4:] = torch.randn(2, 4, 32)  # change only the future
    out2 = attn(x2)
    leak = (out2[:, :4] - out[:, :4]).abs().max().item()
    print(f"early_change={leak:.2e} no_leak={leak < 1e-6}")


if __name__ == "__main__":
    main()
