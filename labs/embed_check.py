"""Chunk 4.1 check — embedding output shape."""
import torch

from src.kit.model import Embeddings, GPTConfig


def main():
    cfg = GPTConfig()
    m = Embeddings(cfg)
    ids = torch.randint(0, cfg.vocab_size, (4, 256))
    out = m(ids)
    print(f"out_shape={tuple(out.shape)} expected=(4, 256, 128)")
    print(f"shape_ok={tuple(out.shape) == (4, 256, 128)}")


if __name__ == "__main__":
    main()
