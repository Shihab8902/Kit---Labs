"""Chunk 4.3 check — shape, param count, init loss near ln(vocab)."""
import math

import torch
import torch.nn.functional as F

from src.kit.model import GPT, GPTConfig


def main():
    torch.manual_seed(0)
    cfg = GPTConfig()
    m = GPT(cfg)
    n = m.count_params()
    # exact: token table + position table + layers(attn 4d2 + mlp 8d2 + 2 norms) + final norm
    expect = (cfg.vocab_size * cfg.d_model + cfg.n_ctx * cfg.d_model
              + cfg.n_layer * (12 * cfg.d_model ** 2 + 2 * cfg.d_model)
              + cfg.d_model)
    print(f"params={n} exact={expect} match={n == expect} (~1.34M, plan said ~1.3M)")
    ids = torch.randint(0, cfg.vocab_size, (4, 256))
    logits = m(ids)
    print(f"logits_shape={tuple(logits.shape)} ok={tuple(logits.shape) == (4, 256, 4096)}")
    # position i must predict token i+1, so compare logits[:, :-1] with ids[:, 1:]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)).item()
    print(f"init_loss={loss:.3f} ln_vocab={math.log(cfg.vocab_size):.3f} near={abs(loss - math.log(cfg.vocab_size)) < 0.3}")


if __name__ == "__main__":
    main()
