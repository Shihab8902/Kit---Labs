"""Kit-1 model — grown one chunk at a time. Chunk 4.3: full transformer."""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_ctx: int = 256


class Embeddings(nn.Module):
    """Token lookup + learned position signal."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.size(1)
        pos = torch.arange(t, device=ids.device)
        return self.wte(ids) + self.wpe(pos)


class SingleHeadAttention(nn.Module):
    """One mixer: each position pulls meaning from earlier positions only."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / (c ** 0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.out(weights @ v)


class MultiHeadAttention(nn.Module):
    """Several mixers side by side, future blocked."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.hd).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.hd).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.hd ** 0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        y = weights @ v
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.out(y)


class Block(nn.Module):
    """One layer: mix with the past, then think per position."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Kit-1: embeddings + 4 blocks + scorer with tied weights."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.emb.wte.weight  # tied: read and write share
        self.apply(self._init_weights)
        for blk in self.blocks:  # smaller start for residual paths
            blk.attn.out.weight.data.normal_(0, 0.02 / (2 * cfg.n_layer) ** 0.5)
            blk.mlp[2].weight.data.normal_(0, 0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(0, 0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
