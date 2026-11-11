"""Kit-1 model — grown one chunk at a time. Chunk 4.3: full transformer."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_ctx: int = 256
    norm: str = "layernorm"  # or "rmsnorm" (Kit-2)
    pos: str = "learned"  # or "rope" (Kit-2)
    mlp: str = "gelu"  # or "swiglu" (Kit-2)


class Embeddings(nn.Module):
    """Token lookup + optional learned position signal."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.cfg.pos == "rope":
            return self.wte(ids)  # positions arrive later, as rotation
        t = ids.size(1)
        pos = torch.arange(t, device=ids.device)
        return self.wte(ids) + self.wpe(pos)


class Rotary(nn.Module):
    """Position as rotation: spin each question/label pair a little per seat."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hd = cfg.d_model // cfg.n_head
        inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(cfg.n_ctx).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-2]
        cos, sin = self.cos[:t], self.sin[:t]  # (T, hd/2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack([rot1, rot2], dim=-1).reshape(x.shape)


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
        self.rope = Rotary(cfg) if cfg.pos == "rope" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.hd).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.hd).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.hd).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        scores = q @ k.transpose(-2, -1) / (self.hd ** 0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        y = weights @ v
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.out(y)


class RMSNorm(nn.Module):
    """Cheaper steadier: rescale by vector size only, no mean shift."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: GPTConfig):
    return RMSNorm(cfg.d_model) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.d_model, bias=False)


class SwiGLU(nn.Module):
    """Gated mixer: one path decides how much of the other path passes."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = int(8 * cfg.d_model / 3)
        self.gate = nn.Linear(cfg.d_model, h, bias=False)
        self.up = nn.Linear(cfg.d_model, h, bias=False)
        self.down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One layer: mix with the past, then think per position."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = make_norm(cfg)
        if cfg.mlp == "swiglu":
            self.mlp = SwiGLU(cfg)
        else:
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
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.emb.wte.weight  # tied: read and write share
        self.apply(self._init_weights)
        for blk in self.blocks:  # smaller start for residual paths
            out_proj = blk.mlp[2] if isinstance(blk.mlp, nn.Sequential) else blk.mlp.down
            blk.attn.out.weight.data.normal_(0, 0.02 / (2 * cfg.n_layer) ** 0.5)
            out_proj.weight.data.normal_(0, 0.02 / (2 * cfg.n_layer) ** 0.5)

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
