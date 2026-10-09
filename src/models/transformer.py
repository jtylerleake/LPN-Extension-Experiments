"""Transformer layer, ported from lpn's src/models/utils.py (Flax) at commit 0adfe56.

Matches the original exactly so the pretrained weights load unchanged
(src/models/pretrained.py):
- pre-norm residual blocks;
- LayerNorm with neither scale nor bias (no parameters), eps 1e-6 (Flax's default);
- multi-head attention without biases, softmax in fp32;
- MLP Dense(emb -> 4 emb) -> SiLU -> Dense(4 emb -> emb), no biases.
Masks are boolean with True = may attend (Flax's convention, also
F.scaled_dot_product_attention's).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.config import ModelConfig

LN_EPS = 1e-6


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    """Flax nn.LayerNorm(use_scale=False, use_bias=False)."""
    return F.layer_norm(x, x.shape[-1:], eps=LN_EPS)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim = cfg.emb_dim
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        def heads(t):  # (..., L, H*D) -> (..., H, L, D)
            return t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(-2, -3).flatten(-2))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        hidden = int(cfg.mlp_factor * cfg.emb_dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.emb_dim, hidden, bias=False), nn.SiLU(),
                                 nn.Linear(hidden, cfg.emb_dim, bias=False), nn.Dropout(cfg.dropout))
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn_dropout(self.attn(layer_norm(x), mask))
        return x + self.mlp(layer_norm(x))
