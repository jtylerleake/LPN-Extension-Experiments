"""LPN encoder q(z | pairs) (PROJECT-SPEC.md section 2.1).

Each (input, output) pair is encoded *independently* by a small transformer
into a diagonal Gaussian (mu_i, logvar_i). Pairs are then mean-pooled (both
mu and logvar, as in the LPN reference implementation): signal shared across
pairs (the tile) reinforces, signal that varies per pair (the anchor)
averages out.

Leave-one-out pooling for training reuses the per-pair posteriors:
mean over j != i = (sum_j - x_i) / (n - 1). One encoder pass per pair, not
n separate (n - 1)-pair passes -- equivalent because each pair is encoded
independently (checked against brute force in tests/test_model.py).
"""
from __future__ import annotations

import torch
from torch import nn

from lpn_interp.config import GRID_SIZE, NUM_COLORS, ModelConfig

NUM_CELLS = GRID_SIZE * GRID_SIZE


def transformer(cfg: ModelConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        cfg.d_model, cfg.num_heads, cfg.mlp_dim, cfg.dropout, activation="gelu",
        batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, cfg.num_layers, enable_nested_tensor=False)


class GridEmbedding(nn.Module):
    """color + cell position (shared by every grid) + segment type."""

    def __init__(self, d_model: int, num_segments: int):
        super().__init__()
        self.color = nn.Embedding(NUM_COLORS, d_model)
        self.position = nn.Parameter(torch.randn(NUM_CELLS, d_model) * 0.02)
        self.segment = nn.Embedding(num_segments, d_model)

    def forward(self, grid: torch.Tensor, segment: int) -> torch.Tensor:
        """grid (..., 4, 4) -> (..., 16, d_model)"""
        cells = grid.flatten(-2)
        return self.color(cells) + self.position[: cells.shape[-1]] + self.segment.weight[segment]


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = GridEmbedding(cfg.d_model, num_segments=2)
        self.cls = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.body = transformer(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.to_posterior = nn.Linear(cfg.d_model, 2 * cfg.latent_dim)

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs/outputs (B, P, 4, 4) -> per-pair (mu, logvar), each (B, P, d)."""
        batch, num_pairs = inputs.shape[:2]
        tokens = torch.cat([self.embed(inputs, 0), self.embed(outputs, 1)], dim=-2).flatten(0, 1)
        cls = self.cls.expand(tokens.shape[0], 1, -1)
        hidden = self.body(torch.cat([cls, tokens], dim=1))
        mu, logvar = self.to_posterior(self.norm(hidden[:, 0])).float().chunk(2, dim=-1)
        return mu.unflatten(0, (batch, num_pairs)), logvar.unflatten(0, (batch, num_pairs))


def pool(mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, P, d) per-pair posteriors -> (B, d) pooled over all pairs."""
    return mu.mean(1), logvar.mean(1)


def loo_pool(mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, P, d) -> (B, P, d): entry i pools every pair except i."""
    num_pairs = mu.shape[1]
    if num_pairs < 2:
        raise ValueError("leave-one-out pooling needs at least 2 pairs")
    return (
        (mu.sum(1, keepdim=True) - mu) / (num_pairs - 1),
        (logvar.sum(1, keepdim=True) - logvar) / (num_pairs - 1),
    )
