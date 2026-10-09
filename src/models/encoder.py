"""LPN encoder q(z | pair), ported from lpn's EncoderTransformer (prompts/PROJECT-SPEC-1.md section 2.1).

Each (input, output) pair is encoded independently into a diagonal Gaussian
(mu_i, logvar_i). Token sequence (37 tokens for 4x4 grids), as in the original:
  [CLS, rows_in, rows_out, cols_in, cols_out, cell(0,0)_in, cell(0,0)_out, cell(0,1)_in, ...]
- cells: color + row-position + col-position + channel (input/output) embeddings,
  input and output cells interleaved per position;
- grid-shape tokens: shape embedding (+ channel embedding). Our grids are always
  4x4, so these are constant tokens and the original's padding mask is all-True
  (omitted here).
The CLS token goes through a parameter-free LayerNorm into two bias-free
projections, mu and logvar.

How pairs are combined into one z (sample per pair, then mean) lives in
src/models/lpn.py.
"""
from __future__ import annotations

import torch
from torch import nn

from src.config import GRID_SIZE, NUM_COLORS, ModelConfig
from src.models.transformer import TransformerLayer, layer_norm


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim = cfg.emb_dim
        self.colors_embed = nn.Embedding(NUM_COLORS, dim)
        self.pos_row_embed = nn.Embedding(GRID_SIZE, dim)
        self.pos_col_embed = nn.Embedding(GRID_SIZE, dim)
        self.channels_embed = nn.Embedding(2, dim)
        self.grid_shapes_row_embed = nn.Embedding(GRID_SIZE, dim)
        self.grid_shapes_col_embed = nn.Embedding(GRID_SIZE, dim)
        self.cls_token = nn.Embedding(1, dim)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(TransformerLayer(cfg) for _ in range(cfg.num_layers))
        self.to_mu = nn.Linear(dim, cfg.latent_dim, bias=False)
        self.to_logvar = nn.Linear(dim, cfg.latent_dim, bias=False)

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs/outputs (..., 4, 4) -> per-pair (mu, logvar), each (..., d) in float32."""
        lead = inputs.shape[:-2]
        pairs = torch.stack([inputs, outputs], dim=-1)  # (..., R, C, 2)
        pos = self.pos_row_embed.weight[:, None, None] + self.pos_col_embed.weight[None, :, None]  # (R, C, 1, D)
        cells = self.colors_embed(pairs) + pos + self.channels_embed.weight  # (..., R, C, 2, D)
        cells = cells.flatten(-4, -2)  # (..., 2*R*C, D): (0,0)in, (0,0)out, (0,1)in, ...

        full = torch.full((), GRID_SIZE - 1, dtype=torch.long, device=inputs.device)  # shape token = size - 1
        rows = self.grid_shapes_row_embed(full) + self.channels_embed.weight  # (2, D): [in, out]
        cols = self.grid_shapes_col_embed(full) + self.channels_embed.weight
        prefix = torch.cat([self.cls_token.weight, rows, cols], dim=0)  # (5, D)
        x = torch.cat([prefix.expand(*lead, -1, -1), cells], dim=-2)
        x = self.embed_dropout(x)

        x = x.reshape(-1, *x.shape[-2:])
        for layer in self.layers:
            x = layer(x)
        cls = layer_norm(x[:, 0]).reshape(*lead, -1)
        return self.to_mu(cls).float(), self.to_logvar(cls).float()
