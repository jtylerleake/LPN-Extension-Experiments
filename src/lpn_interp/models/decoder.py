"""LPN decoder p(y | x, z) (PROJECT-SPEC.md section 2.3).

One sequence per pair: [z token, 16 input cells, output cells 0..14].
Prefix-LM attention: the z token and input cells attend to each other freely;
output cells attend to the prefix and to earlier output cells only. Output
cell j is predicted from position 16 + j (the last input cell for j = 0,
output cell j - 1 otherwise), so no position ever sees the cell it predicts.
The output grid is always 4x4, so there are no shape tokens and greedy
decoding is exactly 16 steps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lpn_interp.config import GRID_SIZE, ModelConfig
from lpn_interp.models.encoder import NUM_CELLS, GridEmbedding, transformer

PREFIX_LEN = 1 + NUM_CELLS  # z token + input cells
SEQ_LEN = PREFIX_LEN + NUM_CELLS - 1


def prefix_lm_mask(device: torch.device | str | None = None) -> torch.Tensor:
    """(SEQ_LEN, SEQ_LEN) bool mask, True = blocked (nn.Transformer convention)."""
    causal = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device).tril()
    causal[:PREFIX_LEN, :PREFIX_LEN] = True
    return ~causal


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = GridEmbedding(cfg.d_model, num_segments=3)  # 0 input, 1 output, 2 z
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.body = transformer(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.to_logits = nn.Linear(cfg.d_model, 10)
        self.register_buffer("mask", prefix_lm_mask(), persistent=False)

    def logits(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Teacher-forced. x, y (..., 4, 4); z (..., d) -> (..., 16, 10) logits for y's cells."""
        lead = x.shape[:-2]
        z_tok = self.from_latent(z.to(self.from_latent.weight.dtype))[..., None, :] + self.embed.segment.weight[2]
        out_tok = self.embed(y, 1)[..., :-1, :]
        seq = torch.cat([z_tok, self.embed(x, 0), out_tok], dim=-2).flatten(0, -3)
        hidden = self.body(seq, mask=self.mask)
        logits = self.to_logits(self.norm(hidden[:, PREFIX_LEN - 1 :]))
        return logits.unflatten(0, lead)

    def nll(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """-log p(y | x, z) summed over the 16 cells -> (...)."""
        logits = self.logits(x, y, z).float()
        return F.cross_entropy(logits.flatten(0, -2), y.flatten(), reduction="none").view(y.shape).sum((-1, -2))

    @torch.no_grad()
    def generate(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Greedy autoregressive decode. x (..., 4, 4), z (..., d) -> (..., 4, 4)."""
        y = torch.zeros_like(x).flatten(-2)
        for cell in range(NUM_CELLS):
            logits = self.logits(x, y.unflatten(-1, (GRID_SIZE, GRID_SIZE)), z)
            y[..., cell] = logits[..., cell, :].argmax(-1)
        return y.unflatten(-1, (GRID_SIZE, GRID_SIZE))
