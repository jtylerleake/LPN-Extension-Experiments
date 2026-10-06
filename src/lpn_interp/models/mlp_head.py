"""MLP head h(z) -> soft tile (PROJECT-SPEC.md section 4.2).

Takes only z -- never x. z doesn't carry the anchor position (mean pooling
cancels it), so the head can only succeed by reading the tile colors out of z.
Returns logits; the soft tile is a softmax per cell (`soft_tile`), and the
hard tile (argmax) is taken at evaluation time only.
"""
from __future__ import annotations

import torch
from torch import nn

from lpn_interp.config import NUM_COLORS, TILE_SIZE, ModelConfig

NUM_TILE_CELLS = TILE_SIZE * TILE_SIZE


class MLPHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, NUM_TILE_CELLS * NUM_COLORS),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., d) -> (..., 4, 10) per-cell color logits (float32, even under autocast)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.net(z.float()).unflatten(-1, (NUM_TILE_CELLS, NUM_COLORS))


def soft_tile(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(-1)


def hard_tile(logits: torch.Tensor) -> torch.Tensor:
    """(..., 4, 10) -> (..., 4) argmax colors. Evaluation only."""
    return logits.argmax(-1)
