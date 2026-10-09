"""MLP heads on z (prompts/PROJECT-SPEC-1.md section 4.2).

h(z) -> soft tile (both architectures). Takes only z -- never x. z doesn't
carry the anchor position (mean pooling cancels it), so the head can only
succeed by reading the tile colors out of z. Returns logits; the soft tile is
a softmax per cell (`soft_tile`), and the hard tile (argmax) is taken at
evaluation time only.

g(z) -> soft anchor map (architecture 2). Same shape of MLP, but one
16-way categorical over the 4x4 grid positions, used in place of the
hardcoded FIND. It also sees only z, so it has no way to know each pair's
anchor; it exists to show what happens when the anchor must come from z.
"""
from __future__ import annotations

import torch
from torch import nn

from src.config import GRID_SIZE, NUM_COLORS, TILE_SIZE, ModelConfig

NUM_TILE_CELLS = TILE_SIZE * TILE_SIZE
NUM_POSITIONS = GRID_SIZE * GRID_SIZE


ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU}


class ResidualBlock(nn.Module):
    """x + act(Linear(LayerNorm(x))): keeps deep heads trainable."""

    def __init__(self, width: int, act: type[nn.Module]):
        super().__init__()
        self.norm, self.linear, self.act = nn.LayerNorm(width), nn.Linear(width, width), act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(self.norm(x)))


def _mlp(cfg: ModelConfig, out_dim: int) -> nn.Sequential:
    """`head_depth` hidden layers of width `head_hidden`. The defaults (2 plain ReLU layers)
    give the original head, parameter names included, so old checkpoints still load."""
    width, act = cfg.head_hidden, ACTIVATIONS[cfg.head_activation]
    layers: list[nn.Module] = [nn.Linear(cfg.latent_dim, width), act()]
    for _ in range(cfg.head_depth - 1):
        layers += [ResidualBlock(width, act)] if cfg.head_residual else [nn.Linear(width, width), act()]
    if cfg.head_residual:
        layers.append(nn.LayerNorm(width))
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class MLPHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(cfg, NUM_TILE_CELLS * NUM_COLORS)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., d) -> (..., 4, 10) per-cell color logits (float32, even under autocast)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.net(z.float()).unflatten(-1, (NUM_TILE_CELLS, NUM_COLORS))


class AnchorHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(cfg, NUM_POSITIONS)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., d) -> (..., 16) logits over flat grid positions (float32, even under autocast)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.net(z.float())


def soft_tile(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(-1)


def hard_tile(logits: torch.Tensor) -> torch.Tensor:
    """(..., 4, 10) -> (..., 4) argmax colors. Evaluation only."""
    return logits.argmax(-1)


def soft_anchor(logits: torch.Tensor) -> torch.Tensor:
    """(..., 16) -> (..., 16) one distribution over all grid positions."""
    return logits.softmax(-1)


def hard_anchor(logits: torch.Tensor) -> torch.Tensor:
    """(..., 16) -> (...) argmax flat position. Evaluation only."""
    return logits.argmax(-1)
