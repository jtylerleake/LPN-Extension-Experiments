"""Differentiable execution module (PROJECT-SPEC.md section 4.3), Architecture 1.

EXECUTE(S_hat, x) = PASTE(S_hat, FIND(x)):
- FIND: the flat index of the single blue (1) pixel in x. Hardcoded lookup,
  no learning -- the explicit version of the attention the LPN decoder must
  learn implicitly.
- PASTE: a per-cell color distribution over the grid. Starts as one-hot(x),
  then the 4 cells under the tile (offset (0, 0) from the anchor) are
  replaced by the soft tile's rows. No hard decisions, so gradients flow
  from the executed grid back into S_hat (and from there into z).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lpn_interp.config import ANCHOR_COLOR, GRID_SIZE, NUM_COLORS
from lpn_interp.data.pattern2d import tile_flat_offsets


def find_anchor(x: torch.Tensor, validate: bool = True) -> torch.Tensor:
    """x (..., 4, 4) -> (...) flat index of the anchor pixel.

    `validate` checks there's exactly one blue pixel per grid (true by task
    construction). It forces a device sync, so hot paths may turn it off.
    """
    is_anchor = x.flatten(-2) == ANCHOR_COLOR
    if validate and not bool((is_anchor.sum(-1) == 1).all()):
        raise ValueError("every input grid must contain exactly one anchor (blue) pixel")
    return is_anchor.long().argmax(-1)


def paste(s_hat: torch.Tensor, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """s_hat (..., 4, 10) soft tile, x (..., 4, 4), anchor (...) flat index
    -> (..., 16, 10) per-cell color distributions of the executed grid.

    Leading dims of s_hat must match x's (expand a per-task tile over pairs
    before calling). The anchor is constrained at generation time so the tile
    always fits; no clipping is done here.
    """
    y_hat = F.one_hot(x.flatten(-2), NUM_COLORS).to(s_hat.dtype)
    cells = anchor[..., None] + tile_flat_offsets().to(anchor.device)  # (..., 4)
    index = cells[..., None].expand(*cells.shape, NUM_COLORS)
    return y_hat.scatter(-2, index, s_hat)


def execute(s_hat: torch.Tensor, x: torch.Tensor, validate: bool = True) -> torch.Tensor:
    return paste(s_hat, x, find_anchor(x, validate=validate))


def grid_of(probs: torch.Tensor) -> torch.Tensor:
    """(..., 16, 10) distributions -> (..., 4, 4) argmax grid (eval only)."""
    return probs.argmax(-1).unflatten(-1, (GRID_SIZE, GRID_SIZE))
