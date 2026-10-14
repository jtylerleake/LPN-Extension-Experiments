"""Differentiable execution module (prompts/PROJECT-SPEC-1.md section 4.3), Architecture 1.

EXECUTE(S_hat, x) = PASTE(S_hat, FIND(x)):
- FIND: the flat index of the single blue (1) pixel in x. Hardcoded lookup,
  no learning -- the explicit version of the attention the LPN decoder must
  learn implicitly.
- PASTE: a per-cell color distribution over the grid. Starts as one-hot(x),
  then the 4 cells under the tile (offset (0, 0) from the anchor) are
  replaced by the soft tile's rows. No hard decisions, so gradients flow
  from the executed grid back into S_hat (and from there into z).

Architecture 2 replaces FIND with a learned soft anchor map A_hat (one
distribution over the 16 grid positions, from g(z)) and pastes in
expectation: Y_hat = sum_p A_hat[p] * PASTE(S_hat, x, p). Every position is
allowed -- an anchor in the last row/column pastes only the tile cells that
land on the grid and drops the rest (no masking, deliberately naive).
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from src.config import ANCHOR_COLOR, GRID_SIZE, NUM_COLORS, TILE_SIZE
from src.data.pattern2d import tile_flat_offsets

NUM_POSITIONS = GRID_SIZE * GRID_SIZE


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


def placements(device: torch.device | str = "cpu") -> torch.Tensor:
    """(16, 4, 16) float: [p, t, cell] = 1 if tile cell t lands on grid cell `cell`
    when the tile's top-left corner is at flat position p. Off-grid cells have no 1.
    Built once per device (it sits in the training hot path); don't modify it in place."""
    return _placements(str(torch.device(device)))


@lru_cache
def _placements(device: str) -> torch.Tensor:
    place = torch.zeros(NUM_POSITIONS, TILE_SIZE * TILE_SIZE, NUM_POSITIONS)
    for p in range(NUM_POSITIONS):
        r, c = divmod(p, GRID_SIZE)
        for t in range(TILE_SIZE * TILE_SIZE):
            rr, cc = r + t // TILE_SIZE, c + t % TILE_SIZE
            if rr < GRID_SIZE and cc < GRID_SIZE:
                place[p, t, rr * GRID_SIZE + cc] = 1.0
    return place.to(device)


def paste_soft_anchor(s_hat: torch.Tensor, x: torch.Tensor, anchor_probs: torch.Tensor) -> torch.Tensor:
    """Expected PASTE under a distribution over anchor positions (architecture 2).

    s_hat (..., 4, 10) soft tile, x (..., 4, 4), anchor_probs (..., 16) summing to 1
    -> (..., 16, 10) per-cell color distributions. A one-hot anchor_probs at a valid
    anchor gives exactly `paste`. Differentiable in both s_hat and anchor_probs.
    """
    place = placements(x.device).to(s_hat.dtype)
    anchor_probs = anchor_probs.to(s_hat.dtype)
    base = F.one_hot(x.flatten(-2), NUM_COLORS).to(s_hat.dtype)
    covered = anchor_probs @ place.sum(1)  # (..., 16): probability each cell is overwritten
    pasted = torch.einsum("...p,ptc,...tk->...ck", anchor_probs, place, s_hat)
    return (1 - covered)[..., None] * base + pasted


def grid_of(probs: torch.Tensor) -> torch.Tensor:
    """(..., 16, 10) distributions -> (..., 4, 4) argmax grid (eval only)."""
    return probs.argmax(-1).unflatten(-1, (GRID_SIZE, GRID_SIZE))
