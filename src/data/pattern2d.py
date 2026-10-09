"""Simplified Pattern-2D task generation (prompts/PROJECT-SPEC-1.md section 3).

A task is one 2x2 color tile T (the program, colors 1-9) plus `num_pairs`
(input, output) pairs on a 4x4 grid:
- input: all black except one blue (1) anchor pixel at (r, c), r, c in {0,1,2}
- output: the input with T pasted with its top-left corner on the anchor
  (overwriting the anchor pixel).

Anchors within a task are distinct, so no pair (in particular an eval query
pair) duplicates another -- with only 9 anchor positions, sampling them
independently would make a held-out pair a copy of a context pair in a large
fraction of tasks (see CLAUDE-CODING-SKILL-10012026.md, "clean rounds").

Tiles are cells in row-major order: (0,0), (0,1), (1,0), (1,1). Tile ids
index `all_tiles()`. A fixed, seeded fraction of tile ids is held out of
training entirely (`split_tile_ids`) to test generalization to unseen
programs.

Everything is plain torch and fully vectorized, so training batches are
generated on the GPU each step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config import ANCHOR_COLOR, GRID_SIZE, TILE_SIZE

TILE_COLORS = tuple(range(1, 10))  # no black tile cells
NUM_TILE_CELLS = TILE_SIZE * TILE_SIZE
NUM_TILES = len(TILE_COLORS) ** NUM_TILE_CELLS
NUM_ANCHORS = (GRID_SIZE - TILE_SIZE + 1) ** 2


def anchor_positions() -> torch.Tensor:
    """(9, 2) valid (row, col) anchors: the tile always fits, no clipping."""
    span = torch.arange(GRID_SIZE - TILE_SIZE + 1)
    rows, cols = torch.meshgrid(span, span, indexing="ij")
    return torch.stack([rows.flatten(), cols.flatten()], dim=-1)


def tile_flat_offsets() -> torch.Tensor:
    """(4,) flat-grid offsets of each tile cell from the anchor's flat index."""
    dr, dc = torch.meshgrid(torch.arange(TILE_SIZE), torch.arange(TILE_SIZE), indexing="ij")
    return (dr * GRID_SIZE + dc).flatten()


def all_tiles() -> torch.Tensor:
    """(6561, 4) every tile, tile id = base-9 number of (color - 1) per cell."""
    ids = torch.arange(NUM_TILES)
    digits = [(ids // len(TILE_COLORS) ** p) % len(TILE_COLORS) for p in reversed(range(NUM_TILE_CELLS))]
    return torch.stack(digits, dim=-1) + TILE_COLORS[0]


def tile_ids(tiles: torch.Tensor) -> torch.Tensor:
    """Inverse of all_tiles(): (..., 4) tiles -> (...) ids."""
    weights = len(TILE_COLORS) ** torch.arange(NUM_TILE_CELLS - 1, -1, -1, device=tiles.device)
    return ((tiles - TILE_COLORS[0]) * weights).sum(-1)


def split_tile_ids(holdout_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(train_ids, heldout_ids), each sorted. Deterministic in (fraction, seed)."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(NUM_TILES, generator=gen)
    num_heldout = round(holdout_fraction * NUM_TILES)
    return perm[num_heldout:].sort().values, perm[:num_heldout].sort().values


@dataclass
class TaskBatch:
    inputs: torch.Tensor  # (B, P, 4, 4) long
    outputs: torch.Tensor  # (B, P, 4, 4) long
    tiles: torch.Tensor  # (B, 4) long, colors 1-9
    anchors: torch.Tensor  # (B, P, 2) long, (row, col)

    def to(self, device: torch.device | str) -> TaskBatch:
        return TaskBatch(*(t.to(device) for t in (self.inputs, self.outputs, self.tiles, self.anchors)))

    def split_query(self) -> tuple[TaskBatch, TaskBatch]:
        """(context: pairs[:-1], query: pairs[-1:]) -- eval tasks carry one extra pair."""
        ctx = TaskBatch(self.inputs[:, :-1], self.outputs[:, :-1], self.tiles, self.anchors[:, :-1])
        qry = TaskBatch(self.inputs[:, -1:], self.outputs[:, -1:], self.tiles, self.anchors[:, -1:])
        return ctx, qry


def render(tiles: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """tiles (B, 4), anchors (B, P, 2) -> (inputs, outputs), each (B, P, 4, 4)."""
    batch, num_pairs = anchors.shape[:2]
    device = tiles.device
    flat_anchor = anchors[..., 0] * GRID_SIZE + anchors[..., 1]  # (B, P)
    inputs = torch.zeros(batch, num_pairs, GRID_SIZE * GRID_SIZE, dtype=torch.long, device=device)
    inputs.scatter_(-1, flat_anchor[..., None], ANCHOR_COLOR)
    cells = flat_anchor[..., None] + tile_flat_offsets().to(device)  # (B, P, 4)
    outputs = inputs.scatter(-1, cells, tiles[:, None, :].expand(-1, num_pairs, -1))
    shape = (batch, num_pairs, GRID_SIZE, GRID_SIZE)
    return inputs.view(shape), outputs.view(shape)


def sample_tasks(
    tile_pool: torch.Tensor, batch_size: int, num_pairs: int, generator: torch.Generator
) -> TaskBatch:
    """`batch_size` tasks with tiles drawn uniformly from `tile_pool` (tile ids)
    and `num_pairs` distinct anchors each. Runs on `generator`'s device (the
    pool must be on the same device)."""
    device = generator.device
    if num_pairs > NUM_ANCHORS:
        raise ValueError(f"num_pairs={num_pairs} exceeds the {NUM_ANCHORS} distinct anchor positions")
    pick = torch.randint(len(tile_pool), (batch_size,), generator=generator, device=device)
    tiles = all_tiles().to(device)[tile_pool[pick]]
    anchor_order = torch.rand(batch_size, NUM_ANCHORS, generator=generator, device=device).argsort(-1)
    anchors = anchor_positions().to(device)[anchor_order[:, :num_pairs]]
    inputs, outputs = render(tiles, anchors)
    return TaskBatch(inputs, outputs, tiles, anchors)


def step_generator(seed: int, step: int, device: torch.device | str) -> torch.Generator:
    """Generator for one training step, derived from (seed, step) alone so a
    resumed run sees exactly the batches an uninterrupted one would."""
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + step)


def make_eval_tasks(tile_pool: torch.Tensor, num_tasks: int, num_pairs: int, seed: int) -> TaskBatch:
    """Fixed, seeded eval set on CPU: `num_pairs` context pairs + 1 query pair per task."""
    return sample_tasks(tile_pool, num_tasks, num_pairs + 1, torch.Generator().manual_seed(seed))
