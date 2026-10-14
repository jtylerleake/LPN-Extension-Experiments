"""Evaluation metrics (prompts/PROJECT-SPEC-1.md section 8).

`evaluate_z0` scores a model with no test-time search (z = z0, the pooled
context mean). It's cheap, so training logs it periodically on fixed seen-
and unseen-tile eval sets. The latent-optimization evaluation (z*) builds on
the same metrics.

Architecture 2 executes at the anchor head's hard anchor p_hat instead of
FIND, and adds anchor metrics: anchor_acc (p_hat is the query pair's anchor)
and anchor_valid (p_hat is one of the 9 positions where a 2x2 tile fits).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config import GRID_SIZE, TILE_SIZE
from src.data.pattern2d import TaskBatch
from src.execution.executor import NUM_POSITIONS, execute, grid_of, paste_soft_anchor
from src.models.lpn import LPN
from src.models.mlp_head import hard_anchor, hard_tile, soft_tile


@torch.no_grad()
def score_latent(model: LPN, ctx: TaskBatch, qry: TaskBatch, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-task metrics for latents z (B, d) on the query pair. Returns (B,) tensors."""
    logits, anchor_logits = model.program_logits(z)
    tile_hat = hard_tile(logits)
    x, y = qry.inputs[:, 0], qry.outputs[:, 0]
    y_dec = model.decoder.generate(x, z)
    anchor_metrics = {}
    if anchor_logits is None:
        y_exec = grid_of(execute(soft_tile(logits), x, validate=False))
    else:
        p_hat = hard_anchor(anchor_logits)
        y_exec = grid_of(paste_soft_anchor(soft_tile(logits), x, F.one_hot(p_hat, NUM_POSITIONS)))
        r, c = p_hat // GRID_SIZE, p_hat % GRID_SIZE
        last = GRID_SIZE - TILE_SIZE
        anchor_metrics = {
            "anchor_acc": (p_hat == qry.anchors[:, 0, 0] * GRID_SIZE + qry.anchors[:, 0, 1]).float(),
            "anchor_valid": ((r <= last) & (c <= last)).float(),
        }
    return {
        "tile_cell_acc": (tile_hat == ctx.tiles).float().mean(-1),
        "tile_full_acc": (tile_hat == ctx.tiles).all(-1).float(),
        "recon_pixel_acc": (y_dec == y).float().flatten(1).mean(-1),
        "recon_grid_acc": (y_dec == y).flatten(1).all(-1).float(),
        "exec_grid_acc": (y_exec == y).flatten(1).all(-1).float(),
        "dec_exec_agree": (y_dec == y_exec).flatten(1).all(-1).float(),
        **anchor_metrics,
    }


@torch.no_grad()
def evaluate_z0(model: LPN, tasks: TaskBatch, batch_size: int, autocast_dtype=None) -> dict[str, float]:
    """Mean metrics at z0 over `tasks` (which carry one extra query pair each)."""
    device = next(model.parameters()).device
    totals: dict[str, float] = {}
    for start in range(0, len(tasks.tiles), batch_size):
        chunk = TaskBatch(*(t[start : start + batch_size] for t in
                            (tasks.inputs, tasks.outputs, tasks.tiles, tasks.anchors))).to(device)
        ctx, qry = chunk.split_query()
        with torch.autocast(device.type, dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None):
            z0 = model.context_latent(ctx.inputs, ctx.outputs)
            metrics = score_latent(model, ctx, qry, z0)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v.sum().item()
    return {k: v / len(tasks.tiles) for k, v in totals.items()}
