"""Head sweep on the frozen pretrained LPN (stage 1): which MLP head reads the tile out of z best?

Why a separate trainer: in stage 1 the encoder is frozen and encodes every pair independently,
so a pair's (mu, logvar) depends only on its tile and anchor -- 6,561 tiles x 9 anchors =
59,049 pairs. `EncodingCache` encodes them all once; a training step is then a lookup plus the
head. The objective is stage 1's (training_losses, probe mode): each pair's z is the
leave-one-out mean of the other pairs' sampled latents, and the loss is
lambda_exec * L_exec (+ lambda_commit * L_commit, off by default).

Tile splits: training tiles as in every run (split_tile_ids(0.1, 0)). The 656 held-out tiles are
split in half: validation ranks configurations, test is scored only in round C.

Rounds (each skips runs already in results.jsonl, so rerunning a round resumes it):
  A  head size: depth x width (ReLU, lr 1e-3, 50k steps); depth >= 4 uses residual blocks.
  B  training: lr x steps x activation, for the 3 best round-A heads.
  C  the 3 best configurations of A + B, 3 seeds each, scored on validation and test.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import GRID_SIZE, LossConfig, ModelConfig
from src.data.pattern2d import NUM_ANCHORS, all_tiles, anchor_positions, render, split_tile_ids
from src.models.lpn import LPN, loo_mean
from src.models.mlp_head import MLPHead
from src.training.losses import commit_entropy
from src.training.train import lr_factor
from src.utils.logging_utils import get_logger
from src.utils.resume import append_jsonl_row, completed_fingerprints, fingerprint, read_jsonl_rows

log = get_logger(__name__)

SWEEP_VERSION = 1  # bump when a code change alters what a given spec trains
NUM_PAIRS, BATCH_SIZE, WARMUP_STEPS, GRAD_CLIP = 4, 256, 500, 1.0
EVAL_EVERY = 10_000
SEEN_EVAL_TASKS, HELDOUT_TASKS_PER_TILE, EVAL_SEED = 2048, 8, 1234
VAL_TEST_SEED = 0

DEPTHS, WIDTHS, RESIDUAL_FROM_DEPTH = (2, 3, 4, 6), (128, 256, 512, 1024), 4
LRS, STEPS, ACTIVATIONS, SEEDS, TOP_K = (3e-4, 1e-3, 3e-3), (50_000, 150_000), ("relu", "silu"), (0, 1, 2), 3
ROUND_A_TRAINING = {"activation": "relu", "lr": 1e-3, "steps": 50_000, "seed": 0}
HEAD_KEYS = ("depth", "width", "residual")
STAGE1_LOSS = LossConfig()  # stage 1's loss weights (lambda_exec 1.0, lambda_commit 0)


def split_heldout(heldout_ids: torch.Tensor, seed: int = VAL_TEST_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Held-out tile ids -> (validation, test), half each, deterministic in `seed`."""
    perm = torch.randperm(len(heldout_ids), generator=torch.Generator().manual_seed(seed))
    half = len(heldout_ids) // 2
    return heldout_ids[perm[:half]].sort().values, heldout_ids[perm[half:]].sort().values


def tile_exec_nll(logits: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
    """losses.exec_nll for architecture 1, from the tile alone. The executed grid's cells off
    the tile are exact copies of the target (zero loss), so the mean over the 16 cells is the
    tile cells' cross-entropy summed / 16. logits (..., 4, 10), tiles (..., 4) -> (...)."""
    ce = F.cross_entropy(logits.flatten(0, -2), tiles.flatten(), reduction="none")
    return ce.view(tiles.shape).sum(-1) / GRID_SIZE**2


class EncodingCache:
    """(mu, logvar) of the frozen encoder for every (tile, anchor) pair; pair id = tile id * 9 + anchor index."""

    def __init__(self, model: LPN, device: torch.device, chunk: int = 8192):
        tiles = all_tiles()
        num_tiles = len(tiles)
        x, y = render(tiles.repeat_interleave(NUM_ANCHORS, 0), anchor_positions().repeat(num_tiles, 1)[:, None])
        model = model.to(device).eval()
        mus, logvars = [], []
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                mu, logvar = model.encoder(x[start:start + chunk].to(device), y[start:start + chunk].to(device))
                mus.append(mu[:, 0])
                logvars.append(logvar[:, 0])
        self.mu, self.logvar = torch.cat(mus), torch.cat(logvars)
        self.tiles = tiles.to(device)
        self.device = device

    def tasks(self, pool: torch.Tensor, num_tasks: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """`num_tasks` tasks: tile ids drawn from `pool` (on this device), 4 distinct anchors each.
        -> (tile ids (B,), pair ids (B, 4))."""
        tile = pool[torch.randint(len(pool), (num_tasks,), generator=gen, device=self.device)]
        anchor = torch.rand(num_tasks, NUM_ANCHORS, generator=gen, device=self.device).argsort(-1)[:, :NUM_PAIRS]
        return tile, tile[:, None] * NUM_ANCHORS + anchor

    def heldout_tasks(self, pool: torch.Tensor, per_tile: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Every tile in `pool`, `per_tile` times, each with fresh distinct anchors."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        tile = pool.repeat(per_tile)
        anchor = torch.rand(len(tile), NUM_ANCHORS, generator=gen, device=self.device).argsort(-1)[:, :NUM_PAIRS]
        return tile, tile[:, None] * NUM_ANCHORS + anchor


class Splits:
    """Tile pools and fixed evaluation task sets (seen / val / test), all on the cache's device."""

    def __init__(self, cache: EncodingCache):
        train_ids, heldout = split_tile_ids(0.1, 0)
        val_ids, test_ids = split_heldout(heldout)
        dev = cache.device
        self.train = train_ids.to(dev)
        gen = torch.Generator(device=dev).manual_seed(EVAL_SEED)
        self.eval = {
            "seen": cache.tasks(self.train, SEEN_EVAL_TASKS, gen),
            "val": cache.heldout_tasks(val_ids.to(dev), HELDOUT_TASKS_PER_TILE, EVAL_SEED + 1),
            "test": cache.heldout_tasks(test_ids.to(dev), HELDOUT_TASKS_PER_TILE, EVAL_SEED + 2),
        }


def head_config(spec: dict) -> ModelConfig:
    return ModelConfig(head_hidden=spec["width"], head_depth=spec["depth"],
                       head_activation=spec["activation"], head_residual=spec["residual"])


@torch.no_grad()
def evaluate(head: MLPHead, cache: EncodingCache, tasks: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    """Tile metrics at z0 = mean of the 4 context pairs' mu (what stage-1 evaluation feeds the head)."""
    tile, pairs = tasks
    hit = head(cache.mu[pairs].mean(1)).argmax(-1) == cache.tiles[tile]
    return {"tile_cell_acc": hit.float().mean().item(), "tile_full_acc": hit.all(-1).float().mean().item()}


def run(spec: dict, cache: EncodingCache, splits: Splits, loss_cfg: LossConfig, with_test: bool,
        eval_every: int = EVAL_EVERY) -> dict:
    """Train one head from scratch on the cache; return its result row (without fingerprint/round)."""
    dev, steps = cache.device, spec["steps"]
    torch.manual_seed(spec["seed"])
    head = MLPHead(head_config(spec)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=spec["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, WARMUP_STEPS, steps))
    gen = torch.Generator(device=dev).manual_seed(spec["seed"])
    curve, start = [], time.perf_counter()
    for step in range(1, steps + 1):
        tile, pairs = cache.tasks(splits.train, BATCH_SIZE, gen)
        mu, logvar = cache.mu[pairs], cache.logvar[pairs]
        z = loo_mean(mu + (0.5 * logvar).exp() * torch.randn(mu.shape, generator=gen, device=dev))
        logits = head(z)
        target = cache.tiles[tile][:, None].expand(-1, NUM_PAIRS, -1)
        loss = loss_cfg.lambda_exec * tile_exec_nll(logits, target).mean()
        if loss_cfg.lambda_commit:
            loss = loss + loss_cfg.lambda_commit * commit_entropy(logits).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            head.eval()
            val = evaluate(head, cache, splits.eval["val"])
            head.train()
            curve.append({"step": step, "loss": round(loss.item(), 5), **{f"val/{k}": round(v, 5) for k, v in val.items()}})
            log.info(f"  step {step}/{steps}: loss={loss.item():.4f} val_full={val['tile_full_acc']:.4f} "
                     f"val_cell={val['tile_cell_acc']:.4f}")
    seconds = time.perf_counter() - start
    head.eval()
    final = {}
    for split in ("seen", "val", "test") if with_test else ("seen", "val"):
        final |= {f"{split}/{k}": round(v, 5) for k, v in evaluate(head, cache, splits.eval[split]).items()}
    return {"spec": spec, "test": with_test, "final": final, "curve": curve,
            "seconds": round(seconds, 1), "steps_per_sec": round(steps / seconds, 1),
            "params": sum(p.numel() for p in head.parameters())}


def spec_fingerprint(spec: dict, with_test: bool, loss_cfg: LossConfig) -> str:
    return fingerprint({**spec, "test": with_test, "loss": loss_cfg.model_dump(), "version": SWEEP_VERSION,
                        "batch": BATCH_SIZE, "warmup": WARMUP_STEPS, "pairs": NUM_PAIRS})


def make_spec(depth: int, width: int, activation: str, lr: float, steps: int, seed: int) -> dict:
    return {"depth": depth, "width": width, "residual": depth >= RESIDUAL_FROM_DEPTH,
            "activation": activation, "lr": lr, "steps": steps, "seed": seed}


def rank_key(row: dict) -> tuple[float, float]:
    return row["final"]["val/tile_full_acc"], row["final"]["val/tile_cell_acc"]


def round_specs(name: str, rows: list[dict], loss_cfg: LossConfig,
                max_steps: int | None = None) -> tuple[list[dict], bool]:
    """The specs a round runs (deduplicated, steps capped at `max_steps` for smoke tests),
    and whether they're scored on test."""
    specs, with_test = _round_specs(name, rows, loss_cfg, max_steps)
    if max_steps:
        specs = [{**sp, "steps": min(sp["steps"], max_steps)} for sp in specs]
    unique = {spec_fingerprint(sp, with_test, loss_cfg): sp for sp in specs}
    return list(unique.values()), with_test


def _round_specs(name, rows, loss_cfg, max_steps):
    if name == "A":
        return [make_spec(d, w, **ROUND_A_TRAINING) for d in DEPTHS for w in WIDTHS], False
    trained = [r for r in rows if not r["test"]]
    if name == "B":
        a_specs = round_specs("A", rows, loss_cfg, max_steps)[0]
        done = {spec_fingerprint(s, False, loss_cfg) for s in a_specs}
        missing = done - completed_fingerprints(trained)
        if missing:
            raise RuntimeError(f"round A isn't finished ({len(missing)} of {len(a_specs)} runs missing): run it first")
        a_rows = sorted((r for r in trained if r["fingerprint"] in done), key=rank_key, reverse=True)
        heads = [{k: r["spec"][k] for k in HEAD_KEYS} for r in a_rows[:TOP_K]]
        return [make_spec(h["depth"], h["width"], act, lr, steps, 0)
                for h in heads for lr in LRS for steps in STEPS for act in ACTIVATIONS], False
    if name == "C":
        if not any(r["round"] == "B" for r in trained):
            raise RuntimeError("round B hasn't run yet: run it first")
        best = sorted(trained, key=rank_key, reverse=True)[:TOP_K]
        return [{**r["spec"], "seed": seed} for r in best for seed in SEEDS], True
    raise ValueError(f"unknown round {name!r}")


def run_round(name: str, results: Path, cache_factory, loss_cfg: LossConfig = STAGE1_LOSS,
              max_steps: int | None = None, eval_every: int = EVAL_EVERY) -> list[dict]:
    """Run every not-yet-finished spec of round `name`, appending one row per run to `results`.
    `cache_factory()` builds (cache, splits); it's only called if there's work to do.
    `max_steps` caps every run's length (smoke tests)."""
    rows = read_jsonl_rows(results)
    specs, with_test = round_specs(name, rows, loss_cfg, max_steps)
    done = completed_fingerprints(rows)
    todo = [s for s in specs if spec_fingerprint(s, with_test, loss_cfg) not in done]
    log.info(f"round {name}: {len(specs)} runs, {len(specs) - len(todo)} already done, {len(todo)} to go")
    if todo:
        cache, splits = cache_factory()
        for i, spec in enumerate(todo, 1):
            log.info(f"[{i}/{len(todo)}] {spec}")
            row = run(spec, cache, splits, loss_cfg, with_test, eval_every)
            row = {"fingerprint": spec_fingerprint(spec, with_test, loss_cfg), "round": name, **row}
            append_jsonl_row(results, row)
            log.info(f"  done in {row['seconds']}s: " + " ".join(f"{k}={v}" for k, v in row["final"].items()))
    rows = read_jsonl_rows(results)
    fps = {spec_fingerprint(s, with_test, loss_cfg) for s in specs}
    return [r for r in rows if r["fingerprint"] in fps]


def summary_table(rows: list[dict]) -> str:
    """Rows ranked by validation full-tile accuracy, one line each."""
    cols = ("depth", "width", "residual", "activation", "lr", "steps", "seed")
    metrics = ("seen/tile_full_acc", "val/tile_full_acc", "val/tile_cell_acc", "test/tile_full_acc", "test/tile_cell_acc")
    lines = ["round " + " ".join(f"{c:>10}" for c in cols) + " " + " ".join(f"{m:>19}" for m in metrics)]
    for r in sorted(rows, key=rank_key, reverse=True):
        vals = " ".join(f"{str(r['spec'][c]):>10}" for c in cols)
        ms = " ".join(f"{r['final'].get(m, float('nan')):>19.4f}" for m in metrics)
        lines.append(f"{r['round']:>5} {vals} {ms}")
    return "\n".join(lines)
