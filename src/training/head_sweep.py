"""Head sweep on the frozen pretrained LPN (stage 1): which MLP head reads the tile out of z best?

Why a separate trainer: in stage 1 the encoder is frozen and encodes every pair independently,
so a pair's (mu, logvar) depends only on its tile and anchor -- 6,561 tiles x 9 anchors =
59,049 pairs. `EncodingCache` encodes them all once; a training step is then a lookup plus the
head. The objective is stage 1's (training_losses, probe mode): each pair's z is the
leave-one-out mean of the other pairs' sampled latents, and the loss is
lambda_exec * L_exec (+ lambda_commit * L_commit, off by default).

Tile splits: training tiles as in every run (split_tile_ids(0.1, 0)). The 656 held-out tiles are
split in half: validation ranks configurations, test is scored only in round C.

Every run: ReLU, 25k steps, validation every 2.5k. Configurations are ranked by validation
tile_cell_acc (cells right out of 4, averaged), then tile_full_acc.

Rounds (each skips runs already in results.jsonl, so rerunning a round resumes it):
  A   head size x lr: depth x width x lr, weight decay 1e-3, no dropout, residual blocks
      for depth >= 4.
  B   regularization, one setting at a time, for the 3 best round-A configurations (a "base"):
      weight decay, dropout, residual flipped; the other settings stay at the base's.
  B2  per base, the best value of each round-B setting combined into one run.
  C   the 3 best configurations of A + B + B2, 3 seeds each, scored on validation and test.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import GRID_SIZE, LossConfig, ModelConfig
from src.data.pattern2d import NUM_ANCHORS, all_tiles, anchor_positions, render, split_tile_ids
from src.models.lpn import LPN, loo_mean
from src.models.mlp_head import MLPHead
from src.training.losses import commit_entropy
from src.training.train import lr_factor
from src.utils.logging_utils import get_logger
from src.utils.resume import append_jsonl_row, completed_fingerprints, fingerprint, read_jsonl_rows

log = get_logger(__name__)

SWEEP_VERSION = 2  # bump when a code change alters what a given spec trains
NUM_PAIRS, BATCH_SIZE, WARMUP_STEPS, GRAD_CLIP = 4, 256, 500, 1.0
EVAL_EVERY = 2_500
SEEN_EVAL_TASKS, HELDOUT_TASKS_PER_TILE, EVAL_SEED = 2048, 8, 1234
VAL_TEST_SEED = 0

DEPTHS, WIDTHS, LRS, RESIDUAL_FROM_DEPTH = (2, 3, 4, 6), (128, 256, 512, 1024), (1e-3, 3e-3, 1e-2), 4
STEPS, BASE_WEIGHT_DECAY, BASE_DROPOUT = 25_000, 1e-3, 0.0
WEIGHT_DECAYS, DROPOUTS = (0.0, 1e-4, 1e-3, 1e-2), (0.0, 0.1, 0.3)
REG_KEYS = ("weight_decay", "dropout", "residual")  # round B's settings
BASE_KEYS = ("depth", "width", "lr")  # what identifies a round-B base
SEEDS, TOP_K = (0, 1, 2), 3
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
    return ModelConfig(head_hidden=spec["width"], head_depth=spec["depth"], head_activation=spec["activation"],
                       head_residual=spec["residual"], head_dropout=spec["dropout"])


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
    opt = torch.optim.AdamW(head.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, WARMUP_STEPS, steps))
    gen = torch.Generator(device=dev).manual_seed(spec["seed"])
    curve, start = [], time.perf_counter()
    bar = tqdm(range(1, steps + 1), desc=f"d{spec['depth']} w{spec['width']} lr{spec['lr']:g}", unit="step",
               leave=False, position=1, mininterval=2.0, dynamic_ncols=True)
    for step in bar:
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
            bar.set_postfix(loss=f"{loss.item():.4f}", val_cell=f"{val['tile_cell_acc']:.3f}",
                            val_full=f"{val['tile_full_acc']:.3f}")
    bar.close()
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


def make_spec(depth: int, width: int, lr: float, seed: int = 0, weight_decay: float = BASE_WEIGHT_DECAY,
              dropout: float = BASE_DROPOUT, residual: bool | None = None) -> dict:
    """Residual defaults to depth >= RESIDUAL_FROM_DEPTH."""
    return {"depth": depth, "width": width, "residual": depth >= RESIDUAL_FROM_DEPTH if residual is None else residual,
            "activation": "relu", "lr": lr, "steps": STEPS, "weight_decay": weight_decay, "dropout": dropout,
            "seed": seed}


def base_spec(spec: dict) -> dict:
    """The round-A spec a run varies (its base): same depth/width/lr, base regularization."""
    return make_spec(*(spec[k] for k in BASE_KEYS))


def rank_key(row: dict) -> tuple[float, float]:
    """Validation cell accuracy first (partial credit), full-tile accuracy breaks ties."""
    return row["final"]["val/tile_cell_acc"], row["final"]["val/tile_full_acc"]


def variations(base: dict) -> list[dict]:
    """Round B for one base: each REG_KEYS setting varied alone (the base itself included)."""
    return ([{**base, "weight_decay": wd} for wd in WEIGHT_DECAYS] + [{**base, "dropout": p} for p in DROPOUTS]
            + [{**base, "residual": r} for r in (False, True)])


def best_combination(base: dict, rows: list[dict]) -> dict:
    """Round B2: for each setting, the value whose one-at-a-time run (or the base) ranked best."""
    spec = dict(base)
    for key in REG_KEYS:
        others = [k for k in REG_KEYS if k != key]
        cands = [r for r in rows if all(r["spec"].get(k) == base[k] for k in (*BASE_KEYS, *others))]
        if cands:
            spec[key] = max(cands, key=rank_key)["spec"][key]
    return spec


def round_specs(name: str, rows: list[dict], loss_cfg: LossConfig,
                max_steps: int | None = None) -> tuple[list[dict], bool]:
    """The specs a round runs (deduplicated, steps capped at `max_steps` for smoke tests),
    and whether they're scored on test."""
    specs, with_test = _round_specs(name, rows, loss_cfg, max_steps)
    if max_steps:
        specs = [{**sp, "steps": min(sp["steps"], max_steps)} for sp in specs]
    unique = {spec_fingerprint(sp, with_test, loss_cfg): sp for sp in specs}
    return list(unique.values()), with_test


def _require(name: str, prior: str, rows: list[dict], loss_cfg: LossConfig, max_steps: int | None) -> list[dict]:
    """The finished rows of round `prior`; raises if any of its runs is missing."""
    specs = round_specs(prior, rows, loss_cfg, max_steps)[0]
    fps = {spec_fingerprint(sp, False, loss_cfg) for sp in specs}
    trained = [r for r in rows if not r["test"]]
    missing = fps - completed_fingerprints(trained)
    if missing:
        raise RuntimeError(f"round {prior} isn't finished ({len(missing)} of {len(specs)} runs missing): "
                           f"run it before round {name}")
    return [r for r in trained if r["fingerprint"] in fps]


def _bases(rows: list[dict], loss_cfg: LossConfig, max_steps: int | None) -> list[dict]:
    """The TOP_K best round-A configurations (with base regularization)."""
    a_rows = sorted(_require("B", "A", rows, loss_cfg, max_steps), key=rank_key, reverse=True)
    return [base_spec(r["spec"]) for r in a_rows[:TOP_K]]


def _round_specs(name, rows, loss_cfg, max_steps):
    if name == "A":
        return [make_spec(d, w, lr) for d in DEPTHS for w in WIDTHS for lr in LRS], False
    if name == "B":
        return [v for b in _bases(rows, loss_cfg, max_steps) for v in variations(b)], False
    if name == "B2":
        b_rows = _require("B2", "B", rows, loss_cfg, max_steps)
        return [best_combination(b, b_rows) for b in _bases(rows, loss_cfg, max_steps)], False
    if name == "C":
        _require("C", "B2", rows, loss_cfg, max_steps)
        best, seen = [], set()
        for r in sorted((r for r in rows if not r["test"]), key=rank_key, reverse=True):
            config = {**r["spec"], "seed": 0}
            if spec_fingerprint(config, False, loss_cfg) not in seen:
                seen.add(spec_fingerprint(config, False, loss_cfg))
                best.append(r["spec"])
            if len(best) == TOP_K:
                break
        return [{**sp, "seed": seed} for sp in best for seed in SEEDS], True
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
        best = max((r["final"]["val/tile_cell_acc"] for r in rows if r["round"] == name), default=0.0)
        # one bar over the round's runs (finished ones count from the start), one over the current run's steps
        with logging_redirect_tqdm(loggers=[log]), tqdm(total=len(specs), initial=len(specs) - len(todo),
                                                        desc=f"round {name}", unit="run", position=0,
                                                        dynamic_ncols=True) as runs:
            for spec in todo:
                log.info(f"start {spec}")
                row = run(spec, cache, splits, loss_cfg, with_test, eval_every)
                row = {"fingerprint": spec_fingerprint(spec, with_test, loss_cfg), "round": name, **row}
                append_jsonl_row(results, row)
                log.info(f"  done in {row['seconds']}s: " + " ".join(f"{k}={v}" for k, v in row["final"].items()))
                best = max(best, row["final"]["val/tile_cell_acc"])
                runs.set_postfix(best_val_cell=f"{best:.3f}", last_run=f"{row['seconds']:.0f}s")
                runs.update()
    rows = read_jsonl_rows(results)
    fps = {spec_fingerprint(s, with_test, loss_cfg) for s in specs}
    return [r for r in rows if r["fingerprint"] in fps]


def summary_table(rows: list[dict]) -> str:
    """Rows ranked by validation cell accuracy (rank_key), one line each."""
    cols = ("depth", "width", "residual", "lr", "weight_decay", "dropout", "seed")
    metrics = ("seen/tile_cell_acc", "val/tile_cell_acc", "val/tile_full_acc", "test/tile_cell_acc", "test/tile_full_acc")
    lines = ["round " + " ".join(f"{c:>10}" for c in cols) + " " + " ".join(f"{m:>19}" for m in metrics)]
    for r in sorted(rows, key=rank_key, reverse=True):
        vals = " ".join(f"{str(r['spec'].get(c)):>10}" for c in cols)
        ms = " ".join(f"{r['final'].get(m, float('nan')):>19.4f}" for m in metrics)
        lines.append(f"{r['round']:>5} {vals} {ms}")
    return "\n".join(lines)
