"""Training loop that survives Colab disconnects (prompts/PROJECT-SPEC-1.md section 7).

A run lives in one directory (normally on Google Drive):
    config.json     resolved config + its fingerprint
    checkpoint.pt   latest full state: model, optimizer, LR schedule, AMP scaler, RNG, step
    metrics.jsonl   one row per log/eval event (kind = train | eval | done)
    final.pt        written once at the end; its presence means "complete"

Running the same command again resumes:
- final.pt exists -> nothing to do, return immediately.
- checkpoint.pt exists -> load it and continue from its step. Metrics rows
  logged after that checkpoint (work about to be redone) are cut, so the
  file never holds the same step twice.
- The config's fingerprint must match the run directory's, so a changed
  config can't silently continue someone else's run. Pass restart=True (or
  a new run name) to start over.

The batch for step s is generated from (seed, s) alone (data.pattern2d.
step_generator), and the global RNG (reparameterization noise) is restored
from the checkpoint, so a resumed run sees the data an uninterrupted one would.
SIGTERM (`colab/colab_runtime.py kill`) saves a checkpoint before exiting.
"""
from __future__ import annotations

import json
import math
import shutil
import signal
import time
from pathlib import Path

import torch

from src.config import INIT_PRETRAINED, INIT_SCRATCH, RunConfig
from src.data.pattern2d import make_eval_tasks, sample_tasks, split_tile_ids, step_generator
from src.eval.evaluate import evaluate_z0
from src.models.lpn import LPN, load_checkpoint
from src.training.losses import training_losses
from src.utils.logging_utils import get_logger
from src.utils.resume import append_jsonl_row, fingerprint, read_jsonl_rows
from src.utils.runtime import amp_dtype, describe_runtime

log = get_logger(__name__)

# Bump when a code change alters what a given config trains, so old run
# directories are never resumed by incompatible code.
TRAIN_VERSION = 1
# Logging/checkpoint cadence doesn't change what's trained.
CADENCE_FIELDS = ("log_every", "eval_every", "checkpoint_every")
QUICK_EVAL_TASKS = 512
INTERRUPTED_EXIT_CODE = 75  # EX_TEMPFAIL: stopped early, safe to resume


def run_fingerprint(cfg: RunConfig) -> str:
    settings = cfg.model_dump(include={"data", "model", "loss", "train"})
    for field in CADENCE_FIELDS:
        settings["train"].pop(field)
    settings["train_version"] = TRAIN_VERSION
    return fingerprint(settings)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak learning rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class Interrupted(Exception):
    pass


def train(
    cfg: RunConfig, run_dir: str | Path, device: torch.device,
    init_path: str | Path | None = None, restart: bool = False,
) -> dict:
    """Train (or resume) the run in `run_dir`. Returns the final metrics row.

    `init_path` is the resolved path when cfg.train.init names one of our checkpoints.
    Raises Interrupted after checkpointing if SIGTERM arrives.
    """
    run_dir = Path(run_dir)
    paths = {k: run_dir / f for k, f in
             (("config", "config.json"), ("ckpt", "checkpoint.pt"), ("final", "final.pt"), ("metrics", "metrics.jsonl"))}
    fp = run_fingerprint(cfg)
    if restart and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if paths["config"].exists():
        existing = json.loads(paths["config"].read_text(encoding="utf-8"))["fingerprint"]
        if existing != fp:
            raise RuntimeError(
                f"{run_dir} holds a run with a different config (fingerprint {existing}, now {fp}). "
                "Use --restart to discard it, or --name to start a separate run."
            )
    if paths["final"].exists():
        log.info(f"{run_dir.name}: already complete ({paths['final']}), nothing to do")
        done = [r for r in read_jsonl_rows(paths["metrics"]) if r.get("kind") == "done"]
        return done[-1] if done else {}

    tc = cfg.train
    torch.manual_seed(tc.seed)
    model = initial_model(cfg, init_path)
    if tc.mode == "probe":
        model.reset_heads()  # fresh heads; they alone are trained
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, tc.warmup_steps, tc.steps))
    dtype = amp_dtype(device, tc.amp)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    step = 0
    if paths["ckpt"].exists():
        state = torch.load(paths["ckpt"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"], device)
        step = state["step"]
        _truncate_metrics(paths["metrics"], step)
        log.info(f"{run_dir.name}: resumed from step {step}/{tc.steps}")
    else:
        _truncate_metrics(paths["metrics"], 0)
        log.info(f"{run_dir.name}: starting fresh ({tc.steps} steps, mode={tc.mode})")
    paths["config"].write_text(json.dumps(
        {"fingerprint": fp, "config": cfg.model_dump(), "runtime": describe_runtime(device)}, indent=2),
        encoding="utf-8")

    def save_checkpoint() -> None:
        tmp = paths["ckpt"].with_suffix(".tmp")
        torch.save({
            "step": step, "fingerprint": fp, "model": model.state_dict(), "model_cfg": cfg.model.model_dump(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, tmp)
        tmp.replace(paths["ckpt"])  # atomic: a disconnect mid-save never corrupts the checkpoint

    train_ids, heldout_ids = split_tile_ids(cfg.data.holdout_fraction, cfg.data.holdout_seed)
    pool = train_ids.to(device)
    eval_sets = {"seen": make_eval_tasks(train_ids, QUICK_EVAL_TASKS, cfg.data.num_pairs, cfg.eval.seed)}
    if len(heldout_ids):
        eval_sets["unseen"] = make_eval_tasks(heldout_ids, QUICK_EVAL_TASKS, cfg.data.num_pairs, cfg.eval.seed)

    def quick_eval() -> dict:
        model.eval()
        row = {"kind": "eval", "step": step}
        for split, tasks in eval_sets.items():
            for k, v in evaluate_z0(model, tasks, cfg.eval.batch_size, dtype).items():
                row[f"{split}/{k}"] = round(v, 5)
        model.train()
        _freeze_mode(model, tc.mode)
        return row

    stop = {"requested": False}
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    model.train()
    _freeze_mode(model, tc.mode)
    window_start, window_steps = time.perf_counter(), 0
    try:
        while step < tc.steps:
            batch = sample_tasks(pool, tc.batch_size, cfg.data.num_pairs, step_generator(tc.seed, step, device))
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                terms = training_losses(model, batch.inputs, batch.outputs, cfg.loss, mode=tc.mode,
                                        latent_steps=tc.train_latent_steps, latent_lr=tc.train_latent_lr)
            opt.zero_grad(set_to_none=True)
            scaler.scale(terms.total).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            window_steps += 1

            if step % tc.log_every == 0 or step == tc.steps:
                elapsed = time.perf_counter() - window_start
                row = {"kind": "train", "step": step, **{k: round(v, 5) for k, v in terms.as_floats().items()},
                       "grad_norm": round(grad_norm.item(), 4), "lr": sched.get_last_lr()[0],
                       "steps_per_sec": round(window_steps / elapsed, 2)}
                append_jsonl_row(paths["metrics"], row)
                log.info(" ".join(f"{k}={v}" for k, v in row.items() if k != "kind"))
                window_start, window_steps = time.perf_counter(), 0
            if step % tc.eval_every == 0 and step < tc.steps:
                row = quick_eval()
                append_jsonl_row(paths["metrics"], row)
                log.info(" ".join(f"{k}={v}" for k, v in row.items() if k != "kind"))
            if step % tc.checkpoint_every == 0 or stop["requested"]:
                save_checkpoint()
            if stop["requested"]:
                log.info(f"{run_dir.name}: SIGTERM, checkpointed at step {step}; rerun the same command to resume")
                raise Interrupted(step)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    save_checkpoint()
    final = {**quick_eval(), "kind": "done"}
    append_jsonl_row(paths["metrics"], final)
    tmp = paths["final"].with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "model_cfg": cfg.model.model_dump(), "fingerprint": fp,
                "step": step}, tmp)
    tmp.replace(paths["final"])
    log.info(f"{run_dir.name}: complete. " + " ".join(f"{k}={v}" for k, v in final.items() if k != "kind"))
    return final


def initial_model(cfg: RunConfig, init_path: str | Path | None = None) -> LPN:
    """The model a run starts from (before any resume): random init, the paper's
    pretrained checkpoint (fresh heads), or one of our own checkpoints."""
    init = cfg.train.init
    if init == INIT_SCRATCH:
        return LPN(cfg.model)
    if init == INIT_PRETRAINED:
        from src.models.pretrained import load_pretrained

        return load_pretrained(**cfg.model.model_dump(exclude=set(cfg.model.backbone())))
    loaded, _ = load_checkpoint(init_path or init)
    if loaded.cfg != cfg.model:
        raise ValueError(f"model config differs from the init checkpoint's ({init})")
    return loaded


def _freeze_mode(model: LPN, mode: str) -> None:
    """Probe mode keeps the frozen encoder/decoder in eval mode (dropout off)."""
    if mode == "probe":
        model.encoder.eval()
        model.decoder.eval()


def _truncate_metrics(path: Path, step: int) -> None:
    """Drop rows past `step` (logged after the checkpoint being resumed from)."""
    rows = read_jsonl_rows(path)
    keep = [r for r in rows if r.get("step", 0) <= step]
    if len(keep) != len(rows):
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
        tmp.replace(path)
