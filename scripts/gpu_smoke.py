"""GPU smoke test + throughput probe for a fresh Colab session.

Checks the whole stack runs on the session's device (data generation on the
GPU, AMP training step on the full-size model, test-time latent gradients,
greedy decoding), reports steps/sec so training budgets can be sized, and
writes a small JSON result to the output root to exercise `fetch`.

    python3 scripts/colab_runtime.py run -- scripts/gpu_smoke.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lpn_interp.config import LossConfig, ModelConfig
from lpn_interp.data.pattern2d import make_eval_tasks, sample_tasks, split_tile_ids, step_generator
from lpn_interp.models.lpn import LPN
from lpn_interp.training.losses import context_loss, training_losses
from lpn_interp.utils.runtime import amp_dtype, describe_runtime, output_root, pick_device


def timed(fn, sync, repeats: int) -> float:
    fn()  # warmup
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - start) / repeats


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-dir", default=str(output_root() / "smoke"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    info = describe_runtime(device)
    print(json.dumps(info, indent=2))
    dtype = amp_dtype(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    torch.manual_seed(0)
    model = LPN(ModelConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    pool = split_tile_ids(0.1, seed=0)[0].to(device)
    losses = []

    def train_step(step=[0]):
        batch = sample_tasks(pool, args.batch_size, 4, step_generator(0, step[0], device))
        step[0] += 1
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            terms = training_losses(model, batch.inputs, batch.outputs, LossConfig())
        opt.zero_grad(set_to_none=True)
        scaler.scale(terms.total).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(terms.total))

    sec_per_step = timed(train_step, sync, args.steps)
    print(f"train: {1 / sec_per_step:.1f} steps/s at batch {args.batch_size} "
          f"(loss {losses[0]:.1f} -> {losses[-1]:.1f}, amp={dtype})")

    model.eval().requires_grad_(False)
    ctx, qry = make_eval_tasks(pool.cpu(), 500, 4, seed=1).to(device).split_query()
    z = model.context_latent(ctx.inputs, ctx.outputs).detach().requires_grad_()

    def latent_grad():
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            loss = context_loss(model, ctx.inputs, ctx.outputs, z, 1.0, 0.1).sum()
        torch.autograd.grad(loss, z)

    sec_latent = timed(latent_grad, sync, 10)
    print(f"latent opt: {sec_latent * 1e3:.1f} ms/step for 500 tasks")
    sec_decode = timed(lambda: model.decoder.generate(qry.inputs[:, 0], z.detach()), sync, 2)
    print(f"greedy decode: {sec_decode:.2f} s for 500 grids")

    result = {**info, "train_steps_per_sec": 1 / sec_per_step, "batch_size": args.batch_size,
              "latent_ms_per_step_500": sec_latent * 1e3, "decode_s_500": sec_decode,
              "loss_first": losses[0], "loss_last": losses[-1]}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out / 'smoke.json'}")
    return result


if __name__ == "__main__":
    main()
