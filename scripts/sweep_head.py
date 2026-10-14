"""Stage-1 head sweep on the frozen pretrained LPN (see src/training/head_sweep.py).

    python scripts/sweep_head.py --round A        # depth x width x lr (48 runs)
    python scripts/sweep_head.py --round B        # weight decay / dropout / residual one at a time, 3 best of A
    python scripts/sweep_head.py --round B2       # per base, the best round-B values combined (3 runs)
    python scripts/sweep_head.py --round C        # 3 best configurations x 3 seeds, scored on test (9 runs)
    python scripts/sweep_head.py --round summary  # ranked table of everything so far
    python scripts/sweep_head.py --round plots    # the standard figures, into <sweep dir>/plots

Results go to <output root>/sweeps/<name>/results.jsonl, one row per finished run (output root =
--output-root, else $LPN_OUTPUT_ROOT, which colab/colab_runtime.py sets on the VM, else ./outputs).
Rerunning a round skips finished runs, so after a disconnect just run the same command again
(only the run in progress is lost). --max-steps caps every run for a smoke test and writes to
<name>_smoke instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.training.head_sweep import EncodingCache, Splits, read_jsonl_rows, run_round, summary_table
from src.utils.runtime import describe_runtime, output_root, pick_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", required=True, choices=["A", "B", "B2", "C", "summary", "plots"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--name", default="head_sweep_v2")
    parser.add_argument("--max-steps", type=int, default=None, help="cap every run's steps (smoke test)")
    args = parser.parse_args(argv)

    name = f"{args.name}_smoke" if args.max_steps else args.name
    results = (Path(args.output_root) if args.output_root else output_root()) / "sweeps" / name / "results.jsonl"
    if args.round == "summary":
        print(summary_table(read_jsonl_rows(results)))
        return 0
    if args.round == "plots":
        from src.viz.sweep_plots import standard_plots

        for path in standard_plots(read_jsonl_rows(results), results.parent / "plots"):
            print(path)
        return 0

    device = pick_device(args.device)
    print(json.dumps(describe_runtime(device)), flush=True)

    def cache_factory():
        from src.models.pretrained import load_pretrained

        cache = EncodingCache(load_pretrained(), device)
        return cache, Splits(cache)

    eval_every = max(1, args.max_steps // 2) if args.max_steps else None
    kwargs = {"max_steps": args.max_steps} | ({"eval_every": eval_every} if eval_every else {})
    rows = run_round(args.round, results, cache_factory, **kwargs)
    print(f"\nround {args.round} ({results}):\n{summary_table(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
