"""Train (or resume) one run.

    python scripts/train.py --config src/configs/stage1_frozen.yaml [--device cuda] [--name stage1_seed1]

Writes to <output root>/runs/<name>/ (output root = --output-root, else
$LPN_OUTPUT_ROOT, which colab/colab_runtime.py sets on the VM, else ./outputs).
Re-running the identical command resumes from the last checkpoint, or
returns immediately if the run is complete -- see src/training/train.py.

Exit codes: 0 done, 75 stopped by SIGTERM after checkpointing (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import INIT_PRETRAINED, INIT_SCRATCH, RunConfig
from src.training.train import INTERRUPTED_EXIT_CODE, Interrupted, train
from src.utils.runtime import describe_runtime, output_root, pick_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--name", default=None, help="run name (default: the config's name)")
    parser.add_argument("--init", default=None, help="override train.init: scratch | pretrained | checkpoint path "
                        "(relative paths resolve against the output root)")
    parser.add_argument("--restart", action="store_true", help="discard this run's existing state and start over")
    args = parser.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    root = Path(args.output_root) if args.output_root else output_root()
    device = pick_device(args.device)
    print(json.dumps(describe_runtime(device)), flush=True)

    if args.init:
        cfg = RunConfig.model_validate({**cfg.model_dump(), "train": {**cfg.train.model_dump(), "init": args.init}})
    init_path = None
    if cfg.train.init not in (INIT_SCRATCH, INIT_PRETRAINED):
        init_path = Path(cfg.train.init)
        init_path = init_path if init_path.is_absolute() else root / init_path
        if not init_path.exists():
            sys.exit(f"init checkpoint not found: {init_path}")

    try:
        train(cfg, root / "runs" / (args.name or cfg.name), device, init_path=init_path, restart=args.restart)
    except Interrupted:
        return INTERRUPTED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
