"""Render one task through both readouts of z as a GIF (see src/viz/pipeline_gif.py).

    python scripts/make_gif.py --ckpt outputs/runs/<name>/checkpoint.pt [--seed 0] [--tile 2,3,4,5] [--out pipeline.gif]

The task (4 context pairs + 1 query, distinct anchors) is sampled from --seed; --tile fixes its
2x2 program (row-major colors 1-9) instead of drawing one uniformly from all 6561 tiles.
"""
from __future__ import annotations

import argparse
import sys

import torch

from src.data.pattern2d import NUM_TILES, sample_tasks, tile_ids
from src.models.lpn import load_checkpoint
from src.viz.pipeline_gif import save_gif, trace_task

NUM_CONTEXT = 4


def parse_tile(text: str) -> torch.Tensor:
    colors = [int(c) for c in text.split(",")]
    if len(colors) != 4 or not all(1 <= c <= 9 for c in colors):
        raise argparse.ArgumentTypeError("--tile takes 4 comma-separated colors in 1-9, e.g. 2,3,4,5")
    return torch.tensor(colors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="checkpoint saved by training (or save_checkpoint)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile", type=parse_tile, default=None)
    parser.add_argument("--out", default="pipeline.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=72)
    args = parser.parse_args(argv)

    model, _ = load_checkpoint(args.ckpt)
    pool = torch.arange(NUM_TILES) if args.tile is None else tile_ids(args.tile)[None]
    task = sample_tasks(pool, 1, NUM_CONTEXT + 1, torch.Generator().manual_seed(args.seed))
    trace = trace_task(model, *task.split_query())
    path = save_gif(trace, args.out, fps=args.fps, dpi=args.dpi)
    print(f"tile {trace.tile.tolist()} (true {task.tiles[0].tolist()}), "
          f"head {'correct' if (trace.head_pred == trace.query_output).all() else 'wrong'}, "
          f"decoder {'correct' if (trace.decoder_pred == trace.query_output).all() else 'wrong'} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
