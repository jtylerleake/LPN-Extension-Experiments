"""Performance-vs-config curves from a head sweep's results.jsonl (see src/viz/sweep_plots.py).

    # round A: val (solid) and seen (dashed) cell accuracy vs depth, a line per width, a panel per lr
    python scripts/plot_sweep.py --x depth --hue width --facet lr --where round=A

    # full-tile accuracy vs lr at width 1024 only
    python scripts/plot_sweep.py --x lr --hue depth --metric tile_full_acc --where round=A width=1024

--where takes key=value (values comma-separated, parsed as numbers/bools where possible) and
should pin every spec key that isn't x/hue/facet/seed, or lines average across configs.
The figure goes to --out, else <sweep dir>/plots/<metric>_vs_<x>[_by_<hue>][_per_<facet>].png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.resume import read_jsonl_rows
from src.utils.runtime import output_root
from src.viz.sweep_plots import plot_sweep


def parse_value(text: str):
    try:
        return json.loads(text.lower() if text.lower() in ("true", "false") else text)
    except json.JSONDecodeError:
        return text


def parse_where(items: list[str]) -> dict:
    where = {}
    for item in items:
        key, _, vals = item.partition("=")
        parsed = [parse_value(v) for v in vals.split(",")]
        where[key] = parsed if len(parsed) > 1 else parsed[0]
    return where


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x", required=True, help="spec key on the x-axis")
    parser.add_argument("--hue", default=None, help="spec key: one line per value")
    parser.add_argument("--facet", default=None, help="spec key: one panel per value")
    parser.add_argument("--metric", default="tile_cell_acc", choices=["tile_cell_acc", "tile_full_acc"])
    parser.add_argument("--heldout", default="val", choices=["val", "test"])
    parser.add_argument("--no-seen", action="store_true", help="hide the dashed seen-tiles lines")
    parser.add_argument("--where", nargs="*", default=[], help="filters, e.g. round=A lr=0.001,0.003")
    parser.add_argument("--title", default=None)
    parser.add_argument("--name", default="head_sweep_v2")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    sweep_dir = (Path(args.output_root) if args.output_root else output_root()) / "sweeps" / args.name
    rows = read_jsonl_rows(sweep_dir / "results.jsonl")
    out = args.out or sweep_dir / "plots" / (
        f"{args.metric}_vs_{args.x}" + (f"_by_{args.hue}" if args.hue else "")
        + (f"_per_{args.facet}" if args.facet else "") + ".png")
    plot_sweep(rows, args.x, args.hue, args.facet, args.metric, args.heldout, not args.no_seen,
               parse_where(args.where), args.title, out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
