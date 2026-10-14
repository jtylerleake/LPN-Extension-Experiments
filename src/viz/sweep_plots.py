"""Performance-vs-config curves for the head sweep (src/training/head_sweep.py).

One figure: x = one spec key (e.g. width), y = a final metric, one line per value of a `hue`
key (e.g. depth), one panel per value of a `facet` key (e.g. lr). Every line is drawn twice:
held-out (solid, `val/...` or `test/...`) and seen tiles (dashed, `seen/...`), so the gap
between them -- memorization -- is visible at a glance. With several seeds per config the
line is the mean and the band spans min..max.

x values are placed evenly (they're grid values like 128/256/512/1024 or 3e-4/1e-3/3e-3),
not on a numeric scale.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical hues in fixed order (assigned by sorted hue value, never by rank)
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#8c8c8c")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def _fmt_short(v) -> str:
    return _fmt(v) if not isinstance(v, bool) else ("res" if v else "plain")


# derived keys usable as x / hue / facet
DERIVED = {
    "base": lambda s: f"d{s['depth']} w{s['width']} lr{_fmt(s['lr'])}",
    "config": lambda s: f"d{s['depth']} w{s['width']} lr{_fmt(s['lr'])}\n"
                        f"wd{_fmt(s.get('weight_decay'))} p{s.get('dropout')} {_fmt_short(s['residual'])}",
}


def spec_value(row: dict, key: str):
    return DERIVED[key](row["spec"]) if key in DERIVED else row["spec"].get(key)


def matches(row: dict, where: dict | None) -> bool:
    """`where` = {spec key: value or list of values}; also accepts 'round'."""
    for k, want in (where or {}).items():
        have = row.get(k) if k == "round" else spec_value(row, k)
        if have not in (want if isinstance(want, (list, tuple, set)) else [want]):
            return False
    return True


def aggregate(rows: list[dict], x: str, metric: str, hue: str | None = None, facet: str | None = None,
              where: dict | None = None) -> dict:
    """-> {facet value: {hue value: {x value: (mean, min, max, n)}}} over seeds (and any other
    unlisted keys -- filter them with `where` so lines don't mix configs)."""
    groups = defaultdict(list)
    for r in rows:
        if not matches(r, where) or metric not in r["final"]:
            continue
        key = (spec_value(r, facet) if facet else None, spec_value(r, hue) if hue else None, spec_value(r, x))
        groups[key].append(r["final"][metric])
    out: dict = defaultdict(lambda: defaultdict(dict))
    for (f, h, xv), vals in groups.items():
        out[f][h][xv] = (float(np.mean(vals)), min(vals), max(vals), len(vals))
    return out


def _sorted(values):
    return sorted(values, key=lambda v: (v is None, v))


def _fmt(v) -> str:
    if isinstance(v, float) and v and (v < 1e-2 or v >= 1e4):
        return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
    return str(v)


def plot_sweep(rows: list[dict], x: str, hue: str | None = None, facet: str | None = None,
               metric: str = "tile_cell_acc", heldout: str = "val", show_seen: bool = True,
               where: dict | None = None, title: str | None = None, out: str | Path | None = None,
               connect: bool = True):
    """Draw the figure (and save it to `out` if given). Returns the matplotlib Figure.
    `connect=False` draws points with min..max error bars instead of lines (x values unordered)."""
    splits = [heldout] + (["seen"] if show_seen else [])
    data = {s: aggregate(rows, x, f"{s}/{metric}", hue, facet, where) for s in splits}
    facets = _sorted({f for d in data.values() for f in d})
    if not facets:
        raise ValueError(f"no rows with metric {heldout}/{metric} match {where}")
    hues = _sorted({h for d in data.values() for f in d.values() for h in f})
    if len(hues) > len(SERIES_COLORS):
        raise ValueError(f"{len(hues)} hue values; at most {len(SERIES_COLORS)} -- facet or filter instead")
    xs = _sorted({xv for d in data.values() for f in d.values() for h in f.values() for xv in h})
    pos = {xv: i for i, xv in enumerate(xs)}
    color = {h: SERIES_COLORS[i] for i, h in enumerate(hues)}

    fig, axes = plt.subplots(1, len(facets), figsize=(3.6 * len(facets) + 1.2, 3.4), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], facets):
        for split in splits:
            style = {"linestyle": "-" if split == heldout else "--", "alpha": 1.0 if split == heldout else 0.75}
            for h in hues:
                pts = data[split].get(f, {}).get(h, {})
                if not pts:
                    continue
                xv = _sorted(pts)
                p = np.array([pos[v] for v in xv])
                mean, lo, hi = (np.array([pts[v][i] for v in xv]) for i in range(3))
                if not connect:
                    ax.errorbar(p, mean, yerr=np.vstack([mean - lo, hi - mean]), color=color[h], linestyle="none",
                                marker="o" if split == heldout else "D", markersize=6, capsize=4,
                                alpha=style["alpha"])
                    continue
                ax.plot(p, mean, color=color[h], linewidth=2, marker="o", markersize=5, **style)
                if any(pts[v][3] > 1 for v in xv):
                    ax.fill_between(p, lo, hi, color=color[h], alpha=0.15, linewidth=0)
        ax.set_xticks(range(len(xs)), [_fmt(v) for v in xs])
        ax.set_xlabel(x, color=MUTED)
        if facet:
            ax.set_title(f"{facet} = {_fmt(f)}", fontsize=10, color=INK)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0][0].set_ylabel(metric, color=MUTED)

    handles = [plt.Line2D([], [], color=color[h], linewidth=2, marker="o", markersize=5,
                          label=f"{hue} = {_fmt(h)}") for h in hues] if hue else []
    if show_seen:
        line = {"linewidth": 2} if connect else {"linestyle": "none", "markersize": 6}
        handles += [plt.Line2D([], [], color=MUTED, label=f"{heldout} (held-out tiles)",
                               **({"linestyle": "-"} | line if connect else {"marker": "o"} | line)),
                    plt.Line2D([], [], color=MUTED, label="seen (training tiles)", alpha=0.75,
                               **({"linestyle": "--"} | line if connect else {"marker": "D"} | line))]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    fig.suptitle(title or f"{metric} vs {x}", color=INK, fontsize=11)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def standard_plots(rows: list[dict], out_dir: str | Path, metric: str = "tile_cell_acc") -> list[Path]:
    """The sweep's figures, for whichever rounds have run:
      round A   accuracy vs depth, a line per width, a panel per lr
      round B   per regularization setting: accuracy vs its value, a line per base
                (the other settings held at the base's values)
      round C   the final configurations, test (mean and min..max over seeds)
    """
    from src.training.head_sweep import REG_KEYS, base_spec  # torch import: keep it out of module load

    out_dir, paths = Path(out_dir), []

    def save(name: str, rows_: list[dict], **kw) -> None:
        if rows_:
            paths.append(out_dir / f"{name}.png")
            plt.close(plot_sweep(rows_, metric=metric, out=paths[-1], **kw))

    save("A_depth_by_width_per_lr", [r for r in rows if r["round"] == "A"], x="depth", hue="width", facet="lr",
         title=f"Round A: {metric} vs depth")
    bases = {spec_value(r, "base") for r in rows if r["round"] == "B"}
    trained = [r for r in rows if not r["test"] and r["round"] in ("A", "B") and spec_value(r, "base") in bases]
    for key in REG_KEYS:
        def at_base(r, key=key):
            base = base_spec(r["spec"])
            return all(r["spec"].get(k) == base[k] for k in REG_KEYS if k != key)
        save(f"B_{key}_by_base", [r for r in trained if at_base(r)], x=key, hue="base",
             title=f"Round B: {metric} vs {key}")
    save("C_final_test", [r for r in rows if r["round"] == "C"], x="config", heldout="test", connect=False,
         title=f"Round C: {metric} of the final configurations (3 seeds)")
    return paths
