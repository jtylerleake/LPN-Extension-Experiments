import pytest

from scripts.plot_sweep import parse_where
from src.training import head_sweep as hs
from src.viz.sweep_plots import aggregate, plot_sweep, standard_plots


def row(depth, width, lr, seed, val, seen, rnd="A"):
    return {"round": rnd, "test": False, "spec": {"depth": depth, "width": width, "lr": lr, "seed": seed},
            "final": {"val/tile_full_acc": val, "seen/tile_full_acc": seen,
                      "val/tile_cell_acc": val, "seen/tile_cell_acc": seen}}


ROWS = [row(d, w, lr, s, val=0.1 * d + 0.01 * s, seen=1.0, rnd="A" if lr == 1e-3 else "B")
        for d in (2, 4) for w in (128, 256) for lr in (1e-3, 3e-3) for s in (0, 1)]


def test_aggregate_averages_seeds_and_respects_where():
    agg = aggregate(ROWS, "width", "val/tile_full_acc", hue="depth", where={"round": "A"})
    mean, lo, hi, n = agg[None][4][256]
    assert n == 2 and (lo, hi, mean) == pytest.approx((0.4, 0.41, 0.405))


def test_plot_writes_a_figure(tmp_path):
    out = tmp_path / "p.png"
    fig = plot_sweep(ROWS, "width", hue="depth", facet="lr", out=out)
    assert out.exists() and len(fig.axes) == 2


def test_plot_rejects_an_empty_selection():
    with pytest.raises(ValueError):
        plot_sweep(ROWS, "width", where={"round": "Z"})


def test_parse_where():
    assert parse_where(["round=A", "lr=0.001,0.003", "residual=true"]) == \
        {"round": "A", "lr": [0.001, 0.003], "residual": True}


def test_standard_plots_cover_every_round(tmp_path):
    def r(spec, rnd, test=False):
        final = {f"{sp}/tile_cell_acc": 0.5 for sp in ("seen", "val", "test")}
        return {"round": rnd, "test": test, "spec": spec, "final": final}

    base = hs.make_spec(4, 256, 1e-3)
    rows = ([r(hs.make_spec(d, w, lr), "A") for d in (2, 4) for w in (128, 256) for lr in (1e-3, 3e-3)]
            + [r(v, "B") for v in hs.variations(base)] + [r({**base, "seed": s}, "C", True) for s in (0, 1, 2)])
    names = {p.stem for p in standard_plots(rows, tmp_path)}
    assert names == {"A_depth_by_width_per_lr", "B_weight_decay_by_base", "B_dropout_by_base",
                     "B_residual_by_base", "C_final_test"}
