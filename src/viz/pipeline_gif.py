"""Side-by-side GIF of one task going through both readouts of z (architecture 1).

Left: the task -- 4 context pairs and the query input. The context pairs pool
into z, which feeds two pipelines that run in lockstep:
- top, the MLP-head program: z -> MLP head -> 2x2x10 tile (one-hot cubes at
  each cell's argmax color) -> FIND(x_q) -> PASTE -> y_hat
- bottom, the decoder: (x_q, z) -> decoder -> y_hat, revealed cell by cell in
  the order greedy decoding produces them
The last frames add the true y_q and mark each prediction right or wrong,
outlining mismatched cells.

`trace_task` runs the model once and keeps everything the drawing needs, so
rendering never touches torch. `draw_frame` redraws the whole scene for frame
t from the timeline in `PHASES`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from src.config import GRID_SIZE, NUM_COLORS, TILE_SIZE  # noqa: E402
from src.data.pattern2d import TaskBatch  # noqa: E402
from src.execution.executor import find_anchor, paste  # noqa: E402
from src.models.lpn import LPN  # noqa: E402
from src.models.mlp_head import hard_tile  # noqa: E402

# Standard ARC palette, indexed by color id.
ARC_COLORS = ("#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
              "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25")
INK, MUTED, FAINT = "#222222", "#888888", "#D0D0D0"
GOOD, BAD, ACTIVE = "#1B9E3E", "#D62728", "#FFF4D6"

# Timeline in frames: (start, end) of each phase. Progress within a phase runs 0 -> 1.
PHASES = {
    "context": (0, 12),  # pairs appear one by one, then the query input
    "pool": (12, 17),  # arrows converge into z
    "run": (17, 27),  # top: MLP head builds the tile; bottom: (x_q, z) enter the decoder
    "find": (27, 37),  # top: FIND highlights the anchor
    "paste": (37, 47),  # top: the tile slides onto the grid
    "decode": (27, 47),  # bottom: output cells appear in generation order
    "verdict": (47, 72),  # target y_q, check marks, mismatch outlines (held to the loop)
}
NUM_FRAMES = PHASES["verdict"][1]

FIG_W, FIG_H = 18.0, 9.0  # scene size in data units (1 unit = 1 inch at the default figsize)
MID_Y = 4.5  # boundary between the top and bottom pipelines


@dataclass
class PipelineTrace:
    ctx_inputs: np.ndarray  # (P, 4, 4)
    ctx_outputs: np.ndarray  # (P, 4, 4)
    query_input: np.ndarray  # (4, 4)
    query_output: np.ndarray  # (4, 4) ground truth
    z: np.ndarray  # (d,)
    tile: np.ndarray  # (4,) head's argmax colors, row-major
    anchor: tuple[int, int]  # FIND(x_q) as (row, col)
    head_pred: np.ndarray  # (4, 4) PASTE(tile, FIND(x_q))
    decoder_pred: np.ndarray  # (4, 4) greedy decode, generated in row-major order


@torch.no_grad()
def trace_task(model: LPN, context: TaskBatch, query: TaskBatch) -> PipelineTrace:
    """Run one task (batch of 1, from TaskBatch.split_query) through both readouts."""
    if model.cfg.architecture != 1:
        raise ValueError("the pipeline GIF draws architecture 1 (hardcoded FIND) only")
    model.eval()
    device = next(model.parameters()).device
    context, query = context.to(device), query.to(device)
    xq, yq = query.inputs[:, 0], query.outputs[:, 0]  # (1, 4, 4)
    z = model.context_latent(context.inputs, context.outputs)  # (1, d)
    logits, _ = model.program_logits(z)
    tile = hard_tile(logits)  # (1, 4)
    anchor = find_anchor(xq)
    head_pred = paste(torch.nn.functional.one_hot(tile, NUM_COLORS).float(), xq, anchor).argmax(-1)
    a = int(anchor[0])
    return PipelineTrace(
        ctx_inputs=context.inputs[0].cpu().numpy(), ctx_outputs=context.outputs[0].cpu().numpy(),
        query_input=xq[0].cpu().numpy(), query_output=yq[0].cpu().numpy(),
        z=z[0].float().cpu().numpy(), tile=tile[0].cpu().numpy(), anchor=(a // GRID_SIZE, a % GRID_SIZE),
        head_pred=head_pred[0].view(GRID_SIZE, GRID_SIZE).cpu().numpy(),
        decoder_pred=model.decoder.generate(xq, z)[0].cpu().numpy(),
    )


# --- timeline -------------------------------------------------------------------------------

def progress(phase: str, t: float) -> float:
    """0 before the phase, 1 after it, linear in between."""
    start, end = PHASES[phase]
    return float(np.clip((t - start) / (end - start), 0.0, 1.0))


def _ease(p: float) -> float:
    return p * p * (3 - 2 * p)


# --- drawing primitives ---------------------------------------------------------------------

def draw_grid(ax, grid: np.ndarray, x: float, y: float, cell: float, alpha: float = 1.0,
              mask: np.ndarray | None = None) -> None:
    """Grid with its top-left corner at (x, y). `mask` (4, 4) bool hides cells (drawn empty)."""
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            shown = mask is None or mask[r, c]
            ax.add_patch(Rectangle((x + c * cell, y - (r + 1) * cell), cell, cell,
                                   facecolor=ARC_COLORS[grid[r, c]] if shown else "white",
                                   edgecolor="#555555", linewidth=0.6, alpha=alpha))


def outline_cells(ax, cells, x: float, y: float, cell: float, color: str, lw: float = 2.5,
                  alpha: float = 1.0) -> None:
    for r, c in cells:
        ax.add_patch(Rectangle((x + c * cell, y - (r + 1) * cell), cell, cell, fill=False,
                               edgecolor=color, linewidth=lw, alpha=alpha, zorder=5))


def box(ax, x: float, y: float, w: float, h: float, label: str, active: float = 0.0,
        alpha: float = 1.0) -> None:
    """Rounded operation box, lower-left at (x, y); `active` in [0, 1] tints it."""
    face = tuple(np.array(matplotlib.colors.to_rgb("white")) * (1 - active)
                 + np.array(matplotlib.colors.to_rgb(ACTIVE)) * active)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=face, edgecolor=INK, linewidth=1.4, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=13, color=INK,
            alpha=alpha, fontweight="bold")


def arrow(ax, start, end, alpha: float = 1.0, color: str = INK, rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                 color=color, alpha=alpha, connectionstyle=f"arc3,rad={rad}"))


def partial_arrow(ax, start, end, p: float, **kw) -> None:
    """Arrow drawn from start toward end, `p` of the way."""
    if p > 0:
        start, end = np.array(start), np.array(end)
        arrow(ax, start, start + (end - start) * p, **kw)


def label(ax, x: float, y: float, text: str, alpha: float = 1.0, size: int = 12, **kw) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=kw.pop("color", INK), alpha=alpha, **kw)


# --- 3D tile ---------------------------------------------------------------------------------

CUBE_S = 0.5  # cube face size
DEPTH = (0.13, 0.085)  # screen offset per color slice (x, y)


def _shade(color: str, f: float):
    return tuple(np.array(matplotlib.colors.to_rgb(color)) * f)


def draw_tile_3d(ax, tile: np.ndarray, x: float, y: float, p: float) -> None:
    """2x2x10 one-hot tile: faint 2x2x10 volume, a cube in each cell's argmax color slice.
    Front-face top-left at (x, y); cubes extrude in as p goes 0 -> 1."""
    if p <= 0:
        return
    dx, dy = DEPTH
    s, n = CUBE_S, TILE_SIZE

    def pt(r, c, k):
        return (x + c * s + k * dx, y - r * s + k * dy)

    # Volume wireframe: front and back faces plus the four depth edges, then slice ticks.
    for k in (0, NUM_COLORS):
        corners = [pt(0, 0, k), pt(0, n, k), pt(n, n, k), pt(n, 0, k)]
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=MUTED, linewidth=0.8, alpha=p))
    for r, c in ((0, 0), (0, n), (n, n), (n, 0)):
        ax.plot(*zip(pt(r, c, 0), pt(r, c, NUM_COLORS), strict=True), color=MUTED, linewidth=0.8, alpha=p)
    for k in range(1, NUM_COLORS):
        ax.plot(*zip(pt(0, 0, k), pt(0, n, k), pt(n, n, k), strict=True), color=FAINT, linewidth=0.5, alpha=p)

    # Cubes, painter's order: far slices first, then bottom rows, then left columns.
    cells = sorted(((int(tile[i]), i // n, i % n) for i in range(n * n)), key=lambda t: (-t[0], -t[1], t[2]))
    for idx, (k, r, c) in enumerate(cells):
        cp = np.clip(p * len(cells) - idx, 0.0, 1.0)  # cubes pop in one after another
        if cp <= 0:
            continue
        color = ARC_COLORS[k]
        front = [pt(r, c, k), pt(r, c + 1, k), pt(r + 1, c + 1, k), pt(r + 1, c, k)]
        top = [pt(r, c, k), pt(r, c, k + 1), pt(r, c + 1, k + 1), pt(r, c + 1, k)]
        side = [pt(r, c + 1, k), pt(r, c + 1, k + 1), pt(r + 1, c + 1, k + 1), pt(r + 1, c + 1, k)]
        for face, f in ((side, 0.6), (top, 0.85), (front, 1.0)):
            ax.add_patch(Polygon(face, closed=True, facecolor=_shade(color, f), edgecolor="#333333",
                                 linewidth=0.6, alpha=cp, zorder=3))
    label(ax, x + n * s / 2, y - n * s - 0.3, "rows x cols", p, size=10, color=MUTED)
    end = pt(0, n, NUM_COLORS)
    ax.text(end[0] + 0.05, end[1], "color (10)", fontsize=10, color=MUTED, alpha=p, ha="left", va="bottom")


# --- scene layout ---------------------------------------------------------------------------

LEFT_CELL = 0.27  # task panel cell size
CELL = 0.32  # pipeline cell size
GRID_W = GRID_SIZE * CELL
Z_POS = (4.75, MID_Y)
# Top pipeline (grids hang from y = TOP_Y; boxes centered on TOP_C).
TOP_Y, TOP_C = 7.25, 6.6
MLP_X, MLP_W = 5.9, 1.35
TILE3D_X = 7.7
TILE3D_W = TILE_SIZE * CUBE_S + NUM_COLORS * DEPTH[0]
FIND_X, FIND_W = 10.4, 1.9
PASTE_X, PASTE_W = 12.75, 1.25
PRED_X = 14.3
# Bottom pipeline.
BOT_Y, BOT_C = 3.05, 2.4
XQ_BOT_X = 6.0
DEC_X, DEC_W = 9.0, 2.6
# Verdict column.
TARGET_X = 16.5


def _task_panel(ax, tr: PipelineTrace, t: float) -> list[tuple[float, float]]:
    """Context pairs + query; returns the right-edge midpoints of the context outputs (arrow sources)."""
    label(ax, 1.65, 8.65, "Task", size=15, fontweight="bold")
    w = GRID_SIZE * LEFT_CELL
    sources = []
    num_ctx = len(tr.ctx_inputs)
    p = progress("context", t) * (num_ctx + 1)  # one slot per context pair, one for the query
    for i in range(num_ctx + 1):
        a = float(np.clip(p - i, 0.0, 1.0))
        top = 8.25 - i * 1.55 - (0.25 if i == num_ctx else 0.0)
        if i == num_ctx:
            ax.plot([0.3, 3.2], [top + 0.3, top + 0.3], color=FAINT, linewidth=1, linestyle="--", alpha=a)
        if a <= 0:
            continue
        x_in, y_in = (tr.ctx_inputs[i], tr.ctx_outputs[i]) if i < num_ctx else (tr.query_input, None)
        draw_grid(ax, x_in, 0.45, top, LEFT_CELL, a)
        arrow(ax, (0.45 + w + 0.1, top - w / 2), (2.05 - 0.1, top - w / 2), a)
        if y_in is not None:
            draw_grid(ax, y_in, 2.05, top, LEFT_CELL, a)
            sources.append((2.05 + w + 0.05, top - w / 2))
        else:
            ax.add_patch(Rectangle((2.05, top - w), w, w, fill=False, edgecolor=MUTED,
                                   linestyle="--", linewidth=1, alpha=a))
            label(ax, 2.05 + w / 2, top - w / 2, "?", a, size=16, color=MUTED)
        name = f"pair {i + 1}" if i < num_ctx else "query"
        ax.text(0.35, top - w / 2, name, rotation=90, ha="right", va="center", fontsize=9, color=MUTED, alpha=a)
    return sources


def draw_frame(ax, tr: PipelineTrace, t: float) -> None:
    ax.clear()
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 3.55, FIG_H, facecolor="#F6F6F6", edgecolor="none"))

    sources = _task_panel(ax, tr, t)

    # Pool: context outputs -> z.
    p_pool = progress("pool", t)
    for src in sources:
        partial_arrow(ax, src, (Z_POS[0] - 0.45, Z_POS[1]), min(1.0, p_pool * 1.6), alpha=0.5, color=MUTED)
    za = float(np.clip(p_pool * 2 - 1, 0.0, 1.0))
    if za > 0:
        ax.add_patch(FancyBboxPatch((Z_POS[0] - 0.4, Z_POS[1] - 0.4), 0.8, 0.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.4",
                                    facecolor="#EDE7F6", edgecolor="#5E35B1", linewidth=1.8, alpha=za))
        label(ax, *Z_POS, "z", za, size=20, fontweight="bold", color="#5E35B1")
        label(ax, Z_POS[0], Z_POS[1] - 0.65, "(" + ", ".join(f"{v:.2f}" for v in tr.z) + ")", za, size=9,
              color="#5E35B1")
        ax.plot([5.45, FIG_W - 0.2], [MID_Y, MID_Y], color=FAINT, linewidth=1, linestyle="--", alpha=za)
        label(ax, 9.3, 8.65, r"MLP-head program:  z → tile → FIND → PASTE → $\hat{y}$", za, size=14, fontweight="bold")
        label(ax, 9.3, 4.1, r"Decoder:  ($x_q$, z) → $\hat{y}$", za, size=14, fontweight="bold")

    p_run, p_find, p_paste = progress("run", t), progress("find", t), progress("paste", t)
    p_dec, p_ver = progress("decode", t), progress("verdict", t)
    active = 1.0 if p_run > 0 and p_run < 1 else 0.0
    xq = tr.query_input

    # ---- top: MLP head -> 3D tile -> FIND -> PASTE -> y_hat ----
    if p_run > 0:
        partial_arrow(ax, (Z_POS[0] + 0.3, Z_POS[1] + 0.35), (MLP_X - 0.05, TOP_C), min(1.0, p_run * 3), rad=-0.25)
        box(ax, MLP_X, TOP_C - 0.55, MLP_W, 1.1, "MLP\nhead", active)
        arrow(ax, (MLP_X + MLP_W + 0.05, TOP_C), (TILE3D_X - 0.1, TOP_C), min(1.0, p_run * 2))
        draw_tile_3d(ax, tr.tile, TILE3D_X, TOP_C + 0.5, _ease(p_run))
        label(ax, TILE3D_X + TILE3D_W / 2, 8.25, "tile  2×2×10", p_run, size=11, color=MUTED)

    if p_find > 0:
        arrow(ax, (TILE3D_X + TILE3D_W + 0.1, TOP_C), (FIND_X - 0.05, TOP_C), min(1.0, p_find * 3))
        find_active = 1.0 if p_find < 1 else 0.0
        box(ax, FIND_X, TOP_C - 1.05, FIND_W, 2.1, "", find_active)
        label(ax, FIND_X + FIND_W / 2, TOP_C + 0.78, r"FIND($x_q$)", size=12, fontweight="bold")
        gx, gy = FIND_X + (FIND_W - GRID_W) / 2, TOP_C + 0.55
        draw_grid(ax, xq, gx, gy, CELL)
        pulse = 0.5 + 0.5 * np.cos(2 * np.pi * p_find * 2.5)
        outline_cells(ax, [tr.anchor], gx, gy, CELL, "#FFB000", lw=2 + 2.5 * pulse)
        label(ax, FIND_X + FIND_W / 2, TOP_C - 0.85, f"anchor = {tr.anchor}", min(1.0, p_find * 2), size=10)

    if p_paste > 0:
        arrow(ax, (FIND_X + FIND_W + 0.05, TOP_C), (PASTE_X - 0.05, TOP_C), min(1.0, p_paste * 3))
        box(ax, PASTE_X, TOP_C - 0.55, PASTE_W, 1.1, "PASTE", 1.0 if p_paste < 1 else 0.0)
        arrow(ax, (PASTE_X + PASTE_W + 0.05, TOP_C), (PRED_X - 0.1, TOP_C), min(1.0, p_paste * 3))
        landed = p_paste >= 1
        draw_grid(ax, tr.head_pred if landed else xq, PRED_X, TOP_Y, CELL)
        if not landed:  # the flat argmax tile slides from the 3D volume onto the anchor
            src = np.array([TILE3D_X, TOP_C + 0.5])
            dst = np.array([PRED_X + tr.anchor[1] * CELL, TOP_Y - tr.anchor[0] * CELL])
            over = np.array([(src[0] + dst[0]) / 2, 9.0])  # arc over the FIND box
            q = _ease(p_paste)
            pos = (1 - q) ** 2 * src + 2 * q * (1 - q) * over + q ** 2 * dst
            draw_grid(ax, tr.tile.reshape(TILE_SIZE, TILE_SIZE), pos[0], pos[1], CELL)
        label(ax, PRED_X + GRID_W / 2, TOP_Y + 0.3, r"$\hat{y}$ (head)", size=12, fontweight="bold")

    # ---- bottom: (x_q, z) -> decoder -> y_hat, cell by cell ----
    if p_run > 0:
        draw_grid(ax, xq, XQ_BOT_X, BOT_Y, CELL, p_run)
        label(ax, XQ_BOT_X + GRID_W / 2, BOT_Y + 0.3, r"$x_q$", p_run, size=12, fontweight="bold")
        partial_arrow(ax, (Z_POS[0] + 0.35, Z_POS[1] - 0.35), (DEC_X + DEC_W / 2, BOT_C + 0.72),
                      min(1.0, p_run * 3), rad=-0.05)
        arrow(ax, (XQ_BOT_X + GRID_W + 0.1, BOT_C), (DEC_X - 0.05, BOT_C), min(1.0, p_run * 2))
        dec_active = 1.0 if 0 < p_run and p_dec < 1 else 0.0
        box(ax, DEC_X, BOT_C - 0.7, DEC_W, 1.4, "Decoder", dec_active)
        label(ax, DEC_X + DEC_W / 2, BOT_C - 1.0, "greedy, one cell at a time", p_run, size=9, color=MUTED)
    if p_dec > 0:
        arrow(ax, (DEC_X + DEC_W + 0.05, BOT_C), (PRED_X - 0.1, BOT_C), min(1.0, p_dec * 3))
        shown = np.arange(GRID_SIZE * GRID_SIZE).reshape(GRID_SIZE, GRID_SIZE) < round(p_dec * GRID_SIZE ** 2)
        draw_grid(ax, tr.decoder_pred, PRED_X, BOT_Y, CELL, mask=shown)
        if p_dec < 1:
            nxt = int(shown.sum())
            outline_cells(ax, [divmod(nxt, GRID_SIZE)], PRED_X, BOT_Y, CELL, "#FFB000", lw=2.5)
        label(ax, PRED_X + GRID_W / 2, BOT_Y + 0.3, r"$\hat{y}$ (decoder)", size=12, fontweight="bold")

    # ---- verdict: target, check marks, mismatch outlines ----
    if p_ver > 0:
        a = min(1.0, p_ver * 4)
        ty = MID_Y + GRID_W / 2
        draw_grid(ax, tr.query_output, TARGET_X, ty, CELL, a)
        label(ax, TARGET_X + GRID_W / 2, ty + 0.3, r"target $y_q$", a, size=12, fontweight="bold")
        for pred, gy, gc in ((tr.head_pred, TOP_Y, TOP_C), (tr.decoder_pred, BOT_Y, BOT_C)):
            wrong = list(zip(*np.nonzero(pred != tr.query_output), strict=True))
            outline_cells(ax, wrong, PRED_X, gy, CELL, BAD, alpha=a)
            ok = not wrong
            label(ax, PRED_X + GRID_W + 0.4, gc, "✓" if ok else "✗", a, size=26,
                  color=GOOD if ok else BAD, fontweight="bold")


def save_gif(tr: PipelineTrace, path: str | Path, fps: int = 10, dpi: int = 72) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    anim = FuncAnimation(fig, lambda t: draw_frame(ax, tr, t), frames=NUM_FRAMES, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
