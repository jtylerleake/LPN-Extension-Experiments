# New Grid Search Skill

How to run (and re-purpose) the stage-1 MLP-head grid search for Architecture 1. It was written
down so the same framework can be reused when other things change in future runs.

Code: `src/training/head_sweep.py` (rounds, training, ranking), `scripts/sweep_head.py` (CLI),
`src/viz/sweep_plots.py` + `scripts/plot_sweep.py` (figures), tests in `tests/test_head_sweep.py`
and `tests/test_sweep_plots.py`.

---

## 1. Goal and ground rules

**Goal:** find the MLP head h(z) -> 2x2 tile that performs best on *held-out tiles* while
memorizing the training tiles as little as possible.

**Why:** in the first sweep (`outputs/sweeps/head_sweep`), the vanilla head (depth 2, width 128)
scored very low, and the "optimized" head (depth 6, width 1024) looked strong only because it
memorized: training-tile full accuracy ~1.0 vs. validation <= 0.46.

**Fixed (never change these in a sweep):**
- the evaluation setup, the dataset, the metrics, the tile splits
- compute: Colab GPU driven by the Colab CLI (`colab/colab_runtime.py`)
- activation: **ReLU only** (SiLU is out; ignore the old SiLU runs)
- training length: **25k steps per run**, not swept (50k is too long). LR warmup is 500 steps,
  then a cosine decay over the run, so a shorter run anneals sooner instead of being cut off.
- batch 256, 4 pairs per task, grad clip 1.0

**Only the architecture / its regularization is searched.**

## 2. Metrics (what "accuracy" means)

Each evaluation task gets **one** predicted 2x2 tile, decoded from z0 = the mean of its 4 context
pairs' mu. Architecture 1's FIND is hardcoded, so cells off the tile are copied exactly and only
the 4 tile cells can be wrong.

| metric | per task | then |
|---|---|---|
| `tile_cell_acc` | correct cells / 4 (0, .25, .5, .75, 1) | averaged over tasks |
| `tile_full_acc` | 1 if all 4 cells right, else 0 | averaged over tasks |

- `tile_full_acc` equals full-grid exact match for Architecture 1.
- Splits: `seen` = training tiles (2,048 tasks); `val` = half of the 656 held-out tiles
  (328 tiles x 8 tasks = 2,624 tasks); `test` = the other half, scored **only in round C**.
- **Ranking: validation `tile_cell_acc`**, then `tile_full_acc` breaks ties. Cell accuracy gives
  partial credit (a 3/4 prediction beats a 1/4 one); full accuracy collapses both to 0.
- **Memorization** = `seen` minus `val` accuracy. Always report it alongside validation. (In v1 the gap
  was ~50-60 points, so a "5-point gap" threshold is not realistic; rank by val, show the gap.)

## 3. The rounds (sweep `head_sweep_v2`)

Every round skips runs already in `results.jsonl`, so rerunning a command resumes it.

| round | what | runs |
|---|---|---|
| **A** | depth {2,3,4,6} x width {128,256,512,1024} x lr {1e-3,3e-3,1e-2}; weight decay 1e-3, dropout 0, residual blocks for depth >= 4 | 48 |
| **B** | the 3 best round-A configs ("bases"), varying **one setting at a time**, others held at the base's: weight decay {0,1e-4,1e-3,1e-2}, dropout {0,0.1,0.3}, residual flipped | 18 new (the bases reuse their A runs) |
| **B2** | per base, the best value of each round-B setting combined into one run | 3 |
| **C** | the 3 best distinct configs of A+B+B2, seeds {0,1,2}, scored on val **and test** | 9 |

Decisions behind this:
- **lr {1e-3, 3e-3, 1e-2}:** 3e-4 was dropped because it hadn't converged even at 50k steps, so
  25k would unfairly penalize it.
- **Two stages (size, then regularization):** this avoids a full grid. All four regularization
  settings crossed would be ~216 runs per head.
- **Weight decay 1e-3 in round A:** round B sweeps it anyway.
- **Not swept:** latent-noise scale (considered, then dropped), batch size, parameter-budget shapes,
  training steps, activation.

## 4. Running it (Colab CLI, from WSL)

```bash
R="python3 colab/colab_runtime.py"
$R up --gpu A100 --drive

# smoke test first (caps steps, writes to head_sweep_v2_smoke)
$R run -- scripts/sweep_head.py --round A --max-steps 200

# the real rounds, in order (each refuses to start until the previous one is complete)
$R launch sweepA  -- scripts/sweep_head.py --round A
$R watch
$R launch sweepB  -- scripts/sweep_head.py --round B
$R launch sweepB2 -- scripts/sweep_head.py --round B2
$R launch sweepC  -- scripts/sweep_head.py --round C

$R run -- scripts/sweep_head.py --round summary   # ranked table
$R run -- scripts/sweep_head.py --round plots     # standard figures
$R fetch --no-checkpoints
$R down
```

After a disconnect: `$R resume` then `$R watch`. Finished runs are kept and only the run in
progress is lost.

**Progress (tqdm):** an outer bar counts runs in the round (finished runs count from the start),
with time remaining, the best val `tile_cell_acc` so far, and the last run's duration. An inner bar counts
the current run's steps, with steps/s, and loss / `val_cell` / `val_full` updated at each eval.
The per-step curve is saved in each row's `curve` field.

**Output:** `<output root>/sweeps/head_sweep_v2/results.jsonl`, one JSON row per finished run:
`spec`, `final` (seen/val[/test] x cell/full acc), `curve` (val every 2.5k steps), `seconds`,
`params`, `round`, `fingerprint`.

## 5. Plots

`--round plots` writes `<sweep dir>/plots/`:

| file | x-axis | lines | panels |
|---|---|---|---|
| `A_depth_by_width_per_lr.png` | depth | width | lr |
| `B_weight_decay_by_base.png`, `B_dropout_by_base.png`, `B_residual_by_base.png` | that setting | base (d/w/lr) | - |
| `C_final_test.png` | final config | - (points with min..max error bars over seeds) | - |

In every figure the y-axis is accuracy (default `tile_cell_acc`). **Solid lines (or circles) are
held-out** (val, or test in C) and **dashed lines (or diamonds) are training tiles**, so the
memorization gap is visible directly. When there are several seeds, the line is the mean and the
band shows min..max.

Custom figures:
```bash
python scripts/plot_sweep.py --x depth --hue width --facet lr --where round=A
python scripts/plot_sweep.py --x lr --hue depth --metric tile_full_acc --where round=A width=1024
```
`--where` must pin every setting that isn't x / hue / facet / seed, or a line averages across
configs. Derived keys `base` and `config` can be used as x / hue / facet.

## 6. Reusing the framework for a future sweep

1. **New sweep name.** Pass `--name head_sweep_v3` (or change the default in
   `scripts/sweep_head.py` and `scripts/plot_sweep.py`) so old results stay untouched.
2. **Bump `SWEEP_VERSION`** in `head_sweep.py` whenever a code change alters what a given spec
   trains. Fingerprints then differ and stale rows aren't reused.
3. **Change the grid:** edit the constants at the top of `head_sweep.py`: `DEPTHS`, `WIDTHS`, `LRS`,
   `STEPS`, `BASE_WEIGHT_DECAY`, `BASE_DROPOUT`, `WEIGHT_DECAYS`, `DROPOUTS`, `SEEDS`, `TOP_K`,
   `EVAL_EVERY`.
4. **Add a new architecture setting** (e.g. normalization, a different head shape):
   - add the field to `ModelConfig` in `src/config.py` (default = current behavior, so old
     checkpoints still load) and use it in `src/models/mlp_head.py`;
   - add it to `make_spec` (with its base value) and `head_config` in `head_sweep.py`;
   - to sweep it in round B, add it to `REG_KEYS` and add its values to `variations`. `best_combination`,
     the B plots, and the summary need no further changes (add a column to `summary_table` if wanted);
   - add tests (shape/behavior in `test_head_sweep.py`; a plot appears in `test_sweep_plots.py`).
5. **Ranking** lives in `rank_key` (val cell acc, then val full acc); change it only on purpose.
6. **Dependencies:** add runtime packages to both `colab/requirements.txt` and
   `docker/requirements.in`, then re-lock with `pwsh docker/compile_requirements.ps1`.
7. **Checklist before launching:** run the tests, do the `--max-steps` smoke on Colab, then run
   the full rounds.

## 7. Practical notes

- Local CPU checks: Docker (`docker compose -f docker/docker-compose.yml run --rm dev pytest`).
  Avoid full Docker runs (45 min+). With the local Anaconda Python, set
  `PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE` (Anaconda OpenMP conflict, local only).
- `tqdm==4.67.1` was pinned into `docker/requirements.txt` by hand while Docker was down.
  Re-lock when Docker is next up.
- Round B/B2/C pick their configs from earlier rounds' results, so run rounds in order on the
  same `results.jsonl`.
