# Claude coding skill — lora_experiments

Lessons from working in this repo, kept so future sessions don't re-discover
them the hard way. Update this file when a new lesson earns its place —
don't let it become a changelog of every session.

## 1. Colab + tqdm + subprocess don't mix

**Never run training code via `!python script.py` (bang/subprocess) in a
Colab cell if it prints a tqdm progress bar.** Subprocess stdout piping
breaks tqdm's in-place `\r` overwrite: Colab's output-capture renders every
bar refresh as its own new printed line instead of overwriting the same
line. The result looks exactly like log spam — a wall of near-identical
lines — and is easy to misdiagnose as "tqdm isn't showing up at all," when
it's actually rendering correctly, just once per step instead of in place.

**Fix: call the entrypoint in-process instead.**
```python
import sys
sys.path.insert(0, "scripts")
from run_experiment import main as run_experiment
run_experiment(["--config", ..., "--device", "cuda"])
```
Running inside the live kernel process lets tqdm overwrite its line
correctly, and lets `tqdm.auto` pick the ipywidgets-based notebook bar
(Colab ships ipywidgets by default) instead of falling back to plain
terminal tqdm.

**If the requirement is "the progress bar is the *only* thing that
renders":** don't chase down every `print()`/`log.info()` call one at a
time. Route the module's own console logger to a file for the duration of
the run instead:
- A `quiet_console(logger)` context manager that temporarily removes the
  logger's own console `StreamHandler` (it still propagates to the root
  logger, so anything captured by a `file_logging`-style context at the
  same time still lands in that file).
- Keep this behavior scoped to the *real* run path only — a `--dry-run`/
  CPU/CI path usually has no progress bar competing for the console and
  should keep printing normally (tests may assert on that console output).

**Library noise (transformers/datasets/huggingface_hub) is a separate
problem and doesn't go through Python's `logging` module at all** — model
config dumps, "some weights were not initialized" boilerplate, and hub
download bars all bypass `logging`. Suppress these at the source, before
those libraries are first imported anywhere (including transitively):
`TRANSFORMERS_VERBOSITY=error`, `HF_HUB_DISABLE_PROGRESS_BARS=1`,
`DATASETS_VERBOSITY=error`, plus each library's own
`disable_progress_bar()` / `set_verbosity_error()` call. Redirecting
`logging` output after the fact never touches this.

## 2. Scaling down a training sweep — what actually moves the needle

In rough order of impact, cheapest-to-implement first:

1. **Epochs.** A straight multiplier on total runtime; 3→1 epoch is often
   the single biggest lever, especially once the dataset itself is capped.
2. **`eval_samples`.** Generation-based eval (autoregressive, one example
   at a time) is far slower than training-loss logging. Cutting eval
   sample count is nearly free — it doesn't change what's being trained,
   only how precisely accuracy is estimated.
3. **Per-task generation length.** A single global `max_new_tokens` sized
   for the hardest task (e.g. multi-step reasoning) forces every task to
   pay that generation budget, even simple classification tasks that only
   ever emit one word. Splitting this by task (or schema field
   `max_new_tokens_by_task` with a fallback) is a real win with no
   accuracy cost on the tasks that don't need it.
4. **Drop the most expensive method from broad sweeps.** Full fine-tuning
   (updates every parameter + full optimizer state) dominates the compute
   budget of any sweep it's part of. Keep it only where it's the actual
   point of the experiment (a direct method-comparison baseline); drop it
   from sweeps that are really about a different axis (rank, target
   modules) and would otherwise be paying for a baseline nobody's
   comparing against apples-to-apples anyway.
5. **Thin a parameter grid.** If a sweep varies one axis across N values
   just to show a trend (e.g. LoRA rank across 6 values), 3 well-chosen
   points (low/mid/high) usually tell the same story for a fraction of the
   runs.
6. **Cap oversized dataset splits**, not all of them uniformly — check
   actual split sizes first. A single global cap sized for the largest
   split is a no-op for splits already smaller than it.
7. **Batch size, last.** Raising `per_device_train_batch_size` (memory
   permitting) reduces step overhead without changing epochs or examples
   seen — a genuine win, but the smallest one on this list, and the first
   thing to revert if you hit an OOM.

## 3. Config/test coupling in this repo

- Every experiment YAML is validated through a pydantic schema
  (`config/schema.py`), and "mini" smoke-test configs are asserted (in
  tests) to structurally mirror their "full" counterpart — same method
  list, same order, same types/ranks, just capped training numbers.
  **Changing a full config's method list requires changing its mini
  counterpart in the same edit**, and updating any test that hardcodes the
  old values (rank lists, method-type sets, run counts in
  docstrings/notebook markdown).
- **Before removing a method from a sweep, grep for it in every
  downstream consumer**, not just the config. A plotting script can have a
  hard semantic dependency on that method being present (e.g. a heatmap
  normalized against a `full_ft` baseline) — removing the data source
  doesn't crash the script, it just silently produces an empty/meaningless
  plot. That's a real design decision, not a mechanical edit: surface it
  and ask rather than deleting it quietly.
- **A glob-based test sweep is a coupling point too.** `tests/test_smoke.py`
  validates every `experiment_*.yaml` through the shared pydantic schema —
  a new experiment whose config *doesn't* fit that schema (e.g. it isn't
  even the same framework, see §6) needs a filename that deliberately
  doesn't match the glob, not a schema workaround.
- **Check for existing WIP state in a file before overwriting it
  wholesale.** A config may already be mid-edit (e.g. methods commented
  out) from earlier work. Diffing that state against what its own tests
  and its mini/full counterpart expect is the fastest way to tell
  "intentional and finished" from "abandoned mid-edit" — don't assume the
  file on disk reflects a deliberate, finished decision just because it's
  already there.

## 4. Tooling notes for this environment

- **Editing `.ipynb` files:** use `NotebookEdit`, not `Edit`/`Write`. The
  `Read` tool renders notebooks as pseudo-XML per-cell blocks (not raw
  JSON), and `NotebookEdit` operates on those same cell IDs.
- **No local Python/pytest outside Docker in this environment.** If Docker
  Desktop isn't running, there's no way to execute the test suite
  in-session — say so explicitly after a change instead of asserting tests
  pass without having run them.

## 5. Dataviz semantics follow the data's job, not just its range

Diverging color (two hues + a neutral midpoint) means "how does this
compare to a baseline" — it only makes sense when that baseline actually
exists in the data. Sequential color (one hue, light→dark) means plain
magnitude. **When a baseline that a chart normalized against gets removed
from the underlying data (e.g. dropping a `full_ft` reference run), the
chart's color encoding has to change with it** — swapping back to raw
values under a diverging/centered colormap silently produces a
meaningless plot, not an error.

## 6. Adapting a model that isn't PyTorch/HF (e.g. JAX/Flax)

Experiment 4 targets LPN, a JAX/Flax model — PEFT (used everywhere else in
this repo) only wraps PyTorch `nn.Module`s, so none of the existing
LoRA infra applies. A few things worth remembering for the next
non-PyTorch model:

- **"LoRA" without PEFT is just pytree math.** Flax modules are immutable —
  they take their params as a plain pytree argument to `.apply()`, they
  don't hold mutable weight attributes the way `torch.nn.Linear` does. So
  "adapting" one means patching that pytree (add `scale * A @ B` onto
  targeted kernel leaves, return a new pytree), not subclassing or wrapping
  any module. No changes to the vendored model's own source are needed —
  it's a pure function around its public `.apply()`.
- **Discover parameter paths by walking the live pytree; don't hardcode
  Flax's auto-generated submodule names** (e.g. `TransformerLayer_0`,
  `Dense_1`). Those names come from instantiation order in the model's
  source and are easy to get subtly wrong by inspection alone — a small
  recursive walk with a predicate (e.g. "every `kernel` leaf under a
  `MlpBlock` inside `decoder`") is both more robust and easier to verify
  than a guessed nested-dict literal.
- **Keep the framework-specific imports (`jax`, `flax`, `optax`) inside
  functions, not at module top-level**, same convention this repo already
  uses for `torch`/`transformers`/`tqdm` (see `scripts/run_experiment.py`,
  `src/training/common.py`). This keeps the new module importable — and its
  pure, framework-agnostic logic (the pytree patch itself, since `@` and
  `+` work identically on numpy or jax arrays) unit-testable with plain
  numpy — in this repo's Docker CPU image, which has no jax/flax installed
  and isn't getting them just for one experiment's dry-run path.
- **A checkpoint's bundled config can reference stale internal paths.**
  LPN's own checkpoint config used `_target_: src_v2.models.utils...` (an
  old internal module name) — instantiating it via Hydra's `_target_`
  resolution would have failed against the current repo layout. Reading the
  plain nested fields and constructing the config dataclasses directly
  (ignoring `_target_`) sidesteps this and doesn't need Hydra as a
  dependency in our own driver code at all.
- **Give the new model its own config schema and entrypoint script rather
  than shoehorning it into the shared pydantic `ExperimentConfig`.** A
  schema built around `model.name` + PEFT `target_modules` will actively
  mislead for a model where neither concept exists — a small, separate
  config class (still pydantic, still a `from_yaml` classmethod, for
  consistency) is clearer than adding a pile of "ignored unless you're
  experiment N" optional fields to the shared one.
- **A vendored repo's pinned `jax`/`jaxlib` version can simply not exist
  for Colab's current Python anymore** (confirmed: `lpn`'s pinned
  `jaxlib==0.4.26`, from mid-2024, has no matching wheel today — `pip
  install -r requirements.txt` aborts the *entire* file when one pin can't
  be satisfied, silently skipping every other package in it, including
  ones with no version problem of their own like `chex`). Same principle as
  `colab_bootstrap.ipynb` already applies to `torch`: don't force a pinned
  `jax`/`jaxlib` over Colab's preinstalled, CUDA-matched build.
- **Don't install a vendored repo's `requirements.txt` on Colab at all.
  List what the code you actually import needs, and install only what
  Colab lacks.** Stripping just the jax lines wasn't enough. lpn's other
  mid-2024 pins still *downgraded* Colab's packages:
  - `huggingface_hub==0.24.6` broke Colab's `transformers`, `datasets` and
    `peft`, which our entrypoints import at startup (`silence_library_noise`).
  - `matplotlib==3.9.0` and `scikit-learn==1.5.1` had no wheels for
    Colab's Python, so pip compiled them from source.

  pip reports all this only as resolver "ERROR" lines, and the cell still
  "succeeds". The fix:
  - Grep the imports reachable from the modules you use. For exp4 that's
    numpy, matplotlib, sklearn, seaborn, networkx, PIL, torch and
    huggingface_hub, all preinstalled on Colab.
  - Install only the rest (flax/optax/chex), pinned to the versions you
    verified locally. Check their `jax>=` floor against Colab's jax.
  - Pin the vendored repo's clone to a commit.
  - End the install cell by importing everything the pipeline uses, your
    own entrypoints included, so a broken install fails right there instead
    of cells later.
- **Unjitted JAX on a small model is dispatch-bound, not compute-bound —
  `jax.jit` the whole per-task computation, and build it once.** Calling
  `model.apply` / `jax.value_and_grad` eagerly in Python loops (over
  leave-one-out rounds and adaptation steps) left exp4's A100 mostly idle;
  jitting each condition's whole per-task function (`vmap` over rounds,
  `lax.scan` over steps) measured 40-230x faster per task on CPU after a
  few seconds' one-time compile. Two things make or break that: construct
  the jitted function **once per condition, outside the task loop** (a
  fresh `jax.jit(closure)` per task recompiles every task and throws the
  win away), and pass params as **arguments**, not closure constants. To
  keep results comparable, derive random keys inside the jitted function
  in the same order the old loop split them (`lax.scan` over
  `jax.random.split`), then check equivalence with `jax.disable_jit()`:
  it should match the old eager code exactly. The jitted version can still
  differ on the odd task, because XLA's float reassociation can flip a
  near-tie in the greedy decode. That's expected, not a bug.
- **When comparing a new adaptation method against a paper's baseline,
  match everything except the thing being compared.** Exp4's first LoRA
  version differed from the paper's gradient ascent on four axes at once:
  objective (leave-one-out + KL vs. all-pairs likelihood), gradient
  clipping, last step vs. best step, and resampled vs. fixed latent noise.
  So a gap between them couldn't be pinned on "weights vs. latent".
  Unclipped SGD at the latent's lr also diverged on some rounds (context
  loss up 180-320x). Read the baseline's optimizer code, not just its
  docstring, and reuse its exact chain, objective, and step selection.
  Then tune each condition's lr separately, on tasks from a different seed
  than the evaluation.
- **Check a procedurally generated eval for test examples that duplicate
  the context.** Pattern-2D has only 9 pattern positions, so ~30% of
  leave-one-out rounds hold out a pair identical to one of their own
  context pairs. Any method that fits the context is then trained on the
  test example, and higher-capacity methods benefit more. Report
  clean-only rounds next to the paper's all-rounds protocol, and select
  hyperparameters on clean rounds.
- **Resuming a sweep safely takes three pieces, not just "skip rows that
  exist".**
  - Fingerprint each row with every setting that affects its result, so a
    changed config redoes the work instead of reusing stale rows. Leave out
    settings like batch size that don't affect results.
  - Derive each task's random key from (seed, task_id) alone. A sequential
    `split` chain gives a resumed run a different key than an
    uninterrupted one.
  - When reading, drop a truncated last line and cut it from the file, or
    the next append glues onto the fragment.

  Put the skip check before the heavy imports, and the resume path becomes
  testable in the CPU image (see `test_run_exp4_resume_*`). When batching
  with `vmap`, pad a short final batch to the full size so there's one
  compile, not two.
