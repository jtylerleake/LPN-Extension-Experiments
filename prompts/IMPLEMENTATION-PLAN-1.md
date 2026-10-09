# Implementation Plan — LPN Interpretability, Architecture 1

Source: `PROJECT-SPEC.md`. Architecture 2 is out of scope for now.
Coding conventions: `CLAUDE-CODING-SKILL-10012026.md`.

## 0. Confirmed decisions (these override the spec where they conflict)

| Topic | Decision |
|---|---|
| Framework | PyTorch, reimplemented from scratch (no JAX, no upstream `lpn` repo) |
| Grid / tile | 4x4 grid, 2x2 tile, n = 4 context pairs |
| Anchor | Fixed blue (1). FIND = the input pixel equal to 1 |
| Anchor positions | rows/cols {0,1,2} (9 positions), so the tile always fits |
| Tile colors | 1–9 only (no black), giving 9^4 = 6,561 programs |
| Latent dim | d = 2 everywhere |
| Head training | Jointly through L_exec only (no ground-truth tile labels) |
| Infra | Docker (CPU tests) + Colab GPUs via the **Colab CLI from WSL** (`colab/colab_runtime.py`). No notebooks |

## 1. Target layout

```
src/
  data/pattern2d.py        task sampler (tile, distinct anchors), torch tensors, seeded
  models/encoder.py        per-pair transformer -> (mu_i, logvar_i); pooling + LOO helpers
  models/decoder.py        causal transformer p(y | x, z); teacher-forced loss + greedy decode
  models/mlp_head.py       h(z): d -> H -> H -> 4x10 logits -> soft tile S_hat
  models/lpn.py            LPN = encoder + decoder + head; forward_train / encode_context
  execution/executor.py    find_anchor(x), paste(S_hat, x, anchor) -> per-cell distributions
  training/losses.py       L_dec, L_exec, L_commit, L_KL
  training/train.py        training loop (AMP, checkpoints, metrics.jsonl, resume)
  inference/latent_opt.py  test-time latent optimization (weights frozen, z requires grad)
  eval/evaluate.py         tile recovery, reconstruction, entropy-vs-step, agreement, trajectories
  config.py                pydantic config + from_yaml
  configs/arch1.yaml, arch1_mini.yaml, vanilla.yaml, vanilla_mini.yaml
  utils/  seeding.py, logging_utils.py, colab.py, resume.py   (trimmed from the existing code)
scripts/  train.py, evaluate.py, plot_results.py
colab/    colab_runtime.py (WSL driver), vm_setup.py, requirements.txt, gpu_smoke.py, check_port_parity.py
docker/   CPU dev image, requirements.in/.txt, compile_requirements.ps1
tests/    test_data.py, test_executor.py, test_model.py, test_losses.py, test_latent_opt.py, test_config.py
```

## 2. Components

**Data.** For each task, sample a tile T in {1..9}^(2x2) and n+1 *distinct* anchors from the 9 valid positions. Pairs 1..n are context and pair n+1 is the query. Distinct anchors mean no query duplicates a context pair (the skill file's clean-round lesson). Training data is generated on the fly on the GPU. Eval sets are fixed and seeded. Each task's RNG is derived from (seed, task_id) so runs can resume.

**Encoder.** Each pair is 32 tokens (16 input + 16 output), embedded as color + 2D position + input/output-type embeddings, plus a CLS token. A small transformer (around 2 layers, dim 128) maps CLS to (mu_i, logvar_i) in R^2.
- **LOO, done efficiently:** encode all n pairs once, then `mu_{-i} = (sum(mu) - mu_i)/(n-1)` (same for logvar). This is O(n), not n separate encoder passes.
- A unit test checks it against brute-force re-encoding.

**Decoder.** Tokens are `[z-token, 16 input tokens, 16 output tokens]`, with causal masking on the output segment. Training uses teacher forcing. Eval uses greedy decoding, 16 steps (output size is fixed, so no shape tokens).

**MLP head.** `Linear(2,H)-ReLU-Linear(H,H)-ReLU-Linear(H,40)`, reshaped to (4,10), then softmax per cell. H = 128. Its forward pass takes only z.

**Executor (vectorized, no Python loops).**
- `find_anchor`: argmax over `x.flatten()==1`, with an assert that exactly one blue pixel exists.
- `paste`: start Y_hat as one-hot(x) over 10 colors, then scatter S_hat into the 4 cells at anchor + {(0,0),(0,1),(1,0),(1,1)} using gather indices.
- The output is a (16,10) probability grid. L_exec is `NLL(log(Y_hat + eps), y)`. Cells outside the tile are exact one-hots and contribute ~0 loss.

**Losses.** `L = L_dec + λ_exec·L_exec + λ_commit·L_commit + β·L_KL`, with defaults λ_exec=1.0, λ_commit=0.1, β=1e-4. All losses are summed over pairs and averaged over the batch. The KL term uses the LOO posteriors.

## 3. Two code paths (kept strictly separate)

**Training** (`training/train.py`), per batch of tasks:
1. Encode with LOO.
2. Reparameterize to get z0^(i).
3. Compute L_dec, L_exec, L_commit, and KL.
4. Backprop into the encoder, decoder, and head.
5. Optional training-time latent steps (`train_latent_steps`, default 0).

**Test time** (`inference/latent_opt.py`):
1. Set z0 = mean of mu over the n context pairs (no LOO).
2. Run K ascent steps on the context objective. `z` is a leaf tensor with `requires_grad`, and the model is in `eval()` with `requires_grad_(False)`.
3. Record z_k, the objective, and H(S_hat) at every step.
4. Decoder path: `y_hat = decode(x_query, z*)`. Head path: `T_hat = argmax h(z*)`.

The whole task batch is optimized at once, one z per task, which is fully batched on the GPU.

## 4. Evaluation outputs

Every metric is reported at z0 (k=0) and at z* (k=K):
- **Tile recovery:** cell-level and full-tile accuracy.
- **Reconstruction:** pixel and full-grid accuracy on the query pair.
- **Entropy:** H(S_hat) vs. step k, mean ± CI, plus the fraction of tasks where it decreases monotonically.
- **Decoder–executor agreement:** a 2x2 table of decoder correct/incorrect vs. executor correct/incorrect.
- **Latent trajectories:** z_0..z_K plotted in 2D for a handful of tasks, over a background showing the head's argmax tile region (d=2 makes this possible).
- **Baseline comparison:** reconstruction accuracy of `arch1` vs. `vanilla` (λ_exec = λ_commit = 0).

Each result row goes to `metrics.jsonl` on Drive with a fingerprint, so a resumed run skips finished rows (pattern reused from the current `resume.py`).

## 5. Cleanup (part of Phase 1)

**Delete:**
- `src/config/`, `src/data/`, `src/eval/metrics.py`, `src/models/registry.py`, `src/training/*`, `src/lpn_exp/*`
- every LoRA/exp4 script (`run_experiment.py`, `run_exp4.py`, `tune_exp4_lr.py`, all `plot_*`, `exp4_plot_style.py`)
- all `notebooks/exp*` notebooks
- `tests/test_smoke.py`

**Keep and adapt:**
- `utils/seeding.py`
- `utils/logging_utils.py` (drop the HF-noise helper)
- `utils/runtime.py` replaces `utils/colab.py` (device, AMP dtype, output root)
- the core of `resume.py`
- `compile_requirements.ps1`
- the Docker files
- (all notebooks removed: GPU runs go through the Colab CLI, see `colab/README.md`)

**Dependencies:** remove transformers, peft, accelerate, bitsandbytes, datasets, and safetensors from `base.in`, then regenerate `base.txt` and `colab.txt`.

**Docs:** rename the project in `pyproject.toml` and rewrite `README.md` and the Colab workflow doc (drop the HF-token step).

## 6. Phases

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | Cleanup + skeleton + config schema | Repo contains only the new layout; ruff and pytest (empty) pass in Docker |
| 2 | Data + executor | Tests pass: anchor validity, distinct anchors, colors 1–9, `paste(one_hot(T)) == y` exactly |
| 3 | Encoder/decoder/head/LPN + losses | Tests pass: shapes, LOO == brute force, causal mask (no leakage), head never receives x |
| 4 | Training loop + mini config | CPU smoke run: loss falls over ~50 steps; checkpoint saves and resumes |
| 5 | Latent optimization + evaluation | Tests pass: weights unchanged after opt, z changes, objective rises on the toy model |
| 6 | Plots + Colab CLI runbook | Mini configs run end to end via `colab_runtime.py launch` |
| 7 | Full runs | `vanilla` and `arch1` trained; results table and figures produced |

Phases 1–5 are verified in the Docker CPU image. If Docker isn't running, I'll say tests were not run (per the skill file).

## 7. Resolved decisions (2026-10-06)

1. **Held-out tiles:** yes. A fixed, seeded 10% of the 6,561 tiles never appear in training. Eval reports seen and unseen tiles separately.
2. **Post-hoc probe:** yes. `probe` mode trains a fresh head on the *frozen* vanilla LPN's z (detached) with L_exec + L_commit.
3. **Test-time optimizer:** plain SGD with gradient clipping (norm 1.0) and best-step selection. K = 100, lr tuned on a separate seed.
4. **Model size / budget:** 2 layers, dim 128, batch 256, 100k–200k steps.

## 8. Pivot: pretrained backbone (2026-10-06)

The authors' Pattern-2D checkpoint (`clement-bonnet/lpn-2d`, `quiet-thunder-789--checkpoint:v0`)
was trained on lpn's `PatternTaskGenerator`. That generator is exactly our task: 4x4 grids,
a blue anchor at rows/cols {0,1,2}, and tile colors 1-9. Its anchors may repeat within a
task, so our distinct-anchor tasks are a subset of its training distribution. Its latent
is 2-dimensional.

Decisions:
- **Port, not JAX.** The paper's architecture is ported to PyTorch (`src/models/`) and
  *replaces* our custom encoder/decoder everywhere. Weights are converted from
  `state.msgpack` with plain msgpack (`src/models/pretrained.py`). The download is pinned
  to an HF revision and checked against its sha256.
- **The paper's semantics are adopted:**
  - one latent is sampled per pair, and the leave-one-out context is the mean of the
    other pairs' samples;
  - the per-pair loss is the shape CE plus the mean cell CE;
  - beta defaults to 1e-3 (the checkpoint's KL coefficient);
  - L_exec and L_commit are per-cell means;
  - test-time search sums over pairs, so the paper's lr 0.1 applies.
- **Deviations, exact for 4x4 grids:** no padding masks; output-shape tokens are
  teacher-forced to 4x4, and shape accuracy is reported as a diagnostic.
- **Stages:**
  1. frozen + head (`mode: probe, init: pretrained`);
  2. all-weights fine-tune at lr 3e-5 for 10k steps (`joint, pretrained`);
  3. retrain from scratch (`joint, scratch`).

  We advance between stages by reviewing results, with no fixed threshold.
- **Verification:**
  - local tests check the conversion round trip, the mask structure, and no target
    leakage;
  - the real checkpoint reaches >90% query pixel accuracy at z0 with no search (≈98%;
    75% of query grids fully correct);
  - `colab/check_port_parity.py` compares against the original JAX code on Colab
    (lpn commit 0adfe56).

## 9. Runtime isolation (2026-10-07)

- **Docker (`docker/`)** is the local dev/test runtime: its own requirements, a lock
  script, and a README.
- **Colab/WSL (`colab/`)** is the real runtime: the driver, `requirements.txt`
  (consumed by `colab/vm_setup.py` on the VM), the parity-check extras, and the
  Colab-only scripts.
- **The contract between them:** every Colab requirement is also a Docker requirement
  (`tests/test_requirements.py`).
- **Shared code** (`src/`, `scripts/train.py`, `tests/`) is runtime-agnostic.
- `tqdm` was dropped as unused.
