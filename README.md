# LPN Interpretability

Extends Latent Program Networks (Macfarlane & Bonnet) with an interpretable
readout. An MLP head decodes the latent program `z` into an explicit 2x2
color tile, and a hardcoded executor (FIND the anchor, PASTE the tile) is
trained to reproduce the outputs from it. The testbed is a simplified
Pattern-2D task: 4x4 grids, one blue anchor, tile colors 1-9.

- `PROJECT-SPEC.md` is the research spec.
- `IMPLEMENTATION-PLAN.md` holds the build plan and the decisions that
  override the spec.
- `CLAUDE-CODING-SKILL-10012026.md` lists coding lessons carried over from
  the previous (LoRA) project.

## Layout

```
src/lpn_interp/
  data/pattern2d.py       task generation (tiles, distinct anchors, held-out tile split)
  models/                 encoder (per-pair + mean/LOO pooling), decoder (prefix-LM), MLP head, LPN
  execution/executor.py   FIND / PASTE (differentiable in the soft tile)
  training/losses.py      L_dec, L_exec, L_commit, L_KL, training-time LOO latent steps
  configs/                arch1, vanilla (baseline), probe (+ _mini smoke versions)
  utils/                  runtime (device/AMP/output root), resume, logging, seeding
scripts/
  colab_runtime.py        drive Colab GPU VMs via the Colab CLI (run from WSL)
  gpu_smoke.py            GPU sanity check + throughput probe
tests/                    pytest suite (CPU)
docker/                   CPU dev image for local tests
```

## Running

- **GPU (Colab CLI, from WSL):** see `docs/colab_workflow.md`. Short version:
  ```bash
  python3 scripts/colab_runtime.py up --gpu A100
  python3 scripts/colab_runtime.py run -- -m pytest -q
  python3 scripts/colab_runtime.py down
  ```
- **Local tests (Docker, CPU):**
  ```powershell
  docker compose -f docker/docker-compose.yml run --rm dev pytest
  ```

## Adding a dependency

1. Add it to `docker/requirements/base.in`.
2. Run `pwsh scripts/compile_requirements.ps1`.
3. If Colab doesn't preinstall it, also add it to `REQUIRED_MODULES` in
   `scripts/colab_runtime.py`.
