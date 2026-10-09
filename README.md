# LPN Interpretability

Extends Latent Program Networks (Macfarlane & Bonnet) with an interpretable
readout. An MLP head decodes the latent program `z` into an explicit 2x2
color tile, and a hardcoded executor (FIND the anchor, PASTE the tile) is
trained to reproduce the outputs from it. The testbed is a simplified
Pattern-2D task: 4x4 grids, one blue anchor, tile colors 1-9.

- `prompts/PROJECT-SPEC-1.md` is the research spec.
- `prompts/IMPLEMENTATION-PLAN-1.md` holds the build plan and the decisions that
  override the spec.
- `CLAUDE-CODING-SKILL-10012026.md` lists coding lessons carried over from
  the previous (LoRA) project.

## Experiment stages

The backbone is the authors' pretrained Pattern-2D LPN (`clement-bonnet/lpn-2d`,
trained on exactly this task), ported to PyTorch. The pretrained model is also
the vanilla-LPN baseline.

1. **Frozen** (`stage1_frozen`): train only the MLP head on the frozen backbone's z.
2. **Light fine-tune** (`stage2_finetune`): all weights, low LR, joint loss.
3. **From scratch** (`stage3_scratch`): the same architecture, randomly initialized.

We move to the next stage after reviewing each stage's results.

**Architecture 2** (`model.architecture: 2`, run frozen as `stage1_frozen_arch2`) adds a
second head g(z) that outputs a 4x4 anchor map, replacing the hardcoded FIND: the program
is (tile, anchor). It's deliberately naive -- g sees only z, which carries no per-pair
anchor -- and is expected to fail; eval adds `anchor_acc` and `anchor_valid`.

**Head sweep** (`scripts/sweep_head.py`, stage 1 only): which MLP head reads the tile out of the
frozen z best. The encoder is frozen, so every (tile, anchor) pair is encoded once and runs train
only the head. Round A sweeps depth x width, round B learning rate x steps x activation for the
3 best heads, round C reruns the 3 best with 3 seeds. The held-out tiles are split into
validation (ranks configurations) and test (scored only in round C).

## Layout

The project code is runtime-agnostic. Each runtime keeps its own
requirements, scripts and docs in its own folder.

```
src/                      the project code (runs anywhere)
  data/pattern2d.py         task generation (tiles, distinct anchors, held-out tile split)
  models/                   PyTorch port of the paper's LPN (encoder, decoder, transformer), MLP heads;
                            pretrained.py loads the authors' Pattern-2D checkpoint (no JAX needed)
  execution/executor.py     FIND / PASTE (differentiable in the soft tile; soft-anchor PASTE for arch 2)
  training/                 losses (L_dec, L_exec, L_commit, L_KL) + resumable training loop
  eval/evaluate.py          tile recovery / reconstruction / agreement metrics
  configs/                  stage1_frozen(_arch2), stage2_finetune, stage3_scratch (+ _mini smoke versions)
  utils/                    runtime (device/AMP/output root), resume, logging, seeding
scripts/train.py          train one run; rerunning the same command resumes it
scripts/sweep_head.py     stage-1 head sweep (rounds A/B/C + summary); rerunning a round resumes it
tests/                    pytest suite (CPU; runs in Docker or on the VM)

colab/                    GPU runtime: Colab VMs driven by the Colab CLI from WSL  -> colab/README.md
  colab_runtime.py          the driver (stdlib only; runs in WSL)
  requirements.txt          what the VM must have (read by `setup`)
  requirements-parity.txt   JAX-side extras for the parity check only
  vm_setup.py               runs on the VM: installs what's missing, never touches Colab's torch/jax
  check_port_parity.py      our port vs. the original JAX model (Colab only)
  gpu_smoke.py              GPU sanity check + throughput probe

docker/                   local dev + testing: CPU image  -> docker/README.md
  Dockerfile.dev, docker-compose.yml
  requirements.in / .txt    runtime libraries (superset of colab/requirements.txt) + dev tools
  compile_requirements.ps1  locks requirements.txt
```

## Dependencies

| Runtime | List | Versions |
|---|---|---|
| Colab (real runs) | `colab/requirements.txt` | Colab's preinstalled builds; missing packages installed; torch never replaced |
| Docker (dev/tests) | `docker/requirements.in` -> `requirements.txt` | locked by pip-compile; CPU torch |
| WSL (driver) | none: stdlib + the `colab` CLI | |

To add a dependency the code needs at run time, put it in **both**
`colab/requirements.txt` and `docker/requirements.in`;
`tests/test_requirements.py` fails if Colab needs something Docker lacks.
Then re-lock: `pwsh docker/compile_requirements.ps1`. Dev-only tools go in
Docker's list only.

## Running

- **GPU (Colab CLI, from WSL):** see `colab/README.md`. Short version:
  ```bash
  python3 colab/colab_runtime.py up --gpu A100 --drive
  python3 colab/colab_runtime.py run -- -m pytest -q
  python3 colab/colab_runtime.py down
  ```
- **Local tests (Docker, CPU):** see `docker/README.md`.
  ```powershell
  docker compose -f docker/docker-compose.yml run --rm dev pytest
  ```
