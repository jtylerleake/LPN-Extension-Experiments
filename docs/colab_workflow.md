# Running on Colab GPUs (Colab CLI, from WSL)

GPU work runs on Google Colab VMs driven by the
[Colab CLI](https://pypi.org/project/google-colab-cli/) (`colab`), not
browser notebooks. The CLI is Linux/macOS only, so everything below runs
in a **WSL terminal** from the repo's Windows checkout:

```bash
cd /mnt/c/Users/Jtyler/Downloads/Research-Public/Research-Code/LPN-Extension-Experiments
```

`scripts/colab_runtime.py` wraps the CLI. It needs only the stdlib, so
WSL's system `python3` is enough and you don't need a local env.

## One-time setup

- Install and authenticate the CLI (`uv tool install google-colab-cli`).
- Check auth with `colab sessions`. It should list sessions or say there
  are none, not fail with 401/403.

## Typical session

```bash
R="python3 scripts/colab_runtime.py"

$R up --gpu A100            # new VM "lpn" + upload code + check deps (T4/L4/G4/A100/H100/cpu)
$R run -- -m pytest -q      # the test suite, on the VM
$R run -- scripts/gpu_smoke.py   # GPU sanity check + throughput numbers

$R launch arch1 -- scripts/train.py --config src/lpn_interp/configs/arch1.yaml
$R jobs                     # state of every background job
$R logs arch1 --follow      # tail until it finishes (Ctrl-C stops tailing, not the job)
$R kill arch1               # stop a job

$R fetch                    # VM's /content/outputs -> ./outputs/ (add --no-checkpoints to skip *.pt)
$R down                     # release the VM. Do this every time: idle VMs burn compute units
```

`-s NAME` (before the subcommand) picks a session other than `lpn`, so
several VMs can run side by side.

## What happens under the hood

- **Code sync.** `run` and `launch` re-upload the working tree first, so a
  local edit is live on the next command. `--no-sync` skips this.
  - The upload covers tracked and untracked-but-not-ignored files under
    `src/`, `scripts/`, `tests/`, and `pyproject.toml`, so it includes
    uncommitted changes.
  - It lands in `/content/lpn`, with a `BUNDLE.json` recording the git
    commit, a dirty flag, and a content hash. Every run logs that manifest.
  - Sync refuses to replace code under a running job; `--force` overrides.
- **Processes.** Code always runs as a subprocess of the session's Jupyter
  kernel, never inside it. The kernel persists between `colab exec` calls,
  so in-kernel imports would keep serving stale modules after a re-sync.
- **Foreground vs. background.**
  - `run` streams output and returns the script's exit code. It's capped by
    `--timeout` (default 1 h), and the run is tied to your terminal's
    connection.
  - `launch` starts a detached process group on the VM that survives the
    CLI disconnecting. It writes `/content/jobs/<name>.log`, plus
    `<name>.exit` when done.
  - Use `launch` for training.
- **Dependencies.** `setup` (part of `up`) only installs packages Colab
  lacks. It never upgrades or downgrades Colab's preinstalled, CUDA-matched
  `torch` (see the coding-skill file §6).
- **Mixed precision.** AMP picks bf16 on L4/A100/H100 and fp16 with a
  GradScaler on T4, which has no bf16 tensor cores
  (`src/lpn_interp/utils/runtime.py`).

## Outputs: VM disk vs. Google Drive

By default, jobs write under `/content/outputs` on the VM (exported to the
script as `$LPN_OUTPUT_ROOT`). That disk disappears with the VM, so `fetch`
before `down`.

For long runs that should survive a VM being reclaimed, write straight to
Drive:

1. In your WSL terminal, run `colab drivemount -s lpn`. It's interactive (a
   browser consent step), so an agent can't do it for you.
2. Pass `--drive` to `run` / `launch` / `fetch`. The output root becomes
   `/content/drive/MyDrive/lpn_interp_outputs`. Training checkpoints and
   metrics rows are written resumably (see `src/lpn_interp/utils/resume.py`),
   so re-launching the same command after a disconnect continues where it
   left off.

## Troubleshooting

- **401/403 from any `colab` command:** this is CLI auth, not the VM. Run
  `colab whoami`, and see `colab skill` for the scopes ADC needs.
- **"Session not found":** the backend reclaimed the VM. Run `colab sessions`,
  then `up` again. VM-local outputs are gone; Drive outputs survive.
- **`up --gpu X` fails with 400:** there's no quota for that accelerator
  right now. Try another (`--gpu T4`, `L4`).
- **Wedged kernel:** run `colab restart-kernel -s lpn`. Background jobs keep
  running, since they're in their own process group.
