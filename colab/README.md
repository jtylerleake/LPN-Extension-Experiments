# Running on Colab GPUs (Colab CLI, from WSL)

GPU work runs on Google Colab VMs driven by the
[Colab CLI](https://pypi.org/project/google-colab-cli/) (`colab`), not
browser notebooks. The CLI is Linux/macOS only, so everything below runs
in a **WSL terminal** from the repo's Windows checkout:

```bash
cd /mnt/c/Users/Jtyler/Downloads/Research-Public/Research-Code/LPN-Extension-Experiments
```

`colab/colab_runtime.py` wraps the CLI. It needs only the stdlib, so
WSL's system `python3` is enough and you don't need a local env.

## One-time setup

- Install and authenticate the CLI (`uv tool install google-colab-cli`).
- Check auth with `colab sessions`. It should list sessions or say there
  are none, not fail with 401/403.

## Typical session

```bash
R="python3 colab/colab_runtime.py"

$R up --gpu A100 --drive    # new VM "lpn" + mount Drive (approve the consent prompt) + upload code + check deps
$R run -- -m pytest -q      # the test suite, on the VM

$R run -- colab/check_port_parity.py   # once: our port vs. the original JAX model
$R launch stage1 -- scripts/train.py --config src/configs/stage1_frozen.yaml
$R watch                    # poll every 5 min until jobs finish; tells you how to recover if the VM is lost
$R jobs                     # state of every job (this VM's and earlier VMs')
$R logs stage1 --follow      # tail until it finishes (Ctrl-C stops tailing, not the job)
$R kill stage1              # stop a job (training checkpoints first)

$R fetch --no-checkpoints   # outputs -> ./outputs/
$R down                     # release the VM. Do this every time: idle VMs burn compute units
```

`-s NAME` (before the subcommand) picks a session other than `lpn`, so
several VMs can run side by side.

## What happens under the hood

- **Code sync.** `run` and `launch` re-upload the working tree first, so a
  local edit is live on the next command. `--no-sync` skips this.
  - The upload covers tracked and untracked-but-not-ignored files under
    `src/`, `scripts/`, `tests/`, `colab/`, and `pyproject.toml`, so it includes
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
- **Dependencies.** `setup` (part of `up`) runs `colab/vm_setup.py` on the
  VM against `colab/requirements.txt`:
  - packages already installed are left alone;
  - missing ones are pip-installed;
  - a version that violates the file's specifier is an error;
  - packages marked `preinstalled` (torch) are checked but never installed,
    because Colab's CUDA-matched builds must not be replaced (coding-skill
    file §6).

  The parity check does the same for `colab/requirements-parity.txt`
  (jax/flax/chex/optax).
- **Mixed precision.** AMP picks bf16 on L4/A100/H100 and fp16 with a
  GradScaler on T4, which has no bf16 tensor cores
  (`src/utils/runtime.py`).

## Mounting Google Drive

`up --drive` mounts Drive for you, and `mount` does the same on an existing
session. Under the hood it's `colab drivemount -s lpn`:

- It prints a Google consent link. Approve it in your browser, and Drive
  appears at `/content/drive`.
- It needs you at the terminal, so an agent can't do it on your behalf.
- Every new VM needs it again.

Once mounted, the session remembers it. `run`, `launch`, and `fetch` then
default to the Drive output root `/content/drive/MyDrive/lpn_interp_outputs`.
`--no-drive` overrides this per command. Without Drive, outputs live on the
VM's own disk (`/content/outputs`) and vanish with the VM.

## Picking up after a disconnect

There are three layers. Together they mean a lost VM costs at most the work
since the last checkpoint (2,500 steps by default, about a minute on an A100).

1. **Run state lives on Drive.** Each training run writes
   `runs/<name>/{config.json, checkpoint.pt, metrics.jsonl, final.pt}` under
   the output root. `checkpoint.pt` holds the model, optimizer, LR schedule,
   AMP scaler, RNG state, and step, and is written atomically.
2. **Scripts resume themselves.** Re-running the same `scripts/train.py`
   command does one of three things:
   - continues from `checkpoint.pt`, cutting any metrics rows logged after it;
   - returns immediately if `final.pt` exists;
   - refuses if the config changed (`--restart` or `--name` to start over).

   Batches are derived from (seed, step), so a resumed run trains on exactly
   the data an uninterrupted one would. This is tested to be bit-identical
   on CPU.
3. **The driver remembers what was running.** Every `launch` is recorded in
   `.colab/state.json` (local, gitignored), which outlives the VM.

After a disconnect:

```bash
$R up --gpu A100 --drive    # fresh VM; it lists the unfinished jobs it found
$R resume                   # relaunches each unfinished job with its original command
$R watch
```

`resume` skips:
- jobs still running;
- jobs that finished;
- jobs you `kill`ed (name them to resume anyway: `resume stage1`);
- jobs that crashed with an error, so it doesn't loop on a bug (check `logs`,
  then name them or pass `--include-failed`).

`forget NAME` drops a job from the registry.

**Keeping sessions alive.** Colab keeps a session alive while its kernel
shows activity, and a detached training process may not count. `watch`'s
polls do count, so leave `watch` running in a WSL terminal during long runs.
## Troubleshooting

- **401/403 from any `colab` command:** this is CLI auth, not the VM. Run
  `colab whoami`, and see `colab skill` for the scopes ADC needs.
- **"Session not found" / `watch` lost contact:** the backend reclaimed the VM. Follow
  "Picking up after a disconnect" above. If `up` says the name is taken, check `colab sessions`.
- **`up --gpu X` fails with 400:** there's no quota for that accelerator
  right now. Try another (`--gpu T4`, `L4`).
- **Wedged kernel:** run `colab restart-kernel -s lpn`. Background jobs keep
  running, since they're in their own process group.
