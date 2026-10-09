# Local dev + testing (Docker, CPU)

Docker is a reproducible CPU environment for editing, linting, unit tests,
and `_mini` smoke runs. It never trains real runs: that happens on Colab
(`colab/`). It also never runs *on* Colab, since Colab VMs can't run
containers.

| File | Role |
|---|---|
| `Dockerfile.dev` | Python 3.11 slim, CPU-only torch, `requirements.txt`, the repo installed editable |
| `docker-compose.yml` | the `dev` service: repo bind-mounted at `/workspace`, `PYTHONPATH=/workspace` |
| `requirements.in` | the hand-edited list: runtime libraries (a superset of `colab/requirements.txt`) plus dev tools |
| `requirements.txt` | the locked list the image installs (an unpinned placeholder until compiled) |
| `compile_requirements.ps1` | regenerates `requirements.txt` by running pip-compile inside the image |

## Use

```powershell
# from the repo root, with Docker Desktop running
docker compose -f docker/docker-compose.yml build dev
docker compose -f docker/docker-compose.yml run --rm dev pytest
docker compose -f docker/docker-compose.yml run --rm dev ruff check .
docker compose -f docker/docker-compose.yml run --rm dev `
    python scripts/train.py --config src/configs/stage3_scratch_mini.yaml --device cpu
```

## Adding a dependency

1. Add it to `requirements.in`. If the code needs it at run time (not just for
   dev), also add it to `colab/requirements.txt`. `tests/test_requirements.py`
   fails if Colab needs something Docker doesn't have.
2. Run `pwsh docker/compile_requirements.ps1`, then rebuild the image.
