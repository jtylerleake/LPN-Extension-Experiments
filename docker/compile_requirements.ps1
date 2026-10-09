# Regenerates docker/requirements.txt (locked) from docker/requirements.in,
# running pip-compile *inside* the dev image so the lock is produced in the
# same environment it will be installed into.
#
# Usage (from the repo root): pwsh docker/compile_requirements.ps1
#
# Only the Docker environment is locked. Colab deliberately isn't: it uses its
# own preinstalled, CUDA-matched builds (see colab/requirements.txt).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "Building dev image (if needed)..."
docker compose -f "$root/docker/docker-compose.yml" build dev
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }

Write-Host "Compiling docker/requirements.txt ..."
docker compose -f "$root/docker/docker-compose.yml" run --rm dev `
    pip-compile docker/requirements.in --output-file docker/requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip-compile failed" }

Write-Host "Done. Review changes with 'git diff docker/requirements.txt', then rebuild the image."
