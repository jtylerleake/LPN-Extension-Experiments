# Regenerates docker/requirements/base.txt and colab.txt from base.in,
# running pip-compile *inside* the dev Docker image so the lock is produced
# in the same environment it will be installed into.
#
# Usage: pwsh scripts/compile_requirements.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "Building dev image (if needed)..."
docker compose -f "$root/docker/docker-compose.yml" build dev
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }

Write-Host "Compiling base.txt (locked, includes torch) ..."
docker compose -f "$root/docker/docker-compose.yml" run --rm dev `
    pip-compile docker/requirements/base.in --output-file docker/requirements/base.txt
if ($LASTEXITCODE -ne 0) { throw "pip-compile for base.txt failed" }

Write-Host "Compiling colab.txt (same deps, torch excluded — Colab ships its own) ..."
docker compose -f "$root/docker/docker-compose.yml" run --rm dev bash -c @'
grep -v "^torch" docker/requirements/base.in > /tmp/colab.in
pip-compile /tmp/colab.in --output-file docker/requirements/colab.txt
'@
if ($LASTEXITCODE -ne 0) { throw "pip-compile for colab.txt failed" }

Write-Host "Done. Review changes with 'git diff docker/requirements/'."
