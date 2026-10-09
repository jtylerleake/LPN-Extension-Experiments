"""Bring a Colab VM up to a requirements file without disturbing Colab's own builds.

Runs ON the VM (colab_runtime.py `setup` calls it after syncing the code):

    python colab/vm_setup.py colab/requirements.txt [more files...]

For each requirement line:
- installed and satisfies its specifier -> left alone (no upgrades/downgrades);
- missing -> pip-installed;
- installed but violating its specifier -> error (we don't silently swap a
  preinstalled library's version);
- marked `# preinstalled` -> only checked, never installed (torch, jax: Colab's
  CUDA-matched builds).
Exit code 0 when everything is satisfied, 1 otherwise.
"""
from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement


def parse(path: str | Path) -> list[tuple[Requirement, bool]]:
    """[(requirement, preinstalled_only)] from a requirements file (comments allowed)."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        body, _, comment = line.partition("#")
        if body.strip() and not body.strip().startswith("-"):  # skip pip options (--extra-index-url ...)
            entries.append((Requirement(body.strip()), "preinstalled" in comment))
    return entries


def plan(entries: list[tuple[Requirement, bool]]) -> tuple[list[str], list[str], list[str]]:
    """(to_install, problems, report lines), from what's installed right now."""
    to_install, problems, report = [], [], []
    for req, preinstalled in entries:
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            if preinstalled:
                problems.append(f"{req.name} is missing but must come preinstalled (not installing it)")
            else:
                to_install.append(str(req))
            report.append(f"  {req.name:<12} missing")
            continue
        ok = req.specifier.contains(version, prereleases=True)
        report.append(f"  {req.name:<12} {version}{'' if ok else f'  (needs {req.specifier})'}")
        if not ok:
            problems.append(f"{req.name} {version} doesn't satisfy {req.specifier}")
    return to_install, problems, report


def main(argv: list[str] | None = None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or ["colab/requirements.txt"]
    entries = [e for p in paths for e in parse(p)]
    to_install, problems, report = plan(entries)
    print("\n".join(report))
    if to_install and not problems:
        print("installing missing:", " ".join(to_install), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *to_install], check=True)
        to_install, problems, _ = plan(entries)
        problems += [f"{name} still missing after install" for name in to_install]
    if any(req.name == "torch" for req, _ in entries):
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        print(f"  cuda: {torch.cuda.is_available()}  gpu: {gpu}")
    for problem in problems:
        print("ERROR:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
