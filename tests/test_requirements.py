"""The two runtimes' requirement lists, and colab/vm_setup.py's install decisions."""
import importlib.util
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("vm_setup", ROOT / "colab" / "vm_setup.py")
vm_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm_setup)


def _names(path: Path) -> set[str]:
    return {req.name.lower() for req, _ in vm_setup.parse(path)}


def test_docker_covers_everything_colab_needs():
    """Code that runs on Colab must have been testable locally."""
    missing = _names(ROOT / "colab" / "requirements.txt") - _names(ROOT / "docker" / "requirements.in")
    assert not missing, f"add to docker/requirements.in: {sorted(missing)}"


def test_gpu_builds_are_never_installed_on_colab():
    flags = {req.name: pre for p in ("requirements.txt", "requirements-parity.txt")
             for req, pre in vm_setup.parse(ROOT / "colab" / p)}
    assert flags["torch"] and flags["jax"]


def test_parse_reads_specifiers_and_flags(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("# header\n\npydantic>=2   # a note\ntorch  # preinstalled (CUDA)\n", encoding="utf-8")
    (pyd, pyd_pre), (torch, torch_pre) = vm_setup.parse(f)
    assert (pyd.name, str(pyd.specifier), pyd_pre) == ("pydantic", ">=2", False)
    assert (torch.name, torch_pre) == ("torch", True)


def test_plan(monkeypatch):
    installed = {"numpy": "2.1.3", "pydantic": "1.10.0", "torch": "2.11.0"}

    def version(name):
        if name not in installed:
            raise vm_setup.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(vm_setup.metadata, "version", version)
    entries = [(Requirement("numpy"), False), (Requirement("msgpack"), False), (Requirement("pydantic>=2"), False),
               (Requirement("torch"), True), (Requirement("jax"), True)]
    to_install, problems, _ = vm_setup.plan(entries)
    assert to_install == ["msgpack"]  # present ones are left alone
    assert any("pydantic" in p for p in problems)  # wrong version: error, not a silent swap
    assert any("jax" in p for p in problems)  # preinstalled-only and missing: error, never installed
