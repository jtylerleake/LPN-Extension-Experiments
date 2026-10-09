import argparse
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "colab" / "colab_runtime.py"
_spec = importlib.util.spec_from_file_location("colab_runtime", _PATH)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


@pytest.mark.parametrize("body", [cr.EXTRACT, cr.FOREGROUND, cr.LAUNCH, cr.JOBS, cr.TAIL, cr.KILL,
                                  cr.PACK, cr.DRIVE_CHECK])
def test_snippets_compile(body):
    compile(cr.snippet(body, X="it's \"quoted\""), "<snippet>", "exec")


@pytest.mark.parametrize("job_state, remote, named, include_failed, relaunch", [
    ("running", None, False, False, True),            # new VM: no record there -> relaunch
    ("running", "died", False, False, True),          # VM-side process vanished
    ("running", "exited(75)", False, False, True),    # SIGTERM'd after checkpointing
    ("running", "running", False, False, False),      # still going
    ("running", "done", False, False, False),
    ("done", None, False, False, False),              # finished on an earlier VM
    ("killed", None, False, False, False),            # deliberate kill...
    ("killed", None, True, False, True),              # ...unless named
    ("running", "exited(1)", False, False, False),    # crashed: don't loop on a bug...
    ("running", "exited(1)", False, True, True),      # ...unless asked
])
def test_resume_decision(job_state, remote, named, include_failed, relaunch):
    reason = cr.resume_decision({"state": job_state}, remote, named, include_failed)
    assert (reason is None) == relaunch, reason


def test_drive_flag_falls_back_to_session_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "STATE_FILE", tmp_path / "state.json")
    cr.save_state({"sessions": {"lpn": {"drive": True}}, "jobs": {}})
    ns = lambda drive: argparse.Namespace(session="lpn", drive=drive)  # noqa: E731
    assert cr.use_drive(ns(None)) is True
    assert cr.use_drive(ns(False)) is False
    assert cr.use_drive(argparse.Namespace(session="other", drive=None)) is False


def test_bundle_includes_code_and_excludes_outputs():
    files = cr.bundle_files()
    assert {"src/training/train.py", "scripts/train.py", "colab/vm_setup.py", "colab/requirements.txt"} <= set(files)
    assert not any(f.startswith(("outputs/", ".colab/")) or "__pycache__" in f for f in files)
