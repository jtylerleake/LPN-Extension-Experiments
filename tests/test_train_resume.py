import signal

import pytest
import torch

import src.training.train as train_mod
from src.config import RunConfig
from src.models.lpn import load_checkpoint
from src.utils.resume import read_jsonl_rows

CPU = torch.device("cpu")


def _cfg(**train_overrides) -> RunConfig:
    train = {"steps": 12, "batch_size": 8, "warmup_steps": 2, "log_every": 1, "eval_every": 1000,
             "checkpoint_every": 3, "amp": False, **train_overrides}
    return RunConfig.model_validate({
        "name": "t",
        "model": {"num_layers": 1, "num_heads": 2, "head_dim": 8, "head_hidden": 16},
        "train": train,
        "eval": {"batch_size": 16},
    })


@pytest.fixture(autouse=True)
def _small_eval(monkeypatch):
    monkeypatch.setattr(train_mod, "QUICK_EVAL_TASKS", 16)


class Crash(Exception):
    pass


def _crash_on_call(monkeypatch, n: int, sigterm: bool = False):
    """Make the n-th training batch either raise (simulated VM loss) or deliver SIGTERM."""
    real, calls = train_mod.sample_tasks, {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            if sigterm:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            else:
                raise Crash()
        return real(*args, **kwargs)

    monkeypatch.setattr(train_mod, "sample_tasks", wrapped)
    return real


def _final_state(run_dir):
    return load_checkpoint(run_dir / "final.pt")[0].state_dict()


def test_resume_after_crash_matches_uninterrupted_run(tmp_path, monkeypatch):
    cfg = _cfg()
    train_mod.train(cfg, tmp_path / "straight", CPU)

    real = _crash_on_call(monkeypatch, n=8)  # dies during step 8; last checkpoint at step 6
    with pytest.raises(Crash):
        train_mod.train(cfg, tmp_path / "resumed", CPU)
    assert torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)["step"] == 6
    assert max(r["step"] for r in read_jsonl_rows(tmp_path / "resumed" / "metrics.jsonl")) == 7

    monkeypatch.setattr(train_mod, "sample_tasks", real)
    train_mod.train(cfg, tmp_path / "resumed", CPU)

    straight, resumed = _final_state(tmp_path / "straight"), _final_state(tmp_path / "resumed")
    for a, b in zip(straight.values(), resumed.values(), strict=True):
        assert torch.equal(a, b)
    steps = [r["step"] for r in read_jsonl_rows(tmp_path / "resumed" / "metrics.jsonl") if r["kind"] == "train"]
    assert steps == list(range(1, 13))  # no step logged twice, none missing


def test_sigterm_checkpoints_then_resumes(tmp_path, monkeypatch):
    cfg = _cfg()
    real = _crash_on_call(monkeypatch, n=5, sigterm=True)
    with pytest.raises(train_mod.Interrupted):
        train_mod.train(cfg, tmp_path / "r", CPU)
    assert torch.load(tmp_path / "r" / "checkpoint.pt", weights_only=False)["step"] == 5
    monkeypatch.setattr(train_mod, "sample_tasks", real)
    train_mod.train(cfg, tmp_path / "r", CPU)
    assert (tmp_path / "r" / "final.pt").exists()


def test_completed_run_is_a_no_op(tmp_path, monkeypatch):
    cfg = _cfg()
    first = train_mod.train(cfg, tmp_path / "r", CPU)
    _crash_on_call(monkeypatch, n=1)  # any training step now would raise
    assert train_mod.train(cfg, tmp_path / "r", CPU) == first


def test_changed_config_refuses_to_resume(tmp_path, monkeypatch):
    _crash_on_call(monkeypatch, n=4)
    with pytest.raises(Crash):
        train_mod.train(_cfg(), tmp_path / "r", CPU)
    with pytest.raises(RuntimeError, match="different config"):
        train_mod.train(_cfg(lr=1e-3), tmp_path / "r", CPU)


def test_cadence_changes_still_resume(tmp_path):
    assert train_mod.run_fingerprint(_cfg()) == train_mod.run_fingerprint(_cfg(log_every=5, checkpoint_every=4))
    assert train_mod.run_fingerprint(_cfg()) != train_mod.run_fingerprint(_cfg(steps=13))


def test_restart_discards_state(tmp_path):
    train_mod.train(_cfg(), tmp_path / "r", CPU)
    train_mod.train(_cfg(lr=1e-3), tmp_path / "r", CPU, restart=True)
    assert len([r for r in read_jsonl_rows(tmp_path / "r" / "metrics.jsonl") if r["kind"] == "done"]) == 1


def test_probe_trains_only_the_head(tmp_path):
    vanilla = _cfg()
    vanilla = vanilla.model_copy(update={"loss": vanilla.loss.model_copy(update={"lambda_exec": 0.0, "lambda_commit": 0.0})})
    train_mod.train(vanilla, tmp_path / "vanilla", CPU)
    probe = _cfg(mode="probe", init=str(tmp_path / "vanilla" / "final.pt"))
    train_mod.train(probe, tmp_path / "probe", CPU)
    before, after = _final_state(tmp_path / "vanilla"), _final_state(tmp_path / "probe")
    for key in before:
        same = torch.equal(before[key], after[key])
        assert same != key.startswith("head."), key


def test_arch2_probe_trains_only_the_heads_and_reports_anchor_metrics(tmp_path):
    def arch2(cfg):
        return cfg.model_copy(update={"model": cfg.model.model_copy(update={"architecture": 2})})
    vanilla = arch2(_cfg())
    vanilla = vanilla.model_copy(update={"loss": vanilla.loss.model_copy(update={"lambda_exec": 0.0, "lambda_commit": 0.0})})
    train_mod.train(vanilla, tmp_path / "vanilla", CPU)
    probe = arch2(_cfg(mode="probe", init=str(tmp_path / "vanilla" / "final.pt")))
    done = train_mod.train(probe, tmp_path / "probe", CPU)
    before, after = _final_state(tmp_path / "vanilla"), _final_state(tmp_path / "probe")
    for key in before:
        assert torch.equal(before[key], after[key]) != key.startswith(("head.", "anchor_head.")), key
    assert {"seen/anchor_acc", "seen/anchor_valid", "unseen/anchor_acc"} <= done.keys()
