from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lpn_interp.config import RunConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "lpn_interp" / "configs"
FULL_CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.stem.endswith("_mini"))


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_config_validates(path):
    assert RunConfig.from_yaml(path).name == path.stem


@pytest.mark.parametrize("full", FULL_CONFIGS, ids=lambda p: p.stem)
def test_mini_mirrors_full(full):
    """A mini config only caps numbers; it must not change what is trained."""
    big = RunConfig.from_yaml(full)
    mini = RunConfig.from_yaml(full.with_name(f"{full.stem}_mini.yaml"))
    assert mini.loss == big.loss
    assert mini.model == big.model
    assert mini.data == big.data
    assert mini.train.mode == big.train.mode
    assert (mini.test_lambda_exec, mini.test_lambda_commit) == (big.test_lambda_exec, big.test_lambda_commit)
    assert mini.train.steps < big.train.steps


def test_vanilla_matches_arch1_except_aux_losses():
    arch1 = RunConfig.from_yaml(CONFIG_DIR / "arch1.yaml")
    vanilla = RunConfig.from_yaml(CONFIG_DIR / "vanilla.yaml")
    assert (vanilla.loss.lambda_exec, vanilla.loss.lambda_commit) == (0.0, 0.0)
    assert vanilla.loss.beta == arch1.loss.beta
    assert vanilla.model == arch1.model
    assert vanilla.train == arch1.train


def test_probe_requires_checkpoint():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "x", "train": {"mode": "probe"}})


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        RunConfig.model_validate(yaml.safe_load("name: x\nloss: {lambda_exce: 1.0}"))


def test_test_time_lambdas_default_to_training_ones():
    cfg = RunConfig.model_validate({"name": "x", "loss": {"lambda_exec": 0.5, "lambda_commit": 0.2}})
    assert (cfg.test_lambda_exec, cfg.test_lambda_commit) == (0.5, 0.2)
