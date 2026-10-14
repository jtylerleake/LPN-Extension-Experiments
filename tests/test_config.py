from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import ModelConfig, RunConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
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
    assert (mini.train.mode, mini.train.init, mini.train.lr) == (big.train.mode, big.train.init, big.train.lr)
    assert (mini.test_lambda_exec, mini.test_lambda_commit) == (big.test_lambda_exec, big.test_lambda_commit)
    assert mini.train.steps < big.train.steps


def test_stage_configs_differ_only_where_intended():
    s1, s2, s3 = (RunConfig.from_yaml(CONFIG_DIR / f"{n}.yaml") for n in ("stage1_frozen", "stage2_finetune", "stage3_scratch"))
    assert (s1.train.mode, s1.train.init) == ("probe", "pretrained")
    assert (s2.train.mode, s2.train.init) == ("joint", "pretrained")
    assert (s3.train.mode, s3.train.init) == ("joint", "scratch")
    # stage 1's head is the head sweep's winner; stages 2/3 keep the default head for now
    assert s2.model == s3.model == ModelConfig()
    assert s1.model.backbone() == s2.model.backbone()
    assert s1.loss == s2.loss == s3.loss


def test_probe_requires_a_trained_backbone():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "x", "train": {"mode": "probe", "init": "scratch"}})


def test_pretrained_init_requires_checkpoint_architecture():
    with pytest.raises(ValidationError, match="backbone"):
        RunConfig.model_validate({"name": "x", "model": {"num_layers": 3}, "train": {"init": "pretrained"}})
    RunConfig.model_validate({"name": "x", "model": {"head_hidden": 64}, "train": {"init": "pretrained"}})


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        RunConfig.model_validate(yaml.safe_load("name: x\nloss: {lambda_exce: 1.0}"))


def test_test_time_lambdas_default_to_training_ones():
    cfg = RunConfig.model_validate({"name": "x", "loss": {"lambda_exec": 0.5, "lambda_commit": 0.2}})
    assert (cfg.test_lambda_exec, cfg.test_lambda_commit) == (0.5, 0.2)
