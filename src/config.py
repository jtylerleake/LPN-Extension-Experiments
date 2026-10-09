"""Pydantic schema for one run's YAML config (see src/configs/).

The model is a PyTorch port of the paper's LPN (src/models/), so a run can
start from the authors' pretrained Pattern-2D checkpoint. The experiment is
staged (configs/stage*.yaml):
  1. frozen   -- mode=probe, init=pretrained: train only the MLP head on the
                 frozen backbone's z.
  2. finetune -- mode=joint, init=pretrained: light fine-tune of all weights
                 on L_dec + lambda_exec L_exec + lambda_commit L_commit + beta L_KL.
  3. scratch  -- mode=joint, init=scratch: the same model trained from random init.
The vanilla-LPN baseline is the pretrained checkpoint itself (evaluated, not trained).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

# Fixed by the testbed (prompts/PROJECT-SPEC-1.md section 3); not configurable.
GRID_SIZE = 4
TILE_SIZE = 2
NUM_COLORS = 10
ANCHOR_COLOR = 1

# `train.init` values with a special meaning; anything else is a path to one
# of our own checkpoints (relative paths resolve against the output root).
INIT_SCRATCH = "scratch"
INIT_PRETRAINED = "pretrained"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    num_pairs: PositiveInt = 4  # context pairs per task; eval adds one query pair
    holdout_fraction: float = 0.1  # tiles never seen in training
    holdout_seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.num_pairs < 2:
            raise ValueError("num_pairs must be >= 2 (leave-one-out needs at least one other pair)")
        # 9 valid anchors; eval needs num_pairs + 1 distinct ones.
        if self.num_pairs + 1 > (GRID_SIZE - TILE_SIZE + 1) ** 2:
            raise ValueError("num_pairs + 1 distinct anchors must fit in the 9 valid positions")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1)")
        return self


class ModelConfig(_Strict):
    """Backbone defaults are the pretrained checkpoint's (clement-bonnet/lpn-2d,
    quiet-thunder-789): 2-layer encoder and decoder, 6 heads x 12 dims, SiLU
    MLP x4, no biases, 2-D variational latent.

    architecture 1: tile head h(z) + hardcoded FIND. architecture 2: adds an
    anchor head g(z) whose 4x4 output replaces FIND (naive on purpose: g never
    sees x, and z carries no per-pair anchor)."""

    latent_dim: PositiveInt = 2
    num_layers: PositiveInt = 2
    num_heads: PositiveInt = 6
    head_dim: PositiveInt = 12
    mlp_factor: float = 4.0
    dropout: float = 0.0
    # MLP heads (ours; not part of the checkpoint)
    head_hidden: PositiveInt = 128  # hidden width
    head_depth: PositiveInt = 2  # number of hidden layers
    head_activation: Literal["relu", "silu"] = "relu"
    head_residual: bool = False  # pre-norm residual blocks (for deep heads)
    architecture: Literal[1, 2] = 1

    @property
    def emb_dim(self) -> int:
        return self.num_heads * self.head_dim

    def backbone(self) -> dict:
        """The fields a checkpoint's weights depend on (the head excluded)."""
        return self.model_dump(include={"latent_dim", "num_layers", "num_heads", "head_dim", "mlp_factor"})


class LossConfig(_Strict):
    """Per-pair terms use the paper's scale: L_dec = shape CE + mean over the
    16 grid cells; L_exec is the executed grid's mean per-cell NLL;
    L_commit is the mean entropy of the 4 tile cells. beta defaults to the
    checkpoint's own KL coefficient."""

    lambda_exec: float = 1.0
    lambda_commit: float = 0.1
    beta: float = 1e-3


class TrainConfig(_Strict):
    mode: Literal["joint", "probe"] = "joint"
    init: str = INIT_SCRATCH  # scratch | pretrained | path to one of our checkpoints
    steps: PositiveInt = 100_000
    batch_size: PositiveInt = 256
    lr: float = 4e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    train_latent_steps: int = 0  # optional training-time latent ascent (spec section 7)
    train_latent_lr: float = 0.1
    amp: bool = True  # bf16/fp16 autocast on CUDA; ignored on CPU
    log_every: PositiveInt = 100
    eval_every: PositiveInt = 5_000
    checkpoint_every: PositiveInt = 2_500
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.mode == "probe" and self.init == INIT_SCRATCH:
            raise ValueError("probe mode freezes a trained backbone: set init to 'pretrained' or a checkpoint")
        if self.train_latent_steps < 0:
            raise ValueError("train_latent_steps must be >= 0")
        return self


class LatentOptConfig(_Strict):
    """Test-time gradient ascent on z (weights frozen): plain SGD, gradient
    clipped to `grad_clip`, best step (by context objective) kept. lr 0.1 is
    the paper's own. The objective's lambdas default to the training ones."""

    steps: int = 100
    lr: float = 0.1
    grad_clip: float = 1.0
    lambda_exec: float | None = None
    lambda_commit: float | None = None


class EvalConfig(_Strict):
    num_tasks: PositiveInt = 2_000  # per split (seen / unseen tiles)
    seed: int = 1234
    batch_size: PositiveInt = 500
    trajectory_tasks: int = 8


class RunConfig(_Strict):
    name: str
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    latent_opt: LatentOptConfig = LatentOptConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if self.train.init == INIT_PRETRAINED and self.model.backbone() != ModelConfig().backbone():
            raise ValueError("init=pretrained needs the checkpoint's backbone: leave model.* (except head_hidden, "
                             "dropout) at their defaults")
        return self

    @property
    def test_lambda_exec(self) -> float:
        lo = self.latent_opt.lambda_exec
        return self.loss.lambda_exec if lo is None else lo

    @property
    def test_lambda_commit(self) -> float:
        lo = self.latent_opt.lambda_commit
        return self.loss.lambda_commit if lo is None else lo

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
