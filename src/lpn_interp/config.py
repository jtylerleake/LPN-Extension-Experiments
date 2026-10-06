"""Pydantic schema for one run's YAML config (see src/lpn_interp/configs/).

A run is one of two training modes:
- `joint`: encoder + decoder + MLP head trained together on
  L_dec + lambda_exec * L_exec + lambda_commit * L_commit + beta * L_KL.
  The vanilla-LPN baseline is `joint` with lambda_exec = lambda_commit = 0
  (the head gets no gradient and is ignored at eval).
- `probe`: load a trained (vanilla) LPN from `init_checkpoint`, freeze it,
  and train a fresh head on its detached z with L_exec + L_commit only --
  tests whether z already encodes the tile without the head shaping it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

# Fixed by the testbed (PROJECT-SPEC.md section 3); not configurable.
GRID_SIZE = 4
TILE_SIZE = 2
NUM_COLORS = 10
ANCHOR_COLOR = 1


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
    latent_dim: PositiveInt = 2
    d_model: PositiveInt = 128
    num_layers: PositiveInt = 2
    num_heads: PositiveInt = 4
    mlp_dim: PositiveInt = 512
    dropout: float = 0.0
    head_hidden: PositiveInt = 128

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


class LossConfig(_Strict):
    lambda_exec: float = 1.0
    lambda_commit: float = 0.1
    beta: float = 1e-4


class TrainConfig(_Strict):
    mode: Literal["joint", "probe"] = "joint"
    init_checkpoint: str | None = None  # required for probe
    steps: PositiveInt = 100_000
    batch_size: PositiveInt = 256
    lr: float = 4e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    train_latent_steps: int = 0  # optional training-time latent ascent (spec section 7)
    train_latent_lr: float = 0.1
    amp: bool = True  # bf16 autocast on CUDA; ignored on CPU
    log_every: PositiveInt = 100
    eval_every: PositiveInt = 5_000
    checkpoint_every: PositiveInt = 5_000
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.mode == "probe" and not self.init_checkpoint:
            raise ValueError("probe mode needs init_checkpoint (a trained LPN to freeze)")
        if self.train_latent_steps < 0:
            raise ValueError("train_latent_steps must be >= 0")
        return self


class LatentOptConfig(_Strict):
    """Test-time gradient ascent on z (weights frozen): plain SGD, gradient
    clipped to `grad_clip`, best step (by context objective) kept.

    The objective's lambdas default to the training ones; probe configs set
    them to 0 so the search uses the decoder alone and the head stays a pure
    reader of z.
    """

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
