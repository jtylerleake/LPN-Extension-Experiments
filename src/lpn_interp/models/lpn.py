"""Full model: encoder + decoder + MLP head (Architecture 1).

The decoder and the head are parallel readouts of the same z: the decoder
produces the prediction, the head produces the explanation (the tile).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from lpn_interp.config import ModelConfig
from lpn_interp.models.decoder import Decoder
from lpn_interp.models.encoder import Encoder, pool
from lpn_interp.models.mlp_head import MLPHead


class LPN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.head = MLPHead(cfg)

    def context_latent(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Test-time z0: the pooled posterior mean over all context pairs (no LOO). -> (B, d)"""
        mu, logvar = self.encoder(inputs, outputs)
        return pool(mu, logvar)[0]


def save_checkpoint(path: str | Path, model: LPN, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg.model_dump(), **extra}, tmp)
    tmp.replace(path)  # atomic: a disconnect mid-save never leaves a corrupt checkpoint


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[LPN, dict]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model = LPN(ModelConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"])
    return model, state
