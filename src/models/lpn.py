"""Full model: the ported LPN encoder + decoder, plus our MLP heads.

The decoder and the heads are parallel readouts of the same z: the decoder
produces the prediction, the heads produce the explanation -- the tile
(both architectures) and, in architecture 2, the anchor position.

Pooling follows the original LPN (lpn's LPN.__call__ / generate_output):
training samples one latent per pair, z_i = mu_i + sigma_i * eps, and pair
i's context is the mean of the *other* pairs' samples (leave-one-out). At test
time we use the posterior means (prompts/PROJECT-SPEC-1.md section 7: z0 = mu), pooled
over all context pairs.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import ModelConfig
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.mlp_head import AnchorHead, MLPHead


def sample_latents(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + (0.5 * logvar).exp() * torch.randn_like(mu)


def loo_mean(latents: torch.Tensor) -> torch.Tensor:
    """(B, P, d) -> (B, P, d): entry i is the mean over every pair except i."""
    num_pairs = latents.shape[1]
    if num_pairs < 2:
        raise ValueError("leave-one-out pooling needs at least 2 pairs")
    return (latents.sum(1, keepdim=True) - latents) / (num_pairs - 1)


class LPN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.reset_heads()

    def reset_heads(self) -> None:
        """Fresh, randomly initialized program heads (the backbone is untouched)."""
        self.head = MLPHead(self.cfg)
        self.anchor_head = AnchorHead(self.cfg) if self.cfg.architecture == 2 else None

    def program_logits(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """z (..., d) -> (tile logits (..., 4, 10), anchor logits (..., 16) or None in architecture 1)."""
        return self.head(z), None if self.anchor_head is None else self.anchor_head(z)

    def context_latent(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Test-time z0: the mean of the per-pair posterior means over all context pairs. -> (B, d)"""
        mu, _ = self.encoder(inputs, outputs)
        return mu.mean(1)


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
