"""Loss terms (PROJECT-SPEC.md section 6).

Per-pair terms are summed over grid cells (a grid's negative log-likelihood)
and over pairs, then averaged over tasks in the batch:
  L = L_dec + lambda_exec * L_exec + lambda_commit * L_commit + beta * L_KL
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from lpn_interp.config import LossConfig
from lpn_interp.execution.executor import execute
from lpn_interp.models.encoder import loo_pool
from lpn_interp.models.lpn import LPN
from lpn_interp.models.mlp_head import soft_tile


def exec_nll(head_logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """-log p(y | EXECUTE(S_hat, x)) summed over cells. logits (..., 4, 10), x/y (..., 4, 4) -> (...).

    Cells outside the tile are exact copies of x in both the executed grid and
    y, so they contribute exactly 0; the clamp only guards log(0) if a target
    ever disagreed with the executor's hard copy (it can't, by construction).
    """
    y_hat = execute(soft_tile(head_logits.float()), x, validate=False)
    p_target = y_hat.gather(-1, y.flatten(-2)[..., None]).squeeze(-1)
    return -p_target.clamp_min(1e-12).log().sum(-1)


def commit_entropy(head_logits: torch.Tensor) -> torch.Tensor:
    """Sum over tile cells of each cell's color entropy. (..., 4, 10) -> (...)."""
    log_p = head_logits.float().log_softmax(-1)
    return -(log_p.exp() * log_p).sum((-1, -2))


def kl_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)) summed over latent dims. (..., d) -> (...)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)


def context_loss(
    model: LPN, inputs: torch.Tensor, outputs: torch.Tensor, z: torch.Tensor,
    lambda_exec: float, lambda_commit: float,
) -> torch.Tensor:
    """Per-task objective (to minimize) of one shared z on its context pairs:
    sum_i [L_dec + lambda_exec L_exec + lambda_commit L_commit]. inputs/outputs
    (B, P, 4, 4), z (B, d) -> (B,). Used by test-time latent optimization.
    """
    z_pairs = z[:, None].expand(-1, inputs.shape[1], -1)
    loss = model.decoder.nll(inputs, outputs, z_pairs).sum(1)
    if lambda_exec or lambda_commit:
        logits = model.head(z)
        n = inputs.shape[1]
        loss = loss + lambda_exec * exec_nll(logits[:, None].expand(-1, n, -1, -1), inputs, outputs).sum(1)
        loss = loss + lambda_commit * n * commit_entropy(logits)
    return loss


@dataclass
class LossTerms:
    total: torch.Tensor
    dec: torch.Tensor
    exec: torch.Tensor
    commit: torch.Tensor
    kl: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {k: float(v) for k, v in vars(self).items()}


def clip_per_row(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip each latent's gradient (last dim) to `max_norm` independently, so
    one task's large gradient never rescales another's."""
    norm = grad.norm(dim=-1, keepdim=True)
    return grad * (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)


def loo_latent_steps(
    model: LPN, inputs: torch.Tensor, outputs: torch.Tensor, z: torch.Tensor,
    steps: int, lr: float, loss_cfg: LossConfig, grad_clip: float = 1.0,
) -> torch.Tensor:
    """Optional training-time latent ascent. z (B, P, d): z[:, i] is pair i's
    LOO latent, refined on pairs j != i only (never on the pair it will
    reconstruct). First-order: the update is detached, so gradients still
    reach the encoder through z0 but not through the inner steps.
    """
    num_pairs = inputs.shape[1]
    others = ~torch.eye(num_pairs, dtype=torch.bool, device=z.device)  # [i, j]: j is context for i
    x = inputs[:, None].expand(-1, num_pairs, -1, -1, -1)
    y = outputs[:, None].expand(-1, num_pairs, -1, -1, -1)
    for _ in range(steps):
        z_leaf = z.detach().requires_grad_()
        z_ij = z_leaf[:, :, None].expand(-1, -1, num_pairs, -1)
        loss = model.decoder.nll(x, y, z_ij)
        if loss_cfg.lambda_exec or loss_cfg.lambda_commit:
            logits = model.head(z_ij)
            loss = loss + loss_cfg.lambda_exec * exec_nll(logits, x, y)
            loss = loss + loss_cfg.lambda_commit * commit_entropy(logits)
        (grad,) = torch.autograd.grad((loss * others).sum(), z_leaf)
        z = z - lr * clip_per_row(grad, grad_clip).detach()
    return z


def training_losses(
    model: LPN, inputs: torch.Tensor, outputs: torch.Tensor, loss_cfg: LossConfig,
    mode: str = "joint", latent_steps: int = 0, latent_lr: float = 0.0,
) -> LossTerms:
    """One batch's losses (PROJECT-SPEC.md section 7, training pipeline).

    joint: LOO-encode, reparameterize z0^(i), then decoder, executor and
      commitment losses on pair i with z0^(i), plus KL of the LOO posteriors.
    probe: the same LOO z0^(i), detached (encoder/decoder frozen); only the
      head's terms are trained. dec and kl are reported as 0.
    """
    mu, logvar = loo_pool(*model.encoder(inputs, outputs))
    z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
    if mode == "probe":
        z = z.detach()
    elif latent_steps:
        z = loo_latent_steps(model, inputs, outputs, z, latent_steps, latent_lr, loss_cfg)

    zero = z.new_zeros(())
    dec = model.decoder.nll(inputs, outputs, z).sum(1).mean() if mode == "joint" else zero
    kl = kl_std_normal(mu, logvar).sum(1).mean() if mode == "joint" else zero
    if loss_cfg.lambda_exec or loss_cfg.lambda_commit or mode == "probe":
        logits = model.head(z)
        ex = exec_nll(logits, inputs, outputs).sum(1).mean()
        commit = commit_entropy(logits).sum(1).mean()
    else:
        ex = commit = zero
    total = dec + loss_cfg.lambda_exec * ex + loss_cfg.lambda_commit * commit + loss_cfg.beta * kl
    return LossTerms(total, dec, ex, commit, kl)
