"""Loss terms (prompts/PROJECT-SPEC-1.md section 6).

Scaled like the original LPN loss, so the pretrained checkpoint's
hyperparameters (KL coefficient, latent-search lr) carry over:
- L_dec per pair = -log p(output shape) + mean over the 16 cells of -log p(cell)
  (Decoder.nll, the original's per-pair loss);
- L_exec per pair = mean over the 16 cells of the executed grid's NLL
  (architecture 2 executes with g(z)'s soft anchor map instead of FIND);
- L_commit = mean over the 4 tile cells of the soft tile's entropy
  (optional: off by default, lambda_commit = 0, see LossConfig);
- L_KL per pair = KL(q(z | pair) || N(0, I)).
Training averages every term over pairs and tasks:
  L = L_dec + lambda_exec * L_exec + beta * L_KL [+ lambda_commit * L_commit]
The commitment entropy is still computed and logged as a diagnostic.
The test-time search objective sums over context pairs instead (as the
original's gradient ascent does), so its lr 0.1 applies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config import LossConfig
from src.execution.executor import execute, paste_soft_anchor
from src.models.lpn import LPN, loo_mean, sample_latents
from src.models.mlp_head import soft_anchor, soft_tile


def exec_nll(
    head_logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, anchor_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """-log p(y | EXECUTE(S_hat, x)), mean over the 16 cells. logits (..., 4, 10), x/y (..., 4, 4),
    anchor_logits (..., 16) or None -> (...).

    None (architecture 1): the anchor comes from FIND. Cells outside the tile
    are exact copies of x in both the executed grid and y, so they contribute
    exactly 0; the clamp only guards log(0) if a target ever disagreed with
    the executor's hard copy (it can't, by construction).
    Given anchor logits (architecture 2): PASTE in expectation over g(z)'s
    softmax. Its probabilities are never exactly 0, so neither is any cell's
    target probability, and the clamp doesn't block gradients in practice.
    """
    s_hat = soft_tile(head_logits.float())
    if anchor_logits is None:
        y_hat = execute(s_hat, x, validate=False)
    else:
        y_hat = paste_soft_anchor(s_hat, x, soft_anchor(anchor_logits.float()))
    p_target = y_hat.gather(-1, y.flatten(-2)[..., None]).squeeze(-1)
    return -p_target.clamp_min(1e-12).log().mean(-1)


def commit_entropy(head_logits: torch.Tensor) -> torch.Tensor:
    """Mean over the 4 tile cells of each cell's color entropy. (..., 4, 10) -> (...)."""
    log_p = head_logits.float().log_softmax(-1)
    return -(log_p.exp() * log_p).sum(-1).mean(-1)


def kl_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)) summed over latent dims. (..., d) -> (...)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)


def context_loss(
    model: LPN, inputs: torch.Tensor, outputs: torch.Tensor, z: torch.Tensor,
    lambda_exec: float, lambda_commit: float,
) -> torch.Tensor:
    """Per-task objective (to minimize) of one shared z on its context pairs:
    sum_i [L_dec + lambda_exec L_exec (+ lambda_commit L_commit)]. inputs/outputs
    (B, P, 4, 4), z (B, d) -> (B,). Used by test-time latent optimization.
    """
    z_pairs = z[:, None].expand(-1, inputs.shape[1], -1)
    loss = model.decoder.nll(inputs, outputs, z_pairs).sum(1)
    if lambda_exec or lambda_commit:
        logits, anchor_logits = model.program_logits(z)
        n = inputs.shape[1]
        if anchor_logits is not None:
            anchor_logits = anchor_logits[:, None].expand(-1, n, -1)
        ex = exec_nll(logits[:, None].expand(-1, n, -1, -1), inputs, outputs, anchor_logits)
        loss = loss + lambda_exec * ex.sum(1)
        if lambda_commit:
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
        return {k: v.item() for k, v in vars(self).items()}


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
            logits, anchor_logits = model.program_logits(z_ij)
            loss = loss + loss_cfg.lambda_exec * exec_nll(logits, x, y, anchor_logits)
            if loss_cfg.lambda_commit:
                loss = loss + loss_cfg.lambda_commit * commit_entropy(logits)
        (grad,) = torch.autograd.grad((loss * others).sum(), z_leaf)
        z = z - lr * clip_per_row(grad, grad_clip).detach()
    return z


def training_losses(
    model: LPN, inputs: torch.Tensor, outputs: torch.Tensor, loss_cfg: LossConfig,
    mode: str = "joint", latent_steps: int = 0, latent_lr: float = 0.0,
) -> LossTerms:
    """One batch's losses (prompts/PROJECT-SPEC-1.md section 7, training pipeline).

    joint: encode each pair, sample one latent per pair, and give pair i the
      mean of the other pairs' samples, z0^(i) (as the original LPN does);
      then decoder, executor and commitment losses on pair i with z0^(i), plus
      each pair's KL to the prior.
    probe: the same z0^(i), detached (encoder/decoder frozen); only the heads'
      terms are trained. dec and kl are reported as 0.
    In architecture 2 the anchor head is trained only through L_exec (the
    commitment term covers the tile alone). `commit` is always reported (as a
    diagnostic) but enters `total` only when lambda_commit > 0.
    """
    mu, logvar = model.encoder(inputs, outputs)
    z = loo_mean(sample_latents(mu, logvar))
    if mode == "probe":
        z = z.detach()
    elif latent_steps:
        z = loo_latent_steps(model, inputs, outputs, z, latent_steps, latent_lr, loss_cfg)

    zero = z.new_zeros(())
    dec = model.decoder.nll(inputs, outputs, z).mean() if mode == "joint" else zero
    kl = kl_std_normal(mu, logvar).mean() if mode == "joint" else zero
    if loss_cfg.lambda_exec or loss_cfg.lambda_commit or mode == "probe":
        logits, anchor_logits = model.program_logits(z)
        ex = exec_nll(logits, inputs, outputs, anchor_logits).mean()
        commit = commit_entropy(logits).mean()
    else:
        ex = commit = zero
    total = dec + loss_cfg.lambda_exec * ex + loss_cfg.beta * kl
    if loss_cfg.lambda_commit:
        total = total + loss_cfg.lambda_commit * commit
    return LossTerms(total, dec, ex, commit, kl)
