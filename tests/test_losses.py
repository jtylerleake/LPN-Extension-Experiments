import math

import torch
import torch.nn.functional as F

from src.config import LossConfig, ModelConfig
from src.data.pattern2d import NUM_TILES, sample_tasks
from src.execution.executor import find_anchor
from src.models.lpn import LPN
from src.training.losses import (
    clip_per_row,
    commit_entropy,
    context_loss,
    exec_nll,
    kl_std_normal,
    training_losses,
)

SMALL = ModelConfig(num_layers=1, num_heads=2, head_dim=8, head_hidden=16)


def _batch(n=8):
    return sample_tasks(torch.arange(NUM_TILES), n, 4, torch.Generator().manual_seed(0))


def test_exec_nll_equals_tile_cell_cross_entropy():
    """Executing then scoring the grid (mean over 16 cells) == CE on the 4 tile cells / 16."""
    b = _batch()
    logits = torch.randn(8, 4, 4, 10)
    cells = find_anchor(b.inputs)[..., None] + torch.tensor([0, 1, 4, 5])
    targets = b.outputs.flatten(-2).gather(-1, cells)
    expected = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none").view(8, 4, 4).sum(-1) / 16
    assert torch.allclose(exec_nll(logits, b.inputs, b.outputs), expected, atol=1e-4)


def test_exec_nll_zero_for_confident_true_tile():
    b = _batch()
    logits = 50 * F.one_hot(b.tiles, 10).float()[:, None].expand(-1, 4, -1, -1)
    assert exec_nll(logits, b.inputs, b.outputs).abs().max() < 1e-4


def test_commit_entropy_bounds():
    assert torch.allclose(commit_entropy(torch.zeros(4, 10)), torch.tensor(math.log(10)))
    assert commit_entropy(100 * F.one_hot(torch.arange(4), 10).float()) < 1e-6


def test_kl_zero_at_prior():
    assert kl_std_normal(torch.zeros(3, 2), torch.zeros(3, 2)).abs().max() == 0


def test_clip_per_row_is_independent():
    g = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
    out = clip_per_row(g, 1.0)
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8]))
    assert torch.equal(out[1], g[1])


def test_joint_losses_backprop_into_all_modules():
    torch.manual_seed(0)
    model, b = LPN(SMALL), _batch()
    terms = training_losses(model, b.inputs, b.outputs, LossConfig())
    terms.total.backward()
    for name in ("encoder", "decoder", "head"):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), name


def test_vanilla_losses_leave_head_untouched():
    torch.manual_seed(0)
    model, b = LPN(SMALL), _batch()
    terms = training_losses(model, b.inputs, b.outputs, LossConfig(lambda_exec=0.0, lambda_commit=0.0))
    terms.total.backward()
    assert all(p.grad is None for p in model.head.parameters())
    assert terms.exec == 0 and terms.commit == 0


def test_probe_trains_only_head():
    torch.manual_seed(0)
    model, b = LPN(SMALL), _batch()
    terms = training_losses(model, b.inputs, b.outputs, LossConfig(), mode="probe")
    terms.total.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    assert any(p.grad is not None for p in model.head.parameters())


def test_training_latent_steps_run_and_backprop():
    torch.manual_seed(0)
    model, b = LPN(SMALL), _batch()
    terms = training_losses(model, b.inputs, b.outputs, LossConfig(), latent_steps=2, latent_lr=0.1)
    terms.total.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())


def test_context_loss_shape():
    torch.manual_seed(0)
    model, b = LPN(SMALL), _batch()
    assert context_loss(model, b.inputs, b.outputs, torch.randn(8, 2), 1.0, 0.1).shape == (8,)


ARCH2 = SMALL.model_copy(update={"architecture": 2})


def test_exec_nll_with_true_anchor_and_tile_is_zero():
    b = _batch()
    logits = 50 * F.one_hot(b.tiles, 10).float()[:, None].expand(-1, 4, -1, -1)
    anchor_logits = 50 * F.one_hot(find_anchor(b.inputs), 16).float()
    assert exec_nll(logits, b.inputs, b.outputs, anchor_logits).abs().max() < 1e-4


def test_arch2_probe_trains_both_heads_only():
    torch.manual_seed(0)
    model, b = LPN(ARCH2), _batch()
    terms = training_losses(model, b.inputs, b.outputs, LossConfig(), mode="probe")
    terms.total.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    for name in ("head", "anchor_head"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name


def test_arch2_joint_losses_latent_steps_and_context_loss_run():
    torch.manual_seed(0)
    model, b = LPN(ARCH2), _batch()
    terms = training_losses(model, b.inputs, b.outputs, LossConfig(), latent_steps=1, latent_lr=0.1)
    terms.total.backward()
    assert any(p.grad is not None for p in model.anchor_head.parameters())
    assert context_loss(model, b.inputs, b.outputs, torch.randn(8, 2), 1.0, 0.1).shape == (8,)
