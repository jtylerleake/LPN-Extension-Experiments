import pytest
import torch
import torch.nn.functional as F

from src.data.pattern2d import NUM_TILES, sample_tasks
from src.execution.executor import execute, find_anchor, grid_of, paste_soft_anchor, placements


def _batch():
    return sample_tasks(torch.arange(NUM_TILES), 64, 4, torch.Generator().manual_seed(0))


def test_find_anchor_matches_generator():
    b = _batch()
    assert torch.equal(find_anchor(b.inputs), b.anchors[..., 0] * 4 + b.anchors[..., 1])


def test_find_anchor_rejects_bad_grids():
    x = torch.zeros(1, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        find_anchor(x)
    x[0, 0, 0] = x[0, 1, 1] = 1
    with pytest.raises(ValueError):
        find_anchor(x)


def test_executing_true_tile_reproduces_output_exactly():
    b = _batch()
    s_true = F.one_hot(b.tiles, 10).float()[:, None].expand(-1, 4, -1, -1)
    probs = execute(s_true, b.inputs)
    assert probs.shape == (64, 4, 16, 10)
    assert torch.equal(grid_of(probs), b.outputs)
    assert torch.allclose(probs.sum(-1), torch.ones(64, 4, 16))


def test_paste_is_differentiable_and_touches_only_tile_cells():
    b = _batch()
    s = torch.softmax(torch.randn(64, 4, 10), -1).requires_grad_()
    probs = execute(s[:, None].expand(-1, 4, -1, -1), b.inputs)
    flat_in = b.inputs.flatten(-2)
    anchor = find_anchor(b.inputs)
    covered = torch.zeros_like(flat_in, dtype=torch.bool)
    for off in (0, 1, 4, 5):
        covered.scatter_(-1, (anchor + off)[..., None], True)
    # Cells outside the tile are untouched one-hots of the input.
    assert torch.equal(probs.argmax(-1)[~covered], flat_in[~covered])
    probs[covered].sum().backward()
    assert s.grad is not None and s.grad.abs().sum() > 0


def _true_anchor_probs(b):
    return F.one_hot(find_anchor(b.inputs), 16).float()


def test_soft_anchor_paste_at_true_anchor_equals_find_paste():
    b = _batch()
    s = torch.softmax(torch.randn(64, 4, 10), -1)[:, None].expand(-1, 4, -1, -1)
    assert torch.allclose(paste_soft_anchor(s, b.inputs, _true_anchor_probs(b)), execute(s, b.inputs))


def test_soft_anchor_paste_is_a_distribution_and_differentiable():
    b = _batch()
    s = torch.softmax(torch.randn(64, 4, 4, 10), -1).requires_grad_()
    a = torch.softmax(torch.randn(64, 4, 16), -1).requires_grad_()
    probs = paste_soft_anchor(s, b.inputs, a)
    assert probs.shape == (64, 4, 16, 10)
    assert torch.allclose(probs.sum(-1), torch.ones(64, 4, 16))
    probs[..., 3].sum().backward()
    assert s.grad.abs().sum() > 0 and a.grad.abs().sum() > 0


def test_off_grid_anchor_drops_the_cells_that_fall_off():
    x = torch.zeros(1, 4, 4, dtype=torch.long)
    x[0, 1, 1] = 1
    s = F.one_hot(torch.tensor([[2, 3, 4, 5]]), 10).float()
    probs = paste_soft_anchor(s, x, F.one_hot(torch.tensor([15]), 16).float())  # bottom-right corner
    expected = x.flatten(-2).clone()
    expected[0, 15] = 2  # only tile cell (0, 0) lands on the grid
    assert torch.equal(probs.argmax(-1), expected)
    assert placements().sum() == 9 * 4 + 6 * 2 + 1  # 9 full fits, 6 edge positions keep 2 cells, corner keeps 1
