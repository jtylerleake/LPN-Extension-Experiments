import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from src.config import ModelConfig
from src.data.pattern2d import NUM_TILES, sample_tasks
from src.models.lpn import LPN, save_checkpoint
from src.viz.pipeline_gif import NUM_FRAMES, PHASES, draw_frame, trace_task

SMALL = ModelConfig(num_layers=1, num_heads=2, head_dim=8, head_hidden=16)


def _trace(cfg=SMALL):
    torch.manual_seed(0)
    task = sample_tasks(torch.arange(NUM_TILES), 1, 5, torch.Generator().manual_seed(0))
    return trace_task(LPN(cfg), *task.split_query()), task


def test_trace_matches_the_task():
    tr, task = _trace()
    assert tr.ctx_inputs.shape == (4, 4, 4) and tr.z.shape == (SMALL.latent_dim,)
    assert tr.anchor == tuple(task.anchors[0, -1].tolist())
    r, c = tr.anchor
    expected = tr.query_input.copy()
    expected[r:r + 2, c:c + 2] = tr.tile.reshape(2, 2)
    np.testing.assert_array_equal(tr.head_pred, expected)


def test_architecture_2_is_rejected():
    with pytest.raises(ValueError, match="architecture 1"):
        _trace(SMALL.model_copy(update={"architecture": 2}))


def test_every_phase_boundary_draws():
    tr, _ = _trace()
    fig, ax = plt.subplots()
    for t in sorted({t for span in PHASES.values() for t in span} | {NUM_FRAMES - 1}):
        draw_frame(ax, tr, t)
    plt.close(fig)


def test_cli_writes_a_gif(tmp_path, monkeypatch):
    from scripts.make_gif import main

    monkeypatch.setattr("src.viz.pipeline_gif.NUM_FRAMES", 3)  # the full 72 frames take ~30 s
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "ckpt.pt", LPN(SMALL))
    out = tmp_path / "pipeline.gif"
    assert main(["--ckpt", str(tmp_path / "ckpt.pt"), "--tile", "2,3,4,5", "--out", str(out), "--dpi", "12"]) == 0
    with Image.open(out) as im:
        assert im.n_frames > 1
