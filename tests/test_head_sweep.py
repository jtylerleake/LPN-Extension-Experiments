import pytest
import torch

from src.config import ModelConfig
from src.data.pattern2d import NUM_TILES, sample_tasks, split_tile_ids, tile_ids
from src.models.lpn import LPN
from src.models.mlp_head import MLPHead, ResidualBlock
from src.training import head_sweep as hs
from src.training.losses import exec_nll
from src.utils.resume import read_jsonl_rows

SMALL = ModelConfig(num_layers=1, num_heads=2, head_dim=8, head_hidden=16)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def cache():
    torch.manual_seed(0)
    return hs.EncodingCache(LPN(SMALL), CPU)


def test_default_head_is_the_original_architecture():
    head = MLPHead(ModelConfig())
    assert [k for k in head.state_dict()] == [f"net.{i}.{p}" for i in (0, 2, 4) for p in ("weight", "bias")]
    assert sum(p.numel() for p in head.parameters()) == 2 * 128 + 128 + 128 * 128 + 128 + 128 * 40 + 40


@pytest.mark.parametrize("depth,residual,act", [(3, False, "silu"), (4, True, "relu"), (6, True, "silu")])
def test_configurable_heads_have_the_right_shape(depth, residual, act):
    cfg = ModelConfig(head_hidden=32, head_depth=depth, head_residual=residual, head_activation=act)
    head = MLPHead(cfg)
    assert head(torch.randn(5, 2)).shape == (5, 4, 10)
    linears = sum(isinstance(m, torch.nn.Linear) for m in head.net)  # top level only: blocks hold their own
    blocks = sum(isinstance(m, ResidualBlock) for m in head.net)
    assert linears - 1 + blocks == depth  # every hidden layer, not counting the output layer


def test_cache_matches_the_live_encoder(cache):
    torch.manual_seed(0)
    model = LPN(SMALL).eval()
    b = sample_tasks(torch.arange(NUM_TILES), 32, 4, torch.Generator().manual_seed(3))
    with torch.no_grad():
        mu, logvar = model.encoder(b.inputs, b.outputs)
    ids = tile_ids(b.tiles)[:, None] * 9 + b.anchors[..., 0] * 3 + b.anchors[..., 1]
    assert torch.allclose(cache.mu[ids], mu, atol=1e-5)
    assert torch.allclose(cache.logvar[ids], logvar, atol=1e-5)


def test_tile_exec_nll_equals_exec_nll():
    b = sample_tasks(torch.arange(NUM_TILES), 16, 4, torch.Generator().manual_seed(0))
    logits = torch.randn(16, 4, 4, 10)
    tiles = b.tiles[:, None].expand(-1, 4, -1)
    assert torch.allclose(hs.tile_exec_nll(logits, tiles), exec_nll(logits, b.inputs, b.outputs), atol=1e-5)


def test_heldout_split_is_disjoint_complete_and_deterministic():
    _, heldout = split_tile_ids(0.1, 0)
    val, test = hs.split_heldout(heldout)
    assert len(set(val.tolist()) & set(test.tolist())) == 0
    assert sorted(val.tolist() + test.tolist()) == heldout.tolist()
    assert torch.equal(val, hs.split_heldout(heldout)[0])


def test_rounds_run_in_order_resume_and_only_c_sees_test(cache, tmp_path):
    results = tmp_path / "results.jsonl"
    splits = hs.Splits(cache)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return cache, splits

    with pytest.raises(RuntimeError, match="round A"):
        hs.run_round("B", results, factory, max_steps=2, eval_every=1)
    a = hs.run_round("A", results, factory, max_steps=2, eval_every=1)
    assert len(a) == len(hs.DEPTHS) * len(hs.WIDTHS)
    assert all("test/tile_full_acc" not in r["final"] for r in a)
    n_rows = len(read_jsonl_rows(results))
    hs.run_round("A", results, factory, max_steps=2)  # resume: nothing left to do
    assert len(read_jsonl_rows(results)) == n_rows and calls["n"] == 1

    b = hs.run_round("B", results, factory, max_steps=2, eval_every=1)
    assert {(r["spec"]["depth"], r["spec"]["width"]) for r in b} <= {(r["spec"]["depth"], r["spec"]["width"]) for r in a}
    c = hs.run_round("C", results, factory, max_steps=2, eval_every=1)
    assert len(c) == hs.TOP_K * len(hs.SEEDS)
    assert all("test/tile_full_acc" in r["final"] for r in c)
    assert "test/tile_full_acc" in hs.summary_table(c)
