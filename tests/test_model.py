import inspect

import torch

from lpn_interp.config import ModelConfig
from lpn_interp.data.pattern2d import NUM_TILES, sample_tasks
from lpn_interp.models.decoder import PREFIX_LEN, SEQ_LEN, prefix_lm_mask
from lpn_interp.models.encoder import loo_pool, pool
from lpn_interp.models.lpn import LPN, load_checkpoint, save_checkpoint
from lpn_interp.models.mlp_head import MLPHead, hard_tile, soft_tile

SMALL = ModelConfig(d_model=32, num_layers=1, num_heads=2, mlp_dim=64, head_hidden=16)


def _model(seed=0):
    torch.manual_seed(seed)
    return LPN(SMALL).eval()


def _batch(n=6, pairs=4):
    return sample_tasks(torch.arange(NUM_TILES), n, pairs, torch.Generator().manual_seed(0))


def test_encoder_shapes_and_pooling():
    b = _batch()
    mu, logvar = _model().encoder(b.inputs, b.outputs)
    assert mu.shape == logvar.shape == (6, 4, 2)
    assert pool(mu, logvar)[0].shape == (6, 2)


def test_loo_pool_equals_brute_force_reencoding():
    model, b = _model(), _batch()
    loo_mu, loo_logvar = loo_pool(*model.encoder(b.inputs, b.outputs))
    for i in range(4):
        keep = [j for j in range(4) if j != i]
        mu, logvar = pool(*model.encoder(b.inputs[:, keep], b.outputs[:, keep]))
        assert torch.allclose(loo_mu[:, i], mu, atol=1e-5)
        assert torch.allclose(loo_logvar[:, i], logvar, atol=1e-5)


def test_encoder_is_permutation_invariant():
    model, b = _model(), _batch()
    perm = torch.tensor([2, 0, 3, 1])
    z = pool(*model.encoder(b.inputs, b.outputs))[0]
    z_perm = pool(*model.encoder(b.inputs[:, perm], b.outputs[:, perm]))[0]
    assert torch.allclose(z, z_perm, atol=1e-5)


def test_prefix_lm_mask():
    mask = prefix_lm_mask()
    assert mask.shape == (SEQ_LEN, SEQ_LEN)
    assert not mask[:PREFIX_LEN, :PREFIX_LEN].any()  # prefix is bidirectional
    assert mask[:PREFIX_LEN, PREFIX_LEN:].all()  # prefix never sees outputs
    out = torch.arange(PREFIX_LEN, SEQ_LEN)
    n = len(out)
    assert torch.equal(mask[out][:, out], torch.ones(n, n, dtype=torch.bool).triu(1))  # causal outputs


def test_decoder_has_no_target_leakage():
    """Logit for output cell j must not change when cells >= j of y change."""
    model, b = _model(), _batch()
    x, y = b.inputs[:, 0], b.outputs[:, 0]
    z = torch.randn(6, 2)
    base = model.decoder.logits(x, y, z)
    for j in (0, 5, 15):
        y2 = y.clone().flatten(-2)
        y2[:, j:] = (y2[:, j:] + 3) % 10
        logits = model.decoder.logits(x, y2.view_as(y), z)
        assert torch.allclose(base[:, : j + 1], logits[:, : j + 1], atol=1e-5)


def test_decoder_nll_and_generate_shapes():
    model, b = _model(), _batch()
    z = torch.randn(6, 4, 2)
    assert model.decoder.nll(b.inputs, b.outputs, z).shape == (6, 4)
    gen = model.decoder.generate(b.inputs, z)
    assert gen.shape == b.outputs.shape and gen.dtype == torch.long


def test_generate_matches_teacher_forced_argmax_on_own_output():
    model, b = _model(), _batch()
    x, z = b.inputs[:, 0], torch.randn(6, 2)
    gen = model.decoder.generate(x, z)
    assert torch.equal(model.decoder.logits(x, gen, z).argmax(-1).view_as(gen), gen)


def test_head_takes_only_z():
    params = list(inspect.signature(MLPHead.forward).parameters)
    assert params == ["self", "z"]
    logits = MLPHead(SMALL)(torch.randn(3, 2))
    assert logits.shape == (3, 4, 10)
    assert torch.allclose(soft_tile(logits).sum(-1), torch.ones(3, 4))
    assert hard_tile(logits).shape == (3, 4)


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    save_checkpoint(tmp_path / "c.pt", model, step=7)
    loaded, state = load_checkpoint(tmp_path / "c.pt")
    assert state["step"] == 7
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
