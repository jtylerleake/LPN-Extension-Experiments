import inspect

import torch

from src.config import ModelConfig
from src.data.pattern2d import NUM_TILES, sample_tasks
from src.models.decoder import INPUT_LEN, SEQ_LEN, attention_mask
from src.models.lpn import LPN, load_checkpoint, loo_mean, save_checkpoint
from src.models.mlp_head import AnchorHead, hard_anchor, MLPHead, hard_tile, soft_tile

SMALL = ModelConfig(num_layers=1, num_heads=2, head_dim=8, head_hidden=16)


def _model(cfg=SMALL, seed=0):
    torch.manual_seed(seed)
    return LPN(cfg).eval()


def _batch(n=6, pairs=4):
    return sample_tasks(torch.arange(NUM_TILES), n, pairs, torch.Generator().manual_seed(0))


def test_default_config_matches_checkpoint_size():
    """The pretrained checkpoint has 254,664 backbone parameters (see tests/test_pretrained.py)."""
    model = LPN(ModelConfig())
    assert sum(p.numel() for n, p in model.named_parameters() if not n.startswith("head.")) == 254_664


def test_encoder_shapes():
    b = _batch()
    mu, logvar = _model().encoder(b.inputs, b.outputs)
    assert mu.shape == logvar.shape == (6, 4, 2)
    assert mu.dtype == torch.float32


def test_encoder_encodes_pairs_independently():
    model, b = _model(), _batch()
    mu, _ = model.encoder(b.inputs, b.outputs)
    mu_one, _ = model.encoder(b.inputs[:, 2], b.outputs[:, 2])
    assert torch.allclose(mu[:, 2], mu_one, atol=1e-5)


def test_loo_mean_equals_brute_force():
    z = torch.randn(3, 4, 2)
    loo = loo_mean(z)
    for i in range(4):
        keep = [j for j in range(4) if j != i]
        assert torch.allclose(loo[:, i], z[:, keep].mean(1), atol=1e-6)


def test_context_latent_is_permutation_invariant():
    model, b = _model(), _batch()
    perm = torch.tensor([2, 0, 3, 1])
    z = model.context_latent(b.inputs, b.outputs)
    assert torch.allclose(z, model.context_latent(b.inputs[:, perm], b.outputs[:, perm]), atol=1e-5)


def test_attention_mask_matches_original_structure():
    mask = attention_mask()
    assert mask.shape == (SEQ_LEN, SEQ_LEN) == (37, 37)
    assert mask[:INPUT_LEN, :INPUT_LEN].all()  # input <-> input
    assert mask[:INPUT_LEN, INPUT_LEN].all()  # inputs see the context token...
    assert not mask[:INPUT_LEN, INPUT_LEN + 1:].any()  # ...but nothing after it
    assert mask[INPUT_LEN:, :INPUT_LEN].all()
    out = mask[INPUT_LEN:, INPUT_LEN:]
    assert torch.equal(out, torch.ones_like(out).tril())


def test_decoder_has_no_target_leakage():
    """Logits for output cell k must not change when cells >= k of y change."""
    model, b = _model(), _batch()
    x, y, z = b.inputs[:, 0], b.outputs[:, 0], torch.randn(6, 2)
    _, _, base = model.decoder(x, y, z)
    for k in (0, 5, 15):
        y2 = y.clone().flatten(-2)
        y2[:, k:] = (y2[:, k:] + 3) % 10
        _, _, logits = model.decoder(x, y2.view_as(y), z)
        assert torch.allclose(base[:, : k + 1], logits[:, : k + 1], atol=1e-5)
        if k < 15:  # later logits do depend on y (the last cell's token feeds no logit)
            assert not torch.allclose(base, logits, atol=1e-5)


def test_decoder_outputs_and_nll():
    model, b = _model(), _batch()
    z = torch.randn(6, 4, 2)
    rows, cols, grid = model.decoder(b.inputs, b.outputs, z)
    assert rows.shape == cols.shape == (6, 4, 4) and grid.shape == (6, 4, 16, 10)
    assert model.decoder.nll(b.inputs, b.outputs, z).shape == (6, 4)


def test_generate_matches_teacher_forced_argmax_on_own_output():
    model, b = _model(), _batch()
    x, z = b.inputs[:, 0], torch.randn(6, 2)
    gen = model.decoder.generate(x, z)
    _, _, logits = model.decoder(x, gen, z)
    assert torch.equal(logits.argmax(-1).view_as(gen), gen)


def test_head_takes_only_z():
    assert list(inspect.signature(MLPHead.forward).parameters) == ["self", "z"]
    logits = MLPHead(SMALL)(torch.randn(3, 2))
    assert logits.shape == (3, 4, 10)
    assert torch.allclose(soft_tile(logits).sum(-1), torch.ones(3, 4))
    assert hard_tile(logits).shape == (3, 4)


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    save_checkpoint(tmp_path / "c.pt", model, step=7)
    loaded, state = load_checkpoint(tmp_path / "c.pt")
    assert state["step"] == 7
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_anchor_head_takes_only_z_and_exists_only_in_architecture_2():
    assert list(inspect.signature(AnchorHead.forward).parameters) == ["self", "z"]
    logits = AnchorHead(SMALL)(torch.randn(3, 2))
    assert logits.shape == (3, 16)
    assert hard_anchor(logits).shape == (3,)
    assert LPN(SMALL).anchor_head is None
    assert not any(k.startswith("anchor_head.") for k in LPN(SMALL).state_dict())
    arch2 = LPN(SMALL.model_copy(update={"architecture": 2}))
    tile_logits, anchor_logits = arch2.program_logits(torch.randn(3, 2))
    assert tile_logits.shape == (3, 4, 10) and anchor_logits.shape == (3, 16)
