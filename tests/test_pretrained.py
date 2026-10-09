"""Weight conversion from the Flax checkpoint.

The round-trip tests need no network: they serialize a random model into the
checkpoint's Flax layout (the inverse of the converter) and convert it back.
Exact agreement with the original JAX model is checked separately on Colab
(colab/check_port_parity.py). The last test uses the real checkpoint if
it's already downloaded (pretrained.cache_dir()), and is skipped otherwise.
"""
import msgpack
import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.models import pretrained as pt
from src.models.lpn import LPN

_INVERSE = {
    pt._dense: lambda w, shape: w.T,
    pt._same: lambda w, shape: w,
    pt._attn_in: lambda w, shape: w.T.reshape(shape),
    pt._attn_out: lambda w, shape: w.T.reshape(shape),
}


def _to_flax_tree(model: LPN) -> dict:
    """Our state_dict -> the checkpoint's nested Flax params layout."""
    cfg, sd, tree = model.cfg, model.state_dict(), {}
    heads, hd, emb = cfg.num_heads, cfg.head_dim, cfg.emb_dim
    for t, f, conv in pt.param_map(cfg):
        shape = (emb, heads, hd) if conv is pt._attn_in else (heads, hd, emb)
        node = tree
        *parents, leaf = f.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _INVERSE[conv](sd[t].numpy(), shape)
    return tree


def _pack_flax_msgpack(tree: dict) -> bytes:
    """flax.serialization.msgpack_serialize's array encoding (ext type 1)."""
    def default(obj):
        if isinstance(obj, np.ndarray):
            return msgpack.ExtType(1, msgpack.packb((obj.shape, obj.dtype.name, obj.tobytes())))
        raise TypeError(type(obj))
    return msgpack.packb({"params": tree, "step": 3}, default=default)


def test_param_map_covers_every_backbone_parameter():
    model = LPN(ModelConfig())
    mapped = {t for t, _, _ in pt.param_map(model.cfg)}
    backbone = {k for k in model.state_dict() if not k.startswith("head.")}
    assert mapped == backbone


def test_roundtrip_through_flax_layout_is_exact():
    torch.manual_seed(0)
    source = LPN(ModelConfig())
    state = pt.read_flax_msgpack(_pack_flax_msgpack(_to_flax_tree(source)))
    sd = pt.flax_params_to_state_dict(state["params"], source.cfg)
    for key, value in sd.items():
        assert torch.equal(value, source.state_dict()[key]), key


def test_converter_rejects_architecture_mismatch():
    tree = _to_flax_tree(LPN(ModelConfig()))
    tree["encoder"]["extra_bias"] = np.zeros(3, np.float32)
    with pytest.raises(KeyError, match="doesn't use"):
        pt.flax_params_to_state_dict(tree, ModelConfig())
    del tree["encoder"]["extra_bias"], tree["decoder"]["context_embed"]
    with pytest.raises(KeyError, match="lacks"):
        pt.flax_params_to_state_dict(tree, ModelConfig())


def _cached_checkpoint():
    return pt.cached_files()


@pytest.mark.skipif(_cached_checkpoint() is None, reason="checkpoint not downloaded (pretrained.download)")
def test_real_checkpoint_solves_most_tasks_without_search():
    from src.data.pattern2d import NUM_TILES, sample_tasks

    model = pt.load_pretrained(*_cached_checkpoint()).eval()
    assert model.cfg.backbone() == ModelConfig().backbone()
    ctx, qry = sample_tasks(torch.arange(NUM_TILES), 128, 5, torch.Generator().manual_seed(0)).split_query()
    with torch.no_grad():
        z = model.context_latent(ctx.inputs, ctx.outputs)
        pred = model.decoder.generate(qry.inputs[:, 0], z)
    assert (pred == qry.outputs[:, 0]).float().mean() > 0.9  # a mis-ported model scores near chance
