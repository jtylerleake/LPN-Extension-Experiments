"""Load the paper's pretrained Pattern-2D LPN into our PyTorch port.

The checkpoint (clement-bonnet/lpn-2d on Hugging Face, pinned revision,
checksum-verified) is a Flax TrainState
serialized with flax.serialization (msgpack, arrays as ext type 1). We decode
it with plain `msgpack` -- no JAX/Flax needed -- and map each Flax parameter
onto the port's state_dict (Flax Dense kernels are (in, out); torch Linear
weights are (out, in); attention kernels are (emb, heads, head_dim) /
(heads, head_dim, emb)). The port's outputs are checked against the original
JAX model by colab/check_port_parity.py.

The returned model's MLP heads are freshly initialized: they aren't part of
the checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from src.config import ModelConfig
from src.models.lpn import LPN

HF_REPO = "clement-bonnet/lpn-2d"
HF_REVISION = "3c6f1d76583d5e899d8153d6049bb43b396964dc"  # pinned repo commit (2024-11-13)
HF_CHECKPOINT = "quiet-thunder-789--checkpoint:v0"
CHECKPOINT_SHA256 = {  # state.msgpack's is the LFS oid Hugging Face publishes
    "config.yaml": "d2da60d9e4556745d2a2ed932ce36d7131fde25cf26ada9ad85b417eb5cbcd63",
    "state.msgpack": "60b3368d8a3c6232867a43b252597f16dbec5de0264bf23168d938a461256a3d",
}
# The lpn repo commit whose model code produced this checkpoint (used by the parity check).
LPN_COMMIT = "0adfe56b86d2cba5ae5794edb02da6399a96d98a"

_MSGPACK_NDARRAY, _MSGPACK_NPSCALAR = 1, 3


def cache_dir() -> Path:
    """Local copy of the checkpoint. The HF folder name contains ':', which
    Windows paths can't hold, so files are stored under a sanitized name."""
    root = Path(os.environ.get("LPN_CACHE", Path.home() / ".cache" / "lpn_interp"))
    return root / HF_REPO.replace("/", "--") / HF_CHECKPOINT.replace(":", "_")


def cached_files() -> tuple[Path, Path] | None:
    paths = tuple(cache_dir() / name for name in CHECKPOINT_SHA256)
    return paths if all(p.exists() for p in paths) else None


def download() -> tuple[Path, Path]:
    """(config.yaml, state.msgpack) local paths, fetched once from the pinned
    HF revision and verified against their SHA-256."""
    import hashlib
    import urllib.parse
    import urllib.request

    target = cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name, sha in CHECKPOINT_SHA256.items():
        path = target / name
        if path.exists():
            continue
        remote = urllib.parse.quote(f"{HF_CHECKPOINT}/{name}")
        url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{remote}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"checksum mismatch for {url}")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return target / "config.yaml", target / "state.msgpack"


def read_flax_msgpack(data: bytes) -> dict:
    """flax.serialization.msgpack_restore, without flax."""
    import msgpack

    def ext_hook(code, payload):
        if code == _MSGPACK_NDARRAY:
            shape, dtype, buf = msgpack.unpackb(payload)
            return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()
        if code == _MSGPACK_NPSCALAR:
            dtype, buf = msgpack.unpackb(payload)
            return np.frombuffer(buf, dtype=np.dtype(dtype))[0]
        return msgpack.ExtType(code, payload)

    return msgpack.unpackb(data, ext_hook=ext_hook, raw=False, strict_map_key=False)


def model_config_from_yaml(cfg: dict) -> ModelConfig:
    enc, dec = cfg["encoder_transformer"], cfg["decoder_transformer"]
    layer = enc["transformer_layer"]
    if enc["num_layers"] != dec["num_layers"] or layer != dec["transformer_layer"]:
        raise ValueError("encoder and decoder transformer settings differ; the port assumes they match")
    if (enc["max_rows"], enc["max_cols"]) != (4, 4) or not enc["variational"]:
        raise ValueError("expected a variational 4x4 checkpoint")
    return ModelConfig(latent_dim=enc["latent_dim"], num_layers=enc["num_layers"], num_heads=layer["num_heads"],
                       head_dim=layer["emb_dim_per_head"], mlp_factor=layer["mlp_dim_factor"])


# (torch suffix, flax path suffix, how to convert the array)
def _dense(a):
    return a.T


def _attn_in(a):  # (emb, heads, head_dim) -> torch (heads*head_dim, emb)
    return a.reshape(a.shape[0], -1).T


def _attn_out(a):  # (heads, head_dim, emb) -> torch (emb, heads*head_dim)
    return a.reshape(-1, a.shape[-1]).T


def _same(a):
    return a


def param_map(cfg: ModelConfig) -> list[tuple[str, str, callable]]:
    """(torch state_dict key, flax param path, converter) for every backbone parameter."""
    entries = []
    for part in ("encoder", "decoder"):
        for i in range(cfg.num_layers):
            t, f = f"{part}.layers.{i}", f"{part}/TransformerLayer_{i}"
            for name in ("query", "key", "value"):
                entries.append((f"{t}.attn.{name}.weight", f"{f}/MultiHeadAttention_0/{name}/kernel", _attn_in))
            entries.append((f"{t}.attn.out.weight", f"{f}/MultiHeadAttention_0/out/kernel", _attn_out))
            entries.append((f"{t}.mlp.0.weight", f"{f}/MlpBlock_0/Dense_0/kernel", _dense))
            entries.append((f"{t}.mlp.2.weight", f"{f}/MlpBlock_0/Dense_1/kernel", _dense))
    for name in ("colors_embed", "pos_row_embed", "pos_col_embed", "grid_shapes_row_embed", "grid_shapes_col_embed"):
        for part in ("encoder", "decoder"):
            entries.append((f"{part}.{name}.weight", f"{part}/{name}/embedding", _same))
    entries += [
        ("encoder.channels_embed.weight", "encoder/channels_embed/embedding", _same),
        ("encoder.cls_token.weight", "encoder/cls_token/embedding", _same),
        ("encoder.to_mu.weight", "encoder/Dense_0/kernel", _dense),
        ("encoder.to_logvar.weight", "encoder/Dense_1/kernel", _dense),
        ("decoder.input_output_embed.weight", "decoder/input_output_embed/embedding", _same),
        ("decoder.context_embed.weight", "decoder/context_embed/kernel", _dense),
        ("decoder.shape_row_logits_proj.weight", "decoder/shape_row_logits_proj/kernel", _dense),
        ("decoder.shape_col_logits_proj.weight", "decoder/shape_col_logits_proj/kernel", _dense),
        ("decoder.grid_logits_proj.weight", "decoder/grid_logits_proj/kernel", _dense),
    ]
    return entries


def _flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        flat |= _flatten(value, path) if isinstance(value, dict) else {path: value}
    return flat


def flax_params_to_state_dict(params: dict, cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Flax params tree -> backbone state_dict. Fails loudly on any unmapped or unused parameter."""
    flat = _flatten(params)
    mapping = param_map(cfg)
    used = {f for _, f, _ in mapping}
    if missing := sorted(used - flat.keys()):
        raise KeyError(f"checkpoint lacks parameters the port needs: {missing}")
    if unused := sorted(flat.keys() - used):
        raise KeyError(f"checkpoint has parameters the port doesn't use (architecture mismatch?): {unused}")
    return {t: torch.from_numpy(np.ascontiguousarray(conv(flat[f]))).float() for t, f, conv in mapping}


def load_pretrained(config_path: str | Path | None = None, state_path: str | Path | None = None,
                    **head_settings) -> LPN:
    """The pretrained LPN as our PyTorch model, with fresh MLP heads. Downloads from HF
    unless paths are given. `head_settings` are ModelConfig fields outside the backbone
    (head_hidden, head_depth, ..., architecture); the backbone always comes from the checkpoint."""
    import yaml

    if config_path is None or state_path is None:
        config_path, state_path = download()
    cfg = model_config_from_yaml(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    if clash := sorted(set(head_settings) & set(cfg.backbone())):
        raise ValueError(f"backbone settings come from the checkpoint, not arguments: {clash}")
    cfg = ModelConfig(**{**cfg.model_dump(), **head_settings})
    state = read_flax_msgpack(Path(state_path).read_bytes())
    model = LPN(cfg)
    backbone = flax_params_to_state_dict(state["params"], cfg)
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    if unexpected or any(not k.startswith(("head.", "anchor_head.")) for k in missing):
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    return model
