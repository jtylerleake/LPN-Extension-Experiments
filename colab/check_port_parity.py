"""Check our PyTorch port against the original JAX/Flax LPN on the pretrained checkpoint.

    python3 colab/colab_runtime.py run -- colab/check_port_parity.py

Runs the original model (github.com/clement-bonnet/lpn at the checkpoint's
commit) and our port on the same random tasks and latents, and compares:
- encoder mu / logvar per pair,
- decoder shape and grid logits (teacher-forced),
- the original per-pair loss (LPN._loss_from_pair_and_context) vs Decoder.nll.
Both sides run on CPU in float32 for an exact comparison.

The original repo's package is also called `src`, so its side runs in a
separate subprocess (cwd = its checkout) and hands arrays back through an
.npz file. Its extra packages are colab/requirements-parity.txt: jax must be
Colab's preinstalled build; flax/chex/optax are installed if missing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.pattern2d import NUM_TILES, sample_tasks
from src.models.pretrained import HF_CHECKPOINT, HF_REPO, LPN_COMMIT, download, load_pretrained

TOLERANCE = 1e-4

REFERENCE = r'''
import sys
import numpy as np, yaml
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax.serialization import msgpack_restore
from src.models.lpn import LPN
from src.models.transformer import DecoderTransformer, EncoderTransformer
from src.models.utils import DecoderTransformerConfig, EncoderTransformerConfig, TransformerLayerConfig

config_path, state_path, data_path, out_path = sys.argv[1:5]
cfg = yaml.safe_load(open(config_path))
strip = lambda d: {k: v for k, v in d.items() if k != "_target_"}
def build(cls, d):
    d = strip(d)
    return cls(**{**d, "transformer_layer": TransformerLayerConfig(**strip(d["transformer_layer"]))})
encoder = EncoderTransformer(build(EncoderTransformerConfig, cfg["encoder_transformer"]))
decoder = DecoderTransformer(build(DecoderTransformerConfig, cfg["decoder_transformer"]))
params = msgpack_restore(open(state_path, "rb").read())["params"]

data = np.load(data_path)
inputs, outputs, z = (jnp.asarray(data[k]) for k in ("inputs", "outputs", "z"))
pairs = jnp.stack([inputs, outputs], axis=-1)
grid_shapes = jnp.full((*inputs.shape[:-2], 2, 2), 4, dtype=jnp.int32)
mu, logvar = encoder.apply({"params": params["encoder"]}, pairs, grid_shapes, dropout_eval=True)

shape_tokens = jnp.full((*inputs.shape[:-2], 2), 4, dtype=inputs.dtype)
flat = lambda g: jnp.concatenate([shape_tokens, g.reshape(*g.shape[:-2], -1)], axis=-1)
rows, cols, grid = decoder.apply({"params": params["decoder"]}, flat(inputs), flat(outputs), z, dropout_eval=True)

lpn = LPN(encoder=encoder, decoder=decoder)
loss, _ = lpn.apply({"params": params}, z, pairs, grid_shapes, True, method=LPN._loss_from_pair_and_context)
np.savez(out_path, mu=mu, logvar=logvar, rows=rows, cols=cols, grid=grid, loss=loss)
print("reference: jax", jax.__version__, "on", jax.devices()[0].platform)
'''


def ensure_reference_repo(lpn_dir: Path) -> None:
    if not (lpn_dir / ".git").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/clement-bonnet/lpn.git", str(lpn_dir)], check=True)
    subprocess.run(["git", "-C", str(lpn_dir), "checkout", "-q", LPN_COMMIT], check=True)


def ensure_parity_packages() -> None:
    """flax/chex/optax if missing; jax must be Colab's own (colab/requirements-parity.txt)."""
    here = Path(__file__).resolve().parent
    setup = [sys.executable, str(here / "vm_setup.py"), str(here / "requirements-parity.txt")]
    rc = subprocess.run(setup, check=False).returncode
    if rc:
        sys.exit("parity-check packages unavailable (see above); this check needs JAX, so run it on Colab")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpn-dir", default="/content/lpn_reference", help="where to clone the original repo")
    parser.add_argument("--num-tasks", type=int, default=64)
    args = parser.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    ensure_parity_packages()
    lpn_dir = Path(args.lpn_dir)
    ensure_reference_repo(lpn_dir)
    config_path, state_path = download()
    print(f"checkpoint {HF_REPO}/{HF_CHECKPOINT}; reference code @ {LPN_COMMIT[:7]}")

    batch = sample_tasks(torch.arange(NUM_TILES), args.num_tasks, 4, torch.Generator().manual_seed(0))
    z = torch.randn(args.num_tasks, 4, 2, generator=torch.Generator().manual_seed(1)) * 1.5

    with tempfile.TemporaryDirectory() as tmp:
        data_path, out_path = Path(tmp) / "data.npz", Path(tmp) / "ref.npz"
        np.savez(data_path, inputs=batch.inputs.numpy().astype(np.int32),
                 outputs=batch.outputs.numpy().astype(np.int32), z=z.numpy())
        subprocess.run([sys.executable, "-c", REFERENCE, str(config_path), str(state_path), str(data_path),
                        str(out_path)], cwd=lpn_dir, env={"PYTHONPATH": str(lpn_dir), **_env()}, check=True)
        ref = dict(np.load(out_path))

    model = load_pretrained(config_path, state_path).eval()
    with torch.no_grad():
        mu, logvar = model.encoder(batch.inputs, batch.outputs)
        rows, cols, grid = model.decoder(batch.inputs, batch.outputs, z)
        loss = model.decoder.nll(batch.inputs, batch.outputs, z)
    ours = {"mu": mu, "logvar": logvar, "rows": rows, "cols": cols, "grid": grid, "loss": loss}

    worst = 0.0
    for key, value in ours.items():
        diff = float(np.abs(value.numpy() - ref[key]).max())
        worst = max(worst, diff)
        print(f"  {key:<7} shape {tuple(value.shape)!s:<16} max |ours - original| = {diff:.2e}")
    ok = worst < TOLERANCE
    print(f"{'PASS' if ok else 'FAIL'}: worst difference {worst:.2e} (tolerance {TOLERANCE:.0e})")
    return 0 if ok else 1


def _env() -> dict:
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # ours points at this repo, whose `src` would shadow the original's
    env["JAX_PLATFORMS"] = "cpu"
    return env


if __name__ == "__main__":
    sys.exit(main())
