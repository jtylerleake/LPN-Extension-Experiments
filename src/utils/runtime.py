"""Device / mixed-precision choices shared by every entrypoint.

The same scripts run on CPU (local tests) and on whatever Colab GPU the
session got (T4, L4, A100, H100, ...). T4 has no bf16 tensor cores, so AMP
uses fp16 + a GradScaler there and bf16 (no scaler needed) everywhere newer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

# Written by colab/colab_runtime.py's `sync` next to the uploaded code.
BUNDLE_MANIFEST = Path(__file__).resolve().parents[2] / "BUNDLE.json"


def output_root() -> Path:
    """Where runs write by default: $LPN_OUTPUT_ROOT (set by colab_runtime.py
    on the VM: /content/outputs, or Drive with --drive), else ./outputs."""
    return Path(os.environ.get("LPN_OUTPUT_ROOT", "outputs"))


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible (did the Colab session get one?)")
    return device


def amp_dtype(device: torch.device, enabled: bool = True) -> torch.dtype | None:
    """Autocast dtype for `device`, or None to run in fp32."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def describe_runtime(device: torch.device) -> dict:
    """Hardware + code provenance, logged at the start of every run."""
    info = {"torch": torch.__version__, "device": str(device), "pid": os.getpid()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info |= {"gpu": props.name, "gpu_mem_gb": round(props.total_memory / 2**30, 1),
                 "bf16": torch.cuda.is_bf16_supported()}
    if BUNDLE_MANIFEST.exists():
        info["bundle"] = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    return info
