"""Reference input and statistics for the HVI-CIDNet parity test (no ``mon`` imports).

``OFFICIAL`` in ``test_hvi_cidnet.py`` comes from the official ``net/*.py``
(https://github.com/Fediory/HVI-CIDNet, MIT) with the Hugging Face releases
(``Fediory/HVI-CIDNet-<name>/model.safetensors``), on CPU, using the official ``eval.py`` gates.
Needs ``einops``, ``huggingface_hub`` and ``safetensors``. Running it executes third-party code: get
the owner's go-ahead first. Regenerate with:

    uv run --with einops --with huggingface_hub --with safetensors \\
        python -I tests/models/lle/hvi_cidnet_reference.py <checkout> <dir with <name>/model.safetensors>
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

_fmt_spec = importlib.util.spec_from_file_location(
    "reference_format", Path(__file__).parents[2] / "reference_format.py",
)
assert _fmt_spec is not None and _fmt_spec.loader is not None
_fmt = importlib.util.module_from_spec(_fmt_spec)
_fmt_spec.loader.exec_module(_fmt)
format_literal = _fmt.format_literal

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)

# (Hugging Face name, gated + alpha_s, gated2 + alpha) as in the official eval.py
CASES = (("LOLv1-woperc", None, None), ("LOLv1-woperc", 1.3, None),
         ("LOLv2-real-bestPSNR", None, 0.8), ("Sony-Total-Dark", None, None))


def reference_input(batch: int = 2, h: int = 24, w: int = 32) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.4]`` (no RNG; sizes divisible by 8; a gray corner)."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(batch, 3, h, w, dtype=torch.float64)
    for b in range(batch):
        for c in range(3):
            wave = torch.sin(0.31 * (c + 1) * i + 0.19 * (b + 1) * j + 0.7 * c) * torch.cos(0.11 * j - 0.27 * i)
            out[b, c] = 0.01 + 0.19 * (1 + wave) * (0.75 + 0.12 * c)
    out = out.float().clamp(0.01, 0.4)
    out[:, :, :2, :3] = out[:, :1, :2, :3]  # R = G = B: the zero-hue branch
    return out


def stats(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().double().flatten()
    return {
        "shape" : list(t.shape),
        "sum"   : flat.sum().item(),
        "sumsq" : (flat**2).sum().item(),
        "values": [flat[k % flat.numel()].item() for k in STATS_INDICES],
    }


def main(checkout: str, weights_dir: str) -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    from safetensors.torch import load_file

    from net.CIDNet import CIDNet  # official code

    x = reference_input()
    result: dict[str, Any] = {}
    for name, alpha_s, alpha in CASES:
        model = CIDNet().eval()
        model.load_state_dict(load_file(str(Path(weights_dir) / name / "model.safetensors")))
        if alpha_s is not None:
            model.trans.gated, model.trans.alpha_s = True, alpha_s
        if alpha is not None:
            model.trans.gated2, model.trans.alpha = True, alpha
        with torch.no_grad():
            rgb = model(x)
            hvi = model.HVIT(x)
        result[f"{name}|{alpha_s}|{alpha}"] = {"enhanced": stats(rgb), "input_hvi": stats(hvi)}
    print(f"# Generated from the official code with torch {torch.__version__}.")
    print(format_literal("OFFICIAL", result))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
