"""Reference input and statistics for the FLOL parity test (no ``mon`` imports).

``OFFICIAL`` in ``test_flol.py`` comes from the official ``model/flol.py`` + ``utils/``
(https://github.com/cidautai/FLOL, MIT) with the released weights (``weights/*.pt``), on CPU.
Needs ``kornia``. Running it executes third-party code: get the owner's go-ahead first.
Regenerate with:

    uv run --with kornia==0.9.2 python -I tests/models/lle/flol_reference.py <checkout> <checkout>/weights [float64]

The default dtype is float64: in float32 the SNR mask (``blur / (|luma - blur| + 1e-4)`` over its
max) carries rounding error that the spatial branch amplifies to ~1e-3, so float32 literals only
match on the machine that made them. Make the literal on x86_64: on arm64, FFT bins that are real for real input
come out ±1e-17 instead of 0, which flips their phase between ±π in the official frequency blocks (~1e-3 in
``enhanced``); mon zeroes them, matching x86. A float64 run needs one patch in the checkout, a no-op in
float32: in ``model/flol.py`` ``get_mask``, ``return mask.float()`` → ``return mask``.
"""

import pprint
import sys
from pathlib import Path
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)

FILES = ("flolv2_all_111439.pt", "flolv2_UHDLL.pt")


def reference_input(batch: int = 2, h: int = 22, w: int = 30) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.4]`` (no RNG; even, not multiples of 8)."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(batch, 3, h, w, dtype=torch.float64)
    for b in range(batch):
        for c in range(3):
            wave = torch.sin(0.31 * (c + 1) * i + 0.19 * (b + 1) * j + 0.7 * c) * torch.cos(0.11 * j - 0.27 * i)
            out[b, c] = 0.01 + 0.19 * (1 + wave) * (0.75 + 0.12 * c)
    return out.float().clamp(0.01, 0.4)


def stats(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().double().flatten()
    return {
        "shape" : list(t.shape),
        "sum"   : flat.sum().item(),
        "sumsq" : (flat**2).sum().item(),
        "values": [flat[k % flat.numel()].item() for k in STATS_INDICES],
    }


def main(checkout: str, weights_dir: str, dtype_name: str = "float64") -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    from model.flol import create_model  # official code

    dtype = getattr(torch, dtype_name)
    x = reference_input().to(dtype)
    h, w = x.shape[-2:]
    result: dict[str, Any] = {}
    for name in FILES:
        model = create_model().eval()
        model.load_state_dict(torch.load(Path(weights_dir) / name, map_location="cpu", weights_only=True)["params"])
        model = model.to(dtype)
        with torch.no_grad():
            enhanced, x_center = model(x, side=True)
        result[name] = {"enhanced": stats(enhanced), "amplitude_enhanced": stats(x_center[:, :, :h, :w])}
    print(f"# Generated from the official code with torch {torch.__version__}, {dtype_name}.")
    print("OFFICIAL: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
