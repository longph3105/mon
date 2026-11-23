"""Reference values for the PairLIE loss parity test (no ``mon`` imports).

``OFFICIAL_LOSS`` in ``test_pairlie.py`` comes from the official loss functions
(https://github.com/zhenqifu/PairLIE, ``utils.py`` ``C_loss``/``R_loss``/``P_loss`` @ ``3fd5c15``, weighted
``1·C + 1·R + 500·P`` as ``main.py`` does; no license: never copied into this repo), on deterministic
inputs, with the loss value and the gradient statistics of each input.

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/pairlie_loss_reference.py <checkout>
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

STATS_INDICES = (0, 7, 101, 523, -1)
SHAPE = (2, 3, 13, 18)


def field(channels: int, phase: float, low: float, high: float) -> torch.Tensor:
    """Deterministic ``[B, channels, H, W]`` values in ``[low, high]`` (no RNG)."""
    b, _, h, w = SHAPE
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(b, channels, h, w, dtype=torch.float64)
    for n in range(b):
        for c in range(channels):
            wave = torch.sin(0.37 * (c + 1) * i + 0.23 * (n + 1) * j + phase) * torch.cos(0.13 * j - 0.29 * i + c)
            out[n, c] = low + (high - low) * (1 + wave) / 2
    return out.float()


def inputs() -> dict[str, torch.Tensor]:
    """``im1`` (low light), ``L1``, ``R1``, ``X1`` (its outputs) and ``R2`` (the pair's reflectance)."""
    return {"im1": field(3, 0.0, 0.02, 0.4), "L1": field(1, 1.0, 0.1, 0.9), "R1": field(3, 2.0, 0.05, 1.0),
            "X1": field(3, 3.0, 0.02, 0.4), "R2": field(3, 4.0, 0.05, 1.0)}


def stats(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().double().flatten()
    return {"sum": flat.sum().item(), "sumsq": (flat**2).sum().item(),
            "values": [flat[k % flat.numel()].item() for k in STATS_INDICES]}


def main(checkout: str) -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    from utils import C_loss, P_loss, R_loss  # official code

    x = {k: v.requires_grad_(True) for k, v in inputs().items()}
    loss = (C_loss(x["R1"], x["R2"]) * 1 + R_loss(x["L1"], x["R1"], x["im1"], x["X1"]) * 1
            + P_loss(x["im1"], x["X1"]) * 500)
    names = ["L1", "R1", "X1", "R2"]
    grads = torch.autograd.grad(loss, [x[n] for n in names])
    result = {"loss": loss.item(), "grads": {n: stats(g) for n, g in zip(names, grads, strict=True)}}
    print(f"# Generated from the official PairLIE loss with torch {torch.__version__}.")
    print(format_literal("OFFICIAL_LOSS", result))


if __name__ == "__main__":
    main(sys.argv[1])
