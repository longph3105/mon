"""Reference input and statistics for the PairLIE parity test (no ``mon`` imports).

``MON_OLD`` in ``test_pairlie.py`` comes from mon_old, whose inference is identical to the official
``net.py`` + ``eval.py`` (https://github.com/zhenqifu/PairLIE). Regenerate in mon_old's conda env:

    PYTHONPATH=../mon_old/libs/mon/src <conda env>/bin/python -I tests/models/lle/pairlie_reference.py \\
        $DATA_DIR/zoo/lle/pairlie/pairlie_sice.pt
"""

import importlib
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


def reference_input(batch: int = 2, h: int = 21, w: int = 30) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.4]`` (no RNG; odd height)."""
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


def main(weights: str) -> None:
    old = importlib.import_module("mon.models.enhance.lle.pairlie.model")  # mon_old
    model = old.PairLIE(name="pairlie", alpha=0.2, weights=None, verbose=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    with torch.no_grad():
        enhanced, illumination, reflectance, denoised, _ = model.eval().forward_step(image=reference_input())
    result = {"enhanced": stats(enhanced), "illumination": stats(illumination),
              "reflectance": stats(reflectance), "denoised": stats(denoised)}
    print(f"# Generated from mon_old with torch {torch.__version__}.")
    print(format_literal("MON_OLD", result))


if __name__ == "__main__":
    main(sys.argv[1])
