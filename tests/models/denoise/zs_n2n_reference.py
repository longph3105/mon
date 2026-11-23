"""Reference input and statistics for the ZS-N2N parity test (no ``mon`` imports).

Regenerate ``MON_OLD`` in ``test_zs_n2n.py`` from mon_old in its conda env:

    PYTHONPATH=../mon_old/libs/mon/src <conda env>/bin/python -I tests/models/denoise/zs_n2n_reference.py
"""

import importlib
import importlib.util
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
ITERATIONS = 100


def reference_input(h: int = 41, w: int = 53) -> torch.Tensor:
    """``[1, 3, H, W]``: a smooth scene plus deterministic pseudo-noise (no RNG; odd size on purpose)."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(1, 3, h, w, dtype=torch.float64)
    for c in range(3):
        clean = 0.5 + 0.3 * torch.sin(0.15 * (c + 1) * i + 0.1 * j) * torch.cos(0.07 * j - 0.05 * c * i)
        noise = torch.frac(torch.sin(12.9898 * (i + 1) + 78.233 * (j + 1) + 37.719 * c) * 43758.5453) - 0.5
        out[0, c] = clean + 0.3 * noise  # uniform noise, std ≈ 0.087
    return out.float().clamp(0, 1)


def stats(t: torch.Tensor) -> dict[str, Any]:
    """Shape, float64 sum and sum of squares, and values at fixed flat indices."""
    flat = t.detach().double().flatten()
    return {
        "shape" : list(t.shape),
        "sum"   : flat.sum().item(),
        "sumsq" : (flat**2).sum().item(),
        "values": [flat[k % flat.numel()].item() for k in STATS_INDICES],
    }


def main() -> None:
    old = importlib.import_module("mon.models.restore.denoise.zs_n2n.model")
    torch.manual_seed(0)
    model = old.ZS_N2N(device=torch.device("cpu"), verbose=False)
    x = reference_input()
    result: dict[str, Any] = {
        "init" : {k.removeprefix("model."): stats(v) for k, v in model.state_dict().items()},
        "loss0": model.denoise_loss(x).item(),
    }
    out = model.fit(x, epochs=ITERATIONS, reset_weights=False)  # defaults: Adam(1e-3), StepLR(1000, 0.5)
    result["restored"] = stats(out["restored"])
    print(f"# Generated from mon_old with torch {torch.__version__}.")
    print(format_literal("MON_OLD", result))


if __name__ == "__main__":
    main()
