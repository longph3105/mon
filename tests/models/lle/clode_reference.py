"""Reference input and statistics for the CLODE parity test (no ``mon`` imports).

``OFFICIAL`` in ``test_clode.py`` comes from the official code (https://github.com/dgjung0220/CLODE,
no license: never copied into this repo) and its released weights (Google Drive folder linked from
its README: ``lowlight.pth``, ``sice.pth``, ``universal.pth``). Needs ``network/conv_node.py``,
``network/torchdiffeq/`` and ``losses.py`` from a checkout, and matplotlib (``uv sync --extra gui``).
Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/clode_reference.py <checkout> <dir with the three .pth files> [float64]

The default dtype is float64: in float32 the outputs depend on the platform's rounding (the
adaptive solver's step sizes, and so ``curve_map``, move with it), so float32 literals only match
on the machine that made them. A float64 run needs one patch in the checkout, a no-op in float32:
in ``network/conv_node.py`` ``pair_downsampler``, ``.to(img.device)`` → ``.to(img)`` (its filters
are ``torch.FloatTensor``).
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

# (official weights file stem, end time T)
CASES = (("universal", 3.0), ("universal", -1.4), ("lowlight", 3.0), ("sice", 3.0))


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


def main(checkout: str, weights_dir: str, dtype_name: str = "float64") -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    from network.conv_node import NODE  # official code

    dtype = getattr(torch, dtype_name)
    x = reference_input().to(dtype)
    result: dict[str, Any] = {}
    for stem, t_end in CASES:
        model = NODE("cpu", (3, 256, 256), 32, augment_dim=0, time_dependent=True, adjoint=True).eval()
        sd = torch.load(Path(weights_dir) / f"{stem}.pth", map_location="cpu", weights_only=True)
        missing, _ = model.load_state_dict(sd, strict=False)
        assert not missing, missing
        model = model.to(dtype)
        t = torch.tensor([0, t_end]).to(dtype)
        with torch.no_grad():
            out   = model(x, t, inference=True)
            curve = model.odefunc.last_curve_map
            nfe   = model.odefunc.nfe
            raw   = model(x, t, inference=False)  # same solve, raw state channels
            state = torch.cat([raw["output"], raw["curve_map"], raw["noise_map"]], 1)
            noise = model.odefunc.denoise(raw["output"])
        result[f"{stem}@{t_end}"] = {"enhanced": stats(out["output"]), "curve_map": stats(curve),
                                     "noise": stats(noise), "state": stats(state), "nfe": nfe}
    print(f"# Generated from the official code with torch {torch.__version__}, {dtype_name}.")
    print(format_literal("OFFICIAL", result))


if __name__ == "__main__":
    main(*sys.argv[1:4])
