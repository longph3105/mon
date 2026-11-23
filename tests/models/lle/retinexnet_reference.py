"""Reference input and statistics for the RetinexNet parity test (no ``mon`` imports).

``REIMPL`` in ``test_retinexnet.py`` comes from the PyTorch reimplementation's ``model.py``
(https://github.com/aasharma90/RetinexNet_PyTorch, no license: never copied into this repo), its
``DecomNet`` and ``RelightNet`` with each checkpoint (``decom_net.*`` / ``enhance_net.*`` keys), on
CPU, composing ``R · L̂`` as its ``forward`` does. Running it executes third-party code: get the
owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/retinexnet_reference.py <checkout> <checkpoint.pt> [...]
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


def main(checkout: str, *checkpoints: str) -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    from model import DecomNet, RelightNet  # reimplementation code

    x = reference_input()
    result: dict[str, Any] = {}
    for path in checkpoints:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        decom, relight = DecomNet().eval(), RelightNet().eval()
        decom.load_state_dict({k.removeprefix("decom_net."): v for k, v in sd.items() if k.startswith("decom_net.")})
        relight.load_state_dict(
            {k.removeprefix("enhance_net."): v for k, v in sd.items() if k.startswith("enhance_net.")},
        )
        with torch.no_grad():
            r, illum = decom(x)
            delta = relight(illum, r)
            enhanced = r * torch.cat([delta, delta, delta], 1)
        result[Path(path).name] = {"enhanced": stats(enhanced), "reflectance": stats(r), "illumination": stats(illum),
                                   "illumination_adjusted": stats(delta)}
    print(f"# Generated from the reimplementation with torch {torch.__version__}.")
    print(format_literal("REIMPL", result))


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
