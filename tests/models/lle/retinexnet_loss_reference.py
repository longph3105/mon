"""Reference values for the RetinexNet loss parity test (no ``mon`` imports).

``REIMPL_LOSS`` in ``test_retinexnet.py`` comes from the PyTorch reimplementation's own loss code
(https://github.com/aasharma90/RetinexNet_PyTorch, ``model.py`` ``RetinexNet.forward`` + ``smooth`` @
``2267510``; no license: never copied into this repo). Its ``forward`` computes both losses from the network
outputs, so the networks are replaced by stubs that return fixed, deterministic ``R``/``L`` maps: the
values below are the reimplementation's ``loss_Decom`` and ``loss_Relight`` on them, with the gradient
statistics of each map. ``Tensor.cuda`` is patched to a no-op (the code hard-codes CUDA).

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/retinexnet_loss_reference.py <checkout>
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
    """The low/normal-light images and the maps the networks would give (``l_adjusted``: RelightNet's)."""
    return {"low": field(3, 0.0, 0.02, 0.4), "high": field(3, 0.5, 0.2, 0.95),
            "r_low": field(3, 1.0, 0.05, 1.0), "l_low": field(1, 2.0, 0.05, 0.6),
            "r_high": field(3, 3.0, 0.05, 1.0), "l_high": field(1, 4.0, 0.3, 1.0),
            "l_adjusted": field(1, 5.0, 0.2, 1.0)}


def stats(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().double().flatten()
    return {"sum": flat.sum().item(), "sumsq": (flat**2).sum().item(),
            "values": [flat[k % flat.numel()].item() for k in STATS_INDICES]}


def main(checkout: str) -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    from model import RetinexNet  # reimplementation code

    x = {k: v.requires_grad_(True) for k, v in inputs().items() if k not in ("low", "high")}
    low, high = inputs()["low"], inputs()["high"]

    class Decom(torch.nn.Module):  # the low image's maps on the first call, the normal-light image's on the second
        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (x["r_low"], x["l_low"]) if torch.equal(image, low) else (x["r_high"], x["l_high"])

    class Relight(torch.nn.Module):
        def forward(self, illumination: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
            return x["l_adjusted"]

    net = RetinexNet()
    net.DecomNet, net.RelightNet = Decom(), Relight()
    net(low.numpy(), high.numpy())  # sets loss_Decom and loss_Relight
    result: dict[str, Any] = {}
    for name, loss, wrt in (("decomposition", net.loss_Decom, ["r_low", "l_low", "r_high", "l_high"]),
                            ("relight", net.loss_Relight, ["r_low", "l_adjusted"])):
        grads = torch.autograd.grad(loss, [x[n] for n in wrt], retain_graph=True)
        result[name] = {"loss": loss.item(), "grads": {n: stats(g) for n, g in zip(wrt, grads, strict=True)}}
    print(f"# Generated from the reimplementation's loss with torch {torch.__version__}.")
    print(format_literal("REIMPL_LOSS", result))


if __name__ == "__main__":
    main(sys.argv[1])
