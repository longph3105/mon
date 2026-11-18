"""Reference input and statistics for the Zero-IG parity tests (no ``mon`` imports).

``OFFICIAL`` in ``test_zero_ig.py`` comes from the official code (https://github.com/Doyle59217/ZeroIG,
no license: never copied into this repo) on CPU (``Tensor.cuda`` is patched to a no-op: its loss
hard-codes CUDA):

- ``inference``: ``Finetunemodel.forward`` with each released weights file;
- ``loss``: ``Network._loss`` in training mode with ``weights/LOL.pt``, plus gradient statistics;
- ``fit``: the official ``train.py`` recipe for ``FIT_STEPS`` steps from seed 2 (Adam without
  ``capturable``, which needs CUDA), then the eval-mode output.

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/zero_ig_reference.py <checkout>
"""

import pprint
import sys
from pathlib import Path
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)
FILES = ("LOL.pt", "LSRW-Huawei.pt", "LSRW-Nikon.pt")
FIT_STEPS = 5


def reference_input(batch: int = 1, h: int = 24, w: int = 30) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.4]`` (no RNG; even sizes)."""
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


def main(checkout: str) -> None:
    root = Path(checkout).resolve()
    sys.path.insert(0, str(root))
    # Keep the official code on CPU (its loss hard-codes .cuda()).
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]
    from torch import nn

    from model import Finetunemodel, Network  # official code

    x = reference_input()
    result: dict[str, Any] = {"inference": {}}
    for name in FILES:
        net = Network().eval()
        net.load_state_dict(torch.load(root / "weights" / name, map_location="cpu", weights_only=True))
        with torch.no_grad():
            h2, h3 = Finetunemodel.forward(net, x)
        result["inference"][name] = {"enhanced": stats(h3), "enhanced_noisy": stats(h2)}

    net = Network().train()
    net.load_state_dict(torch.load(root / "weights" / "LOL.pt", map_location="cpu", weights_only=True))
    loss = net._loss(x)
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in net.parameters()])
    result["loss"] = {"value": loss.item(), "grad": stats(grads)}

    torch.manual_seed(2)
    net = Network()
    for part in (net.enhance.in_conv, net.enhance.conv, net.enhance.out_conv):
        part.apply(net.enhance_weights_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=3e-4)
    net.train()
    for _ in range(FIT_STEPS):
        optimizer.zero_grad()
        step_loss = net._loss(x)
        step_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    net.eval()
    with torch.no_grad():
        outs = net(x)
    result["fit"] = {"steps": FIT_STEPS, "last_loss": step_loss.item(), "enhanced": stats(outs[13])}  # H3

    print(f"# Generated from the official code with torch {torch.__version__}.")
    print("OFFICIAL: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
