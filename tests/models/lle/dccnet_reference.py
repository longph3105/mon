"""Reference input and statistics for the DCC-Net parity test (no ``mon`` imports).

``OFFICIAL`` in ``test_dccnet.py`` comes from the official ``src/model.py``
(https://github.com/Ian0926/DCC-Net, MIT) with the released LOL checkpoint on CPU (its ``module.``
prefix stripped: the official ``src/test.py`` wraps the net in ``nn.DataParallel``, which moves it to the
GPU, where TF32 convolutions change the outputs by ~1e-4). Running it executes third-party code: get
the owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/dccnet_reference.py <checkout>/src $DATA_DIR/zoo/lle/dccnet/dccnet_lol_v1.pt
"""

import pprint
import sys
from pathlib import Path
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)


def reference_input(batch: int = 2, h: int = 24, w: int = 32) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.4]`` (no RNG; sizes divisible by 8)."""
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


def main(src: str, weights: str) -> None:
    sys.path.insert(0, str(Path(src).resolve()))
    import model as official  # official src/model.py

    net = official.color_net()
    sd  = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    with torch.no_grad():
        gray, color_hist, enhanced = net.eval()(reference_input())
    result = {"enhanced": stats(enhanced), "gray": stats(gray), "color_hist": stats(color_hist)}
    print(f"# Generated from the official code with torch {torch.__version__}.")
    print("OFFICIAL: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
