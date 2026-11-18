"""Reference input and statistics for the SCI parity tests (no ``mon`` imports).

``MON_OLD`` in ``test_sci.py`` comes from mon_old, whose inference networks and loss are copies of
the official code (https://github.com/vis-opt-group/SCI). Regenerate in mon_old's conda env:

    PYTHONPATH=../mon_old/libs/mon/src <conda env>/bin/python -I tests/models/lle/sci_reference.py \\
        $DATA_DIR/zoo/lle
"""

import importlib
import pprint
import sys
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)
CHECKPOINTS = {  # name -> (mon_old class, file under zoo/lle)
    "sci_easy"     : ("SCI", "sci/sci_easy.pt"),
    "sci_medium"   : ("SCI", "sci/sci_medium.pt"),
    "sci_difficult": ("SCI", "sci/sci_difficult.pt"),
    "sci++_tpami"  : ("SCI_PP", "sci++/sci++_1_3500.pt"),
}


def reference_input(batch: int = 2, h: int = 23, w: int = 31) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.45]`` (no RNG; odd size, W not a multiple of 3)."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(batch, 3, h, w, dtype=torch.float64)
    for b in range(batch):
        for c in range(3):
            wave = torch.sin(0.41 * (c + 1) * i + 0.27 * (b + 1) * j + 0.9 * c) * torch.cos(0.13 * j - 0.31 * i)
            out[b, c] = 0.01 + 0.22 * (1 + wave) * (0.7 + 0.15 * c)
    return out.float().clamp(0.01, 0.45)


def stats(t: torch.Tensor) -> dict[str, Any]:
    """Shape, float64 sum and sum of squares, and values at fixed flat indices."""
    flat = t.detach().double().flatten()
    return {
        "shape" : list(t.shape),
        "sum"   : flat.sum().item(),
        "sumsq" : (flat**2).sum().item(),
        "values": [flat[k % flat.numel()].item() for k in STATS_INDICES],
    }


def main(zoo_lle: str) -> None:
    model_mod = importlib.import_module("mon.models.enhance.lle.sci.model")  # mon_old
    loss_mod  = importlib.import_module("mon.models.enhance.lle.sci.loss")
    x = reference_input()
    result: dict[str, Any] = {}
    with torch.no_grad():
        for name, (cls_name, file) in CHECKPOINTS.items():
            model = getattr(model_mod, cls_name)(name=name, weights=None, verbose=False)
            model.load_state_dict(torch.load(f"{zoo_lle}/{file}", map_location="cpu", weights_only=True))
            enhanced, illumination = model.eval().forward_step(image=x)
            result[name] = {"enhanced": stats(enhanced), "illumination": stats(illumination)}
            if name == "sci_medium":
                loss = loss_mod.LossFunction()
                result["loss"] = {
                    "total"     : loss(x, illumination).item(),
                    "smoothness": loss_mod.SmoothLoss()(x, illumination).item(),
                }
    print(f"# Generated from mon_old with torch {torch.__version__}.")
    print("MON_OLD: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
