"""Reference input and statistics for the CoLIE parity tests (no ``mon`` imports).

``MON_OLD`` in ``test_colie.py`` comes from mon_old's ``CoLIE.forward_step``, whose network, losses,
and optimizer match the official code. A spy on its ``guided_filter_upsample`` captures the
low-resolution Value, illumination, and enhanced Value before mon_old's (buggy) upscaling.
Regenerate in mon_old's conda env:

    PYTHONPATH=../mon_old/libs/mon/src <conda env>/bin/python -I tests/models/lle/colie_reference.py
"""

import importlib
import pprint
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, -1)
SETTINGS = {"window": 7, "hidden_dim": 32, "epochs": 100, "exposure": 0.5}  # mon_old: down_size = hidden_dim


def reference_input(h: int = 40, w: int = 52) -> torch.Tensor:
    """``[1, 3, H, W]`` deterministic low-light image in ``[0.01, 0.3]`` (no RNG)."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(1, 3, h, w, dtype=torch.float64)
    for c in range(3):
        wave = torch.sin(0.23 * (c + 1) * i + 0.17 * j + 0.5 * c) * torch.cos(0.09 * j - 0.21 * i)
        out[0, c] = 0.01 + 0.145 * (1 + wave) * (0.8 + 0.1 * c)
    return out.float().clamp(0.01, 0.3)


def stats(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().double().flatten()
    return {
        "shape" : list(t.shape),
        "sum"   : flat.sum().item(),
        "sumsq" : (flat**2).sum().item(),
        "values": [flat[k % flat.numel()].item() for k in STATS_INDICES],
    }


def main() -> None:
    model_mod = importlib.import_module("mon.models.enhance.lle.colie.model")  # mon_old
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    real = model_mod.guided_filter_upsample

    def spy(x_lr: torch.Tensor, y_hr: torch.Tensor, y_lr: torch.Tensor | None = None, r: int = 1) -> torch.Tensor:
        calls.append((x_lr.detach().clone(), y_lr.detach().clone()))
        return real(x_lr, y_hr, y_lr, r)

    model_mod.guided_filter_upsample = spy
    x = reference_input()
    torch.manual_seed(0)
    model = model_mod.CoLIE(name="colie", window_size=SETTINGS["window"], hidden_dim=SETTINGS["hidden_dim"],
                            num_layers=4, add_layers=2, epochs=SETTINGS["epochs"], optimizer=None,
                            device=torch.device("cpu"), verbose=False)
    enhanced = model.forward_step(image=x, E=SETTINGS["exposure"])[0]
    (enhanced_v_lr, value_lr), _, (illumination_lr, _) = calls  # calls: r, i_res, i_fixed
    result = {
        "value_lr"       : stats(value_lr),
        "illumination_lr": stats(illumination_lr),
        "enhanced_v_lr"  : stats(enhanced_v_lr),
        "mon_old_output_mean": enhanced.mean().item(),  # swapped guided filter: darker than the input
        "input_mean"     : x.mean().item(),
    }
    print(f"# Generated from mon_old with torch {torch.__version__}.")
    print("MON_OLD: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main()
