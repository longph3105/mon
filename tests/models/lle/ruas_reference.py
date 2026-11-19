"""Reference input and statistics for the RUAS parity tests (no ``mon`` imports).

``MON_OLD`` in ``test_ruas.py`` comes from mon_old's archived RUAS network
(``archive/enhance_/lle/ruas/ruas/{module,operations,genotypes}.py``), whose forward pass is a copy of the
official ``model.py`` (https://github.com/dut-media-lab/RUAS, 22b1cd6; checked by reading). Those three files
import only torch, so they are loaded by path without mon_old's package (its ``__init__`` imports mon_old).
``Network()`` loads its bundled ``pretrained/denoise.pt``; each released checkpoint then replaces all weights.
Regenerate with any torch (mon_old's ``mon`` env, or this repo's):

    uv run python -I tests/models/lle/ruas_reference.py ../mon_old $DATA_DIR/zoo/lle
"""

import importlib.util
import pprint
import sys
import types
from pathlib import Path
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)
CHECKPOINTS = {"lol": "ruas/ruas_lol_v1.pt", "upe": "ruas/ruas_upe.pt", "dark": "ruas/ruas_darkface.pt"}
ARCHIVE = "libs/mon/src/archive/enhance_/lle/ruas/ruas"


def reference_input(batch: int = 2, h: int = 23, w: int = 31) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.01, 0.45]`` (no RNG; odd size)."""
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


def _load_archive(mon_old: str) -> types.ModuleType:
    """Import ``module.py`` with its sibling ``operations``/``genotypes`` as package ``ruas_archive``."""
    folder = Path(mon_old) / ARCHIVE
    pkg = types.ModuleType("ruas_archive")
    pkg.__path__ = [str(folder)]
    sys.modules["ruas_archive"] = pkg
    for name in ("genotypes", "operations", "module"):
        spec = importlib.util.spec_from_file_location(f"ruas_archive.{name}", folder / f"{name}.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ruas_archive.module"]


def main(mon_old: str, zoo_lle: str) -> None:
    module = _load_archive(mon_old)
    x = reference_input()
    result: dict[str, Any] = {}
    with torch.no_grad():
        for name, file in CHECKPOINTS.items():
            model = module.Network()
            model.load_state_dict(torch.load(f"{zoo_lle}/{file}", map_location="cpu", weights_only=True))
            u_list, t_list = model.eval()(x)
            result[name] = {  # u_list: the 3 IEM outputs, then the denoised one
                "enhanced"    : stats(u_list[-1]),
                "pre_denoise" : stats(u_list[-2]),
                "illumination": stats(t_list[-1]),
            }
    print(f"# Generated from mon_old's archived RUAS with torch {torch.__version__}.")
    print("MON_OLD: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
