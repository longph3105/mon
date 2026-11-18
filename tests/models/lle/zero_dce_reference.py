"""Reference input and output statistics for the Zero-DCE parity tests (no ``mon`` imports).

Run as a script to regenerate ``OFFICIAL`` in ``test_zero_dce.py`` with the released weights,
from either implementation (their forward math is the same for these cases):

    # mon_old, in its conda env (forward_step bypasses mon_old's TensorDict __call__)
    PYTHONPATH=../mon_old/libs/mon/src <conda env>/bin/python -I tests/models/lle/zero_dce_reference.py \\
        mon_old - - <zero_dce.pt> <zero_dce++.pt>
    # official code (the repos have no license, so it isn't copied into mon)
    uv run python -I tests/models/lle/zero_dce_reference.py \\
        official <Zero-DCE>/Zero-DCE_code <Zero-DCE_extension>/Zero-DCE++ <zero_dce.pt> <zero_dce++.pt>
"""

import importlib.util
import pprint
import sys
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)


def reference_input(batch: int = 2, h: int = 24, w: int = 36) -> torch.Tensor:
    """Deterministic low-light-like images in ``[0.02, 0.5]`` (no RNG: identical on every machine)."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(batch, 3, h, w, dtype=torch.float64)
    for b in range(batch):
        for c in range(3):
            wave = torch.sin(0.37 * (c + 1) * i + 0.23 * (b + 1) * j + 1.1 * c) * torch.cos(0.11 * j - 0.29 * i)
            out[b, c] = 0.02 + 0.24 * (1 + wave) * (0.6 + 0.2 * c)
    return out.float().clamp(0.02, 0.5)


def stats(t: torch.Tensor) -> dict[str, Any]:
    """Shape, float64 sum and sum of squares, and values at fixed flat indices."""
    flat = t.detach().double().flatten()
    return {
        "shape" : list(t.shape),
        "sum"   : flat.sum().item(),
        "sumsq" : (flat**2).sum().item(),
        "values": [flat[k].item() for k in STATS_INDICES],
    }


def _official_models(zdce_dir: str, zdcepp_dir: str) -> tuple[Any, Any]:
    def module(code_dir: str, name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, f"{code_dir}/model.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    zdce, zdcepp = module(zdce_dir, "official_zero_dce"), module(zdcepp_dir, "official_zero_dce_pp")

    def run_zdce(net: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, enhanced, curve = net(x)
        return enhanced, curve
    return (zdce.enhance_net_nopool, run_zdce), (zdcepp.enhance_net_nopool, lambda net, x: net(x))


def _mon_old_models() -> tuple[Any, Any]:
    import importlib  # mon_old's package re-exports a `zero_dce` function that shadows the subpackage name

    old_module = importlib.import_module("mon.models.enhance.lle.zero_dce.module")
    old_model  = importlib.import_module("mon.models.enhance.lle.zero_dce.model")
    ZeroDCE, ZeroDCEPP = old_model.ZeroDCE, old_model.ZeroDCEPP

    # mon_old bug: DSConv.apply(weights_init) matches "Conv" in "DSConv" and reads its missing .weight, so
    # ZeroDCEPP() raises AttributeError. Initialization is irrelevant here (weights are loaded), so skip it.
    old_module.weights_init = lambda m: None

    def run(net: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enhanced, curve = net.forward_step(image=x)
        return enhanced, curve
    return (lambda: ZeroDCE(verbose=False), run), (lambda s: ZeroDCEPP(scale_factor=s, verbose=False), run)


def main(source: str, zdce_dir: str, zdcepp_dir: str, zdce_weights: str, zdcepp_weights: str) -> None:
    (make_zdce, run_zdce), (make_zdcepp, run_zdcepp) = (
        _official_models(zdce_dir, zdcepp_dir) if source == "official" else _mon_old_models())
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)  # noqa: E731
    x = reference_input()
    result: dict[str, Any] = {}
    with torch.no_grad():
        net = make_zdce().eval()
        result["zero_dce_state"] = {k: list(v.shape) for k, v in net.state_dict().items()}
        net.load_state_dict(load(zdce_weights))
        enhanced, curve = run_zdce(net, x.clone())
        result["zero_dce"] = {"enhanced": stats(enhanced), "curve": stats(curve)}
        for scale in (1, 12):
            net = make_zdcepp(scale).eval()
            result.setdefault("zero_dce++_state", {k: list(v.shape) for k, v in net.state_dict().items()})
            net.load_state_dict(load(zdcepp_weights))
            enhanced, curve = run_zdcepp(net, x.clone())
            result[f"zero_dce++_x{scale}"] = {"enhanced": stats(enhanced), "curve": stats(curve)}
    print(f"# Generated from {source} with torch {torch.__version__}.")
    print("OFFICIAL: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(*sys.argv[1:6])
