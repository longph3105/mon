"""Tests for ``mon.models.lle.hvi_cidnet``: parity with the official network, the HVI transform, and loading."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.models.lle.hvi_cidnet import HVICIDNet, HVICIDNet_Weights, hvi_to_rgb, rgb_to_hvi
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("hvi_cidnet_reference", Path(__file__).with_name("hvi_cidnet_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# Official net/*.py (Fediory/HVI-CIDNet eb43d7d) with the Hugging Face releases, CPU; keys
# "<HF name>|<alpha_s or None>|<alpha or None>". Output of hvi_cidnet_reference.py.
# Generated from the official code with torch 2.15.0+cu132.
OFFICIAL: dict = {'LOLv1-woperc|None|None': {'enhanced': {'shape': [2, 3, 24, 32],
                                         'sum': 1129.2215253945906,
                                         'sumsq': 339.1546131776123,
                                         'values': [0.22164420783519745,
                                                    0.35353517532348633,
                                                    0.39150747656822205,
                                                    0.03290744498372078,
                                                    0.261116087436676,
                                                    0.4083861708641052,
                                                    0.28021377325057983]},
                            'input_hvi': {'shape': [2, 3, 24, 32],
                                          'sum': 235.9927314266788,
                                          'sumsq': 354.77655057814013,
                                          'values': [0.0,
                                                     -0.03849716857075691,
                                                     0.5949091911315918,
                                                     -0.47993433475494385,
                                                     0.005770974326878786,
                                                     0.30106842517852783,
                                                     0.20031194388866425]}},
 'LOLv1-woperc|1.3|None': {'enhanced': {'shape': [2, 3, 24, 32],
                                        'sum': 1031.5932161613891,
                                        'sumsq': 311.23335299265,
                                        'values': [0.22009612619876862,
                                                   0.34320446848869324,
                                                   0.39150747656822205,
                                                   0.0,
                                                   0.25610217452049255,
                                                   0.4083861708641052,
                                                   0.28021377325057983]},
                           'input_hvi': {'shape': [2, 3, 24, 32],
                                         'sum': 235.9927314266788,
                                         'sumsq': 354.77655057814013,
                                         'values': [0.0,
                                                    -0.03849716857075691,
                                                    0.5949091911315918,
                                                    -0.47993433475494385,
                                                    0.005770974326878786,
                                                    0.30106842517852783,
                                                    0.20031194388866425]}},
 'LOLv2-real-bestPSNR|None|0.8': {'enhanced': {'shape': [2, 3, 24, 32],
                                               'sum': 880.1122975621547,
                                               'sumsq': 199.42042116047418,
                                               'values': [0.1909405142068863,
                                                          0.26607462763786316,
                                                          0.2736375331878662,
                                                          0.0453667975962162,
                                                          0.24364785850048065,
                                                          0.33005842566490173,
                                                          0.22483809292316437]},
                                  'input_hvi': {'shape': [2, 3, 24, 32],
                                                'sum': 230.6728789102017,
                                                'sumsq': 374.8536103584282,
                                                'values': [0.0,
                                                           -0.03992530331015587,
                                                           0.6168034672737122,
                                                           -0.49751561880111694,
                                                           0.006070312112569809,
                                                           0.30106842517852783,
                                                           0.20031194388866425]}},
 'Sony-Total-Dark|None|None': {'enhanced': {'shape': [2, 3, 24, 32],
                                            'sum': 4540.410309970379,
                                            'sumsq': 4476.100768247702,
                                            'values': [0.970368504524231,
                                                       0.937261164188385,
                                                       0.984266459941864,
                                                       1.0,
                                                       1.0,
                                                       1.0,
                                                       1.0]},
                               'input_hvi': {'shape': [2, 3, 24, 32],
                                             'sum': 227.98225196369822,
                                             'sumsq': 385.29027480108016,
                                             'values': [0.0,
                                                        -0.040643636137247086,
                                                        0.6278137564659119,
                                                        -0.506355881690979,
                                                        0.006222463678568602,
                                                        0.30106842517852783,
                                                        0.20031194388866425]}}}

# Hugging Face release → zoo weights (same tensors).
ZOO = {"LOLv1-woperc": HVICIDNet_Weights.LOL_V1_WO_PERC, "LOLv2-real-bestPSNR": HVICIDNet_Weights.LOL_V2_REAL_BEST_PSNR,
       "Sony-Total-Dark": HVICIDNet_Weights.SID_SONY}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(member: HVICIDNet_Weights) -> dict:
    try:
        return member.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

@pytest.mark.parametrize("case", list(OFFICIAL))
def test_released_weights_load_strictly_and_match_official(case):
    name, alpha_s, alpha = case.split("|")
    model = HVICIDNet(saturation_gain=1.0 if alpha_s == "None" else float(alpha_s),
                      output_gain=1.0 if alpha == "None" else float(alpha)).eval()
    model.load_state_dict(released(ZOO[name]))
    x = reference_input()
    with torch.no_grad():
        out = model(x)
        hvi = rgb_to_hvi(x, model.trans.density_k)
    assert_matches(out["enhanced"], OFFICIAL[case]["enhanced"])
    assert_matches(hvi, OFFICIAL[case]["input_hvi"])


@pytest.mark.parametrize("member", list(HVICIDNet_Weights))
def test_every_listed_release_loads_strictly(member):
    HVICIDNet().load_state_dict(released(member))


# --- HVI transform ---

def test_hvi_round_trip_and_gray_pixels():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 8, dtype=torch.float64) * 0.9 + 0.05
    x[:, :, 0, 0] = 0.3                                   # gray: zero hue and saturation
    hvi = rgb_to_hvi(x, 0.2)
    assert hvi.shape == x.shape and hvi[:, :2].abs().max() <= 1 and torch.equal(hvi[:, 2], x.max(1)[0])
    assert torch.allclose(hvi[:, :2, 0, 0], torch.zeros(2, 2, dtype=torch.float64), atol=1e-6)
    # Exact up to the official eps terms: sqrt(H² + V² + 1e-8) gives gray pixels saturation ~1e-4 (error ~3e-5).
    assert torch.allclose(hvi_to_rgb(hvi, 0.2), x, atol=5e-5)


def test_saturation_and_output_gains():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 16, 16) * 0.4
    base = HVICIDNet().eval()
    gained = HVICIDNet(output_gain=0.5).eval()
    gained.load_state_dict(base.state_dict())
    with torch.no_grad():
        assert torch.allclose(gained(x)["enhanced"], base(x)["enhanced"] * 0.5)
    hvi = rgb_to_hvi(x, 0.2)
    sat = lambda rgb: (rgb.max(1)[0] - rgb.min(1)[0]) / (rgb.max(1)[0] + 1e-8)  # noqa: E731
    assert (sat(hvi_to_rgb(hvi, 0.2, saturation_gain=1.3)) >= sat(hvi_to_rgb(hvi, 0.2)) - 1e-6).all()


# --- Forward ---

def test_forward_shapes_sizes_and_batch_independence():
    torch.manual_seed(0)
    model = HVICIDNet().eval()
    x = torch.rand(3, 3, 16, 24) * 0.4
    with torch.no_grad():
        out = model(x)
        assert out["enhanced"].shape == out["hvi"].shape == (3, 3, 16, 24)
        assert torch.allclose(out["enhanced"][1:2], model(x[1:2])["enhanced"], atol=1e-5)
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.rand(1, 3, 20, 24))


def test_density_k_gets_gradients_through_the_forward_transform():
    model = HVICIDNet()
    model(torch.rand(1, 3, 16, 16) * 0.4)["enhanced"].mean().backward()
    assert model.trans.density_k.grad is not None


# --- Registry ---

def test_registry_and_default_weights():
    assert MODELS.get("hvi_cidnet") is HVICIDNet and HVICIDNet_Weights.DEFAULT is HVICIDNet_Weights.LOL_V1_WO_PERC
    model = load("hvi_cidnet", weights=None, saturation_gain=1.3)
    assert isinstance(model, HVICIDNet) and model.saturation_gain == 1.3


# --- README benchmarks ---

@pytest.mark.full
@pytest.mark.parametrize(("dataset", "member", "gains", "expected"), [
    ("LOLv1", HVICIDNet_Weights.LOL_V1_WO_PERC, {"saturation_gain": 1.3}, 23.5029),         # README 23.5000
    ("LOLv1", HVICIDNet_Weights.LOL_V1_W_PERC, {"saturation_gain": 1.3}, 23.8092),          # README 23.8091
    ("LOLv2Real", HVICIDNet_Weights.LOL_V2_REAL_BEST_PSNR, {"output_gain": 0.8}, 23.9046),  # README 23.9040
    ("LOLv2Real", HVICIDNet_Weights.LOL_V2_REAL_BEST_SSIM, {"output_gain": 0.82}, 23.8977), # README 23.8975
])
def test_readme_psnr(dataset: str, member: HVICIDNet_Weights, gains: dict, expected: float,
                     regression_device: torch.device):
    """Official protocol (eval.py + measure.py): clamp, truncate to 8 bit (``ToPILImage``), PSNR with +1e-8 in
    the MSE. Measured 2026-11-18 on CPU; CUDA within 4e-3."""
    from mon.core.path import datasets_dir
    from mon.data.zoo import lol
    cls = getattr(lol, dataset)
    try:
        root = datasets_dir() / cls.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = HVICIDNet(**gains).eval().to(regression_device)
    model.load_state_dict(released(member))
    psnrs = []
    with torch.no_grad():
        for sample in cls("test"):
            x      = (sample["image"].float() / 255).to(regression_device)
            target = (sample["target"].double()).to(regression_device)
            y8 = (model(x[None])["enhanced"][0].clamp(0, 1) * 255).to(torch.uint8).double()
            psnrs.append(10 * math.log10(255.0**2 / (((y8 - target) ** 2).mean().item() + 1e-8)))
    assert sum(psnrs) / len(psnrs) == pytest.approx(expected, abs=5e-3)
