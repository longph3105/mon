"""Tests for ``mon.models.lle.flol``: parity with the official network and checkpoint loading."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.models.lle.flol import FLOL, FLOL_Weights, _gaussian_blur5
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("flol_reference", Path(__file__).with_name("flol_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# Official model/flol.py (cidautai/FLOL d6b66c3, kornia 0.9.2) with the released weights, CPU.
# Output of flol_reference.py, in float64 (see its docstring: in float32 the SNR mask's rounding is
# amplified to ~1e-3 in `enhanced`, so float32 literals only match on the machine that made them).
# Generated from the official code with torch 2.15.0, float64.
OFFICIAL: dict = {'flolv2_all_111439.pt': {'enhanced': {'shape': [2, 3, 22, 30],
                                       'sum': 1871.6337882994771,
                                       'sumsq': 952.9419889122137,
                                       'values': [0.40847318785774106,
                                                  0.6194765350931614,
                                                  0.2652154046896732,
                                                  0.005871604918570135,
                                                  0.5391348238690856,
                                                  0.2591141925186481,
                                                  0.6040954091060242]},
                          'amplitude_enhanced': {'shape': [2, 3, 22, 30],
                                                 'sum': 2901.238249621736,
                                                 'sumsq': 2203.948743544997,
                                                 'values': [0.7251857115089704,
                                                            0.823534342601713,
                                                            0.6751293867624961,
                                                            0.38058667104449273,
                                                            0.6890318068822785,
                                                            0.6188235212895438,
                                                            0.7867948277287665]}},
 'flolv2_UHDLL.pt': {'enhanced': {'shape': [2, 3, 22, 30],
                                  'sum': 1783.411994296215,
                                  'sumsq': 880.3566797590265,
                                  'values': [0.423368096701919,
                                             0.5920589573538488,
                                             0.40176415562015033,
                                             -0.00358530846052141,
                                             0.4744760608509345,
                                             0.3188294134457659,
                                             0.4256484812702502]},
                     'amplitude_enhanced': {'shape': [2, 3, 22, 30],
                                            'sum': 1891.6074370398403,
                                            'sumsq': 944.7344315501076,
                                            'values': [0.42505961055143804,
                                                       0.5189805291552655,
                                                       0.41266897534262936,
                                                       0.23027150886102976,
                                                       0.42505758070596306,
                                                       0.42722741850806134,
                                                       0.4881051036971298]}}}

# Official file → zoo weights (same tensors).
ZOO = {"flolv2_all_111439.pt": FLOL_Weights.LOL_V2_REAL, "flolv2_UHDLL.pt": FLOL_Weights.UHD_LL}


# Parity runs in float64, where mon and the official code agree to 0 on this machine.
ATOL = 1e-8

def assert_matches(actual: torch.Tensor, expected: dict, atol: float = ATOL) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(member: FLOL_Weights) -> dict:
    try:
        return member.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

@pytest.mark.parametrize("name", list(OFFICIAL))
def test_released_weights_load_strictly_and_match_official(name):
    model = FLOL().eval()
    model.load_state_dict(released(ZOO[name]))
    model = model.double()
    with torch.no_grad():
        out = model(reference_input().double())
    for key in ("enhanced", "amplitude_enhanced"):
        assert out[key].dtype == torch.float64
        assert_matches(out[key], OFFICIAL[name][key])


def test_gaussian_blur_matches_a_direct_2d_convolution():
    # kornia's separable 5×5 Gaussian (sigma 1.5, reflect border), checked against the 2-D kernel.
    torch.manual_seed(0)
    x   = torch.rand(2, 3, 9, 11, dtype=torch.float64)
    pos = torch.arange(5, dtype=torch.float64) - 2
    k   = torch.exp(-pos**2 / 4.5)
    k2  = torch.outer(k, k) / k.sum() ** 2
    ref = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect"),
                                     k2.view(1, 1, 5, 5).repeat(3, 1, 1, 1), groups=3)
    assert torch.allclose(_gaussian_blur5(x), ref, atol=1e-12)


# --- Checkpoint layout ---

def test_params_wrapper_is_unwrapped_and_plain_keys_load():
    plain = FLOL().state_dict()
    assert len(plain) == 182 and next(iter(plain)) == "AmpNet.0.conv_init.weight"
    FLOL().load_state_dict({"params": plain})  # strict
    FLOL().load_state_dict(plain)              # strict


# --- Forward ---

def test_forward_shapes_padding_and_odd_sizes():
    torch.manual_seed(0)
    model = FLOL().eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 18, 26) * 0.3)   # even, not multiples of 8: padded inside
    assert out["enhanced"].shape == out["amplitude_enhanced"].shape == (1, 3, 18, 26)
    with pytest.raises(ValueError, match="even"):
        model(torch.rand(1, 3, 19, 26))


def test_batch_independence():
    torch.manual_seed(0)
    model = FLOL().eval()
    x = torch.rand(3, 3, 16, 24) * 0.3
    with torch.no_grad():
        assert torch.allclose(model(x)["enhanced"][1:2], model(x[1:2])["enhanced"], atol=1e-5)


def test_layer_norm_gradient_flows():
    model = FLOL()
    model(torch.rand(1, 3, 16, 16))["enhanced"].mean().backward()
    assert model.AmpNet[0].conv1.norm1.weight.grad is not None


# --- Registry ---

def test_registry_and_default_weights():
    assert MODELS.get("flol") is FLOL and FLOL_Weights.DEFAULT is FLOL_Weights.LOL_V2_REAL
    assert isinstance(load("flol", weights=None), FLOL)


# --- LOL-v2-real ---

@pytest.mark.full
def test_lol_v2_real_psnr_regression(regression_device: torch.device):
    """Measured 2026-11-18 on CPU (CUDA within 2e-3). The official ``evaluation.py`` protocol (float outputs, no
    clamp, peak ``max(target)``) gives the README's 21.75; 8-bit clamped outputs score higher."""
    from torch.nn import functional as F

    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv2Real
    try:
        root = datasets_dir() / LOLv2Real.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = FLOL().eval().to(regression_device)
    model.load_state_dict(released(FLOL_Weights.LOL_V2_REAL))
    psnr_8bit, psnr_official = [], []
    with torch.no_grad():
        for sample in LOLv2Real("test"):
            x      = (sample["image"].float() / 255).to(regression_device)
            target = (sample["target"].float() / 255).to(regression_device)
            y = model(x[None])["enhanced"][0]
            psnr_official.append((20 * torch.log10(target.max() / torch.sqrt(F.mse_loss(y, target)))).item())
            y8 = (y.clamp(0, 1) * 255).round() / 255
            psnr_8bit.append(10 * torch.log10(1 / F.mse_loss(y8, target)).item())
    assert sum(psnr_official) / len(psnr_official) == pytest.approx(21.752, abs=5e-3)
    assert sum(psnr_8bit) / len(psnr_8bit) == pytest.approx(22.100, abs=5e-3)
