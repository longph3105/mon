"""Tests for ``mon.models.lle.dccnet``: parity with the official network and checkpoint loading."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.models.lle.dccnet import DCCNet, DCCNet_Weights
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("dccnet_reference", Path(__file__).with_name("dccnet_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# Official src/model.py (Ian0926/DCC-Net 8b57f9e) with the released LOL checkpoint. Output of dccnet_reference.py.
# Generated from the official code with torch 2.15.0+cu132.
OFFICIAL: dict = {'enhanced': {'shape': [2, 3, 24, 32],
              'sum': 226.56184634566307,
              'sumsq': 15.306887691863249,
              'values': [0.08794993162155151,
                         0.09200048446655273,
                         0.12125259637832642,
                         0.02122873067855835,
                         0.05673101544380188,
                         0.04256829619407654,
                         0.007556110620498657]},
 'gray': {'shape': [2, 1, 24, 32],
          'sum': 378.1917074620724,
          'sumsq': 177.43246235047394,
          'values': [0.13535287976264954,
                     0.0825941264629364,
                     0.003061354160308838,
                     0.3454046845436096,
                     0.6687653064727783,
                     0.2226562798023224,
                     0.14903414249420166]},
 'color_hist': {'shape': [2, 3, 64],
                'sum': 6.000000120620825,
                'sumsq': 0.17727708900416836,
                'values': [7.606990402564406e-05,
                           0.008905158378183842,
                           0.019107231870293617,
                           0.018624041229486465,
                           0.013724105432629585,
                           0.0004425482184160501,
                           0.0006423912127502263]}}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released() -> dict:
    try:
        return DCCNet_Weights.DEFAULT.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

def test_released_weights_load_strictly_and_match_official():
    model = DCCNet().eval()
    model.load_state_dict(released())
    with torch.no_grad():
        out = model(reference_input())
    for key in ("enhanced", "gray", "color_hist"):
        assert_matches(out[key], OFFICIAL[key])


# --- Checkpoint layout ---

def test_data_parallel_prefix_is_stripped_and_plain_keys_still_load():
    model = DCCNet()
    plain = model.state_dict()
    assert len(plain) == 113 and next(iter(plain)) == "g_net.Encoder.0.main.0.weight"
    DCCNet().load_state_dict({f"module.{k}": v for k, v in plain.items()})  # strict
    DCCNet().load_state_dict(plain)                                         # strict
    with pytest.raises(RuntimeError, match="Unexpected key"):
        DCCNet().load_state_dict(plain | {"module.extra": torch.zeros(1)})  # mixed: nothing stripped


def test_nested_load_strips_only_its_own_prefix():
    outer = torch.nn.Sequential(DCCNet())
    sd = {f"0.module.{k}": v for k, v in DCCNet().state_dict().items()}
    outer.load_state_dict(sd)


# --- Forward ---

def test_forward_shapes_ranges_and_histogram():
    torch.manual_seed(0)
    model = DCCNet(d_hist=16).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 16, 40))
    assert out["enhanced"].shape == (2, 3, 16, 40) and out["gray"].shape == (2, 1, 16, 40)
    assert out["color_hist"].shape == (2, 3, 16)
    assert torch.allclose(out["color_hist"].sum(-1), torch.ones(2, 3))
    assert 0 <= out["enhanced"].min() and out["enhanced"].max() <= 1


def test_sizes_must_be_divisible_by_8():
    with pytest.raises(ValueError, match="divisible by 8"):
        DCCNet()(torch.rand(1, 3, 20, 24))


def test_batch_independence():
    # Per-sample norms only; batch-size-dependent conv algorithms leave float noise that the
    # InstanceNorm over the 2×2 bottleneck amplifies to ~2e-5.
    torch.manual_seed(0)
    model = DCCNet().eval()
    x = torch.rand(3, 3, 16, 16)
    with torch.no_grad():
        assert torch.allclose(model(x)["enhanced"][1:2], model(x[1:2])["enhanced"], atol=1e-4)


# --- Registry ---

def test_registry_and_default_weights():
    assert MODELS.get("dccnet") is DCCNet and DCCNet_Weights.DEFAULT is DCCNet_Weights.LOL_V1
    model = load("dccnet", weights=None, d_hist=32)
    assert isinstance(model, DCCNet) and model.d_hist == 32 and not model.training


# --- LOL-v1 ---

@pytest.mark.full
def test_lol_v1_psnr_regression(regression_device: torch.device):
    """Mean PSNR of 8-bit outputs vs targets (full 600×400 images, CPU ~1 min); measured 2026-11-18. The
    official ``results/`` images score the same and differ from these outputs by at most one 8-bit level."""
    from torch.nn import functional as F

    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv1
    try:
        root = datasets_dir() / LOLv1.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = DCCNet().eval().to(regression_device)
    model.load_state_dict(released())
    psnrs = []
    with torch.no_grad():
        for sample in LOLv1("test"):
            x      = (sample["image"].float() / 255).to(regression_device)
            target = (sample["target"].float() / 255).to(regression_device)
            y = (model(x[None])["enhanced"][0] * 255).round() / 255
            psnrs.append(10 * torch.log10(1 / F.mse_loss(y, target)).item())
    assert sum(psnrs) / len(psnrs) == pytest.approx(22.976, abs=5e-3)
