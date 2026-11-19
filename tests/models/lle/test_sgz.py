"""Tests for ``mon.models.lle.sgz``: the released weights through the official forward pass."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mon.models.lle.sgz import SGZ, SGZ_Weights
from mon.models.lle.zero_dce import ZeroDCEPP, ZeroDCEPP_Weights
from mon.models.registry import MODELS, load


def released() -> dict:
    try:
        return SGZ_Weights.SICE_ME.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


def official_forward(state: dict, x: torch.Tensor, scale_factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``enhance_net_nopool(scale_factor, "dsc").forward`` (official ``modeling/model.py``, MIT), written out."""
    def dsc(name: str, t: torch.Tensor) -> torch.Tensor:
        dw = state[f"{name}.depth_conv.weight"]
        t = F.conv2d(t, dw, state[f"{name}.depth_conv.bias"], padding=1, groups=dw.shape[0])
        return F.conv2d(t, state[f"{name}.point_conv.weight"], state[f"{name}.point_conv.bias"])

    x_down = x if scale_factor == 1 else F.interpolate(x, scale_factor=1 / scale_factor, mode="bilinear")
    x1 = F.relu(dsc("e_conv1", x_down))
    x2 = F.relu(dsc("e_conv2", x1))
    x3 = F.relu(dsc("e_conv3", x2))
    x4 = F.relu(dsc("e_conv4", x3))
    x5 = F.relu(dsc("e_conv5", torch.cat([x3, x4], 1)))
    x6 = F.relu(dsc("e_conv6", torch.cat([x2, x5], 1)))
    r = torch.tanh(dsc("e_conv7", torch.cat([x1, x6], 1)))
    if scale_factor != 1:
        r = nn.UpsamplingBilinear2d(scale_factor=scale_factor)(r)
    y = x
    for _ in range(8):
        y = y + r * (y**2 - y)
    return y, r


@pytest.mark.parametrize("scale_factor", [1, 12])
def test_released_weights_match_official_forward(scale_factor: int):
    state = released()
    model = SGZ(scale_factor=scale_factor).eval()
    model.load_state_dict(state)  # strict: official names
    torch.manual_seed(0)
    x = torch.rand(2, 3, 48, 72) * 0.4  # a multiple of 12, as the official test crops to
    with torch.no_grad():
        out = model(x)
        enhanced, curve = official_forward(state, x, scale_factor)
    assert torch.allclose(out["curve"], curve, atol=1e-6)
    assert torch.allclose(out["enhanced"], enhanced, atol=1e-6)


def test_is_zero_dce_pp_with_its_own_weights():
    assert issubclass(SGZ, ZeroDCEPP) and SGZ.weights_enum is SGZ_Weights
    assert SGZ_Weights.SICE_ME.weights.meta["test_scale_factor"] == 12
    assert MODELS.entry("sgz").meta["arch"] == "sgz"
    state = released()
    try:
        other = ZeroDCEPP_Weights.SICE_ME.weights.load_state_dict(download=False)
    except FileNotFoundError:
        return
    assert state.keys() == other.keys() and not all(torch.equal(state[k], other[k]) for k in state)


def test_load_any_size():
    try:
        model = load("sgz", scale_factor=12)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")
    with torch.no_grad():
        out = model(torch.rand(1, 3, 50, 61) * 0.3)  # not a multiple of 12: no crop needed
    assert out["enhanced"].shape == (1, 3, 50, 61)
