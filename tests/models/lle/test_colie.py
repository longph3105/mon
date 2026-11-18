"""Tests for ``mon.models.lle.colie``: parity with mon_old (= official training), building blocks, pipeline."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mon.models.lle.colie import (
    CoLIE, CoLIELoss, _coords, _patches, colie_enhance, fast_guided_filter, fit_illumination, replace_value,
)
from mon.models.registry import MODELS, load
from mon.nn.losses.zero_reference import IlluminationSmoothnessLoss

_spec = importlib.util.spec_from_file_location("colie_reference", Path(__file__).with_name("colie_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats, SETTINGS = _ref.reference_input, _ref.stats, _ref.SETTINGS

# mon_old CoLIE.forward_step after torch.manual_seed(0), low-resolution results (output of colie_reference.py).
# Generated from mon_old with torch 2.14.0.
MON_OLD: dict = {'value_lr': {'shape': [1, 1, 32, 32],
              'sum': 201.62577287480235,
              'sumsq': 42.29143461708906,
              'values': [0.27701330184936523,
                         0.19042789936065674,
                         0.22851742804050446,
                         0.27340295910835266,
                         0.2503262460231781,
                         0.28053203225135803]},
 'illumination_lr': {'shape': [1, 1, 32, 32],
                     'sum': 559.2204475998878,
                     'sumsq': 307.28463862330557,
                     'values': [0.6103748083114624,
                                0.5351461172103882,
                                0.580457866191864,
                                0.6122844815254211,
                                0.6000648736953735,
                                0.6374959945678711]},
 'enhanced_v_lr': {'shape': [1, 1, 32, 32],
                   'sum': 364.74534829705954,
                   'sumsq': 134.98997336734033,
                   'values': [0.45376694202423096,
                              0.355776309967041,
                              0.3936169445514679,
                              0.4464563727378845,
                              0.41709578037261963,
                              0.439983993768692]},
 'mon_old_output_mean': 0.2561054825782776,
 'input_mean': 0.14065666496753693}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


# --- Parity: the per-image fit ---

def test_fit_matches_mon_old():
    x = reference_input()
    value_lr = F.interpolate(x.amax(1, keepdim=True), size=(SETTINGS["hidden_dim"],) * 2)
    assert_matches(value_lr, MON_OLD["value_lr"], atol=1e-12)
    illumination = fit_illumination(value_lr, window=SETTINGS["window"], hidden_dim=SETTINGS["hidden_dim"],
                                    epochs=SETTINGS["epochs"], loss=CoLIELoss(exposure=SETTINGS["exposure"]), seed=0)
    assert_matches(illumination, MON_OLD["illumination_lr"], atol=1e-6)
    assert_matches(value_lr / (illumination + 1e-4), MON_OLD["enhanced_v_lr"], atol=1e-6)


# --- Building blocks vs their definitions ---

def test_patches_and_coords_match_official_construction():
    v = torch.rand(1, 1, 9, 9)
    for k in (1, 3, 7):  # official get_patches: one-hot conv kernels on a reflect-padded image
        kernel = torch.zeros(k * k, 1, k, k)
        for i in range(k):
            for j in range(k):
                kernel[i * k + j, 0, i, j] = 1
        expected = F.conv2d(F.pad(v, (k // 2,) * 4, mode="reflect") if k > 1 else v, kernel)[0].movedim(0, -1)
        assert torch.allclose(_patches(v, k), expected, atol=1e-6)  # conv may use Winograd (inexact); unfold copies
    expected = torch.from_numpy(np.dstack(np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 12)))).float()
    assert torch.equal(_coords(12, torch.device("cpu")), expected)


def test_guided_filter_matches_definition():
    torch.manual_seed(0)
    g_lr, t_lr = torch.rand(2, 1, 20, 24, dtype=torch.float64), torch.rand(2, 1, 20, 24, dtype=torch.float64)
    g_hr = torch.rand(2, 1, 50, 61, dtype=torch.float64)
    box = lambda t: F.avg_pool2d(t, 3, 1, 1, count_include_pad=False)  # noqa: E731  (mean over the truncated window)
    a = (box(g_lr * t_lr) - box(g_lr) * box(t_lr)) / (box(g_lr**2) - box(g_lr) ** 2 + 1e-8)
    b = box(t_lr) - a * box(g_lr)
    up = lambda t: F.interpolate(t, (50, 61), mode="bilinear", align_corners=True)  # noqa: E731
    assert torch.allclose(fast_guided_filter(g_lr, t_lr, g_hr), up(a) * g_hr + up(b), atol=1e-9)
    assert torch.allclose(fast_guided_filter(g_lr, 2 * g_lr + 0.1, g_hr), 2 * g_hr + 0.1, atol=1e-6)  # linear map kept
    with pytest.raises(ValueError, match="must exceed"):
        fast_guided_filter(torch.rand(1, 1, 3, 8), torch.rand(1, 1, 3, 8), torch.rand(1, 1, 9, 9))


def hsv_round_trip(rgb: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    """Replace V through HSV, written from the standard definition (as the official ``color.py``)."""
    cmax, idx = rgb.max(dim=1, keepdim=True)
    delta = cmax - rgb.min(dim=1, keepdim=True)[0]
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    safe = torch.where(delta > 0, delta, 1.0)
    h = torch.where(idx == 0, ((g - b) / safe) % 6, torch.where(idx == 1, (b - r) / safe + 2, (r - g) / safe + 4))
    h = torch.where(delta > 0, h, 0.0) / 6
    s = torch.where(cmax > 0, delta / torch.where(cmax > 0, cmax, 1.0), 0.0)
    c = v_new * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    zero = torch.zeros_like(c)
    sector = (h * 6).floor().clamp(0, 5)
    table = [(c, x, zero), (x, c, zero), (zero, c, x), (zero, x, c), (x, zero, c), (c, zero, x)]
    out = torch.zeros_like(rgb)
    for k, (cr, cg, cb) in enumerate(table):
        mask = sector == k
        out = torch.where(mask, torch.cat([cr, cg, cb], 1), out)
    return out + (v_new - c)


def test_replace_value_equals_hsv_round_trip():
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 9, 11, dtype=torch.float64)
    rgb[0, :, 0, 0] = 0.0          # black
    rgb[0, :, 0, 1] = 0.4          # gray
    rgb[1, :, 0, 0] = torch.tensor([0.7, 0.7, 0.2], dtype=torch.float64)  # two channels at the max
    v_new = torch.rand(2, 1, 9, 11, dtype=torch.float64)
    out = replace_value(rgb, v_new)
    assert torch.allclose(out, hsv_round_trip(rgb, v_new), atol=1e-12)
    assert torch.allclose(out.amax(1, keepdim=True), v_new)


def test_loss_terms():
    torch.manual_seed(0)
    v = torch.rand(2, 1, 32, 32) * 0.3
    illumination = v + torch.rand(2, 1, 32, 32) * 0.5
    loss = CoLIELoss(alpha=1, beta=20, gamma=8, delta=5, exposure=0.5)
    expected = (F.mse_loss(illumination, v) + 20 * IlluminationSmoothnessLoss()(illumination)
                + 8 * ((F.avg_pool2d(illumination, 16) ** 0.5 - 0.5) ** 2).mean() + 5 * (v / (illumination + 1e-4)).mean())
    assert torch.allclose(loss(illumination, v), expected)
    per_sample = CoLIELoss(reduction="none")(illumination, v)
    assert per_sample.shape == (2,) and torch.allclose(per_sample.mean(), CoLIELoss()(illumination, v))
    with pytest.raises(ValueError, match="reduction"):
        CoLIELoss(reduction="max")  # type: ignore[arg-type]


def test_siren_initialization():
    torch.manual_seed(0)
    net = CoLIE(window=7, hidden_dim=64)
    first, hidden = net.patch_net[0].linear.weight, net.patch_net[1].linear.weight
    assert first.abs().max() <= 1 / 49 and hidden.abs().max() <= math.sqrt(6 / 64) / 30
    assert net.spatial_net[0].linear.weight.abs().max() <= 1 / 2
    out = net(torch.rand(5, 6, 49), torch.rand(5, 6, 2))
    assert out.shape == (5, 6, 1) and out.min() > 0 and out.max() < 1


# --- Pipeline ---

def small(**kwargs):
    return {"down_size": 32, "hidden_dim": 32, "epochs": 15, "seed": 0, **kwargs}


def test_enhance_brightens_and_rescales():
    x = reference_input()
    y = colie_enhance(x, **small())
    assert y.shape == x.shape and y.mean() > 1.5 * x.mean()
    assert y.max().item() == pytest.approx(1.0) and y.min() >= 0
    clamped = colie_enhance(x, **small(rescale_output=False))
    assert clamped.max() <= 1 and clamped.mean() < y.mean()
    assert not torch.allclose(colie_enhance(x, **small(rescale_input=False)), y)


def test_batch_items_are_independent_and_unbatched_input():
    a, b = reference_input(), reference_input().flip(-1)
    batched = colie_enhance(torch.cat([a, b]), **small())
    assert torch.allclose(batched[1:], colie_enhance(b, **small()), atol=1e-6)
    assert colie_enhance(a[0], **small()).shape == a[0].shape


def test_seed_reproducible_global_rng_untouched_and_inference_mode():
    x = reference_input()
    torch.manual_seed(5)
    expected_next = torch.rand(1)
    torch.manual_seed(5)
    first = colie_enhance(x, **small())
    assert torch.equal(torch.rand(1), expected_next)
    with torch.inference_mode():
        assert torch.equal(colie_enhance(x, **small()), first)
    assert not torch.equal(colie_enhance(x, **small(seed=1)), first)


@pytest.mark.parametrize(("x", "kwargs", "match"), [
    (torch.rand(1, 1, 20, 20), {}, r"\[B, 3, H, W\]"),
    (torch.rand(1, 3, 20, 20), {"down_size": 8}, "n >= 16"),
    (torch.rand(1, 3, 20, 20), {"epochs": 0}, "epochs must be >= 1"),
])
def test_errors(x: torch.Tensor, kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        colie_enhance(x, **{**small(), **kwargs})


def test_registry():
    entry = MODELS.entry("colie")
    assert entry.obj is CoLIE and entry.meta["zero_shot"] is True
    with pytest.raises(ValueError, match="no pretrained weights"):
        load("colie")
    assert isinstance(load("colie", weights=None), CoLIE)
