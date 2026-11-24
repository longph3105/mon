"""Tests for ``mon.nn.losses.zero_reference``: parity with the official Zero-DCE code (reference copies below)."""

import pytest
import torch
from torch.nn import functional as F

from mon.nn.losses.zero_reference import (
    ColorConstancyLoss, ExposureControlLoss, IlluminationSmoothnessLoss, SpatialConsistencyLoss,
)

# --- Reference: Zero-DCE Myloss.py (as in mon_old/models/enhance/lle/zero_dce/loss.py), batch-level ---


def ref_col(x):
    mean_rgb = torch.mean(x, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    d_rg, d_rb, d_gb = (mr - mg) ** 2, (mr - mb) ** 2, (mb - mg) ** 2
    return torch.mean(torch.pow(d_rg**2 + d_rb**2 + d_gb**2, 0.5))


def ref_spa(org, enhance):
    def k(m):
        return torch.tensor(m, dtype=torch.float32)[None, None]

    kernels = [k([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), k([[0, 0, 0], [0, 1, -1], [0, 0, 0]]),
               k([[0, -1, 0], [0, 1, 0], [0, 0, 0]]), k([[0, 0, 0], [0, 1, 0], [0, -1, 0]])]
    org_pool = F.avg_pool2d(torch.mean(org, 1, keepdim=True), 4)
    enh_pool = F.avg_pool2d(torch.mean(enhance, 1, keepdim=True), 4)
    e = sum((F.conv2d(org_pool, kk, padding=1) - F.conv2d(enh_pool, kk, padding=1)) ** 2 for kk in kernels)
    return torch.mean(e)


def ref_exp(x, patch_size=16, mean_val=0.6):
    mean = F.avg_pool2d(torch.mean(x, 1, keepdim=True), patch_size)
    return torch.mean((mean - mean_val) ** 2)


def ref_tv(x):
    b, _, h, w = x.shape
    count_h, count_w = (h - 1) * w, h * (w - 1)
    h_tv = ((x[:, :, 1:, :] - x[:, :, : h - 1, :]) ** 2).sum()
    w_tv = ((x[:, :, :, 1:] - x[:, :, :, : w - 1]) ** 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / b


@pytest.fixture
def images():
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, 3, 64, 48, generator=g) * 0.3, torch.rand(3, 3, 64, 48, generator=g)  # low, enhanced


# --- Parity ---

def test_color_constancy_matches_reference(images):
    _, enh = images
    assert torch.allclose(ColorConstancyLoss()(enh), ref_col(enh), atol=1e-7)


def test_color_constancy_gradient_at_equal_means():
    """Equal channel means (a fully saturated crop): finite loss and gradient (the official sqrt(0) gives NaN)."""
    x = torch.ones(2, 3, 8, 8, requires_grad=True)
    loss = ColorConstancyLoss()(x)
    loss.backward()
    assert torch.isfinite(loss) and float(loss) <= 1e-6
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spatial_consistency_matches_reference(images):
    low, enh = images
    assert torch.allclose(SpatialConsistencyLoss()(enh, low), ref_spa(low, enh), atol=1e-7)


@pytest.mark.parametrize(("patch", "target"), [(16, 0.6), (8, 0.5), (16, 0.45)])
def test_exposure_matches_reference(images, patch, target):
    _, enh = images
    assert torch.allclose(ExposureControlLoss(patch, target)(enh), ref_exp(enh, patch, target), atol=1e-7)


def test_illumination_smoothness_matches_reference():
    curves = torch.randn(2, 24, 33, 17, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(IlluminationSmoothnessLoss()(curves), ref_tv(curves), rtol=1e-6)


def test_odd_sizes_match_reference():
    g = torch.Generator().manual_seed(2)
    low, enh = torch.rand(2, 3, 37, 53, generator=g), torch.rand(2, 3, 37, 53, generator=g)  # pooling floors
    assert torch.allclose(SpatialConsistencyLoss()(enh, low), ref_spa(low, enh), atol=1e-7)
    assert torch.allclose(ExposureControlLoss()(enh), ref_exp(enh), atol=1e-7)


# --- Reduction, gradients, devices ---

@pytest.mark.parametrize("make", [
    lambda: (ColorConstancyLoss, lambda loss, x, y: loss(y)),
    lambda: (ExposureControlLoss, lambda loss, x, y: loss(y)),
    lambda: (SpatialConsistencyLoss, lambda loss, x, y: loss(y, x)),
    lambda: (IlluminationSmoothnessLoss, lambda loss, x, y: loss(y)),
])
def test_reductions_and_gradients(images, make):
    cls, call = make()
    low, enh = images
    enh = enh.clone().requires_grad_(True)
    none = call(cls(reduction="none"), low, enh)
    assert none.shape == (3,)
    assert torch.allclose(call(cls(reduction="mean"), low, enh), none.mean())
    assert torch.allclose(call(cls(reduction="sum"), low, enh), none.sum())
    call(cls(), low, enh).backward()
    assert enh.grad is not None and torch.isfinite(enh.grad).all()


def test_per_sample_values_are_independent(images):
    low, enh = images
    full = SpatialConsistencyLoss(reduction="none")(enh, low)
    single = SpatialConsistencyLoss(reduction="none")(enh[1:2], low[1:2])
    assert torch.allclose(full[1:2], single)


def test_float64_and_buffer_not_in_state_dict(images):
    low, enh = images
    assert SpatialConsistencyLoss()(enh.double(), low.double()).dtype == torch.float64
    assert SpatialConsistencyLoss().state_dict() == {}  # kernels are a non-persistent buffer


# --- Validation ---

def test_errors():
    x = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="reduction"):
        ColorConstancyLoss(reduction="avg")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="3 channels"):
        ColorConstancyLoss()(torch.rand(1, 1, 8, 8))
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        ExposureControlLoss()(torch.rand(3, 8, 8))
    with pytest.raises(ValueError, match="smaller than patch_size"):
        ExposureControlLoss(16)(x)
    with pytest.raises(ValueError, match="same shape"):
        SpatialConsistencyLoss()(x, torch.rand(1, 3, 8, 4))
    with pytest.raises(ValueError, match="2×2"):
        IlluminationSmoothnessLoss()(torch.rand(1, 3, 1, 8))
    with pytest.raises(ValueError, match="patch_size"):
        ExposureControlLoss(0)
