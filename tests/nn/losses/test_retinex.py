"""Tests for ``mon.nn.losses.retinex``: the edge-aware smoothness against its written-out definition."""

import pytest
import torch

from mon.nn.losses.retinex import EdgeAwareSmoothnessLoss


def reference(pred: torch.Tensor, guide: torch.Tensor, official: bool, sigma: float = 10.0) -> torch.Tensor:
    """All 24 neighbor terms within distance 2, per sample (weights from the YCbCr guide)."""
    mat  = torch.tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]])
    bias = torch.tensor([16.0, 128.0, 128.0]) / 255
    if official:  # the official view(-1, 3) on NCHW memory
        g = (guide.contiguous().view(-1, 3) @ mat + bias).view(guide.shape)
    else:
        g = (guide.permute(0, 2, 3, 1) @ mat + bias).permute(0, 3, 1, 2)
    h, w  = guide.shape[-2:]
    total = torch.zeros(pred.shape[0])
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if (dy, dx) == (0, 0):
                continue
            ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
            xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
            weight = torch.exp(-(g[..., ys, xs] - g[..., yd, xd]).pow(2).sum(1, keepdim=True) / (2 * sigma**2))
            diff   = (pred[..., ys, xs] - pred[..., yd, xd]).abs().sum(1, keepdim=True)
            total  = total + (weight * diff).flatten(1).mean(1)
    return total


@pytest.mark.parametrize("official", [True, False])
@pytest.mark.parametrize("channels", [1, 3])
def test_matches_the_24_neighbor_definition(official: bool, channels: int):
    torch.manual_seed(0)
    pred, guide = torch.rand(2, channels, 13, 17), torch.rand(2, 3, 13, 17)
    loss = EdgeAwareSmoothnessLoss(official_ycbcr=official, reduction="none")
    assert torch.allclose(loss(pred, guide), reference(pred, guide, official), rtol=1e-5)


def test_sigma_reductions_and_constant_map():
    torch.manual_seed(1)
    pred, guide = torch.rand(3, 1, 12, 12), torch.rand(3, 3, 12, 12)
    per_sample = EdgeAwareSmoothnessLoss(sigma=0.5, reduction="none")(pred, guide)
    assert torch.allclose(per_sample, reference(pred, guide, official=True, sigma=0.5), rtol=1e-5)
    assert torch.allclose(EdgeAwareSmoothnessLoss(sigma=0.5)(pred, guide), per_sample.mean())
    assert torch.allclose(EdgeAwareSmoothnessLoss(sigma=0.5, reduction="sum")(pred, guide), per_sample.sum())
    assert float(EdgeAwareSmoothnessLoss()(torch.full_like(pred, 0.3), guide)) == 0.0
    with pytest.raises(ValueError, match="reduction"):
        EdgeAwareSmoothnessLoss(reduction="max")  # type: ignore[arg-type]
