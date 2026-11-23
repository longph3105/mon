"""Tests for ``mon.models.denoise.tv_denoise``: parity with scikit-image's Chambolle TV."""

import numpy as np
import pytest
import torch
from skimage.restoration import denoise_tv_chambolle

from mon.models.denoise.tv_denoise import TVDenoise, tv_denoise
from mon.models.registry import MODELS, load


def noisy_image(h: int, w: int, c: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((h, w, c)) * 0.5 + 0.25 + rng.normal(0, 0.1, (h, w, c)), 0, 1)


# --- Parity with scikit-image ---

@pytest.mark.parametrize(("h", "w", "c"), [(37, 52, 3), (8, 5, 1), (2, 2, 3), (64, 64, 3)])
@pytest.mark.parametrize(("weight", "iterations"), [(0.1, 1), (0.1, 2), (0.1, 50), (0.3, 200)])
def test_matches_skimage_chambolle(h: int, w: int, c: int, weight: float, iterations: int):
    x = noisy_image(h, w, c)
    ref = denoise_tv_chambolle(x, weight=weight, eps=0, max_num_iter=iterations, channel_axis=-1)
    ours = tv_denoise(torch.from_numpy(x).permute(2, 0, 1), weight, iterations).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_float32_close_to_skimage():
    x = noisy_image(40, 48)
    ref = denoise_tv_chambolle(x, weight=0.1, eps=0, max_num_iter=50, channel_axis=-1)
    ours = tv_denoise(torch.from_numpy(x).float().permute(2, 0, 1)).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)


# --- Behavior ---

def test_channels_and_batch_items_are_independent():
    x = torch.from_numpy(noisy_image(24, 30)).permute(2, 0, 1)
    batch = torch.stack([x, x.flip(-1)])
    out = tv_denoise(batch)
    assert torch.allclose(out[1], tv_denoise(x.flip(-1)))
    assert torch.allclose(out[0, 1:2], tv_denoise(x[1:2]))  # one channel alone gives the same result


def test_denoising_reduces_error_and_weight_controls_strength():
    g = torch.Generator().manual_seed(0)
    clean = torch.linspace(0.2, 0.8, 64).expand(1, 3, 48, 64).clone()
    noisy = clean + 0.1 * torch.randn(clean.shape, generator=g)

    def err(y):
        return (y - clean).pow(2).mean().item()

    assert err(tv_denoise(noisy, 0.1)) < err(noisy) / 3

    def tv(y):
        return (y.diff(dim=-1).abs().sum() + y.diff(dim=-2).abs().sum()).item()

    assert tv(tv_denoise(noisy, 0.3)) < tv(tv_denoise(noisy, 0.05)) < tv(noisy)


def test_one_iteration_returns_input_and_constant_image_is_fixed():
    x = torch.rand(1, 3, 10, 12)
    assert torch.equal(tv_denoise(x, iterations=1), x)
    c = torch.full((3, 9, 9), 0.4)
    assert torch.allclose(tv_denoise(c), c)


def test_any_leading_dims_and_dtype():
    x = torch.rand(2, 2, 3, 16, 16, dtype=torch.float64)
    out = tv_denoise(x)
    assert out.shape == x.shape and out.dtype == torch.float64
    assert torch.allclose(out[1, 0], tv_denoise(x[1, 0]))
    assert tv_denoise(torch.rand(16, 16)).shape == (16, 16)


def test_differentiable():
    x = torch.rand(1, 3, 12, 12, requires_grad=True)
    tv_denoise(x, iterations=10).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize(("kwargs", "exc", "match"), [
    ({"weight": 0}, ValueError, "weight must be > 0"),
    ({"iterations": 0}, ValueError, "iterations must be >= 1"),
])
def test_argument_errors(kwargs: dict, exc: type[Exception], match: str):
    with pytest.raises(exc, match=match):
        tv_denoise(torch.rand(3, 8, 8), **kwargs)


def test_input_errors():
    with pytest.raises(TypeError, match="float"):
        tv_denoise(torch.zeros(3, 8, 8, dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"\[\.\.\., H, W\]"):
        tv_denoise(torch.rand(8))


@pytest.mark.gpu
def test_gpu_matches_cpu():
    from mon.core.runtime import available_devices
    devices = [d for d in available_devices() if d != "cpu"]
    if not devices:
        pytest.skip("no GPU")
    x = torch.rand(2, 3, 40, 52)
    for device in devices:
        assert torch.allclose(tv_denoise(x.to(device)).cpu(), tv_denoise(x), atol=1e-5), device


# --- Module and registry ---

def test_module_and_registry():
    model = load("tv_denoise", weight=0.2)
    assert isinstance(model, TVDenoise) and list(model.parameters()) == [] and "weight=0.2" in repr(model)
    x = torch.rand(1, 3, 16, 16)
    assert torch.equal(model(x), tv_denoise(x, 0.2))
    assert "tv_denoise" in MODELS.names(tasks="denoise")
