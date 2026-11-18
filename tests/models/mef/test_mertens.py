"""Tests for ``mon.models.mef.mertens``: parity with OpenCV (pyramids, ``MergeMertens``) and the paper's formulas."""

import cv2
import numpy as np
import pytest
import torch

import mon.models.mef.mertens as mertens_module
from mon.models.mef.mertens import Mertens, exposure_fusion, fusion_weights, pyr_down, pyr_up
from mon.models.registry import MODELS, load

GAINS = (0.25, 0.7, 1.6, 3.5)


def textured_stack(h: int, w: int, seed: int = 0) -> np.ndarray:
    """``[N, H, W, 3]`` uint8 exposure bracket of a random texture (well conditioned: contrast everywhere)."""
    base = np.random.default_rng(seed).random((h, w, 3))
    return np.stack([np.clip(255 * (base * g) ** 0.8, 0, 255) for g in GAINS]).astype(np.uint8)


def smooth_stack(h: int = 96, w: int = 128) -> torch.Tensor:
    """``[N, 3, H, W]`` uint8 bracket of a smooth 8-bit scene with clipped highlights: many pixels where every
    true weight is tiny, so float32 rounding noise in the quality measures would decide the blend."""
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None]
    base = torch.stack([0.5 + 0.45 * torch.sin(0.05 * (c + 1) * i + 0.03 * j) * torch.cos(0.04 * j - 0.02 * c * i)
                        for c in range(3)])
    base = (base * 255).round() / 255
    return torch.stack([((base * g).clamp(0, 1) ** 0.8 * 255).round() for g in GAINS]).to(torch.uint8)


def to_tensor(stack: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(stack).permute(0, 3, 1, 2).float() / 255


# --- Pyramids vs OpenCV ---

@pytest.mark.parametrize(("h", "w"), [(1, 1), (1, 4), (2, 3), (5, 8), (17, 23), (40, 61)])
def test_pyr_down_matches_cv2(h: int, w: int):
    x = np.random.default_rng(h * w).random((h, w, 3)).astype(np.float32)
    ours = pyr_down(torch.from_numpy(x).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(ours, cv2.pyrDown(x), atol=1e-6)


@pytest.mark.parametrize(("h", "w"), [(1, 1), (2, 3), (5, 8), (17, 23)])
@pytest.mark.parametrize("odd", [False, True])
def test_pyr_up_matches_cv2_for_even_and_odd_sizes(h: int, w: int, odd: bool):
    x = np.random.default_rng(h + w).random((h, w, 3)).astype(np.float32)
    size = (max(2 * h - odd, 1), max(2 * w - odd, 1))
    ours = pyr_up(torch.from_numpy(x).permute(2, 0, 1), size).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(ours, cv2.pyrUp(x, dstsize=size[::-1]), atol=1e-6)


def test_pyr_up_rejects_other_sizes():
    with pytest.raises(ValueError, match="needs size"):
        pyr_up(torch.zeros(1, 4, 4), (10, 8))


def test_pyramid_round_trip_is_exact():
    x = torch.rand(2, 3, 37, 50)
    band = x - pyr_up(pyr_down(x), x.shape[-2:])
    assert torch.allclose(band + pyr_up(pyr_down(x), x.shape[-2:]), x)
    assert torch.allclose(exposure_fusion(x[None, 0]), x[0], atol=1e-6)  # one exposure: reconstruction


# --- Fusion vs OpenCV MergeMertens ---

@pytest.mark.parametrize(("h", "w"), [(48, 64), (100, 75), (37, 211), (256, 266)])
def test_fusion_matches_cv2_merge_mertens(h: int, w: int):
    stack = textured_stack(h, w)
    ref = cv2.createMergeMertens(1.0, 1.0, 1.0).process(list(stack))
    ours = exposure_fusion(to_tensor(stack)).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_opencv_defaults_are_exposure_weight_zero():
    stack = textured_stack(40, 52, seed=3)
    ref = cv2.createMergeMertens().process(list(stack))  # contrast 1, saturation 1, exposure 0
    ours = exposure_fusion(to_tensor(stack), exposure=0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_float32_matches_float64_on_near_saturated_stack(monkeypatch: pytest.MonkeyPatch):
    # OpenCV's float32 result is itself rounding-noise driven here (it differs from float64 on ~30% of 8-bit
    # pixels of a real bracketed photo); the noise floor makes float32 match float64.
    u8 = smooth_stack()
    exact = exposure_fusion(u8.double() / 255)
    assert (exposure_fusion(u8.float() / 255).double() - exact).abs().max() < 1e-4
    monkeypatch.setattr(mertens_module, "_FLOOR", 0.0)
    assert (exposure_fusion(u8.float() / 255).double() - exact).abs().max() > 1e-2  # the floor is what fixes it


# --- Weights follow the paper ---

def test_weights_follow_the_paper_formulas():
    x = torch.rand(3, 3, 20, 24, dtype=torch.float64)
    gray = (x * torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(3, 1, 1)).sum(1, keepdim=True)
    p = torch.nn.functional.pad(gray, (1, 1, 1, 1), mode="reflect")
    lap = (p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4 * gray).abs()
    sat = x.std(dim=1, unbiased=False, keepdim=True)
    expo = torch.exp(-((x - 0.5) ** 2) / (2 * 0.2**2)).prod(dim=1, keepdim=True)
    w = lap * sat * expo + 1e-12
    assert torch.allclose(fusion_weights(x), w / w.sum(0, keepdim=True))
    w2 = lap**2 * sat**0.5 * expo**3 + 1e-12
    assert torch.allclose(fusion_weights(x, contrast=2, saturation=0.5, exposure=3), w2 / w2.sum(0, keepdim=True))


def test_well_exposedness_uses_two_sigma_squared():
    # Only the exposure term: two flat gray images at 0.1 and 0.5. Paper: exp(-0.16 / 0.08) per channel;
    # mon_old's exp(-0.16·3 / 0.4) gave 0.30 instead of 0.0025 relative weight.
    x = torch.stack([torch.full((3, 8, 8), 0.1), torch.full((3, 8, 8), 0.5)])
    w = fusion_weights(x, contrast=0, saturation=0)
    expected = np.exp(-0.16 / 0.08) ** 3
    assert w[0].mean().item() == pytest.approx(expected / (1 + expected), rel=1e-4)


def test_gray_stack_is_not_averaged():
    stack = to_tensor(textured_stack(32, 40))[:, :1]  # one channel
    w = fusion_weights(stack)
    assert w.shape == (4, 1, 32, 40) and not torch.allclose(w, torch.full_like(w, 0.25))
    assert not torch.allclose(exposure_fusion(stack), stack.mean(0), atol=1e-3)


def test_constant_images_give_equal_weights_and_no_nan():
    x = torch.full((3, 3, 16, 16), 0.5)
    assert torch.allclose(fusion_weights(x), torch.full((3, 1, 16, 16), 1 / 3))
    assert torch.allclose(exposure_fusion(x), torch.full((3, 16, 16), 0.5))


# --- Shapes, batching, options ---

def test_batched_equals_per_scene():
    stacks = torch.stack([to_tensor(textured_stack(30, 44, seed=s)) for s in (1, 2)])  # [B, N, C, H, W]
    batched = Mertens()(stacks)
    assert batched.shape == (2, 3, 30, 44)
    for b in range(2):
        assert torch.allclose(batched[b], exposure_fusion(stacks[b]), atol=1e-6)


def test_levels_option():
    stack = to_tensor(textured_stack(32, 32))
    weighted_average = (fusion_weights(stack) * stack).sum(0)
    assert torch.allclose(exposure_fusion(stack, levels=0), weighted_average)
    assert not torch.allclose(exposure_fusion(stack, levels=2), exposure_fusion(stack))
    with pytest.raises(ValueError, match="levels must be >= 0"):
        exposure_fusion(stack, levels=-1)


@pytest.mark.parametrize(("x", "exc", "match"), [
    (torch.rand(3, 32, 32), ValueError, r"\[N, C, H, W\]"),
    (torch.rand(2, 2, 8, 8), ValueError, "1 or 3 channels"),
    (torch.zeros(2, 3, 8, 8, dtype=torch.uint8), TypeError, "float"),
])
def test_input_errors(x: torch.Tensor, exc: type[Exception], match: str):
    with pytest.raises(exc, match=match):
        exposure_fusion(x)


def test_module_requires_batched_stacks_and_has_no_parameters():
    model = Mertens(exposure=0.5, levels=3)
    assert list(model.parameters()) == [] and "exposure=0.5" in repr(model)
    with pytest.raises(ValueError, match=r"\[B, N, C, H, W\]"):
        model(torch.rand(3, 3, 8, 8))


def test_gradients_are_finite():
    stack = to_tensor(textured_stack(24, 24))
    stack[:, :, :4, :4] = 0.5  # gray pixels: zero saturation and contrast
    stack.requires_grad_(True)
    exposure_fusion(stack).sum().backward()
    assert stack.grad is not None and torch.isfinite(stack.grad).all()


def test_float64_and_dtype_preserved():
    stack = to_tensor(textured_stack(16, 20)).double()
    assert exposure_fusion(stack).dtype == torch.float64


# --- Registry ---

def test_registered_and_loadable_without_weights():
    assert MODELS.names(tasks="mef") == ["mertens"]
    model = load("mertens", exposure=0)
    assert isinstance(model, Mertens) and model.exposure == 0 and not model.training


@pytest.mark.gpu
def test_gpu_matches_cpu():
    from mon.core.runtime import available_devices
    devices = [d for d in available_devices() if d != "cpu"]
    if not devices:
        pytest.skip("no GPU")
    stacks = torch.stack([to_tensor(textured_stack(64, 80))])
    cpu = Mertens()(stacks)
    for device in devices:
        assert torch.allclose(Mertens()(stacks.to(device)).cpu(), cpu, atol=1e-5), device
