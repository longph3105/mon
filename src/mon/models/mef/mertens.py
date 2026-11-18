"""Mertens exposure fusion: blend an exposure stack by contrast, saturation, and well-exposedness.

Each exposure gets a per-pixel weight ``C^wc · S^ws · E^we``: ``C`` is the absolute Laplacian
of the grayscale image, ``S`` the standard deviation across color channels, and ``E`` the
product over channels of ``exp(-(x - 0.5)² / (2σ²))``. Weights are normalized across the
stack, and images are blended with Laplacian pyramids weighted by Gaussian pyramids of the
weights, which avoids seams. No training, no parameters; differentiable and batched on GPU.

Follows the paper and OpenCV's ``MergeMertens`` (same pyramids: 5-tap Burt–Adelson kernel,
reflect-101 borders, ``floor(log2(min(H, W)))`` levels above the base; the tests compare with
``cv2.pyrDown``/``pyrUp`` and ``cv2.createMergeMertens``). Paper defaults ``contrast = saturation
= exposure = 1``; OpenCV's defaults are ``1, 1, 0`` (no well-exposedness term).

One deliberate difference from OpenCV: contrast and saturation below ``1e-6`` count as 0. In
float32 they are rounding noise there (the smallest real 8-bit contrast is ~4e-5), and in pixels
where every true weight is tiny (near-saturated stacks) that noise decides the blend. On a real
bracketed photo, OpenCV's float32 output differs from a float64 computation on ~28% of 8-bit
pixels; this implementation in float32 matches float64 on all of them.

Output is not clipped: Laplacian blending can overshoot ``[0, 1]`` (down to -0.33 on that photo,
0.3% of values), as in OpenCV; clamp when saving.

Differences from ``mon_old`` (and the PyTorch code it came from, Jamy-L/Pytorch-Exposure-Fusion):

- Well-exposedness divided by ``2σ`` instead of ``2σ²`` (0.4 instead of 0.08), so the term
  barely mattered: a pixel at 0.1 or 0.9 kept 67% of its weight instead of 13%.
- Fixed 4 pyramid levels with a 3×3 kernel: on large images the coarsest level stayed at 1/8
  resolution, leaving low-frequency seams. Now the full depth (``levels=None``) by default.
- Grayscale stacks were averaged: saturation is 0 for one channel, so all weights were 0 and
  fell back to uniform. Saturation is now skipped for one channel, as in OpenCV.
- The Laplacian zero-padded the image (false contrast at the borders); now reflect-101.
- The PyTorch path crashed on every input with 2+ levels: the Laplacian pyramid was built
  coarse-to-fine but zipped with the fine-to-coarse weight pyramid (size mismatch).
- The batch dimension was the exposure stack; inputs now have an explicit exposure axis.
- Removed the NumPy path (it aligned the inputs in place with ``AlignMTB``, returned uint8
  HWC, and ignored ``n_levels``) and ``benchmark`` (crashed: the model has no parameters).

Examples:
    >>> import torch
    >>> scene = torch.rand(1, 3, 40, 60)
    >>> stack = torch.stack([(scene * g).clamp(0, 1) for g in (0.3, 1.0, 3.0)], dim=1)  # [B, N, C, H, W]
    >>> Mertens()(stack).shape
    torch.Size([1, 3, 40, 60])
    >>> fusion_weights(stack[0]).sum(dim=0).allclose(torch.ones(1, 40, 60))  # one scene: [N, C, H, W]
    True

References:
    - Mertens, Kautz, Van Reeth, "Exposure Fusion", Pacific Graphics 2007.
      https://doi.org/10.1109/PG.2007.17
    - OpenCV ``MergeMertens``: https://docs.opencv.org/4.x/d7/dd6/classcv_1_1MergeMertens.html
    - https://github.com/Jamy-L/Pytorch-Exposure-Fusion (MIT; ported by ``mon_old``)
"""

__all__ = [
    "Mertens",
    "exposure_fusion",
    "fusion_weights",
    "pyr_down",
    "pyr_up",
]

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_BINOMIAL = (1.0, 4.0, 6.0, 4.0, 1.0)
_LUMA     = (0.299, 0.587, 0.114)  # RGB → gray, as cv2.COLOR_RGB2GRAY
_FLOOR = 1e-6  # quality measures below this are float32 rounding noise (see fusion_weights)


# --- Pyramids (match cv2.pyrDown / cv2.pyrUp) ---

def _reflect101_index(n: int, pad: int, device: torch.device) -> Tensor:
    """Indices of ``range(-pad, n + pad)`` mapped into ``[0, n)`` by reflect-101 (``cv2.BORDER_REFLECT_101``)."""
    i = torch.arange(-pad, n + pad, device=device)
    if n == 1:
        return torch.zeros_like(i)
    period = 2 * n - 2
    i = i.abs() % period
    return torch.where(i > n - 1, period - i, i)


def _pad_reflect101(x: Tensor, pad: int) -> Tensor:
    """Reflect-101 padding of the last two dims; works for any size (``F.pad`` needs ``pad < size``)."""
    h, w = x.shape[-2:]
    x = x.index_select(-2, _reflect101_index(h, pad, x.device))
    return x.index_select(-1, _reflect101_index(w, pad, x.device))


def _binomial_conv(x: Tensor, scale: float, stride: int = 1) -> Tensor:
    """Depthwise 5×5 binomial filter (``outer([1, 4, 6, 4, 1])·scale``) on reflect-101-padded ``x``."""
    k1 = torch.tensor(_BINOMIAL, dtype=x.dtype, device=x.device)
    kernel = (torch.outer(k1, k1) * scale)[None, None]
    shape = x.shape
    flat = _pad_reflect101(x.reshape(-1, 1, *shape[-2:]), 2)
    out = F.conv2d(flat, kernel, stride=stride)
    return out.reshape(*shape[:-2], *out.shape[-2:])


def pyr_down(x: Tensor) -> Tensor:
    """Blur and drop every other row and column: ``[..., H, W]`` → ``[..., ⌈H/2⌉, ⌈W/2⌉]`` (``cv2.pyrDown``)."""
    return _binomial_conv(x, 1 / 256, stride=2)


def pyr_up(x: Tensor, size: Sequence[int]) -> Tensor:
    """Upsample ``[..., h, w]`` to ``size`` (``2h`` or ``2h - 1``, same for width), as ``cv2.pyrUp(dstsize=)``.

    Raises:
        ValueError: If ``size`` isn't ``2h`` or ``2h - 1`` (and the same for the width).
    """
    h, w = x.shape[-2:]
    if size[0] not in (2 * h, 2 * h - 1) or size[1] not in (2 * w, 2 * w - 1):
        raise ValueError(f"pyr_up from {h}×{w} needs size (2h or 2h-1, 2w or 2w-1), got {tuple(size)}")
    up = x.new_zeros(*x.shape[:-2], 2 * h, 2 * w)
    up[..., ::2, ::2] = x
    return _binomial_conv(up, 1 / 64)[..., : size[0], : size[1]]  # odd sizes: OpenCV crops the even result


# --- Weights ---

def _check_images(images: Tensor) -> None:
    if images.ndim not in (4, 5):
        raise ValueError(f"images must be [N, C, H, W] or [B, N, C, H, W], got shape {tuple(images.shape)}")
    if images.shape[-3] not in (1, 3):
        raise ValueError(f"images must have 1 or 3 channels, got {images.shape[-3]}")
    if not images.is_floating_point():
        raise TypeError(f"images must be float in [0, 1], got {images.dtype}")


def fusion_weights(
    images    : Tensor,
    *,
    contrast  : float = 1.0,
    saturation: float = 1.0,
    exposure  : float = 1.0,
    sigma     : float = 0.2,
) -> Tensor:
    """Per-exposure weight maps, normalized to sum to 1 over the stack.

    Args:
        images: ``[N, C, H, W]`` (one stack) or ``[B, N, C, H, W]``, float RGB or gray in ``[0, 1]``.
        contrast, saturation, exposure: Exponents of the three quality measures (0 disables one).
            Saturation is ignored for gray images.
        sigma: Width of the well-exposedness Gaussian around 0.5.

    Returns:
        ``[N, 1, H, W]`` or ``[B, N, 1, H, W]``.
    """
    _check_images(images)
    c = images.shape[-3]
    if c == 3:
        luma = torch.tensor(_LUMA, dtype=images.dtype, device=images.device).view(3, 1, 1)
        gray = (images * luma).sum(dim=-3, keepdim=True)
    else:
        gray = images
    laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=images.dtype,
                             device=images.device)[None, None]
    flat_gray = _pad_reflect101(gray.reshape(-1, 1, *gray.shape[-2:]), 1)
    measure_c = F.conv2d(flat_gray, laplacian).abs().reshape(gray.shape)
    # Below _FLOOR, contrast is float32 rounding noise (~1e-7 on [0, 1] images; the smallest real 8-bit
    # Laplacian is ~4e-5). Where every true weight is tiny (near-saturated stacks), noise would decide the blend.
    measure_c = torch.where(measure_c < _FLOOR, 0.0, measure_c)
    weights   = measure_c.pow(contrast)
    if c == 3 and saturation != 0:
        var = (images - images.mean(dim=-3, keepdim=True)).pow(2).mean(dim=-3, keepdim=True)
        # Same noise floor as contrast (gray pixels: the channel mean is off by an ulp); also keeps sqrt's
        # gradient finite at 0.
        pos = var >= _FLOOR**2
        measure_s = torch.where(pos, torch.where(pos, var, 1.0).sqrt(), 0.0)
        weights = weights * measure_s.pow(saturation)
    if exposure != 0:
        measure_e = torch.exp(-(images - 0.5).pow(2) / (2 * sigma**2)).prod(dim=-3, keepdim=True)
        weights = weights * measure_e.pow(exposure)
    weights = weights + 1e-12  # as OpenCV: all-zero pixels get equal weights
    return weights / weights.sum(dim=-4, keepdim=True)


# --- Fusion ---

def exposure_fusion(
    images    : Tensor,
    *,
    contrast  : float = 1.0,
    saturation: float = 1.0,
    exposure  : float = 1.0,
    sigma     : float = 0.2,
    levels    : int | None = None,
) -> Tensor:
    """Fuse an exposure stack into one image (Mertens et al. 2007).

    Args:
        images: ``[N, C, H, W]`` (one stack) or ``[B, N, C, H, W]``, float RGB or gray in ``[0, 1]``,
            aligned (no motion compensation is done).
        contrast, saturation, exposure, sigma: See ``fusion_weights``.
        levels: Pyramid levels above full resolution; ``None`` = ``floor(log2(min(H, W)))`` (as OpenCV).

    Returns:
        ``[C, H, W]`` or ``[B, C, H, W]``; not clipped to ``[0, 1]``.

    Raises:
        ValueError: For other shapes or channel counts, or a negative ``levels``.
        TypeError: For non-float inputs.
    """
    weights = fusion_weights(images, contrast=contrast, saturation=saturation, exposure=exposure, sigma=sigma)
    h, w = images.shape[-2:]
    levels = (min(h, w).bit_length() - 1) if levels is None else levels
    if levels < 0:
        raise ValueError(f"levels must be >= 0, got {levels}")

    image_g, weight_g = [images], [weights]
    for _ in range(levels):
        image_g.append(pyr_down(image_g[-1]))
        weight_g.append(pyr_down(weight_g[-1]))
    fused = []
    for lvl in range(levels + 1):
        band = image_g[lvl] if lvl == levels else image_g[lvl] - pyr_up(image_g[lvl + 1], image_g[lvl].shape[-2:])
        fused.append((band * weight_g[lvl]).sum(dim=-4))
    out = fused[-1]
    for lvl in range(levels - 1, -1, -1):
        out = pyr_up(out, fused[lvl].shape[-2:]) + fused[lvl]
    return out


class Mertens(nn.Module):
    """Mertens exposure fusion as a parameter-free module (see ``exposure_fusion``).

    Args:
        contrast, saturation, exposure, sigma, levels: See ``exposure_fusion``.

    Shapes:
        ``forward(images [B, N, C, H, W] in [0, 1])`` → ``[B, C, H, W]``.
    """

    def __init__(
        self,
        contrast  : float = 1.0,
        saturation: float = 1.0,
        exposure  : float = 1.0,
        sigma     : float = 0.2,
        levels    : int | None = None,
    ) -> None:
        super().__init__()
        self.contrast   = contrast
        self.saturation = saturation
        self.exposure   = exposure
        self.sigma      = sigma
        self.levels     = levels

    def extra_repr(self) -> str:
        return (f"contrast={self.contrast}, saturation={self.saturation}, exposure={self.exposure}, "
                f"sigma={self.sigma}, levels={self.levels}")

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"images must be [B, N, C, H, W], got shape {tuple(images.shape)}")
        return exposure_fusion(images, contrast=self.contrast, saturation=self.saturation, exposure=self.exposure,
                               sigma=self.sigma, levels=self.levels)
