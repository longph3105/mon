"""Full-reference fidelity metrics: PSNR and SSIM.

Defaults follow pyiqa (``psnr``, ``ssim``), the toolbox ``mon_old`` evaluated with, so earlier
numbers stay comparable: PSNR on RGB floats; SSIM on luma rounded to 8-bit levels (MATLAB's
``ssim`` on ``rgb2gray`` of a ``uint8`` image), without MATLAB's automatic downsampling.

Known differences:
    - pyiqa clamps SSIM's contrast-structure term at 0 (``relu``); so does ``SSIM`` here. Wang et al.
      and scikit-image don't; the two agree wherever that term is non-negative (similar images).
    - PSNR adds ``eps = 1e-8`` to the MSE (pyiqa): identical images score 80 dB, not ``inf``.

References:
    - Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error
      visibility to structural similarity," IEEE TIP 13(4), 2004. Reference code:
      https://ece.uwaterloo.ca/~z70wang/research/ssim/
    - pyiqa (https://github.com/chaofengc/IQA-PyTorch, PolyForm Noncommercial): protocol read, code not copied.
"""

__all__ = [
    "PSNR",
    "SSIM",
    "gaussian_window",
    "psnr",
    "ssim",
]

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from mon.metrics.base import Metric, MetricMode, check_pair, crop_border, rgb_to_luma, to_8bit_levels


def psnr(pred: Tensor, target: Tensor, *, y_channel: bool = False, border: int = 0, eps: float = 1e-8) -> Tensor:
    """Per-image PSNR in dB of ``[0, 1]`` images (data range 1), as float64 ``[B]``.

    Examples:
        >>> x = torch.full((2, 3, 4, 4), 0.5)
        >>> psnr(x, x + 0.1).round(decimals=4).tolist()
        [20.0, 20.0]
    """
    check_pair(pred, target)
    pred, target = crop_border(pred, border).double(), crop_border(target, border).double()
    if y_channel:
        pred, target = rgb_to_luma(pred), rgb_to_luma(target)
    mse = (pred - target).square().mean(dim=(1, 2, 3))
    return 10 * torch.log10(1 / (mse + eps))


def gaussian_window(size: int = 11, sigma: float = 1.5) -> Tensor:
    """Normalized ``size × size`` Gaussian (MATLAB ``fspecial('gaussian', size, sigma)``), float64."""
    r = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma**2))
    g[g < torch.finfo(torch.float64).eps * g.max()] = 0
    return g / g.sum()


def ssim(
    pred      : Tensor,
    target    : Tensor,
    *,
    y_channel : bool  = True,
    border    : int   = 0,
    downsample: bool  = False,
    window    : int   = 11,
    sigma     : float = 1.5,
) -> Tensor:
    """Per-image mean SSIM of ``[0, 1]`` images, as float64 ``[B]``.

    Computed on 8-bit levels (``to_8bit_levels``; data range 255, ``K1 = 0.01``, ``K2 = 0.03``) with
    a Gaussian window and no padding ("valid"). RGB input with ``y_channel=False`` averages the
    three channel SSIMs.

    Args:
        pred, target: ``[B, 1|3, H, W]`` floats in ``[0, 1]``, same shape.
        y_channel: Score luma (rounded to 8-bit levels) instead of RGB.
        border: Pixels cropped on every side first.
        downsample: Average-pool by ``max(1, round(min(H, W) / 256))`` first, as Wang's MATLAB
            ``ssim.m`` does.
        window, sigma: Gaussian window size and standard deviation.

    Raises:
        ValueError: If the (cropped, downsampled) image is smaller than the window.

    Examples:
        >>> x = torch.rand(2, 3, 32, 32)
        >>> ssim(x, x).tolist()
        [1.0, 1.0]
    """
    check_pair(pred, target)
    pred, target = crop_border(pred, border), crop_border(target, border)
    if y_channel:
        pred, target = rgb_to_luma(pred.double()), rgb_to_luma(target.double())
    x, y = to_8bit_levels(pred), to_8bit_levels(target)
    factor = max(1, round(min(x.shape[-2:]) / 256))
    if downsample and factor > 1:
        x, y = F.avg_pool2d(x, factor), F.avg_pool2d(y, factor)
    if min(x.shape[-2:]) < window:
        raise ValueError(f"image {tuple(x.shape[-2:])} is smaller than the {window}×{window} SSIM window")

    c = x.shape[1]
    w = gaussian_window(window, sigma).to(x.device).expand(c, 1, window, window)

    def blur(t: Tensor) -> Tensor:
        return F.conv2d(t, w, groups=c)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x, mu_y = blur(x), blur(y)
    var_x  = blur(x * x) - mu_x**2
    var_y  = blur(y * y) - mu_y**2
    cov_xy = blur(x * y) - mu_x * mu_y
    cs     = F.relu((2 * cov_xy + c2) / (var_x + var_y + c2))
    lum    = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    return (lum * cs).mean(dim=(1, 2, 3))


class PSNR(Metric):
    """Peak signal-to-noise ratio (dB, higher is better); see ``psnr``.

    Examples:
        >>> PSNR()(torch.zeros(1, 3, 8, 8), torch.full((1, 3, 8, 8), 0.1)).round(decimals=4).tolist()
        [20.0]
    """

    mode         : ClassVar[MetricMode] = "fr"
    higher_better: ClassVar[bool] = True

    def __init__(self, *, y_channel: bool = False, crop_border: int = 0) -> None:
        super().__init__()
        self.y_channel   = y_channel
        self.crop_border = crop_border

    def extra_repr(self) -> str:
        return f"y_channel={self.y_channel}, crop_border={self.crop_border}"

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return psnr(pred, target, y_channel=self.y_channel, border=self.crop_border)


class SSIM(Metric):
    """Structural similarity (``[0, 1]``, higher is better); see ``ssim``.

    Examples:
        >>> metric = SSIM()
        >>> metric
        SSIM(y_channel=True, crop_border=0, downsample=False)
        >>> metric(torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16)).tolist()
        [1.0]
    """

    mode         : ClassVar[MetricMode] = "fr"
    higher_better: ClassVar[bool] = True

    def __init__(self, *, y_channel: bool = True, crop_border: int = 0, downsample: bool = False) -> None:
        super().__init__()
        self.y_channel   = y_channel
        self.crop_border = crop_border
        self.downsample  = downsample

    def extra_repr(self) -> str:
        return f"y_channel={self.y_channel}, crop_border={self.crop_border}, downsample={self.downsample}"

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ssim(pred, target, y_channel=self.y_channel, border=self.crop_border, downsample=self.downsample)

