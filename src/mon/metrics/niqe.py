"""NIQE: natural image quality evaluator (no-reference; lower is better).

Each image is converted to luma rounded to 8-bit levels (the official MATLAB code runs
``rgb2gray`` on the ``uint8`` image; pyiqa's ``niqe`` does the same), cropped to whole 96×96
blocks, and described by 36 natural-scene statistics per block (AGGD fits of the normalized
luminance and of four neighbor products, at full and half resolution). The score is the distance
between the multivariate Gaussian of these features and the pristine model released with the paper.

Ported from BasicSR's ``basicsr/metrics/niqe.py`` (Apache-2.0, https://github.com/XPixelGroup/BasicSR,
``8d56e3a``), vectorized over blocks and computed in float64 throughout (BasicSR resizes in
float32). BasicSR's default ``convert_to="y"`` (Y of YCbCr) is a super-resolution convention; here,
as in MATLAB and pyiqa, the input is luma. The pristine parameters are BasicSR's
``niqe_pris_params.npz`` (from the official ``modelparameters.mat``), downloaded into
``zoo_dir() / "metrics/niqe"``.

References:
    - A. Mittal, R. Soundararajan, A. C. Bovik, "Making a 'Completely Blind' Image Quality
      Analyzer," IEEE SPL 20(3), 2013. Official MATLAB: http://live.ece.utexas.edu/research/quality/niqe_release.zip
"""

__all__ = [
    "NIQE",
    "NIQE_Weights",
    "matlab_resize_matrix",
]

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mon.core.weights import Weights, WeightsEnum, WeightsLike, resolve_weights
from mon.metrics.base import Metric, MetricMode, YColorSpace, crop_border, rgb_to_y, to_8bit_levels
from mon.metrics.fidelity import gaussian_window


class NIQE_Weights(WeightsEnum):
    """Pristine multivariate-Gaussian model (``mu_pris_param`` [1, 36], ``cov_pris_param`` [36, 36])."""

    PRISTINE = Weights(
        path   = "metrics/niqe/niqe_pristine.npz",
        name   = "NIQE pristine model (BasicSR niqe_pris_params.npz)",
        url    = "https://raw.githubusercontent.com/XPixelGroup/BasicSR/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/"
                 "basicsr/metrics/niqe_pris_params.npz",
        sha256 = "2a7c182a68c9e7f1b2e2e5ec723279d6f65d912b6fcaf37eb2bf03d7367c4296",
        meta   = {"source": "official modelparameters.mat via BasicSR", "license": "Apache-2.0 (BasicSR)"},
    )
    DEFAULT = PRISTINE


BLOCK = 96
"""Block side (the official recommended size; blocks don't overlap)."""

_SHIFTS = ((0, 1), (1, 0), (1, 1), (1, -1))
_GAMMA_GRID = 0.2 + 0.001 * torch.arange(9801, dtype=torch.float64)  # 0.2, 0.201, …, 10.0


# --- MATLAB imresize (bicubic, antialiased) ---

def _cubic(x: Tensor) -> Tensor:
    a = x.abs()
    return ((1.5 * a**3 - 2.5 * a**2 + 1) * (a <= 1)
            + (-0.5 * a**3 + 2.5 * a**2 - 4 * a + 2) * ((a > 1) & (a <= 2)))


def matlab_resize_matrix(in_len: int, scale: float | None = None, *, out_len: int | None = None,
                         antialiasing: bool = True) -> Tensor:
    """``[out_len, in_len]`` float64 matrix of MATLAB ``imresize`` (bicubic) along one axis.

    Give ``scale`` (``out_len = ceil(in_len * scale)``) or ``out_len`` (``scale = out_len / in_len``,
    as ``imresize(x, [h w])``).

    Rows hold the normalized kernel weights; taps beyond the edges are mirrored (symmetric, edge
    sample repeated) as in MATLAB. ``M_h @ img @ M_w.T`` resizes a 2-D image.

    Examples:
        >>> m = matlab_resize_matrix(8, 0.5)
        >>> m.shape, torch.allclose(m.sum(1), torch.ones(4, dtype=torch.float64))
        (torch.Size([4, 8]), True)
    """
    if (scale is None) == (out_len is None):
        raise ValueError("give exactly one of scale and out_len")
    if scale is None:
        assert out_len is not None
        scale = out_len / in_len
    else:
        out_len = math.ceil(in_len * scale)
    width   = 4.0 / scale if (scale < 1 and antialiasing) else 4.0
    x       = torch.arange(1, out_len + 1, dtype=torch.float64)
    u       = x / scale + 0.5 * (1 - 1 / scale)
    left    = torch.floor(u - width / 2)
    taps    = math.ceil(width) + 2
    idx     = left[:, None] + torch.arange(taps, dtype=torch.float64)[None, :]  # 1-based
    dist    = u[:, None] - idx
    w       = scale * _cubic(dist * scale) if (scale < 1 and antialiasing) else _cubic(dist)
    w       = w / w.sum(dim=1, keepdim=True)
    mirror  = torch.cat([torch.arange(in_len), torch.arange(in_len - 1, -1, -1)])  # 0-based symmetric
    cols    = mirror[(idx.long() - 1) % (2 * in_len)]
    m = torch.zeros(out_len, in_len, dtype=torch.float64)
    m.scatter_add_(1, cols, w)
    return m


def _imresize_half(img: Tensor) -> Tensor:
    h, w = img.shape
    return matlab_resize_matrix(h, 0.5).to(img.device) @ img @ matlab_resize_matrix(w, 0.5).to(img.device).T


# --- Features ---

def _aggd(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-row AGGD fit of ``[N, n]`` samples: ``(alpha, beta_left, beta_right)``, each ``[N]``."""
    neg, pos = x < 0, x > 0
    left  = ((x.square() * neg).sum(1) / neg.sum(1)).sqrt()
    right = ((x.square() * pos).sum(1) / pos.sum(1)).sqrt()
    g     = left / right
    rhat  = x.abs().mean(1).square() / x.square().mean(1)
    rnorm = rhat * (g**3 + 1) * (g + 1) / (g**2 + 1).square()
    grid  = _GAMMA_GRID.to(x.device)
    r_gam = torch.exp(2 * torch.lgamma(2 / grid) - torch.lgamma(1 / grid) - torch.lgamma(3 / grid))
    alpha = grid[(r_gam[None, :] - rnorm[:, None]).square().argmin(dim=1)]
    ratio = torch.exp(torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).sqrt()
    return alpha, left * ratio, right * ratio


def _block_features(blocks: Tensor) -> Tensor:
    """``[N, s, s]`` normalized-luminance blocks → ``[N, 18]`` features."""
    flat = blocks.flatten(1)
    alpha, bl, br = _aggd(flat)
    feats = [alpha, (bl + br) / 2]
    for shift in _SHIFTS:
        prod = (blocks * torch.roll(blocks, shift, dims=(1, 2))).flatten(1)
        alpha, bl, br = _aggd(prod)
        mean = (br - bl) * torch.exp(torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha))
        feats += [alpha, mean, bl, br]
    return torch.stack(feats, dim=1)


def _blocks(img: Tensor, size: int) -> Tensor:
    """Non-overlapping ``size × size`` blocks of a 2-D image (whose sides are multiples of it)."""
    h, w = img.shape
    return img.reshape(h // size, size, w // size, size).permute(0, 2, 1, 3).reshape(-1, size, size)


def _niqe_one(levels: Tensor, mu_pris: Tensor, cov_pris: Tensor, window: Tensor) -> Tensor:
    """NIQE of one 2-D float64 image in ``[0, 255]`` levels."""
    h, w   = levels.shape
    nh, nw = h // BLOCK, w // BLOCK
    if nh * nw < 2:
        raise ValueError(f"NIQE needs at least two {BLOCK}×{BLOCK} blocks, got a {h}×{w} image")
    img = levels[: nh * BLOCK, : nw * BLOCK]
    pad = window.shape[-1] // 2
    feats = []
    for scale in (1, 2):
        x     = img[None, None]
        mu    = F.conv2d(F.pad(x, [pad] * 4, mode="replicate"), window)
        sigma = (F.conv2d(F.pad(x * x, [pad] * 4, mode="replicate"), window) - mu.square()).abs().sqrt()
        norm  = ((x - mu) / (sigma + 1))[0, 0]
        feats.append(_block_features(_blocks(norm, BLOCK // scale)))
        if scale == 1:
            img = _imresize_half(img / 255) * 255
    dist  = torch.cat(feats, dim=1)                     # [blocks, 36]
    mu    = dist.nanmean(dim=0)
    clean = dist[~dist.isnan().any(dim=1)]
    cov   = torch.cov(clean.T)
    diff  = mu_pris - mu
    inv: Tensor = torch.linalg.pinv((cov_pris + cov) / 2)
    score: Tensor = (diff @ inv @ diff).sqrt()
    return score


class NIQE(Metric):
    """NIQE score per image (lower is better; natural images score about 2–6).

    Args:
        crop_border: Pixels cropped on every side first.
        color_space: Gray channel: ``"yiq"`` (luma, the default, as MATLAB and pyiqa) or ``"ycbcr"``
            (BT.601 Y, as the PIRM perceptual index; see ``mon.metrics.pi``).
        weights: Pristine model: ``"default"`` (``NIQE_Weights.PRISTINE``), a member, or a ``.npz``
            path with ``mu_pris_param`` and ``cov_pris_param``.

    Raises:
        ValueError: (on call) If an image has fewer than two 96×96 blocks (needs ``H, W >= 96`` and
            ``H * W >= 2 * 96²``).
    """

    mode         : ClassVar[MetricMode] = "nr"
    higher_better: ClassVar[bool] = False
    mu_pris      : Tensor
    cov_pris     : Tensor
    window       : Tensor

    def __init__(self, *, crop_border: int = 0, color_space: YColorSpace = "yiq",
                 weights: WeightsLike = "default") -> None:
        super().__init__()
        if color_space not in ("yiq", "ycbcr"):
            raise ValueError(f"color_space must be 'yiq' or 'ycbcr', got {color_space!r}")
        self.crop_border = crop_border
        self.color_space = color_space
        spec = resolve_weights(weights, NIQE_Weights)
        if spec is None:
            raise ValueError("NIQE needs its pristine model; weights=None is not supported")
        with np.load(spec.resolve(), allow_pickle=False) as params:
            mu, cov = params["mu_pris_param"], params["cov_pris_param"]
        self.register_buffer("mu_pris", torch.from_numpy(np.ravel(mu)).double(), persistent=False)
        self.register_buffer("cov_pris", torch.from_numpy(cov).double(), persistent=False)
        # MATLAB fspecial('gaussian', 7, 7/6), as the official code (equals BasicSR's stored window).
        self.register_buffer("window", gaussian_window(7, 7 / 6)[None, None], persistent=False)

    def extra_repr(self) -> str:
        color = "" if self.color_space == "yiq" else f", color_space={self.color_space!r}"
        return f"crop_border={self.crop_border}{color}"

    def forward(self, pred: Tensor) -> Tensor:
        if pred.ndim != 4 or pred.shape[1] not in (1, 3) or not pred.is_floating_point():
            raise ValueError(f"pred must be a float [B, 1 or 3, H, W] batch, got {tuple(pred.shape)} {pred.dtype}")
        levels = to_8bit_levels(rgb_to_y(crop_border(pred, self.crop_border).double(), self.color_space))
        mu, cov, win = self.mu_pris.to(levels.device), self.cov_pris.to(levels.device), self.window.to(levels.device)
        return torch.stack([_niqe_one(levels[i, 0], mu, cov, win) for i in range(levels.shape[0])])
