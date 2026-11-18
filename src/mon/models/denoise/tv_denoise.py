"""Total-variation denoising (ROF model) with Chambolle's projection algorithm.

Solves ``min_u TV(u) + ||u - f||² / (2·weight)`` per channel: larger ``weight`` removes more
noise and more texture. Same iteration as ``skimage.restoration.denoise_tv_chambolle`` (with
``channel_axis`` set and ``eps=0``): ``iterations`` here equals its ``max_num_iter``. A fixed
iteration count (no early stopping) keeps batches in lockstep; the default 50 is about as close
to convergence as skimage's default early stopping. No parameters; differentiable; any device.

Differences from ``mon_old`` (``ops/image/denoise.py``):

- It ran one iteration more than skimage for the same count (50 matched skimage's 51) and
  added ``1e-8`` inside the square root; both removed, so results differ slightly.
- It ran under ``torch.no_grad()``; now differentiable (use ``torch.inference_mode`` when
  gradients aren't needed).
- Removed the NumPy path: it returned all zeros for uint8 input (skimage's ``[0, 1]`` result
  cast back to uint8) and guessed channel order from the shape.

Examples:
    >>> import torch
    >>> clean = torch.linspace(0, 1, 64).expand(1, 3, 48, 64)
    >>> noisy = (clean + 0.1 * torch.randn(1, 3, 48, 64, generator=torch.Generator().manual_seed(0))).clamp(0, 1)
    >>> denoised = TVDenoise(weight=0.1)(noisy)
    >>> bool((denoised - clean).pow(2).mean() < (noisy - clean).pow(2).mean())
    True

References:
    - Chambolle, "An Algorithm for Total Variation Minimization and Applications", JMIV 2004.
      https://doi.org/10.1023/B:JMIV.0000011325.36760.1e
    - Rudin, Osher, Fatemi, "Nonlinear Total Variation Based Noise Removal Algorithms",
      Physica D 1992.
    - scikit-image ``denoise_tv_chambolle`` (the tests compare with it).
"""

__all__ = [
    "TVDenoise",
    "tv_denoise",
]

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_TAU = 0.25  # step size 1 / (2·ndim) for 2-D images, as skimage


def _forward_diff(x: Tensor, dim: int) -> Tensor:
    """``x[i+1] - x[i]`` along ``dim`` (-2 or -1), 0 at the last index."""
    d = torch.diff(x, dim=dim)
    return F.pad(d, (0, 1) if dim == -1 else (0, 0, 0, 1))


def _neg_divergence(px: Tensor, py: Tensor) -> Tensor:
    """``-div p`` with the boundary convention matching ``_forward_diff`` (its negative adjoint)."""
    shifted_x = F.pad(px[..., :-1, :], (0, 0, 1, 0))
    shifted_y = F.pad(py[..., :, :-1], (1, 0))
    return shifted_x - px + shifted_y - py


def tv_denoise(image: Tensor, weight: float = 0.1, iterations: int = 50) -> Tensor:
    """Total-variation denoise each channel of ``image``.

    Args:
        image: ``[..., H, W]`` float, typically ``[B, C, H, W]`` in ``[0, 1]`` (``weight`` is
            relative to that range).
        weight: Denoising strength (> 0).
        iterations: Chambolle iterations (>= 1; 1 returns the input).

    Returns:
        Denoised image, same shape and dtype.

    Raises:
        ValueError: If ``weight <= 0``, ``iterations < 1``, or ``image`` has fewer than 2 dims.
        TypeError: If ``image`` isn't floating point.
    """
    if image.ndim < 2:
        raise ValueError(f"image must be [..., H, W], got shape {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError(f"image must be float, got {image.dtype}")
    if weight <= 0:
        raise ValueError(f"weight must be > 0, got {weight}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    px = torch.zeros_like(image)
    py = torch.zeros_like(image)
    out = image
    for i in range(iterations):
        if i > 0:
            out = image + _neg_divergence(px, py)
        if i == iterations - 1:
            break  # the last dual update would not change the returned image
        gx, gy = _forward_diff(out, -2), _forward_diff(out, -1)
        norm = 1 + torch.sqrt(gx**2 + gy**2) * (_TAU / weight)
        px = (px - _TAU * gx) / norm
        py = (py - _TAU * gy) / norm
    return out


class TVDenoise(nn.Module):
    """Total-variation denoising as a parameter-free module (see ``tv_denoise``).

    Args:
        weight: Denoising strength (> 0).
        iterations: Chambolle iterations (>= 1).

    Shapes:
        ``forward(image [B, C, H, W] in [0, 1])`` → ``[B, C, H, W]``.
    """

    def __init__(self, weight: float = 0.1, iterations: int = 50) -> None:
        super().__init__()
        self.weight     = weight
        self.iterations = iterations

    def extra_repr(self) -> str:
        return f"weight={self.weight}, iterations={self.iterations}"

    def forward(self, image: Tensor) -> Tensor:
        return tv_denoise(image, self.weight, self.iterations)
