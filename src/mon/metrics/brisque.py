"""BRISQUE: blind/referenceless image spatial quality evaluator (no-reference; lower is better).

Each image is converted to luma rounded to 8-bit levels (pyiqa's ``brisque`` default), normalized
by a local Gaussian mean and deviation (MSCN coefficients, 7×7 window, σ 7/6, zero padding), and
described by 18 statistics per scale at full and half resolution (MATLAB bicubic ``imresize``): a
GGD fit of the MSCN coefficients and AGGD fits of their products with four neighbors. The 36
features are scaled to ``[-1, 1]`` with the official feature ranges and scored by the RBF support
vector regressor (``γ = 0.05``, ``ρ = -153.591``) released with the paper (LIVE IQA).

Ported from piq's ``piq/brisque.py`` (Apache-2.0, https://github.com/photosynthesis-team/piq), the
implementation behind pyiqa's ``brisque``: neighbor products use ``torch.roll`` (wrapping at the
borders), as there. The SVR (support vectors and coefficients) is piq's ``brisque_svm_weights.pt``
(v0.4.0 release; byte-identical to pyiqa's copy), downloaded into ``zoo_dir() / "metrics/brisque"``.
Computed in float64 (piq and pyiqa run in the input's float32).

References:
    - A. Mittal, A. K. Moorthy, A. C. Bovik, "No-Reference Image Quality Assessment in the Spatial
      Domain," IEEE TIP 21(12), 2012. Official MATLAB: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
"""

__all__ = [
    "BRISQUE",
    "BRISQUE_Weights",
    "brisque_features",
]

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from mon.core.weights import Weights, WeightsEnum, WeightsLike, resolve_weights
from mon.metrics.base import Metric, MetricMode, crop_border, rgb_to_luma, to_8bit_levels
from mon.metrics.fidelity import gaussian_window
from mon.metrics.niqe import matlab_resize_matrix


class BRISQUE_Weights(WeightsEnum):
    """RBF support vector regressor trained on LIVE IQA (``sv_coef`` [774], ``sv`` [774, 36])."""

    LIVE = Weights(
        path   = "metrics/brisque/brisque_svm_live.pt",
        name   = "BRISQUE SVR, LIVE IQA (piq brisque_svm_weights.pt)",
        url    = "https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt",
        sha256 = "e5b048ef5325bae6c57965cf077fadfa5834fcefb02e13611b4ea2d36cf3c52c",
        meta   = {"source": "official MATLAB allmodel via piq", "license": "Apache-2.0 (piq)",
                  "gamma": 0.05, "rho": -153.591},
    )
    DEFAULT = LIVE


# Official MATLAB feature ranges (``allrange``): row k scales feature k to [-1, 1].
_RANGES = torch.tensor([
    [0.338, 10], [0.017204, 0.806612],
    [0.236, 1.642], [-0.123884, 0.20293], [0.000155, 0.712298], [0.001122, 0.470257],
    [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456], [0.000975, 0.470984],
    [0.249, 1.555], [-0.135687, 0.100858], [0.000174, 0.684173], [0.000913, 0.534174],
    [0.258, 1.561], [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508],
    [0.471, 3.264], [0.012809, 0.703171],
    [0.218, 1.046], [-0.094876, 0.187459], [1.5e-05, 0.442057], [0.001272, 0.40803],
    [0.222, 1.042], [-0.115772, 0.162604], [1.6e-05, 0.444362], [0.001374, 0.40243],
    [0.227, 0.996], [-0.117188, 0.098323], [3e-05, 0.531903], [0.001122, 0.369589],
    [0.228, 0.99], [-0.12243, 0.098658], [2.8e-05, 0.530092], [0.001118, 0.370399],
], dtype=torch.float64)

_SHIFTS = ((0, 1), (1, 0), (1, 1), (-1, 1))
_GRID   = 0.2 + 0.001 * torch.arange(9801, dtype=torch.float64)  # 0.2, 0.201, …, 10.0


# --- Features ---

def _ggd(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row GGD fit of ``[N, n]`` samples: ``(alpha, sigma²)``."""
    grid  = _GRID.to(x.device)
    r_tab = torch.exp(torch.lgamma(1 / grid) + torch.lgamma(3 / grid) - 2 * torch.lgamma(2 / grid))
    var   = x.square().mean(1)
    rho   = var / x.abs().mean(1).square()
    return grid[(rho[:, None] - r_tab[None, :]).abs().argmin(dim=1)], var


def _aggd(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-row AGGD fit of ``[N, n]`` samples: ``(alpha, sigma_left, sigma_right)``."""
    neg, pos = x < 0, x > 0
    left  = ((x.square() * neg).sum(1) / neg.sum(1)).sqrt()
    right = ((x.square() * pos).sum(1) / pos.sum(1)).sqrt()
    g     = left / right
    rhat  = x.abs().mean(1).square() / x.square().mean(1)
    rnorm = rhat * (g**3 + 1) * (g + 1) / (g**2 + 1).square()
    grid  = _GRID.to(x.device)
    r_gam = torch.exp(2 * torch.lgamma(2 / grid) - torch.lgamma(1 / grid) - torch.lgamma(3 / grid))
    return grid[(r_gam[None, :] - rnorm[:, None]).abs().argmin(dim=1)], left, right


def _mscn(x: Tensor, window: Tensor) -> Tensor:
    """Mean-subtracted, contrast-normalized coefficients of ``[B, 1, H, W]`` (zero padding)."""
    pad   = window.shape[-1] // 2
    mu    = F.conv2d(x, window, padding=pad)
    sigma = (F.conv2d(x * x, window, padding=pad) - mu.square()).abs().sqrt()
    return (x - mu) / (sigma + 1)


def _scale_features(mscn: Tensor) -> Tensor:
    """``[B, 1, H, W]`` MSCN → ``[B, 18]``."""
    flat = mscn.flatten(1)
    alpha, var = _ggd(flat)
    feats = [alpha, var]
    for shift in _SHIFTS:
        prod = (mscn * torch.roll(mscn, shift, dims=(-2, -1))).flatten(1)
        alpha, sl, sr = _aggd(prod)
        eta = (sr - sl) * torch.exp(torch.lgamma(2 / alpha) - (torch.lgamma(1 / alpha) + torch.lgamma(3 / alpha)) / 2)
        feats += [alpha, eta, sl.square(), sr.square()]
    return torch.stack(feats, dim=1)


def brisque_features(levels: Tensor) -> Tensor:
    """The 36 unscaled BRISQUE features of ``[B, 1, H, W]`` float64 luma in ``[0, 255]`` → ``[B, 36]``."""
    window = gaussian_window(7, 7 / 6).to(levels.device)[None, None]
    x = levels
    feats = []
    for scale in range(2):
        feats.append(_scale_features(_mscn(x, window)))
        if scale == 0:
            h, w = x.shape[-2:]
            m_h = matlab_resize_matrix(h, 0.5).to(x.device)
            m_w = matlab_resize_matrix(w, 0.5).to(x.device)
            x = m_h @ x @ m_w.T
    return torch.cat(feats, dim=1)


# --- Metric ---

class BRISQUE(Metric):
    """BRISQUE score per image (lower is better; roughly 0–100 for natural images).

    Args:
        crop_border: Pixels cropped on every side first.
        weights: SVR: ``"default"`` (``BRISQUE_Weights.LIVE``), a member, or a ``.pt`` path holding
            ``(sv_coef [N], sv [N, 36])``.
    """

    mode         : ClassVar[MetricMode] = "nr"
    higher_better: ClassVar[bool] = False
    sv           : Tensor
    sv_coef      : Tensor

    def __init__(self, *, crop_border: int = 0, weights: WeightsLike = "default") -> None:
        super().__init__()
        self.crop_border = crop_border
        spec = resolve_weights(weights, BRISQUE_Weights)
        if spec is None:
            raise ValueError("BRISQUE needs its SVR; weights=None is not supported")
        sv_coef, sv = torch.load(spec.resolve(), map_location="cpu", weights_only=True)
        self.register_buffer("sv_coef", sv_coef.double(), persistent=False)
        self.register_buffer("sv", sv.double(), persistent=False)
        self.gamma = float(spec.meta.get("gamma", 0.05))
        self.rho   = float(spec.meta.get("rho", -153.591))

    def extra_repr(self) -> str:
        return f"crop_border={self.crop_border}"

    def forward(self, pred: Tensor) -> Tensor:
        if pred.ndim != 4 or pred.shape[1] not in (1, 3) or not pred.is_floating_point():
            raise ValueError(f"pred must be a float [B, 1 or 3, H, W] batch, got {tuple(pred.shape)} {pred.dtype}")
        levels = to_8bit_levels(rgb_to_luma(crop_border(pred, self.crop_border).double()))
        feats  = brisque_features(levels)
        ranges = _RANGES.to(feats.device)
        scaled = -1 + 2 * (feats - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
        sv, coef = self.sv.to(feats.device), self.sv_coef.to(feats.device)
        kernel = torch.exp(-self.gamma * (scaled[:, None, :] - sv[None]).square().sum(-1))
        score: Tensor = kernel @ coef - self.rho
        return score
