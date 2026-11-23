"""RUAS: Retinex-inspired unrolling with architecture search for low-light enhancement.

The enhancement network unrolls three illumination estimation modules (IEMs). Stage ``k`` turns a
prior ``t̂`` (``k = 0``: the 2×2 local maximum of the input ``y``; later: ``max(u) - (u - y) / 2``) into an
illumination map ``t = clamp(σ(cell(t̂)), 0.001, 1)`` and an enhanced image ``u = clamp(y / t, 0, 1)``.
A noise removal module (NRM: a stem conv, three cells, a conv to 3 channels) then predicts noise ``n``
and returns ``u - n``. The cells' layers were found by architecture search; the searched genotypes
(``IEM``, ``NRM`` below) are fixed here. 3,438 parameters.

The official test script saves the denoised output for its ``lol`` checkpoint and the last IEM output
(no denoising) for ``upe`` and ``dark``. ``forward`` returns both (``enhanced``, ``pre_denoise``); each
``RUAS_Weights`` entry names the official one in ``meta["output_key"]``, which ``mon predict`` uses
unless ``output_key`` is set. ``enhanced`` isn't clamped (the official code clips when saving).

Published numbers aren't reproducible on a standard split: the paper's LOL scores (Table 1: PSNR 18.226,
SSIM 0.717) use 100 randomly sampled LOL pairs, training on the other 689. On LOL-v1 ``eval15`` the
``lol_v1`` checkpoint gives PSNR 16.446, SSIM 0.706, LPIPS 0.270 (``mon evaluate`` on saved 8-bit
outputs, 2026-11-19). Complexity matches the paper's Table 2 (RUAS_i+n: 0.003 M parameters, 0.832 G at
600×400): ``mon benchmark`` counts 3,438 parameters and 0.784 GMACs (0.266 for the IEMs alone vs RUAS_i's
0.281); the paper's counter evidently reports MACs plus some elementwise work.

Parameter names match the official code, so the released checkpoints load with ``strict=True``. This
is a re-implementation (the official repository has no license), checked against mon_old's copy of
the official forward pass on the released weights (``tests/models/lle/test_ruas.py``).

Differences from ``mon_old`` (``archive/enhance_/lle/ruas``, never ported to its model zoo):

- Its predictor built ``RUAS()`` without loading the requested weights (only the bundled
  ``denoise.pt`` for the NRM) and always saved the denoised output.
- Training (bilevel architecture search, two SGD optimizers, a loss with hard-coded ``.cuda()``)
  isn't ported.

Examples:
    >>> import torch
    >>> out = RUAS().eval()(torch.rand(1, 3, 37, 50))
    >>> sorted(out), out["enhanced"].shape
    (['enhanced', 'illumination', 'noise', 'pre_denoise'], torch.Size([1, 3, 37, 50]))
    >>> from mon.models.registry import load
    >>> model = load("ruas", "lol_v1")  # also "upe" (default), "darkface"; needs DATA_DIR  # doctest: +SKIP

References:
    - Liu, Ma, Zhang, Zhang, Fan, Luo, "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement", CVPR 2021. https://arxiv.org/abs/2012.05609
    - https://github.com/dut-media-lab/RUAS (22b1cd6; no license)
"""

__all__ = [
    "RUAS",
    "RUAS_Weights",
]

from collections.abc import Callable
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum

_REPO = "https://raw.githubusercontent.com/dut-media-lab/RUAS/22b1cd65ccdd87bc7de8f9e4dfad06aaa3ed8539/ckpt"


# --- Weights ---

class RUAS_Weights(WeightsEnum):
    """Released RUAS checkpoints (official ``ckpt/``); ``meta["output_key"]`` is the official output."""

    LOL_V1 = Weights(
        path   = "lle/ruas/ruas_lol_v1.pt",
        name   = "RUAS LOL (official ckpt/lol.pt)",
        url    = f"{_REPO}/lol.pt",
        sha256 = "f45a881a9b836fff4a3172481f28c9eb4d91bc4df54ec064092f39ce40abcdcd",
        meta   = {"train": "lol_v1", "output_key": "enhanced"},
    )
    UPE = Weights(
        path   = "lle/ruas/ruas_upe.pt",
        name   = "RUAS UPE (official ckpt/upe.pt; MIT-Adobe FiveK split of DeepUPE)",
        url    = f"{_REPO}/upe.pt",
        sha256 = "2252d3d909e346b547232c7015f034685aac75181171c6caf90ee12d87ca36c1",
        meta   = {"train": "upe", "output_key": "pre_denoise"},
    )
    DARKFACE = Weights(
        path   = "lle/ruas/ruas_darkface.pt",
        name   = "RUAS DarkFace (official ckpt/dark.pt)",
        url    = f"{_REPO}/dark.pt",
        sha256 = "59083ec4dd4a2b7926c182a4add1519b9646a3aca6c39cfdb7d496fe885828e4",
        meta   = {"train": "darkface", "output_key": "pre_denoise"},
    )
    DEFAULT = UPE  # the official test script's default


# --- Searched cells ---

class _Conv(nn.Module):
    """``op`` = same-size conv; ``residual`` adds the input (the official ``ConvBlock``/``ResBlock``)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, residual: bool = False) -> None:
        super().__init__()
        self.op       = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2 * dilation,
                                  dilation=dilation)
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.op(x)
        return y + x if self.residual else y


_OPS: dict[str, Callable[[int], nn.Module]] = {
    "skip_connect"  : lambda c: nn.Identity(),
    "conv_1x1"      : lambda c: _Conv(c, 1),
    "conv_3x3"      : lambda c: _Conv(c, 3),
    "dilconv_3x3"   : lambda c: _Conv(c, 3, dilation=2),
    "resconv_1x1"   : lambda c: _Conv(c, 1, residual=True),
    "resconv_3x3"   : lambda c: _Conv(c, 3, residual=True),
    "resdilconv_3x3": lambda c: _Conv(c, 3, dilation=2, residual=True),
}

# Searched genotypes (official ``genotypes.py``): ops for c1_d, c1_r, c2_d, c2_r, c3_d, c3_r, c4.
IEM_GENOTYPE = ("skip_connect", "resconv_1x1", "resdilconv_3x3", "conv_3x3", "conv_3x3", "skip_connect", "conv_3x3")
NRM_GENOTYPE = ("resconv_1x1", "resconv_1x1", "resdilconv_3x3", "skip_connect", "resconv_1x1", "resconv_1x1",
                "skip_connect")


class _Cell(nn.Module):
    """Searched block (official ``SearchBlock``): three distill/refine steps, then a 1×1 fusion of 4 branches."""

    def __init__(self, channels: int, genotype: tuple[str, ...]) -> None:
        super().__init__()
        self.c1_d, self.c1_r, self.c2_d, self.c2_r, self.c3_d, self.c3_r, self.c4 = (_OPS[g](channels) for g in genotype)
        self.act = nn.LeakyReLU(0.05)
        self.c5  = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        d1 = self.act(self.c1_d(x))
        r1 = self.act(self.c1_r(x) + x)
        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1) + r1)
        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2) + r2)
        r4 = self.act(self.c4(r3))
        out: Tensor = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        return out


def _local_max(x: Tensor) -> Tensor:
    """Max over each pixel's 2×2 window toward the bottom right, zero beyond the border (official ``max_operation``)."""
    return F.max_pool2d(F.pad(x, (0, 1, 0, 1)), kernel_size=2, stride=1)


class _IEM(nn.Module):
    """Illumination estimation module: prior → ``t`` → ``u = y / t``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cell = _Cell(channels, IEM_GENOTYPE)

    def forward(self, y: Tensor, u: Tensor | None) -> tuple[Tensor, Tensor]:
        prior = _local_max(y) if u is None else _local_max(u) - 0.5 * (u - y)
        t = torch.sigmoid(self.cell(prior)).clamp(0.001, 1.0)
        return (y / t).clamp(0.0, 1.0), t


class _EnhanceNet(nn.Module):
    def __init__(self, iterations: int, channels: int = 3) -> None:
        super().__init__()
        self.iems = nn.ModuleList([_IEM(channels) for _ in range(iterations)])


class _DenoiseNet(nn.Module):
    def __init__(self, layers: int, channels: int) -> None:
        super().__init__()
        self.stem     = nn.Conv2d(3, channels, 3, padding=1)
        self.nrms     = nn.ModuleList([_Cell(channels, NRM_GENOTYPE) for _ in range(layers)])
        self.activate = nn.Sequential(nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        """The predicted noise."""
        feat = self.stem(x)
        for nrm in self.nrms:
            feat = nrm(feat)
        noise: Tensor = self.activate(feat)
        return noise


# --- Model ---

class RUAS(nn.Module):
    """RUAS (CVPR 2021) with the searched IEM and NRM cells.

    Args:
        iterations: Unrolled IEM stages (official: 3).
        denoise_layers: NRM cells (official: 3).
        denoise_channels: NRM width (official: 6).

    ``forward(image)`` returns ``enhanced`` (denoised), ``pre_denoise`` (last IEM output), and the
    last stage's ``illumination`` and the NRM's ``noise``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = RUAS_Weights

    def __init__(self, iterations: int = 3, denoise_layers: int = 3, denoise_channels: int = 6) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.enhance_net = _EnhanceNet(iterations)
        self.denoise_net = _DenoiseNet(denoise_layers, denoise_channels)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        u: Tensor | None = None
        t = image
        for iem in self.enhance_net.iems:
            u, t = iem(image, u)
        assert u is not None
        noise = self.denoise_net(u)
        return {"enhanced": u - noise, "pre_denoise": u, "illumination": t, "noise": noise}
