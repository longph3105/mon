"""PairLIE: low-light enhancement learned from pairs of low-light images of the same scene.

A denoising branch ``N`` maps the input to a cleaner image ``X``; two branches decompose ``X``
into a one-channel illumination ``L`` and a reflectance ``R`` (Retinex: ``X ≈ L·R``). The
enhanced image is ``L^α · R``: ``α < 1`` brightens the illumination (``0.2`` by default; the
official code suggests ``0.14`` for LOL). Training needs only two differently exposed low-light
images per scene: their reflectances must agree (``PairLIELoss``). Each branch is five
reflection-padded 3×3 convolutions with 64 channels.

Parameter names match the official code, so the released checkpoint loads with ``strict=True``.

Differences from ``mon_old`` (``models/enhance/lle/pairlie``), whose inference matched the
official code:

- Its total-variation term added the vertical and horizontal gradient maps elementwise (shapes
  ``[H-2, W]`` and ``[H, W-2]``: no broadcast, so training crashed); the official code adds their
  means.
- Weights loading in ``__init__``, ``forward_patch`` (use ``mon.nn.tiling.tiled_forward``),
  ``benchmark``, and the image-pasting helpers were removed.

Examples:
    >>> import torch
    >>> out = PairLIE().eval()(torch.rand(1, 3, 21, 30))
    >>> sorted(out), out["illumination"].shape
    (['denoised', 'enhanced', 'illumination', 'reflectance'], torch.Size([1, 1, 21, 30]))
    >>> from mon.models.registry import load
    >>> model = load("pairlie")  # official release; needs DATA_DIR or a download  # doctest: +SKIP

References:
    - Fu, Yang, Tu, Huang, Ding, Ma, "Learning a Simple Low-Light Image Enhancer From Paired
      Low-Light Instances", CVPR 2023.
      https://openaccess.thecvf.com/content/CVPR2023/html/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.html
    - https://github.com/zhenqifu/PairLIE (no license stated; read, not copied)
"""

__all__ = [
    "PairLIE",
    "PairLIELoss",
    "PairLIE_Weights",
]

from typing import ClassVar

import torch
from torch import Tensor, nn

from mon.core.weights import Weights, WeightsEnum
from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss


class PairLIE_Weights(WeightsEnum):
    """Released PairLIE weights."""

    SICE = Weights(
        path   = "lle/pairlie/pairlie_sice.pt",
        name   = "PairLIE, official release (weights/PairLIE.pth)",
        url    = "https://raw.githubusercontent.com/zhenqifu/PairLIE/3fd5c1586aa5f8f0736382de36d5bf46a4e208c2/"
                 "weights/PairLIE.pth",
        sha256 = "e223ec9e66504550e69f65b631ce520be9b5c4e8e0aba882768d06af355b0631",
        meta   = {"source": "official weights/PairLIE.pth",
                  "train": "the official PairLIE training set ('sice' in the file name is from the old zoo)"},
    )
    DEFAULT = SICE


class _Branch(nn.Module):
    """Five reflection-padded 3×3 convolutions with ReLU between them, then a sigmoid.

    The ``Sequential`` is stored under ``attr`` (``"L_net"``, ``"R_net"``, ``"N_net"``), as in the
    official code, so checkpoint keys read ``L_net.L_net.1.weight``.
    """

    def __init__(self, attr: str, out_channels: int, channels: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for k in range(5):
            cin, cout = (3 if k == 0 else channels), (out_channels if k == 4 else channels)
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(cin, cout, 3, 1, 0)]
            if k < 4:
                layers.append(nn.ReLU())
        self.attr = attr
        self.add_module(attr, nn.Sequential(*layers))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(getattr(self, self.attr)(x))


class PairLIE(nn.Module):
    """PairLIE network: denoise (``N_net``), then decompose into illumination and reflectance.

    Args:
        alpha: Illumination exponent of the enhanced image ``L^α·R`` (official: 0.2; 0.14 for LOL).
        channels: Feature channels per branch (64 in the paper).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W], "illumination":
        [B, 1, H, W], "reflectance": [B, 3, H, W], "denoised": [B, 3, H, W]}``; any ``H, W >= 2``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = PairLIE_Weights

    def __init__(self, alpha: float = 0.2, channels: int = 64) -> None:
        super().__init__()
        self.alpha = alpha
        self.L_net = _Branch("L_net", 1, channels)
        self.R_net = _Branch("R_net", 3, channels)
        self.N_net = _Branch("N_net", 3, channels)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        denoised     = self.N_net(image)
        illumination = self.L_net(denoised)
        reflectance  = self.R_net(denoised)
        enhanced     = torch.pow(illumination, self.alpha) * reflectance
        return {"enhanced": enhanced, "illumination": illumination, "reflectance": reflectance, "denoised": denoised}


class PairLIELoss(nn.Module):
    """The PairLIE training objective for two low-light images of one scene.

    ``MSE(R₁, R₂) + [MSE(L₁R₁, X₁) + MSE(R₁, X₁ / sg(L₁)) + MSE(L₁, max_c(I₁)) + TV(L₁)] +
    500·MSE(I₁, X₁)``: reflectance consistency, Retinex decomposition (``sg`` = stop-gradient;
    ``TV`` = mean absolute difference between pixels two apart, vertically plus horizontally),
    and projection. With ``reduction="mean"`` it equals the official loss.

    Args:
        projection_weight: Weight of ``MSE(I₁, X₁)`` (official: 500).
        reduction: ``"mean"``, ``"sum"``, or ``"none"`` (per-sample ``[B]``).
    """

    def __init__(self, projection_weight: float = 500.0, reduction: Reduction = "mean") -> None:
        super().__init__()
        check_reduction(reduction)
        self.projection_weight = projection_weight
        self.reduction         = reduction

    def forward(self, image1: Tensor, out1: dict[str, Tensor], out2: dict[str, Tensor]) -> Tensor:
        """Args: the first image and the model outputs for both images (``PairLIE.forward``)."""
        def mse(a: Tensor, b: Tensor) -> Tensor:
            return (a - b).pow(2).flatten(1).mean(1)

        illu, refl, den = out1["illumination"], out1["reflectance"], out1["denoised"]
        consistency = mse(refl, out2["reflectance"])
        tv = ((illu[..., 2:, :] - illu[..., :-2, :]).abs().flatten(1).mean(1)
              + (illu[..., :, 2:] - illu[..., :, :-2]).abs().flatten(1).mean(1))
        retinex = (mse(illu * refl, den) + mse(refl, den / illu.detach())
                   + mse(illu, image1.amax(dim=1, keepdim=True)) + tv)
        projection = mse(image1, den)
        return reduce_loss(consistency + retinex + self.projection_weight * projection, self.reduction)
