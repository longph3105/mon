"""Zero-DCE and Zero-DCE++: low-light enhancement by zero-reference curve estimation.

A small CNN predicts per-pixel curve parameters ``r`` in ``[-1, 1]``; the image is enhanced
by applying the quadratic curve ``x ← x + r·(x² − x)`` ``num_iters`` times, which keeps
values in ``[0, 1]``. Zero-DCE predicts one curve map per iteration (``3·num_iters``
channels); Zero-DCE++ predicts one shared map with depthwise-separable convolutions and can
run on a ``scale_factor``-times downsampled copy of the input. Training needs no reference
images (``ZeroDCELoss``).

Parameter names match the official code, so the released checkpoints load with
``strict=True``. Differences from the official code and from ``mon_old``:

- ``forward`` returns ``{"enhanced", "curve"}``. The official Zero-DCE also returns the
  image after 4 iterations, which neither its training nor its test script uses.
- Zero-DCE++ upsamples the curve to the input's exact size. The official code upsamples by
  ``scale_factor``, which fails unless height and width are multiples of it (its test script
  crops them); the results are identical when they are.
- Zero-DCE++ keeps PyTorch's default initialization, as the official training does
  (``weights_init`` is commented out there); ``mon_old`` applied ``weights_init`` to it.
- Removed from ``mon_old``: ``out_channels`` (any value but ``channels`` broke the curve
  split) and the unused ``maxpool``/``upsample`` layers (no parameters, so checkpoints are
  unaffected). ``forward_patch`` is replaced by ``mon.nn.tiling.tiled_forward``.

Examples:
    >>> import torch
    >>> model = ZeroDCE().eval()
    >>> out = model(torch.rand(1, 3, 32, 48))
    >>> out["enhanced"].shape, out["curve"].shape
    (torch.Size([1, 3, 32, 48]), torch.Size([1, 24, 32, 48]))
    >>> ZeroDCEPP(scale_factor=4)(torch.rand(1, 3, 30, 45))["curve"].shape  # any size
    torch.Size([1, 3, 30, 45])
    >>> from mon.models.registry import load
    >>> model = load("zero_dce")  # SICE_ME, needs DATA_DIR or a download  # doctest: +SKIP

References:
    - Zero-DCE: Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement", CVPR 2020. https://arxiv.org/abs/2001.06826,
      https://github.com/Li-Chongyi/Zero-DCE
    - Zero-DCE++: Li et al., "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
      Estimation", IEEE TPAMI 2022. https://arxiv.org/abs/2103.00860,
      https://github.com/Li-Chongyi/Zero-DCE_extension
"""

__all__ = [
    "ZeroDCE",
    "ZeroDCELoss",
    "ZeroDCEPP",
    "ZeroDCEPP_Weights",
    "ZeroDCE_Weights",
    "apply_curves",
]

from collections.abc import Sequence
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum
from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss
from mon.nn.losses.zero_reference import (
    ColorConstancyLoss, ExposureControlLoss, IlluminationSmoothnessLoss, SpatialConsistencyLoss,
)


# --- Weights ---

class ZeroDCE_Weights(WeightsEnum):
    """Released Zero-DCE weights."""

    SICE_ME = Weights(
        path   = "lle/zero_dce/zero_dce_sice_me.pt",
        name   = "Zero-DCE, official release (SICE multi-exposure)",
        url    = "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/43f9da935d55758d93b54a47ce5872c96b342668/"
                 "Zero-DCE_code/snapshots/Epoch99.pth",
        sha256 = "a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612",
        meta   = {"source": "official snapshots/Epoch99.pth", "train": "sice_me"},
    )
    DEFAULT = SICE_ME


class ZeroDCEPP_Weights(WeightsEnum):
    """Released Zero-DCE++ weights (the official test script runs them with ``scale_factor=12``)."""

    SICE_ME = Weights(
        path   = "lle/zero_dce++/zero_dce++_sice_me.pt",
        name   = "Zero-DCE++, official release (SICE multi-exposure)",
        url    = "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/"
                 "99ab5265054f685c874da58033ece00579514425/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth",
        sha256 = "ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84",
        meta   = {"source": "official snapshots_Zero_DCE++/Epoch99.pth", "train": "sice_me",
                  "test_scale_factor": 12},
    )
    DEFAULT = SICE_ME


# --- Curves ---

def apply_curves(image: Tensor, curves: Sequence[Tensor]) -> Tensor:
    """Apply the light-enhancement curve ``x ← x + r·(x² − x)`` once per map in ``curves``.

    Args:
        image: ``[B, C, H, W]`` in ``[0, 1]``.
        curves: Curve maps broadcastable to ``image``, values in ``[-1, 1]``; applied in order.

    Returns:
        The enhanced image, still in ``[0, 1]``.
    """
    x = image
    for r in curves:
        x = x + r * (torch.pow(x, 2) - x)
    return x


def _conv_weights_init(module: nn.Module) -> None:
    """Official Zero-DCE initialization: conv weights ~ N(0, 0.02), biases left at PyTorch's default."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)


# --- Models ---

class ZeroDCE(nn.Module):
    """Zero-DCE: a 7-layer CNN predicting one curve map per iteration.

    Args:
        channels: Image channels.
        hidden_dim: Feature channels (32 in the paper).
        num_iters: Curve iterations (8 in the paper).

    Shapes:
        ``forward(image [B, C, H, W] in [0, 1])`` → ``{"enhanced": [B, C, H, W],
        "curve": [B, C·num_iters, H, W]}``; any ``H, W``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = ZeroDCE_Weights

    def __init__(self, channels: int = 3, hidden_dim: int = 32, num_iters: int = 8) -> None:
        super().__init__()
        self.channels  = channels
        self.num_iters = num_iters
        self.e_conv1 = nn.Conv2d(channels,       hidden_dim, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(hidden_dim,     hidden_dim, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(hidden_dim,     hidden_dim, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(hidden_dim,     hidden_dim, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(hidden_dim * 2, channels * num_iters, 3, 1, 1)
        self.apply(_conv_weights_init)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        x1 = F.relu(self.e_conv1(image))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        enhanced = apply_curves(image, torch.split(curve, self.channels, dim=1))
        return {"enhanced": enhanced, "curve": curve}


class _DSConv(nn.Module):
    """Depthwise 3×3 then pointwise 1×1 convolution (``CSDN_Tem`` in the official code)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.point_conv(self.depth_conv(x))
        return out


class ZeroDCEPP(nn.Module):
    """Zero-DCE++: depthwise-separable layers, one curve map shared by all iterations.

    Args:
        channels: Image channels.
        hidden_dim: Feature channels (32 in the paper).
        num_iters: Curve iterations (8 in the paper).
        scale_factor: Estimate the curve on the input downsampled by this factor (bilinear),
            then upsample it to the input size. ``1`` (training default) uses full resolution;
            the official test script uses ``12``.

    Shapes:
        ``forward(image [B, C, H, W] in [0, 1])`` → ``{"enhanced": [B, C, H, W],
        "curve": [B, C, H, W]}``; any ``H, W`` with ``H, W >= scale_factor``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = ZeroDCEPP_Weights

    def __init__(self, channels: int = 3, hidden_dim: int = 32, num_iters: int = 8, scale_factor: float = 1.0) -> None:
        super().__init__()
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
        self.channels     = channels
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        self.e_conv1 = _DSConv(channels,       hidden_dim)
        self.e_conv2 = _DSConv(hidden_dim,     hidden_dim)
        self.e_conv3 = _DSConv(hidden_dim,     hidden_dim)
        self.e_conv4 = _DSConv(hidden_dim,     hidden_dim)
        self.e_conv5 = _DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv6 = _DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv7 = _DSConv(hidden_dim * 2, channels)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if self.scale_factor == 1:
            x = image
        else:
            h, w = image.shape[-2:]
            if min(h, w) < self.scale_factor:
                raise ValueError(f"input {h}×{w} is smaller than scale_factor={self.scale_factor}")
            x = F.interpolate(image, scale_factor=1 / self.scale_factor, mode="bilinear")
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            curve = F.interpolate(curve, size=image.shape[-2:], mode="bilinear", align_corners=True)
        enhanced = apply_curves(image, [curve] * self.num_iters)
        return {"enhanced": enhanced, "curve": curve}


# --- Loss ---

class ZeroDCELoss(nn.Module):
    """The Zero-DCE training objective: weighted sum of the four zero-reference losses.

    ``loss = spatial + exposure_weight·exposure + color_weight·color + tv_weight·smoothness``,
    with the official defaults for Zero-DCE. For Zero-DCE++ use ``tv_weight=1600`` (its
    official training value). With ``reduction="mean"`` the value equals the official
    training loss.

    Args:
        spatial_weight: Weight of ``SpatialConsistencyLoss`` (pred vs input).
        exposure_weight: Weight of ``ExposureControlLoss``.
        color_weight: Weight of ``ColorConstancyLoss``.
        tv_weight: Weight of ``IlluminationSmoothnessLoss`` on the curve maps.
        patch_size: Exposure patch size.
        exposure: Target exposure level ``E``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"`` (per-sample ``[B]``).
    """

    def __init__(
        self,
        spatial_weight : float = 1.0,
        exposure_weight: float = 10.0,
        color_weight   : float = 5.0,
        tv_weight      : float = 200.0,
        patch_size     : int   = 16,
        exposure       : float = 0.6,
        reduction      : Reduction = "mean",
    ) -> None:
        super().__init__()
        check_reduction(reduction)
        self.weights    = {"spatial": spatial_weight, "exposure": exposure_weight, "color": color_weight,
                           "smoothness": tv_weight}
        self.reduction  = reduction
        self.spatial    = SpatialConsistencyLoss(reduction="none")
        self.exposure   = ExposureControlLoss(patch_size, exposure, reduction="none")
        self.color      = ColorConstancyLoss(reduction="none")
        self.smoothness = IlluminationSmoothnessLoss(reduction="none")

    def terms(self, enhanced: Tensor, curve: Tensor, image: Tensor) -> dict[str, Tensor]:
        """Weighted per-term values (reduced by ``reduction``), e.g. for logging; they sum to ``forward``."""
        raw = {
            "spatial"   : self.spatial(enhanced, image),
            "exposure"  : self.exposure(enhanced),
            "color"     : self.color(enhanced),
            "smoothness": self.smoothness(curve),
        }
        return {k: reduce_loss(self.weights[k] * v, self.reduction) for k, v in raw.items()}

    def forward(self, enhanced: Tensor, curve: Tensor, image: Tensor) -> Tensor:
        """Args: model outputs ``enhanced`` and ``curve``, and the low-light ``image`` input."""
        total: Tensor = sum(self.terms(enhanced, curve, image).values())  # type: ignore[assignment]
        return total
