"""RetinexNet: Retinex decomposition, then illumination adjustment (the PyTorch reimplementation's variant).

``DecomNet`` splits an image into reflectance ``R`` (3 channels) and illumination ``L`` (1
channel), both sigmoid outputs: a 9×9 conv on ``[max_c(x), x]``, five 3×3 ReLU convs, a 3×3
reconstruction conv. ``RelightNet`` (here ``enhance_net``) predicts a new illumination ``L̂`` from
``[R, L]`` with a three-stage stride-2 encoder, a nearest-upsampling decoder with skips, and a
multi-scale fusion; the output is ``R · L̂`` (not clamped). All convs use replicate padding.

The official code (TensorFlow, weichen582/RetinexNet) and its weights don't fit this network: its
decoder *adds* each skip after a 64-channel conv, while the PyTorch reimplementation
(aasharma90/RetinexNet_PyTorch, which ``mon_old`` copied and every listed checkpoint was trained
with) *concatenates* the skip before a 128-channel conv; the reimplementation also pads the 1×1
fusion conv by one pixel and leaves the 3×3 output conv unpadded, and replicate-pads where
TensorFlow zero-pads (``same``). This module follows the reimplementation; parameter names follow
``mon_old`` (``decom_net.*``, ``enhance_net.*``).

``forward`` accepts any ``H, W >= 2`` (decoder upsampling goes to the encoder's sizes). The
training losses (decomposition and relighting, with illumination smoothness) aren't ported yet.

Examples:
    >>> import torch
    >>> out = RetinexNet().eval()(torch.rand(1, 3, 21, 30))
    >>> sorted(out), out["illumination_adjusted"].shape
    (['enhanced', 'illumination', 'illumination_adjusted', 'reflectance'], torch.Size([1, 1, 21, 30]))
    >>> from mon.models.registry import load
    >>> model = load("retinexnet")  # the reimplementation's release; needs DATA_DIR  # doctest: +SKIP

References:
    - Wei, Wang, Yang, Liu, "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018.
      https://arxiv.org/abs/1808.04560
    - https://github.com/weichen582/RetinexNet (official, TensorFlow, MIT; architecture read).
    - https://github.com/aasharma90/RetinexNet_PyTorch (commit ``2267510``; no license stated: read, not
      copied). Its released checkpoints are ``ckpts/Decom/9200.tar`` and ``ckpts/Relight/9200.tar``.
"""

__all__ = [
    "RetinexNet",
    "RetinexNet_Weights",
]

from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum

_REIMPL = "https://raw.githubusercontent.com/aasharma90/RetinexNet_PyTorch/22675105f52432715a7935db16562f5117c9d369/"


class RetinexNet_Weights(WeightsEnum):
    """RetinexNet checkpoints; none is an official release (the official weights are TensorFlow, another decoder)."""

    LOL_V1_PSNR = Weights(
        path   = "lle/retinexnet/retinexnet_lol_v1_psnr.pt",
        name   = "RetinexNet, owner's mon_old training on LOL-v1 (relight net with the best PSNR)",
        sha256 = "85345bc6338fd375c11d06e89c46fe865529792f897b2123543e04bb18bf1957",
        meta   = {"source": "mon_old training, 2026-03-20; same tensors as lle/retinexnet/retinexnet_lol_v1.pt",
                  "train": "lol_v1", "selection": "best val PSNR"},
    )
    LOL_V1_SSIM = Weights(
        path   = "lle/retinexnet/retinexnet_lol_v1_ssim.pt",
        name   = "RetinexNet, owner's mon_old training on LOL-v1 (relight net with the best SSIM)",
        sha256 = "aa2c3230e7df3dd1d27514c900b2676a0fbabf2b00b95c06c52279c3fd7bdd45",
        meta   = {"source": "mon_old training, 2026-03-20; same tensors as lle/retinexnet/retinexnet_lol_v1_ssimc.pt; "
                            "same decomposition net as LOL_V1_PSNR",
                  "train": "lol_v1", "selection": "best val SSIM"},
    )
    REIMPL_9200 = Weights(
        path   = "lle/retinexnet/retinexnet_reimpl_9200.pt",
        name   = "RetinexNet, PyTorch reimplementation release (epoch 9200), merged into one state dict",
        sha256 = "59b1545d4bc3da0fd81796756acef5f12278e9a5a4f351e09d64de6df5ca83e7",
        meta   = {"source": f"{_REIMPL}ckpts/Decom/9200.tar + ckpts/Relight/9200.tar (keys prefixed decom_net. / "
                            "enhance_net.)",
                  "source_sha256": {
                      "Decom/9200.tar"  : "40c090e3bb884e697f6b396fe7fddade67172b30b428daa316c68a0b9247d4b5",
                      "Relight/9200.tar": "513ae3a0bf503b9a6ecce1a46112135aefcfeb393cf61531e18cc3356701efde",
                  },
                  "train": "the LOL training set (per the reimplementation's README)"},
    )
    DEFAULT = REIMPL_9200


def _conv(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int | None = None) -> nn.Conv2d:
    pad = kernel_size // 2 if padding is None else padding
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, padding_mode="replicate")


class _DecomNet(nn.Module):
    """Image → reflectance ``[B, 3, H, W]`` and illumination ``[B, 1, H, W]``."""

    def __init__(self, channels: int = 64, layers: int = 5) -> None:
        super().__init__()
        self.net1_conv0 = _conv(4, channels, 9)
        self.net1_convs = nn.Sequential(*[m for _ in range(layers) for m in (_conv(channels, channels, 3), nn.ReLU())])
        self.net1_recon = _conv(channels, 4, 3)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([image.max(1, keepdim=True)[0], image], 1)
        y = self.net1_recon(self.net1_convs(self.net1_conv0(x)))
        return torch.sigmoid(y[:, :3]), torch.sigmoid(y[:, 3:4])


class _RelightNet(nn.Module):
    """``[R, L]`` → adjusted illumination ``[B, 1, H, W]`` (no output activation)."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        c = channels
        self.net2_conv0_1   = _conv(4, c, 3)
        self.net2_conv1_1   = _conv(c, c, 3, stride=2)
        self.net2_conv1_2   = _conv(c, c, 3, stride=2)
        self.net2_conv1_3   = _conv(c, c, 3, stride=2)
        self.net2_deconv1_1 = _conv(c * 2, c, 3)
        self.net2_deconv1_2 = _conv(c * 2, c, 3)
        self.net2_deconv1_3 = _conv(c * 2, c, 3)
        self.net2_fusion    = _conv(c * 3, c, 1, padding=1)                 # as the reimplementation: grows by 2 px
        self.net2_output    = nn.Conv2d(c, 1, 3, padding=0)                 # ... and shrinks back

    def forward(self, reflectance: Tensor, illumination: Tensor) -> Tensor:
        x0 = self.net2_conv0_1(torch.cat([reflectance, illumination], 1))
        x1 = F.relu(self.net2_conv1_1(x0))
        x2 = F.relu(self.net2_conv1_2(x1))
        x3 = F.relu(self.net2_conv1_3(x2))
        d1 = F.relu(self.net2_deconv1_1(torch.cat([F.interpolate(x3, size=x2.shape[-2:]), x2], 1)))
        d2 = F.relu(self.net2_deconv1_2(torch.cat([F.interpolate(d1, size=x1.shape[-2:]), x1], 1)))
        d3 = F.relu(self.net2_deconv1_3(torch.cat([F.interpolate(d2, size=x0.shape[-2:]), x0], 1)))
        size  = reflectance.shape[-2:]
        fused = self.net2_fusion(torch.cat([F.interpolate(d1, size=size), F.interpolate(d2, size=size), d3], 1))
        out: Tensor = self.net2_output(fused)
        return out


class RetinexNet(nn.Module):
    """RetinexNet: ``decom_net`` (reflectance, illumination), then ``enhance_net`` (adjusted illumination).

    Args:
        channels: Feature channels of both networks (64 in the paper).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W] (R·L̂, not clamped),
        "reflectance": [B, 3, H, W], "illumination": [B, 1, H, W], "illumination_adjusted": [B, 1, H, W]}``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = RetinexNet_Weights

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.decom_net   = _DecomNet(channels)
        self.enhance_net = _RelightNet(channels)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        reflectance, illumination = self.decom_net(image)
        adjusted = self.enhance_net(reflectance, illumination)
        return {"enhanced": reflectance * adjusted, "reflectance": reflectance, "illumination": illumination,
                "illumination_adjusted": adjusted}
