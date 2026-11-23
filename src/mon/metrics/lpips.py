"""LPIPS: learned perceptual image patch similarity (lower is better).

Deep features of both images (AlexNet or VGG-16, torchvision ImageNet weights) are unit-normalized
per position, their squared differences weighted per channel by learned 1×1 layers (v0.1, trained
on BAPPS), averaged over space, and summed over five layers. Same computation as the official
``lpips`` package (``LPIPS(net, version="0.1")(pred, target, normalize=True)``) and pyiqa's ``lpips``
/ ``lpips-vgg``: inputs in ``[0, 1]`` are mapped to ``[-1, 1]`` and then by the official scaling layer.

The linear-layer weights download from the official repo (pinned commit, SHA-256 checked) into
``zoo_dir() / "metrics/lpips"``; the backbones come through torchvision (``TORCH_HOME``).

References:
    - R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The Unreasonable Effectiveness of Deep
      Features as a Perceptual Metric," CVPR 2018.
    - Code and weights: https://github.com/richzhang/PerceptualSimilarity (BSD-2-Clause).
"""

__all__ = [
    "LPIPS",
    "LPIPS_Weights",
]

from itertools import pairwise
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn
from torchvision import models

from mon.core.weights import Weights, WeightsEnum, WeightsLike, resolve_weights
from mon.metrics.base import Metric, MetricMode, check_pair

_REPO = "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/082bb24f84c091ea94de2867d34c4544f68e0963/"


class LPIPS_Weights(WeightsEnum):
    """Official LPIPS v0.1 linear layers (``lpips/weights/v0.1/<net>.pth``)."""

    ALEX = Weights(
        path   = "metrics/lpips/lpips_alex_bapps.pth",
        name   = "LPIPS v0.1 linear layers, AlexNet",
        url    = _REPO + "lpips/weights/v0.1/alex.pth",
        sha256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
        meta   = {"net": "alex", "license": "BSD-2-Clause"},
    )
    VGG = Weights(
        path   = "metrics/lpips/lpips_vgg_bapps.pth",
        name   = "LPIPS v0.1 linear layers, VGG-16",
        url    = _REPO + "lpips/weights/v0.1/vgg.pth",
        sha256 = "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
        meta   = {"net": "vgg", "license": "BSD-2-Clause"},
    )
    DEFAULT = ALEX


# Backbone layer index ranges ending at each tapped ReLU, and their channel counts.
_SLICES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "alex": ((0, 2, 5, 8, 10, 12), (64, 192, 384, 256, 256)),
    "vgg" : ((0, 4, 9, 16, 23, 30), (64, 128, 256, 512, 512)),
}


def _backbone(net: str, pretrained: bool) -> nn.Sequential:
    if net == "alex":
        model: nn.Module = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    features = model.features
    assert isinstance(features, nn.Sequential)
    return features


class LPIPS(Metric):
    """LPIPS distance per image (``[B]``, float64; 0 for identical images).

    Args:
        net: Backbone, ``"alex"`` (default, the paper's recommended metric) or ``"vgg"``.
        weights: Linear-layer weights: ``"default"`` (the official file for ``net``), an
            ``LPIPS_Weights`` member, a path, or ``None`` (untrained, uniform layers: for tests).
        pretrained_backbone: Load torchvision's ImageNet weights (``False``: random, for tests).

    Examples:
        >>> metric = LPIPS(weights=None, pretrained_backbone=False)
        >>> x = torch.rand(2, 3, 32, 32)
        >>> metric(x, x).tolist()
        [0.0, 0.0]
    """

    mode         : ClassVar[MetricMode] = "fr"
    higher_better: ClassVar[bool] = False
    shift        : Tensor
    scale        : Tensor

    def __init__(
        self,
        net                : Literal["alex", "vgg"] = "alex",
        *,
        weights            : WeightsLike = "default",
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        if net not in _SLICES:
            raise ValueError(f"net must be 'alex' or 'vgg', got {net!r}")
        self.net = net
        bounds, channels = _SLICES[net]
        features = _backbone(net, pretrained_backbone)
        self.slices = nn.ModuleList(nn.Sequential(*features[a:b]) for a, b in pairwise(bounds))
        lins = [nn.Conv2d(c, 1, 1, bias=False) for c in channels]
        self.lins = nn.ModuleList(lins)
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1), persistent=False)

        if isinstance(weights, str) and weights.strip().lower() == "default":
            weights = LPIPS_Weights.ALEX if net == "alex" else LPIPS_Weights.VGG
        spec = resolve_weights(weights, LPIPS_Weights)
        if spec is None:
            for lin in lins:
                nn.init.constant_(lin.weight, 1 / lin.in_channels)
        else:
            state = spec.load_state_dict()
            # Official keys: ``lin<k>.model.1.weight`` (index 0 is a Dropout).
            for k, lin in enumerate(lins):
                lin.weight.data.copy_(state[f"lin{k}.model.1.weight"])
        self.requires_grad_(False)
        self.eval()

    def extra_repr(self) -> str:
        return f"net={self.net!r}"

    def train(self, mode: bool = True) -> "LPIPS":
        """Stay in eval mode: a metric has no training behavior (the backbones have none either)."""
        return super().train(False)

    def _features(self, x: Tensor) -> list[Tensor]:
        h = (x * 2 - 1 - self.shift) / self.scale
        feats = []
        for block in self.slices:
            h = block(h)
            feats.append(h / (h.square().sum(dim=1, keepdim=True).sqrt() + 1e-10))
        return feats

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        check_pair(pred, target)
        if pred.shape[1] != 3:
            raise ValueError(f"LPIPS needs RGB images, got {pred.shape[1]} channel(s)")
        dtype = self.shift.dtype
        f0, f1 = self._features(pred.to(dtype)), self._features(target.to(dtype))
        total = sum(lin((a - b).square()).mean(dim=(2, 3)) for lin, a, b in zip(self.lins, f0, f1, strict=True))
        return torch.as_tensor(total).flatten().double()
