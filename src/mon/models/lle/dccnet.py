"""DCC-Net: deep color consistent low-light enhancement (gray, color, and fusion branches).

Three U-Net-style branches of instance-normalized GELU convolutions with residual blocks:
``G-Net`` predicts a gray (luminance) image; ``C-Net`` encodes the colors into features at 1/4
resolution and a per-channel color histogram (``d_hist`` bins over 256 pooled features); ``R-Net``
restores the image from the input and the gray image, and its decoder receives the encoder skip
features after *pyramid color embedding* (PCE): at each scale a color-matching attention (CMA) adds
the color features weighted by their L1 and dot-product similarity to the gray features, then
upsamples the color features for the next scale. Inputs must have ``H`` and ``W`` divisible by 8
(three stride-2 stages); pad and crop otherwise.

Parameter names match the official code. The released checkpoint was saved from
``nn.DataParallel`` (keys start with ``module.``); ``DCCNet`` strips that prefix on load, so it
loads with ``strict=True``.

Differences from the official code (``src/model.py``, ``src/test.py``):

- ``forward`` returns a dict (``enhanced``, ``gray``, ``color_hist``) instead of the tuple
  ``(gray, color_hist, enhanced)``; the branch classes are private (``_GNet``, ``_CNet``,
  ``_RNet``) and share one encoder/decoder builder.
- The last CMA's color upsampling, whose output the official code discards, isn't computed.
- No training code was released (no loss to port); ``color_hist`` is returned for a future loss.

``mon_old`` (``models/enhance/lle/dccnet``) matches the official network; its weights loading in
``__init__``, ``forward_patch`` (use ``mon.nn.tiling.tiled_forward``), and ``benchmark`` were
removed. Its predictor resized inputs to 512 (``mon`` pads instead).

Examples:
    >>> import torch
    >>> out = DCCNet().eval()(torch.rand(1, 3, 16, 24))
    >>> {k: tuple(v.shape) for k, v in out.items()}
    {'enhanced': (1, 3, 16, 24), 'gray': (1, 1, 16, 24), 'color_hist': (1, 3, 64)}
    >>> from mon.models.registry import load
    >>> model = load("dccnet")  # official LOL release; needs DATA_DIR or a download  # doctest: +SKIP

References:
    - Zhang, Zheng, Hong, Xu, Yan, Wang, "Deep Color Consistent Network for Low-Light Image
      Enhancement", CVPR 2022.
      https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html
    - https://github.com/Ian0926/DCC-Net (commit ``8b57f9e``). Network adapted from ``src/model.py``
      under the MIT License, Copyright (c) 2022 Ian.
"""

__all__ = [
    "DCCNet",
    "DCCNet_Weights",
]

from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from mon.core.weights import Weights, WeightsEnum


class DCCNet_Weights(WeightsEnum):
    """Released DCC-Net weights."""

    LOL_V1 = Weights(
        path   = "lle/dccnet/dccnet_lol_v1.pt",
        name   = "DCC-Net, official release (LOL)",
        url    = "https://drive.usercontent.google.com/download?id=134wM6wz0GdC6QXaeyrtQy6tyHpuRZ8Jp&export=download"
                 "&confirm=t",
        sha256 = "12e2bdc20cd4dd93d547543643a49e1183ee96d209092ab476ef02f3dae67ae5",
        meta   = {"source": "official pretrained model (README link, Google Drive)", "train": "lol_v1",
                  "keys": "saved from nn.DataParallel ('module.' prefix)"},
    )
    DEFAULT = LOL_V1


# --- Layers ---

class _BasicConv(nn.Module):
    """Conv (or 4×4 stride-2 transposed conv) → InstanceNorm → GELU; no bias before the norm."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 transpose: bool = False) -> None:
        super().__init__()
        conv: nn.Module
        if transpose:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2 - 1, bias=False)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.main = nn.Sequential(conv, nn.InstanceNorm2d(out_channels), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.main(x)
        return out


class _RB(nn.Module):
    """Residual block: two ``_BasicConv`` 3×3 layers plus the input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer_1 = _BasicConv(channels, channels, 3, 1)
        self.layer_2 = _BasicConv(channels, channels, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.layer_2(self.layer_1(x)) + x
        return out


class _Scale(nn.Module):
    """Stride-2 down (3×3 conv, channels ×2) or up (4×4 transposed conv, channels ÷2)."""

    def __init__(self, in_channels: int, up: bool) -> None:
        super().__init__()
        if up:
            self.main = _BasicConv(in_channels, in_channels // 2, 4, 2, transpose=True)
        else:
            self.main = _BasicConv(in_channels, in_channels * 2, 3, 2)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.main(x)
        return out


def _encoder(base: int, depth: tuple[int, ...], last_down: bool) -> nn.ModuleList:
    """Per level: conv, residual blocks (skip output), downscale (except the last level unless ``last_down``)."""
    layers: list[nn.Module] = []
    for level, d in enumerate(depth[:3]):
        ch = base * 2**level
        layers += [_BasicConv(ch, ch, 3, 1), nn.Sequential(*[_RB(ch) for _ in range(d)])]
        if level < 2 or last_down:
            layers.append(_Scale(ch, up=False))
    return nn.ModuleList(layers)


def _decoder(base: int, depth: tuple[int, ...]) -> nn.ModuleList:
    """Per level, from the bottom: upscale, (concat skip) conv halving channels, residual blocks."""
    layers: list[nn.Module] = []
    for level in (2, 1, 0):
        ch = base * 2**level
        layers += [_Scale(ch * 2, up=True), _BasicConv(ch * 2, ch, 3, 1),
                   nn.Sequential(*[_RB(ch) for _ in range(depth[level])])]
    return nn.ModuleList(layers)


def _encode(blocks: nn.ModuleList, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    skips = []
    for i, block in enumerate(blocks):
        x = block(x)
        if i % 3 == 1:  # after each level's residual blocks
            skips.append(x)
    return x, skips


def _decode(blocks: nn.ModuleList, x: Tensor, skips: list[Tensor]) -> Tensor:
    for i, block in enumerate(blocks):
        if i % 3 == 1:  # before each level's conv: concat the matching skip
            x = torch.cat([x, skips[len(skips) - 1 - i // 3]], 1)
        x = block(x)
    return x


# --- Branches ---

class _GNet(nn.Module):
    """Gray branch: U-Net → one-channel image in ``(0, 1)``."""

    def __init__(self, base: int = 32, depth: tuple[int, ...] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.Encoder    = _encoder(base, depth, last_down=True)
        self.middle     = nn.Sequential(*[_RB(base * 8) for _ in range(depth[3])])
        self.Decoder    = _decoder(base, depth)
        self.conv_first = _BasicConv(3, base, 3, 1)
        self.conv_last  = nn.Conv2d(base, 1, 3, 1, 1)

    def forward(self, image: Tensor) -> Tensor:
        x, skips = _encode(self.Encoder, self.conv_first(image))
        x = _decode(self.Decoder, self.middle(x), skips)
        return (torch.tanh(self.conv_last(x)) + 1) / 2


class _CNet(nn.Module):
    """Color branch: encoder to 1/4 resolution → color features and a ``[B, 3, d_hist]`` histogram."""

    def __init__(self, d_hist: int = 64, base: int = 32, depth: tuple[int, ...] = (2, 2, 2)) -> None:
        super().__init__()
        self.Encoder    = _encoder(base, depth, last_down=False)
        self.conv_first = _BasicConv(3, base, 3, 1)
        self.conv_color = _BasicConv(base * 4, 256 * 3, 3, 1)
        self.pooling    = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Linear(256, d_hist)
        self.softmax    = nn.Softmax(dim=2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features, _ = _encode(self.Encoder, self.conv_first(image))
        pooled      = self.pooling(self.conv_color(features)).reshape(-1, 3, 256)
        return self.softmax(self.fc(pooled)), features


class _CMA(nn.Module):
    """Color-matching attention: embed color features into gray features by similarity."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.InstanceNorm2d(out_channels),
                                  nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode="nearest"))

    def forward(self, color: Tensor, gray: Tensor) -> Tensor:
        sim_l1  = 2 * torch.sigmoid(-torch.abs(gray - color).sum(1, keepdim=True))  # (0, 1]
        sim_cos = torch.tanh((gray * color).sum(1, keepdim=True))
        return gray + color * (sim_l1 * sim_cos)

    def upsample(self, color: Tensor) -> Tensor:
        """Color features for the next (2× finer) scale."""
        out: Tensor = self.conv(color)
        return out


class _PCE(nn.Module):
    """Pyramid color embedding over the three ``R-Net`` skip scales (coarse to fine)."""

    def __init__(self) -> None:
        super().__init__()
        self.cma_3 = _CMA(128, 64)
        self.cma_2 = _CMA(64, 32)
        self.cma_1 = _CMA(32, 16)  # its upsampling is unused, as in the official code

    def forward(self, color: Tensor, skips: list[Tensor]) -> list[Tensor]:
        x3 = self.cma_3(color, skips[2])
        c2 = self.cma_3.upsample(color)
        x2 = self.cma_2(c2, skips[1])
        x1 = self.cma_1(self.cma_2.upsample(c2), skips[0])
        return [x1, x2, x3]


class _RNet(nn.Module):
    """Restoration branch: U-Net on ``[image, gray]`` with color-embedded skips → RGB in ``(0, 1)``."""

    def __init__(self, base: int = 32, depth: tuple[int, ...] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.Encoder    = _encoder(base, depth, last_down=True)
        self.middle     = nn.Sequential(*[_RB(base * 8) for _ in range(depth[3])])
        self.Decoder    = _decoder(base, depth)
        self.conv_first = _BasicConv(4, base, 3, 1)
        self.conv_last  = nn.Conv2d(base, 3, 3, 1, 1)
        self.pce        = _PCE()

    def forward(self, image: Tensor, gray: Tensor, color: Tensor) -> Tensor:
        x, skips = _encode(self.Encoder, self.conv_first(torch.cat([image, gray], 1)))
        x = _decode(self.Decoder, self.middle(x), self.pce(color, skips))
        return (torch.tanh(self.conv_last(x)) + 1) / 2


# --- Model ---

def _strip_data_parallel_prefix(module: nn.Module, state_dict: dict[str, Any], prefix: str, *args: Any) -> None:
    """Load-state-dict pre-hook: drop a ``module.`` prefix that every key carries (``nn.DataParallel``)."""
    keys = [k for k in state_dict if k.startswith(prefix)]
    full = prefix + "module."
    if keys and all(k.startswith(full) for k in keys):
        for k in keys:
            state_dict[prefix + k[len(full):]] = state_dict.pop(k)


class DCCNet(nn.Module):
    """DCC-Net: gray (``g_net``), color (``c_net``), and restoration (``r_net``) branches.

    Args:
        d_hist: Histogram bins per color channel (official: 64).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W], "gray": [B, 1, H,
        W], "color_hist": [B, 3, d_hist]}`` (values in ``(0, 1)``; histograms sum to 1);
        ``H, W`` divisible by 8.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = DCCNet_Weights
    input_multiple: ClassVar[int] = 8  # H, W must be multiples (three stride-2 stages); runners pad

    def __init__(self, d_hist: int = 64) -> None:
        super().__init__()
        self.d_hist = d_hist
        self.g_net  = _GNet()
        self.c_net  = _CNet(d_hist)
        self.r_net  = _RNet()
        self.register_load_state_dict_pre_hook(_strip_data_parallel_prefix)  # type: ignore[no-untyped-call]

    def extra_repr(self) -> str:
        return f"d_hist={self.d_hist}"

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        h, w = image.shape[-2:]
        if h % 8 or w % 8:
            raise ValueError(f"DCCNet needs H and W divisible by 8, got {h}×{w}; pad the input (pad_to_multiple)")
        gray               = self.g_net(image)
        color_hist, color  = self.c_net(image)
        enhanced           = self.r_net(image, gray, color)
        return {"enhanced": enhanced, "gray": gray, "color_hist": color_hist}
