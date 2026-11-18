"""HVI-CIDNet: low-light enhancement in the HVI color space with two cross-attending branches.

**HVI space.** Hue becomes a point ``(H, V) = C_k(I)·S·(cos 2πh, sin 2πh)`` on a disk whose radius
shrinks with intensity ``I = max(R, G, B)`` (``C_k(I) = sin(πI/2)^k``, ``k`` learned), which
removes the red-hue discontinuity of HSV and collapses dark colors. ``HVIT`` maps RGB to HVI;
``PHVIT`` maps back (via HSV).

**CIDNet.** An intensity branch (on ``I``) and an HV branch (on all of HVI) are U-Nets with three
bilinear (``align_corners=True``) down/up stages; at each scale *lightweight cross attention*
(LCA) lets each branch attend to the other (channel attention as in Restormer, then a gated
depthwise feed-forward). The output is the HVI input plus the predicted residual, mapped back to
RGB.

**Inference gains** (official ``eval.py``): ``saturation_gain`` multiplies the saturation in
``PHVIT`` (the official ``gated``/``alpha_s``: 1.3 for LOLv1); ``output_gain`` multiplies the RGB
output (``gated2``/``alpha``: 0.8 or 0.82 for the LOLv2-real weights). Both default to 1 (off);
each weights entry records its recommended values in ``meta["eval"]``. Outputs aren't clamped
(the official evaluation clamps to ``[0, 1]``). ``H`` and ``W`` must be divisible by 8; the
official evaluation reflect-pads to the next multiple.

Parameter names match the official code (191 keys incl. ``trans.density_k``); checkpoints load
with ``strict=True``.

Kept from the official code because the weights were trained with it: the third encoder stage
downsamples the features *before* the second LCA (the LCA outputs feed only the skips), and the
intensity decoder's second stage upsamples the features before ``I_LCA5`` (whose output is unused,
so it isn't computed; its parameters stay); ``HV_LCA``
has no residual around its feed-forward (``I_LCA`` does); ``PHVIT`` uses ``k`` as a constant (no
gradient through the inverse transform).

Differences from the official code (``net/*.py``, ``eval.py``): ``einops`` replaced by reshapes;
``HVIT`` starts hue from zeros instead of an uninitialized tensor; the ``gated``/``gated2`` flags
are constructor arguments; ``forward`` returns a dict (adds the HVI output); the Hugging Face
mixin is dropped. The training loss (L1 + SSIM + edge + VGG perceptual, in RGB and HVI) isn't
ported yet. ``mon_old`` (``models/enhance/lle/hvi_cidnet``) vendored the official network.

Examples:
    >>> import torch
    >>> out = HVICIDNet().eval()(torch.rand(1, 3, 16, 24))
    >>> {k: tuple(v.shape) for k, v in out.items()}
    {'enhanced': (1, 3, 16, 24), 'hvi': (1, 3, 16, 24)}
    >>> from mon.models.registry import load
    >>> model = load("hvi_cidnet", weights="lol_v1_wo_perc", saturation_gain=1.3)  # doctest: +SKIP

References:
    - Yan, Feng, Wang, et al., "HVI: A New Color Space for Low-light Image Enhancement", CVPR 2025.
      https://arxiv.org/abs/2502.20272
    - https://github.com/Fediory/HVI-CIDNet (commit ``eb43d7d``). Network adapted under the MIT
      License, Copyright (c) 2024 Fediory Feng. Weights: https://huggingface.co/Fediory.
"""

__all__ = [
    "HVICIDNet",
    "HVICIDNet_Weights",
    "hvi_to_rgb",
    "rgb_to_hvi",
]

import math
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum

_EPS = 1e-8


def _hf(repo: str, revision: str, sha256: str, train: str, **eval_settings: float) -> dict[str, object]:
    return {"source": f"https://huggingface.co/Fediory/HVI-CIDNet-{repo} (model.safetensors, rev {revision}); "
                      "same tensors", "source_sha256": sha256, "train": train, "eval": eval_settings}


class HVICIDNet_Weights(WeightsEnum):
    """Released HVI-CIDNet weights (zoo ``.pt`` files hold the same tensors as the official Hugging Face files).

    ``meta["eval"]`` holds the official evaluation settings (``saturation_gain``, ``output_gain``).
    Three zoo files (``hvi_cidnet_lol_v1.pt``, ``…_lol_v2_real_w_perc.pt``, ``…_lol_v2_syn.pt``) match no
    Hugging Face release and aren't listed.
    """

    LOL_V1_W_PERC = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_v1_w_perc.pt",
        name   = "HVI-CIDNet, official LOLv1/w_perc",
        sha256 = "9206385b514014b60bf9396151a011cabf68abb380c2ad87ffde3d53e8227926",
        meta   = _hf("LOLv1-wperc", "1fa9b69", "1163da200c800681…", "lol_v1", saturation_gain=1.3),
    )
    LOL_V1_WO_PERC = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_v1_wo_perc.pt",
        name   = "HVI-CIDNet, official LOLv1/wo_perc",
        sha256 = "2c5afb426fee910dd9806b4e6214f05914fb5988b7982a4624560815aefce2b5",
        meta   = _hf("LOLv1-woperc", "de3f94c", "928bc914bcec03ab…", "lol_v1", saturation_gain=1.3),
    )
    LOL_V2_REAL_BEST_PSNR = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_v2_real.pt",
        name   = "HVI-CIDNet, official LOLv2_real/best_PSNR",
        sha256 = "44a7f92cf02a1da322c1a34b3aa940341dc93fd53840b7f19b49f461200b00a6",
        meta   = _hf("LOLv2-real-bestPSNR", "c317f57", "f03da46e8dd7181a…", "lol_v2_real", output_gain=0.8),
    )
    LOL_V2_REAL_BEST_SSIM = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_v2_real_ssim.pt",
        name   = "HVI-CIDNet, official LOLv2_real/best_SSIM",
        sha256 = "3d7ad3e93d65effaeb937cf833b1f2b3fb78c7529a4db95c589ff0ca569347c3",
        meta   = _hf("LOLv2-real-bestSSIM", "5f759d5", "fad7bef532a018dc…", "lol_v2_real", output_gain=0.82),
    )
    LOL_V2_SYN_W_PERC = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_v2_syn_w_perc.pt",
        name   = "HVI-CIDNet, official LOLv2_syn/w_perc",
        sha256 = "460a9956018935cab61ddef3cea14f1dbbc02a65e12ee4d5b5ab45d704d2575b",
        meta   = _hf("LOLv2-syn-wperc", "001d921", "aafd2a73dae9770b…", "lol_v2_syn"),
    )
    LOL_V2_SYN_WO_PERC = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_v2_syn_wo_perc.pt",
        name   = "HVI-CIDNet, official LOLv2_syn/wo_perc",
        sha256 = "519cc5fc55491cf36b47b8295c0204dde29ba887feeed06a77e716072469e55e",
        meta   = _hf("LOLv2-syn-woperc", "e30d947", "59a5998b6cacc310…", "lol_v2_syn"),
    )
    LOL_BLUR = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_lol_blur.pt",
        name   = "HVI-CIDNet, official LOL-Blur",
        sha256 = "eba53573097b4138be26ffc219826cf87eaeafbc0a6ae90cf4b35330070b5494",
        meta   = _hf("LOL-Blur", "8faacc8", "6b9761dfa3d0f35d…", "lol_blur"),
    )
    SICE = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_sice.pt",
        name   = "HVI-CIDNet, official SICE",
        sha256 = "7b0824026124427fe0419465fc06a6be0129356c949794d1d1f52c4173091607",
        meta   = _hf("SICE", "dc35250", "77b36ad42df18087…", "sice"),
    )
    SID_SONY = Weights(
        path   = "lle/hvi_cidnet/hvi_cidnet_sid_sony.pt",
        name   = "HVI-CIDNet, official Sony-Total-Dark (SID.pth)",
        sha256 = "49678f69495610a2691bb5b2665d2766b57a6a79201a236f7afa84f67c5fec5f",
        meta   = _hf("Sony-Total-Dark", "bf347b9", "49851ed73c47d61a…", "sid_sony"),
    )
    DEFAULT = LOL_V1_WO_PERC


# --- HVI color space ---

def rgb_to_hvi(image: Tensor, k: Tensor | float) -> Tensor:
    """RGB ``[B, 3, H, W]`` in ``[0, 1]`` → HVI ``[B, 3, H, W]`` (``H, V`` in ``[-1, 1]``, ``I = max(RGB)``)."""
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    value   = image.max(1)[0]
    vmin    = image.min(1)[0]
    delta   = value - vmin + _EPS
    hue     = torch.zeros_like(value)
    hue     = torch.where(b == value, 4.0 + (r - g) / delta, hue)
    hue     = torch.where(g == value, 2.0 + (b - r) / delta, hue)
    hue     = torch.where(r == value, ((g - b) / delta) % 6, hue)
    hue     = torch.where(vmin == value, torch.zeros_like(hue), hue) / 6.0
    sat     = (value - vmin) / (value + _EPS)
    sat     = torch.where(value == 0, torch.zeros_like(sat), sat)
    hue, sat, value = hue.unsqueeze(1), sat.unsqueeze(1), value.unsqueeze(1)
    cs      = (torch.sin(value * 0.5 * math.pi) + _EPS).pow(k)
    return torch.cat([cs * sat * torch.cos(2 * math.pi * hue), cs * sat * torch.sin(2 * math.pi * hue), value], 1)


def hvi_to_rgb(hvi: Tensor, k: float, saturation_gain: float = 1.0) -> Tensor:
    """HVI → RGB (values in ``[0, 1]``); ``saturation_gain`` scales the saturation (official ``alpha_s``)."""
    h_, v_  = hvi[:, 0].clamp(-1, 1), hvi[:, 1].clamp(-1, 1)
    value   = hvi[:, 2].clamp(0, 1)
    cs      = (torch.sin(value * 0.5 * math.pi) + _EPS).pow(k)
    h_      = (h_ / (cs + _EPS)).clamp(-1, 1)
    v_      = (v_ / (cs + _EPS)).clamp(-1, 1)
    hue     = (torch.atan2(v_ + _EPS, h_ + _EPS) / (2 * math.pi)) % 1
    sat     = torch.sqrt(h_**2 + v_**2 + _EPS)
    if saturation_gain != 1.0:
        sat = sat * saturation_gain
    sat     = sat.clamp(0, 1)

    sector  = torch.floor(hue * 6.0)
    f       = hue * 6.0 - sector
    p       = value * (1.0 - sat)
    q       = value * (1.0 - f * sat)
    t       = value * (1.0 - (1.0 - f) * sat)
    zero    = torch.zeros_like(hue)
    # (r, g, b) per HSV sector 0..5; sectors outside 0..5 (none in practice) stay black, as in the official code.
    table   = ((value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q))
    rgb     = [zero, zero, zero]
    for s, channels in enumerate(table):
        mask = sector == s
        rgb  = [torch.where(mask, src, dst) for src, dst in zip(channels, rgb, strict=True)]
    return torch.stack(rgb, 1)


class _RGBHVI(nn.Module):
    """Holds the learned density ``k`` (``trans.density_k``; the paper's ``1/k``)."""

    def __init__(self) -> None:
        super().__init__()
        self.density_k = nn.Parameter(torch.full([1], 0.2))


# --- Layers ---

class _LayerNorm(nn.Module):
    """Channels-first LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x - u) / torch.sqrt(s + self.eps)) + self.bias[:, None, None]


class _NormDownsample(nn.Module):
    """3×3 conv → bilinear ×0.5 (``align_corners=True``) → PReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.prelu = nn.PReLU()
        self.down  = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.UpsamplingBilinear2d(scale_factor=0.5))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.prelu(self.down(x))
        return out


class _NormUpsample(nn.Module):
    """3×3 conv → bilinear ×2 → concat skip → 1×1 conv → PReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.prelu    = nn.PReLU()
        self.up_scale = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up       = nn.Conv2d(out_ch * 2, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        out: Tensor = self.prelu(self.up(torch.cat([self.up_scale(x), skip], 1)))
        return out


class _CAB(nn.Module):
    """Cross attention over channels: queries from ``x``, keys and values from ``y`` (per head)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q           = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_dwconv    = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.kv          = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.kv_dwconv   = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q    = self.q_dwconv(self.q(x)).reshape(b, self.num_heads, c // self.num_heads, h * w)
        k, v = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
        k    = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v    = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        out: Tensor = self.project_out((attn.softmax(dim=-1) @ v).reshape(b, c, h, w))
        return out


class _IEL(nn.Module):
    """Gated depthwise feed-forward (intensity enhancement layer; tanh-residual depthwise convs)."""

    def __init__(self, dim: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden           = int(dim * expansion)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=False)
        self.dwconv1     = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False)
        self.dwconv2     = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        x1 = torch.tanh(self.dwconv1(x1)) + x1
        x2 = torch.tanh(self.dwconv2(x2)) + x2
        out: Tensor = self.project_out(x1 * x2)
        return out


class _LCA(nn.Module):
    """Lightweight cross attention; ``residual_ffn`` is True for the intensity branch (``I_LCA``)."""

    def __init__(self, dim: int, num_heads: int, residual_ffn: bool) -> None:
        super().__init__()
        self.gdfn         = _IEL(dim)
        self.norm         = _LayerNorm(dim)
        self.ffn          = _CAB(dim, num_heads)
        self.residual_ffn = residual_ffn

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.ffn(self.norm(x), self.norm(y))
        out: Tensor = self.gdfn(self.norm(x))
        return x + out if self.residual_ffn else out


def _stem(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_ch, out_ch, 3, 1, 0, bias=False))


# --- Model ---

class HVICIDNet(nn.Module):
    """CIDNet in HVI space.

    Args:
        channels: Feature channels per scale (official: ``(36, 36, 72, 144)``).
        heads: Attention heads per scale (official: ``(1, 2, 4, 8)``).
        saturation_gain: Saturation multiplier in the inverse transform (official LOLv1 eval: 1.3).
        output_gain: RGB output multiplier (official LOLv2-real eval: 0.8 / 0.82).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W] (RGB, not clamped above
        ``output_gain``), "hvi": [B, 3, H, W] (predicted HVI, before the inverse transform)}``;
        ``H, W`` divisible by 8.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = HVICIDNet_Weights
    input_multiple: ClassVar[int] = 8  # H, W must be multiples (three 2x downsamplings); runners pad

    def __init__(
        self,
        channels       : tuple[int, int, int, int] = (36, 36, 72, 144),
        heads          : tuple[int, int, int, int] = (1, 2, 4, 8),
        saturation_gain: float = 1.0,
        output_gain    : float = 1.0,
    ) -> None:
        super().__init__()
        ch1, ch2, ch3, ch4 = channels
        _, h2, h3, h4      = heads
        self.saturation_gain = saturation_gain
        self.output_gain     = output_gain

        self.HVE_block0 = _stem(3, ch1)
        self.HVE_block1 = _NormDownsample(ch1, ch2)
        self.HVE_block2 = _NormDownsample(ch2, ch3)
        self.HVE_block3 = _NormDownsample(ch3, ch4)
        self.HVD_block3 = _NormUpsample(ch4, ch3)
        self.HVD_block2 = _NormUpsample(ch3, ch2)
        self.HVD_block1 = _NormUpsample(ch2, ch1)
        self.HVD_block0 = _stem(ch1, 2)

        self.IE_block0 = _stem(1, ch1)
        self.IE_block1 = _NormDownsample(ch1, ch2)
        self.IE_block2 = _NormDownsample(ch2, ch3)
        self.IE_block3 = _NormDownsample(ch3, ch4)
        self.ID_block3 = _NormUpsample(ch4, ch3)
        self.ID_block2 = _NormUpsample(ch3, ch2)
        self.ID_block1 = _NormUpsample(ch2, ch1)
        self.ID_block0 = _stem(ch1, 1)

        for i, (dim, nh) in enumerate([(ch2, h2), (ch3, h3), (ch4, h4), (ch4, h4), (ch3, h3), (ch2, h2)], 1):
            self.add_module(f"HV_LCA{i}", _LCA(dim, nh, residual_ffn=False))
            self.add_module(f"I_LCA{i}", _LCA(dim, nh, residual_ffn=True))

        self.trans = _RGBHVI()

    def extra_repr(self) -> str:
        return f"saturation_gain={self.saturation_gain}, output_gain={self.output_gain}"

    def _lca(self, branch: str, i: int, x: Tensor, y: Tensor) -> Tensor:
        out: Tensor = getattr(self, f"{branch}_LCA{i}")(x, y)
        return out

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        h, w = image.shape[-2:]
        if h % 8 or w % 8:
            raise ValueError(f"HVICIDNet needs H and W divisible by 8, got {h}×{w}; pad the input (pad_to_multiple)")
        hvi   = rgb_to_hvi(image, self.trans.density_k)
        i     = hvi[:, 2:3]

        # Encoder.
        i_0   = self.IE_block0(i)
        hv_0  = self.HVE_block0(hvi)
        i_1   = self.IE_block1(i_0)
        hv_1  = self.HVE_block1(hv_0)
        i_s1  = self._lca("I", 1, i_1, hv_1)
        hv_s1 = self._lca("HV", 1, hv_1, i_1)
        i_2   = self.IE_block2(i_s1)
        hv_2  = self.HVE_block2(hv_s1)
        i_s2  = self._lca("I", 2, i_2, hv_2)
        hv_s2 = self._lca("HV", 2, hv_2, i_2)
        i_3   = self.IE_block3(i_2)    # official: downsamples i_2, not i_s2 (LCA2 feeds only the skip)
        hv_3  = self.HVE_block3(hv_2)

        # Bottleneck.
        i_4   = self._lca("I", 3, i_3, hv_3)
        hv_4  = self._lca("HV", 3, hv_3, i_3)
        i_d4  = self._lca("I", 4, i_4, hv_4)
        hv_d4 = self._lca("HV", 4, hv_4, i_4)

        # Decoder.
        hv_d3 = self.HVD_block3(hv_d4, hv_s2)
        i_d3  = self.ID_block3(i_d4, i_s2)
        hv_l5 = self._lca("HV", 5, hv_d3, i_d3)  # I_LCA5's output is unused in the official network: skipped
        hv_d2 = self.HVD_block2(hv_l5, hv_s1)
        i_d2  = self.ID_block2(i_d3, i_s1)
        i_l6  = self._lca("I", 6, i_d2, hv_d2)
        hv_l6 = self._lca("HV", 6, hv_d2, i_d2)
        i_out = self.ID_block0(self.ID_block1(i_l6, i_0))
        hv_out = self.HVD_block0(self.HVD_block1(hv_l6, hv_0))

        out_hvi = torch.cat([hv_out, i_out], 1) + hvi
        rgb     = hvi_to_rgb(out_hvi, float(self.trans.density_k.item()), self.saturation_gain)
        if self.output_gain != 1.0:
            rgb = rgb * self.output_gain
        return {"enhanced": rgb, "hvi": out_hvi}
