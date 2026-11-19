"""FLOL: fast low-light enhancement in the frequency, then the spatial domain.

**Frequency stage.** A small amplitude network (three frequency-gated ``SFBlock``\\ s at half
resolution, then a sigmoid) predicts a per-pixel, per-channel divisor of the image's FFT
magnitude; dividing by it and inverting the FFT with the original phase brightens the image
(``x_center``). **Spatial stage.** An encoder (two stride-2 convs) and residual blocks extract
features at 1/4 resolution; an SNR map of ``x_center`` (luminance over its deviation from a 5×5
Gaussian blur, normalized per image) blends a frequency-domain branch (``SFNet``: 4 FFT
amplitude/phase blocks) where the SNR is low with a residual-block branch where it's high; a
pixel-shuffle decoder with skips predicts a residual added to the input.

``H`` and ``W`` must be even (the amplitude network halves and re-doubles the image); the model
reflect-pads to a multiple of 8 itself and crops back. Outputs aren't clamped (the official
inference clamps to ``[0, 1]`` before saving).

Parameter names match the official code. The released files are BasicSR-style ``{"params":
state_dict}`` checkpoints; ``FLOL`` unwraps that on load, so both zoo files load with
``strict=True``.

Differences from the official code (``model/flol.py``, ``utils/utils.py``, ``inference.py``):

- The SNR map's Gaussian blur is written out (separable, reflect border) instead of calling
  ``kornia.filters.gaussian_blur2d``; no ``kornia`` dependency.
- ``LayerNorm2d`` uses autograd instead of the hand-written backward (same gradient).
- Unused modules (``SpaBlock``; ``ProcessBlock`` always runs without its spatial path, as in the
  official ``SFNet``) and the ``side`` output flag were removed; ``forward`` returns a dict.
- No training code was released (no loss to port).
- The frequency blocks set the imaginary part of the real-valued ``rfft2`` bins to exactly 0 before ``angle``:
  the official output depends on whether the FFT library returns ``±1e-17`` or 0 there (``-π`` vs ``+π`` phase,
  ~1e-3 apart in ``enhanced``). Exactly 0 is what x86 produces, so this matches the official code on x86.

``mon_old`` (``models/enhance/lle/flol``) copied the official network; its predictor resized inputs
to 512 (``mon`` pads instead, as the official ``inference.py`` does).

Examples:
    >>> import torch
    >>> out = FLOL().eval()(torch.rand(1, 3, 20, 30))
    >>> {k: tuple(v.shape) for k, v in out.items()}
    {'enhanced': (1, 3, 20, 30), 'amplitude_enhanced': (1, 3, 20, 30)}
    >>> from mon.models.registry import load
    >>> model = load("flol")  # official LOL-v2-real release; needs DATA_DIR or a download  # doctest: +SKIP

References:
    - Benito, Feijoo, Garcia, Conde, "FLOL: Fast Baselines for Real-World Low-Light Enhancement",
      arXiv 2501.09718, 2025. https://arxiv.org/abs/2501.09718
    - https://github.com/cidautai/FLOL (commit ``d6b66c3``). Network adapted from ``model/flol.py``
      and ``utils/utils.py`` under the MIT License, Copyright (c) 2025 cidautai.
"""

__all__ = [
    "FLOL",
    "FLOL_Weights",
]

from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum

_REPO = "https://raw.githubusercontent.com/cidautai/FLOL/d6b66c3a6f5edcb4b7d122cf126fd2e3c951c2a9/weights/"


class FLOL_Weights(WeightsEnum):
    """Released FLOL weights (the paper's FLOL+)."""

    LOL_V2_REAL = Weights(
        path   = "lle/flol/flol_lol_v2_real.pt",
        name   = "FLOL, official release (weights/flolv2_all_111439.pt)",
        url    = _REPO + "flolv2_all_111439.pt",
        sha256 = "1e43dbdfc5a8d1ea015f8038544dd96bb530b9918e9e542e93b06a504cc17231",
        meta   = {"source": "official weights/flolv2_all_111439.pt", "train": "lol_v2_real",
                  "readme_psnr": {"lol_v2_real": 21.75}},
    )
    UHD_LL = Weights(
        path   = "lle/flol/flol_lol_v2_real_uhd_ll.pt",
        name   = "FLOL, official release (weights/flolv2_UHDLL.pt), unwrapped from 'params'",
        sha256 = "5311856caf6ec624863c174a47db17b6b0f7caa9988ca55d956b9d37fc493ca2",
        meta   = {"source": f"official weights/flolv2_UHDLL.pt ({_REPO}flolv2_UHDLL.pt); same tensors",
                  "source_sha256": "4415a5e5fd44c96c2e7e7d8a78196cc76c83cfb28ab2ec3e87620f9ce8ef87bb",
                  "train": "uhd_ll", "readme_psnr": {"uhd_ll": 25.01}},
    )
    DEFAULT = LOL_V2_REAL


# --- Layers ---

class _LayerNorm2d(nn.Module):
    """LayerNorm over channels of a ``[B, C, H, W]`` tensor (NAFNet's ``LayerNorm2d``)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias   = nn.Parameter(torch.zeros(channels))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        mu  = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y   = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class _ResidualBlockNoBN(nn.Module):
    """``x + conv(relu(conv(x)))``."""

    def __init__(self, nf: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = x + self.conv2(F.relu(self.conv1(x)))
        return out


def _simple_gate(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=1)
    return x1 * x2


class _SGE(nn.Module):
    """Gate whose first half passes a depthwise 3×3 conv."""

    def __init__(self, dw_channel: int) -> None:
        super().__init__()
        c = dw_channel // 2
        self.dwc = nn.Conv2d(c, c, 3, 1, 1, groups=c)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        out: Tensor = self.dwc(x1) * x2
        return out


def _mlp(nc: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(nc, nc, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nc, nc, 1, 1, 0))


class _FreBlock(nn.Module):
    """1×1 conv, then 1×1-conv MLPs on the rFFT amplitude and phase; residual."""

    def __init__(self, nc: int) -> None:
        super().__init__()
        self.fpre     = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = _mlp(nc)  # amplitude
        self.process2 = _mlp(nc)  # phase

    def forward(self, x: Tensor) -> Tensor:
        h, w   = x.shape[-2:]
        x_freq = _exact_real_bins(torch.fft.rfft2(self.fpre(x), norm="backward"), h, w)
        mag    = self.process1(torch.abs(x_freq))
        pha    = self.process2(torch.angle(x_freq))
        out: Tensor = torch.fft.irfft2(torch.complex(mag * torch.cos(pha), mag * torch.sin(pha)), s=(h, w),
                                       norm="backward")
        return out + x


def _exact_real_bins(x_freq: Tensor, h: int, w: int) -> Tensor:
    """Zero the imaginary part of the ``rfft2`` bins that are real for real input (rows 0 and ``h/2``, columns 0
    and ``w/2``; even sizes only).

    FFT libraries leave ``±1e-17`` there or exactly 0, so ``angle`` gives ``+π`` or ``-π`` for a negative real
    value depending on the platform, and the learned phase MLP turns that into ~1e-3 output differences. Exactly 0
    gives ``+π`` everywhere (what x86 already produces).
    """
    rows = [0] + ([h // 2] if h % 2 == 0 else [])
    cols = [0] + ([w // 2] if w % 2 == 0 else [])
    mask = torch.zeros(x_freq.shape[-2:], dtype=torch.bool, device=x_freq.device)
    for r in rows:
        mask[r, cols] = True
    return torch.complex(x_freq.real, x_freq.imag.masked_fill(mask, 0.0))


class _SFBlock(nn.Module):
    """NAFNet-style block whose attention is a depthwise gate plus a ``_FreBlock``."""

    def __init__(self, nc: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw, ffn    = nc * dw_expand, nc * ffn_expand
        self.conv1 = nn.Conv2d(nc, dw, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, nc, 1, 1, 0)
        self.fatt  = _FreBlock(dw // 2)
        self.sge   = _SGE(dw)
        self.conv4 = nn.Conv2d(nc, ffn, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn // 2, nc, 1, 1, 0)
        self.norm1 = _LayerNorm2d(nc)
        self.norm2 = _LayerNorm2d(nc)
        self.beta  = nn.Parameter(torch.zeros((1, nc, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv3(self.fatt(self.sge(self.conv2(self.conv1(self.norm1(x))))))
        y = x + x * self.beta
        x = self.conv5(_simple_gate(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class _ProcessBlock(nn.Module):
    """Frequency block → 1×1 conv, residual (the official block with ``spatial=False``)."""

    def __init__(self, nc: int) -> None:
        super().__init__()
        self.frequency_process = _FreBlock(nc)
        self.cat               = nn.Conv2d(nc, nc, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.cat(self.frequency_process(x)) + x
        return out


class _SFNet(nn.Module):
    """``n`` process blocks with a global residual."""

    def __init__(self, nc: int, n: int = 4) -> None:
        super().__init__()
        self.block = nn.Sequential(*[_ProcessBlock(nc) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = x + self.block(x)
        return out


class _AmplitudeNet(nn.Module):
    """Predicts the FFT-magnitude divisor at half resolution (bilinear down and up)."""

    def __init__(self, nc: int = 8) -> None:
        super().__init__()
        self.conv_init = nn.Conv2d(3, nc, 1, 1, 0)
        self.conv1     = _SFBlock(nc)
        self.conv2     = _SFBlock(nc)
        self.conv3     = _SFBlock(nc)
        self.conv_out  = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        x = self.conv_out(self.conv3(self.conv2(self.conv1(self.conv_init(x)))))
        return F.interpolate(x, scale_factor=2, mode="bilinear")


# --- SNR map ---

def _gaussian_blur5(x: Tensor, sigma: float = 1.5) -> Tensor:
    """Separable 5×5 Gaussian blur with reflect border (``kornia.filters.gaussian_blur2d`` defaults)."""
    c   = x.shape[1]
    pos = torch.arange(5, device=x.device, dtype=x.dtype) - 2
    k   = torch.exp(-pos.pow(2) / (2 * sigma**2))
    k   = k / k.sum()
    x   = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)


def _luma(x: Tensor) -> Tensor:
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def _snr_map(image: Tensor) -> Tensor:
    """SNR map (SNR-Aware LLIE): ``blur/|image − blur|`` on luminance, over its per-image max, in ``[0, 1]``."""
    light = _luma(_gaussian_blur5(image))
    noise = torch.abs(_luma(image) - light)
    mask  = light / (noise + 1e-4)
    peak  = mask.flatten(1).max(1).values.view(-1, 1, 1, 1)
    return torch.clamp(mask / (peak + 1e-4), 0, 1)


# --- Model ---

def _unwrap_params(module: nn.Module, state_dict: dict[str, Any], prefix: str, *args: Any) -> None:
    """Load-state-dict pre-hook: unwrap a BasicSR ``{"params": state_dict}`` checkpoint."""
    if prefix == "" and set(state_dict) == {"params"} and isinstance(state_dict["params"], dict):
        state_dict.update(state_dict.pop("params"))


class FLOL(nn.Module):
    """FLOL network (``nf=16`` is the released FLOL+).

    Args:
        nf: Feature channels of the spatial stage (16 in the release).
        amp_channels: Feature channels of the amplitude network (8 in the release).
        sf_blocks: Frequency process blocks in the low-SNR branch (4 in the release).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W] (not clamped),
        "amplitude_enhanced": [B, 3, H, W] (frequency-stage output)}``; ``H, W`` even.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = FLOL_Weights

    def __init__(self, nf: int = 16, amp_channels: int = 8, sf_blocks: int = 4) -> None:
        super().__init__()
        self.nf                 = nf
        self.AmpNet             = nn.Sequential(_AmplitudeNet(amp_channels), nn.Sigmoid())
        self.conv_first_1       = nn.Conv2d(3 * 2, nf, 3, 1, 1)
        self.conv_first_2       = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv_first_3       = nn.Conv2d(nf, nf, 3, 2, 1)
        self.feature_extraction = nn.Sequential(_ResidualBlockNoBN(nf))
        self.recon_trunk        = nn.Sequential(_ResidualBlockNoBN(nf))
        self.upconv1            = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)
        self.upconv2            = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)
        self.pixel_shuffle      = nn.PixelShuffle(2)
        self.HRconv             = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.conv_last          = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu              = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer        = _SFNet(nf, n=sf_blocks)
        self.recon_trunk_light  = nn.Sequential(*[_ResidualBlockNoBN(nf) for _ in range(6)])
        self.register_load_state_dict_pre_hook(_unwrap_params)  # type: ignore[no-untyped-call]

    def extra_repr(self) -> str:
        return f"nf={self.nf}"

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        h, w = image.shape[-2:]
        if h % 2 or w % 2:
            raise ValueError(f"FLOL needs even H and W, got {h}×{w}; pad the input (pad_to_multiple)")

        # Frequency stage: divide the FFT magnitude by the predicted curve, keep the phase.
        fft     = torch.fft.fft2(image, norm="backward")
        mag     = torch.abs(fft) / (self.AmpNet(image) + 1e-8)
        pha     = torch.angle(fft)
        spec    = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        x_amp   = torch.fft.ifft2(spec, s=(h, w), norm="backward").real

        # Spatial stage on inputs reflect-padded to a multiple of 8.
        pad_h, pad_w = -h % 8, -w % 8
        x_center, x  = x_amp, image
        if pad_h or pad_w:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x        = F.pad(x, (0, pad_w, 0, pad_h), "reflect")
        fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center, x), 1)))
        fea_2 = self.lrelu(self.conv_first_2(fea_1))
        fea_3 = self.lrelu(self.conv_first_3(fea_2))

        fea       = self.feature_extraction(fea_3)
        fea_light = self.recon_trunk_light(fea)
        mask      = F.interpolate(_snr_map(x_center), size=fea.shape[-2:], mode="nearest")
        fea       = self.transformer(fea) * (1 - mask) + fea_light * mask   # SNR-based interaction

        out = torch.cat([self.recon_trunk(fea), fea_3], 1)
        out = torch.cat([self.lrelu(self.pixel_shuffle(self.upconv1(out))), fea_2], 1)
        out = torch.cat([self.lrelu(self.pixel_shuffle(self.upconv2(out))), fea_1], 1)
        out = self.conv_last(self.lrelu(self.HRconv(out))) + x
        return {"enhanced": out[:, :, :h, :w], "amplitude_enhanced": x_amp}
