"""Zero-IG: zero-shot, illumination-guided joint denoising and enhancement, fitted to each image.

Three small networks, fitted together to one low-light image ``x``: ``denoise_1`` removes noise
from the input (``L = x − D₁(x)``), the enhancer estimates an illumination map ``s`` from ``L``
(three residual conv–BN–ReLU blocks that **share one module**, then a sigmoid, clamped to
``[1e-4, 1]``), and the Retinex-style division ``x / s`` gives the enhanced image, which
``denoise_2`` cleans jointly with ``s`` (6 channels in and out). ``forward`` returns the denoised
enhanced image (``enhanced``) and the intermediate maps.

The loss (``ZeroIGLoss``) combines an illumination prior (a gamma-like target from the mean
luminance, an edge-aware smoothness and a TV term), Zero-Shot Noise2Noise residual and
consistency terms on pair-downsampled views (for both denoisers), color (blurred images),
illumination and texture-consistency terms, and a local-variance term. ``zero_ig_enhance`` fits a
fresh network to each image with the official recipe (Adam, lr 3e-4, weight decay 3e-4, gradient
norm clipped at 5, 2001 steps, BatchNorm in training mode), then runs it in eval mode.

The official repository also releases dataset-trained weights (LOL, LSRW-Huawei, LSRW-Nikon); its
README says they are *not* the models behind the paper's results and recommends fitting each image.
They load with ``strict=True`` (``load("zero_ig", weights=...)``) for direct inference; the model is
registered ``zero_shot=True``.

Kept from the official code because training depends on it: luminance uses weights
``0.299, 0.587, 0.144`` (not 0.114) on channels in B, G, R order; the edge-aware smoothness
converts to YCbCr by reshaping NCHW memory into rows of three values (which mixes neighboring
pixels rather than converting each one); the second texture-consistency term reuses the first
view's image; ``enhanced`` is clamped to ``[1e-4, 1]`` and the input gets ``+1e-4`` first.

Differences from the official code (``model.py``, ``loss.py``, ``utils.py``, ``train.py``,
``test.py``): the loss is rewritten (no code copied: the repository has no license); the 24
directional smoothness terms are 12 directions counted twice, as in the official sum; the loss is
per-sample (``reduction="mean"`` equals the official value for any batch); the Gaussian blur kernel
is built on the input's device instead of CUDA; ``Finetunemodel``'s non-strict loading is replaced
by strict loading. ``mon_old`` (``models/enhance/lle/zero_ig``) vendored the official network.

Examples:
    >>> import torch
    >>> out = ZeroIG().eval()(torch.rand(1, 3, 12, 16) * 0.3)
    >>> sorted(out)
    ['denoised_input', 'enhanced', 'enhanced_noisy', 'illumination']
    >>> y = zero_ig_enhance(torch.rand(1, 3, 12, 16) * 0.3, iterations=2, seed=0)
    >>> y.shape
    torch.Size([1, 3, 12, 16])

References:
    - Shi, Liu, Zhang, Tian, Xia, Fu, "ZERO-IG: Zero-Shot Illumination-Guided Joint Denoising and
      Adaptive Enhancement for Low-Light Images", CVPR 2024.
      https://openaccess.thecvf.com/content/CVPR2024/html/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.html
    - https://github.com/Doyle59217/ZeroIG (commit ``5af4b8a``; no license stated: read, not copied).
"""

__all__ = [
    "ZeroIG",
    "ZeroIGLoss",
    "ZeroIG_Weights",
    "zero_ig_enhance",
]

from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum
from mon.models.denoise.zs_n2n import pair_downsample
from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss

_REPO = "https://raw.githubusercontent.com/Doyle59217/ZeroIG/5af4b8aca6b114732f05d7fea70e0fbc874d63ab/weights/"
_NOTE  = "dataset-trained release; the official README says these are not the paper's models"


class ZeroIG_Weights(WeightsEnum):
    """Released Zero-IG weights (dataset-trained; fitting each image is the paper's method)."""

    LOL_V1 = Weights(
        path   = "lle/zero_ig/zero_ig_lol_v1.pt",
        name   = "Zero-IG, official weights/LOL.pt",
        url    = _REPO + "LOL.pt",
        sha256 = "716e66142f27d9f2227429f6d5c8c036f81d9429ba9fb8e1a871fc68dbb04c2a",
        meta   = {"source": "official weights/LOL.pt", "train": "lol_v1", "note": _NOTE},
    )
    LSRW_HUAWEI = Weights(
        path   = "lle/zero_ig/zero_ig_lsrw_huawei.pt",
        name   = "Zero-IG, official weights/LSRW-Huawei.pt",
        url    = _REPO + "LSRW-Huawei.pt",
        sha256 = "8b3b86eae59d1b0e98167200408b9d3cb7522730e204be64b2a8fb60bb2928cb",
        meta   = {"source": "official weights/LSRW-Huawei.pt", "train": "lsrw_huawei", "note": _NOTE},
    )
    LSRW_NIKON = Weights(
        path   = "lle/zero_ig/zero_ig_lsrw_nikon.pt",
        name   = "Zero-IG, official weights/LSRW-Nikon.pt",
        url    = _REPO + "LSRW-Nikon.pt",
        sha256 = "23c0e5e037312b5270edd925f3f8cefdc15c335eff3d9b8d8653f48ec99d7681",
        meta   = {"source": "official weights/LSRW-Nikon.pt", "train": "lsrw_nikon", "note": _NOTE},
    )
    DEFAULT = LOL_V1


# --- Networks ---

class _Denoiser(nn.Module):
    """Noise estimator: two 3×3 convs (LeakyReLU 0.2) and a 1×1 conv; ``x − D(x)`` is the denoised input."""

    def __init__(self, channels: int, embed: int = 48) -> None:
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(channels, embed, 3, padding=1)
        self.conv2 = nn.Conv2d(embed, embed, 3, padding=1)
        self.conv3 = nn.Conv2d(embed, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv3(self.act(self.conv2(self.act(self.conv1(x)))))
        return out


class _Enhancer(nn.Module):
    """Illumination estimator: in-conv, ``layers`` residual applications of ONE shared conv–BN–ReLU, sigmoid."""

    def __init__(self, layers: int = 3, channels: int = 64) -> None:
        super().__init__()
        self.in_conv  = nn.Sequential(nn.Conv2d(3, channels, 3, 1, 1), nn.ReLU())
        self.conv     = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks   = nn.ModuleList([self.conv] * layers)   # shared, as in the official code
        self.out_conv = nn.Sequential(nn.Conv2d(channels, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        fea = self.in_conv(x)
        for block in self.blocks:
            fea = fea + block(fea)
        return torch.clamp(self.out_conv(fea), 0.0001, 1)


def _enhance_init(module: nn.Module) -> None:
    """The official enhancer initialization: conv weights N(0, 0.02), biases 0; BN weights N(1, 0.02)."""
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0.0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)


class ZeroIG(nn.Module):
    """Zero-IG networks: ``denoise_1`` (input), ``enhance`` (illumination), ``denoise_2`` (joint, 6 channels).

    Args:
        layers: Applications of the shared enhancer block (official: 3).
        channels: Enhancer width (official: 64).
        embed: Width of both denoisers (official: 48).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W] (denoised, in [1e-4, 1]),
        "enhanced_noisy": x/s, "illumination": s, "denoised_input": L}`` (all ``[B, 3, H, W]``).
    """

    weights_enum: ClassVar[type[WeightsEnum]] = ZeroIG_Weights

    def __init__(self, layers: int = 3, channels: int = 64, embed: int = 48) -> None:
        super().__init__()
        self.enhance   = _Enhancer(layers, channels)
        self.denoise_1 = _Denoiser(3, embed)
        self.denoise_2 = _Denoiser(6, embed)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        eps = 1e-4
        x   = image + eps
        lo  = torch.clamp(x - self.denoise_1(x), eps, 1)
        s   = self.enhance(lo)
        h   = torch.clamp(x / s, eps, 1)
        hs  = torch.cat([h, s], 1)
        out = torch.clamp(hs.detach() - self.denoise_2(hs), eps, 1)
        return {"enhanced": out[:, :3], "enhanced_noisy": h, "illumination": s, "denoised_input": lo}


# --- Loss helpers ---

def _luma(x: Tensor) -> Tensor:
    """Official luminance: ``0.299·c₂ + 0.587·c₁ + 0.144·c₀`` (channel order and 0.144 as written there)."""
    return x[:, 2] * 0.299 + x[:, 1] * 0.587 + x[:, 0] * 0.144


def _gauss_kernel(size: int, nsig: float, channels: int, like: Tensor) -> Tensor:
    """Square Gaussian kernel from differences of the normal CDF, sqrt of the outer product, normalized."""
    interval = (2 * nsig + 1.0) / size
    t        = torch.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, size + 1, device=like.device)
    k1d      = torch.diff(0.5 * (1 + torch.erf(t / torch.sqrt(torch.tensor(2.0, device=like.device)))))
    k2d      = torch.sqrt(torch.outer(k1d, k1d))
    return (k2d / k2d.sum()).view(1, 1, size, size).repeat(channels, 1, 1, 1).to(like.dtype)


def _blur(x: Tensor) -> Tensor:
    """21×21 Gaussian (``nsig=1``), reflect border."""
    kernel = _gauss_kernel(21, 1, x.shape[1], x)
    return F.conv2d(F.pad(x, (10, 10, 10, 10), mode="reflect"), kernel, groups=x.shape[1])


def _patches(x: Tensor, size: int = 5, mode: str = "reflect") -> Tensor:
    """``[B, C, H, W]`` → ``[B, C, H, W, size, size]`` windows around each pixel."""
    p = size // 2
    x = F.pad(x, (p, p, p, p), mode=mode) if mode == "reflect" else F.pad(x, (p, p, p, p))
    return x.unfold(2, size, 1).unfold(3, size, 1)


def _local_mean(x: Tensor) -> Tensor:
    return _patches(x).mean(dim=(4, 5))


def _texture_agreement(a: Tensor, b: Tensor, c: float = 1e-5, threshold: float = 0.975) -> Tensor:
    """1 where the 5×5 local standard deviations agree: ``2σ₁σ₂ / (σ₁² + σ₂² + c) > threshold``."""
    def local_std(x: Tensor) -> Tensor:
        w = _patches(_luma_gray(x))
        return torch.sqrt(((w - w.mean(dim=(4, 5), keepdim=True)) ** 2).mean(dim=(4, 5)) + 1e-9)

    sa, sb = local_std(a), local_std(b)
    ratio  = 2 * sa * sb / (sa**2 + sb**2 + c)
    return (ratio > threshold).to(a.dtype)


def _luma_gray(x: Tensor) -> Tensor:
    """Official grayscale for the texture map: ``0.144·c₀ + 0.587·c₁ + 0.299·c₂``, ``[B, 1, H, W]``."""
    return (0.144 * x[:, 0] + 0.5870 * x[:, 1] + 0.299 * x[:, 2]).unsqueeze(1)


def _local_variance(x: Tensor) -> Tensor:
    """Mean over each 5×5 window of the squared deviation from the local 5×5 mean (zero border)."""
    mean = F.avg_pool2d(x, 5, 1, 2)
    return ((_patches(x, mode="zeros") - _patches(mean, mode="zeros")) ** 2).mean(dim=(4, 5))


def _tv(x: Tensor) -> Tensor:
    """Squared-difference total variation per sample (Zero-DCE's ``L_TV``)."""
    h, w = x.shape[-2:]
    h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).flatten(1).sum(1)
    w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).flatten(1).sum(1)
    return 2 * (h_tv / ((h - 1) * w) + w_tv / (h * (w - 1)))


_SMOOTH_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2),
                      (2, 1), (2, -1), (1, 2), (1, -2), (2, 2), (2, -2))


def _pairs(x: Tensor, dy: int, dx: int) -> tuple[Tensor, Tensor]:
    """Pixels ``p`` and ``p + (dy, dx)`` over the region where both exist."""
    h, w = x.shape[-2:]
    a = x[:, :, : h - dy, max(0, -dx) : w - max(0, dx)]
    b = x[:, :, dy:, max(0, dx) : w - max(0, -dx)]
    return a, b


def _smoothness(guide: Tensor, output: Tensor, sigma: float = 10.0) -> Tensor:
    """Edge-aware smoothness of ``output`` weighted by ``guide`` (per sample)."""
    b, _, h, w = guide.shape
    mat  = guide.new_tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]])
    bias = guide.new_tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0])
    ycc  = (guide.contiguous().view(-1, 3) @ mat + bias).view(b, 3, h, w)  # official memory reshape
    k    = -1.0 / (2 * sigma * sigma)
    total = output.new_zeros(b)
    for dy, dx in _SMOOTH_DIRECTIONS:
        ga, gb = _pairs(ycc, dy, dx)
        oa, ob = _pairs(output, dy, dx)
        weight = torch.exp((ga - gb).pow(2).sum(1, keepdim=True) * k)
        term   = (weight * (oa - ob).abs().sum(1, keepdim=True)).flatten(1).mean(1)
        total  = total + term + term                                    # both orders of each pair, as official
    return total


def _mse(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).pow(2).flatten(1).mean(1)


class ZeroIGLoss(nn.Module):
    """The Zero-IG objective for ``model`` on ``image`` (``loss(model, image)``; builds the training graph).

    Args:
        reduction: ``"mean"`` (equals the official loss), ``"sum"``, or ``"none"`` (per-sample ``[B]``).
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        check_reduction(reduction)
        self.reduction = reduction

    def forward(self, model: ZeroIG, image: Tensor) -> Tensor:
        eps = 1e-4
        x   = image + eps

        # Training graph (the official Network.forward).
        l11, l12   = pair_downsample(x)
        l_pred1    = l11 - model.denoise_1(l11)
        l_pred2    = l12 - model.denoise_1(l12)
        lo         = torch.clamp(x - model.denoise_1(x), eps, 1)
        s2         = model.enhance(lo.detach())
        s21, s22   = pair_downsample(s2)
        h2         = torch.clamp(x / s2, eps, 1)
        h11        = torch.clamp(l11 / s21, eps, 1)
        h12        = torch.clamp(l12 / s22, eps, 1)

        def joint(h: Tensor, s: Tensor) -> Tensor:
            hs = torch.cat([h, s], 1)
            return torch.clamp(hs.detach() - model.denoise_2(hs), eps, 1)

        h3_pred    = joint(h11, s21)
        h4_pred    = joint(h12, s22)
        h5_pred    = joint(h2, s2)
        h3, s3     = h5_pred[:, :3], h5_pred[:, 3:]
        h3_d1, h3_d2 = pair_downsample(h3)
        agree      = _texture_agreement(h3_d1, h3_d2)
        h2_blur    = _blur(torch.clamp(lo / s2, 0, 1))
        h3_blur    = _blur(h3)

        # Illumination prior: target exposure from the mean luminance of the denoised input.
        tiny       = 1e-9
        lo_d       = lo.detach()
        factor     = torch.clamp(0.5 / (_luma(lo_d).mean(dim=(1, 2)) + tiny), 1, 25).view(-1, 1, 1, 1)
        ratio      = torch.pow(0.7, -factor) / factor
        target_s   = torch.clamp(torch.pow(lo_d * factor, factor) * ratio, tiny, 1)
        normalized = torch.clamp(lo_d / s2, tiny, 0.8)
        brightened = torch.clamp(lo_d * factor, tiny, 1)

        loss  = _mse(s2, target_s) * 700
        loss  = loss + _mse(normalized, brightened) * 1000
        loss  = loss + _smoothness(lo_d, s2) * 5
        loss  = loss + _tv(s2) * 1600
        # Residual and consistency (denoise_1), on views of the input (+1e-9, as official).
        v1, v2 = pair_downsample(image + tiny)
        loss  = loss + _mse(v1, l_pred2) * 1000
        loss  = loss + _mse(v2, l_pred1) * 1000
        d1, d2 = pair_downsample(lo)
        loss  = loss + _mse(l_pred1, d1) * 1000
        loss  = loss + _mse(l_pred2, d2) * 1000
        # Residual and consistency (denoise_2).
        loss  = loss + _mse(h3_pred, torch.cat([h12.detach(), s22.detach()], 1)) * 1000
        loss  = loss + _mse(h4_pred, torch.cat([h11.detach(), s21.detach()], 1)) * 1000
        loss  = loss + _mse(h3_pred[:, :3], h3_d1) * 1000
        loss  = loss + _mse(h4_pred[:, :3], h3_d2) * 1000
        # Color, illumination.
        loss  = loss + _mse(h2_blur.detach(), h3_blur) * 10000
        loss  = loss + _mse(s2.detach(), s3) * 1000
        # Texture consistency (the second term reuses the first view, as official).
        loss  = loss + _mse(h3_d1, (1 - agree) * _local_mean(h3_d1) + h3_d1 * agree) * 10000
        loss  = loss + _mse(h3_d2, (1 - agree) * _local_mean(h3_d2) + h3_d1 * agree) * 10000
        # Local variance of the removed noise vs the noisy enhanced image.
        loss  = loss + _mse(_local_variance(h2), _local_variance(h3 - h2)) * 1000
        return reduce_loss(loss, self.reduction)


# --- Zero-shot fitting ---

def zero_ig_enhance(
    image       : Tensor,
    *,
    iterations  : int   = 2001,
    lr          : float = 3e-4,
    weight_decay: float = 3e-4,
    clip_norm   : float = 5.0,
    seed        : int | None = None,
) -> Tensor:
    """Enhance and denoise each image by fitting a fresh ``ZeroIG`` to it (works under ``torch.inference_mode``).

    Official recipe: enhancer re-initialized (``N(0, 0.02)``), Adam (``lr``, ``weight_decay``), gradient norm
    clipped at ``clip_norm``, ``iterations`` steps with BatchNorm in training mode, then one eval-mode forward.

    Args:
        image: ``[B, 3, H, W]`` or ``[3, H, W]`` float in ``[0, 1]`` at native resolution (``H, W`` even).
        iterations: Adam steps per image (official: 2001 epochs over one image).
        lr, weight_decay: Adam settings (official: 3e-4 both).
        clip_norm: Gradient-norm clip (official: 5).
        seed: Seed for the initialization (each image starts from the same weights; the global RNG is left
            untouched); ``None`` uses the global RNG. The official script seeds 2.

    Returns:
        The enhanced, denoised images (the model's ``enhanced``), same shape as ``image``.

    Raises:
        ValueError: For other shapes, odd sizes, or ``iterations < 1``.
    """
    if image.ndim not in (3, 4):
        raise ValueError(f"image must be [B, 3, H, W] or [3, H, W], got shape {tuple(image.shape)}")
    if image.shape[-1] % 2 or image.shape[-2] % 2:
        raise ValueError(f"zero_ig_enhance needs even H and W, got {tuple(image.shape[-2:])}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    batch   = image if image.ndim == 4 else image[None]
    loss_fn = ZeroIGLoss()
    outputs = []
    with torch.inference_mode(False), torch.enable_grad():
        for i in range(batch.shape[0]):
            x = batch[i : i + 1].clone()  # clone: inference tensors can't be saved for backward
            with torch.random.fork_rng(devices=[], enabled=seed is not None):
                if seed is not None:
                    torch.manual_seed(seed)
                model = ZeroIG()  # initialized on CPU: same weights on any device
                for part in (model.enhance.in_conv, model.enhance.conv, model.enhance.out_conv):
                    part.apply(_enhance_init)
            model = model.to(device=x.device, dtype=x.dtype).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
            for _ in range(iterations):
                optimizer.zero_grad()
                loss = loss_fn(model, x)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            with torch.no_grad():
                outputs.append(model.eval()(x)["enhanced"])
    out = torch.cat(outputs)
    return out if image.ndim == 4 else out[0]
