"""CoLIE: low-light enhancement by fitting a neural implicit illumination field to each image.

The HSV Value channel is downscaled to ``down_size × down_size``; a SIREN MLP maps each pixel's
coordinates and its ``window × window`` neighborhood to a residual illumination, trained for
``epochs`` Adam steps on zero-reference losses (``CoLIELoss``). The enhanced Value
``V / (V + residual + 1e-4)`` is upscaled with a fast guided filter (guide: V), put back into
the image keeping hue and saturation, and the result is divided by its maximum. No training
data and no weights: ``colie_enhance`` fits a fresh network to every image.

Defaults follow the paper (window 7, ``L = 0.5``, α/β/γ/δ = 1/20/8/5, lr 1e-5, 100 epochs,
256 px, 256 channels) and the official code (weight decay 3e-4; input and output divided by
their maximum; nearest-neighbor downscaling to a square, which distorts the aspect ratio until
the guided filter restores the full size). The official CLI defaults to ``window=1``; its
notebook uses ``L=0.1``, γ = 9.

Measured (2026-11-18, LOL-v1 test, 8-bit outputs, seed 0): 13.784 dB mean PSNR with these
defaults, 9.925 dB without the input/output rescaling; ~23 s per image on an Apple M1 GPU.

Differences from ``mon_old`` (``models/enhance/lle/colie``), whose network, losses, optimizer,
and guided filter matched the official code (low-resolution results agree to 6e-8):

- The "best epoch" was saved as ``state_dict()`` tensors that alias the live weights, so the
  final weights were always used (the official code uses the final weights too; no selection
  now).
- No input/output rescaling by the maximum (outputs were clamped instead), so results were
  darker than the official script's; ``down_size`` was tied to ``hidden_dim``; one image per
  call; ``forward_patch`` (use ``mon.nn.tiling.tiled_forward``) and ``benchmark`` removed.

Examples:
    >>> import torch
    >>> x = torch.rand(1, 3, 40, 52) * 0.2
    >>> y = colie_enhance(x, down_size=32, hidden_dim=32, epochs=20, seed=0)
    >>> y.shape, bool(y.mean() > x.mean()), float(y.max())
    (torch.Size([1, 3, 40, 52]), True, 1.0)

References:
    - Chobola, Liu, Zhang, Schnabel, Peng, "Fast Context-Based Low-Light Image Enhancement via
      Neural Implicit Representations", ECCV 2024. https://arxiv.org/abs/2407.12511
    - https://github.com/ctom2/colie (Apache-2.0)
    - Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
      (SIREN), NeurIPS 2020.
    - Wu et al., "Fast End-to-End Trainable Guided Filter", CVPR 2018 (``fast_guided_filter``).
"""

__all__ = [
    "CoLIE",
    "CoLIELoss",
    "colie_enhance",
    "fast_guided_filter",
    "fit_illumination",
    "replace_value",
]

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss
from mon.nn.losses.zero_reference import IlluminationSmoothnessLoss


# --- Network ---

class _SirenLayer(nn.Module):
    """``sin(w0 · linear(x))``, or ``sigmoid(linear(x))`` for the last layer (SIREN initialization)."""

    def __init__(self, in_features: int, out_features: int, w0: float = 30.0, is_first: bool = False,
                 is_last: bool = False) -> None:
        super().__init__()
        self.w0      = w0
        self.is_last = is_last
        self.linear  = nn.Linear(in_features, out_features)
        if not is_last:
            bound = 1 / in_features if is_first else math.sqrt(6 / in_features) / w0
            with torch.no_grad():
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        return torch.sigmoid(y) if self.is_last else torch.sin(self.w0 * y)


class CoLIE(nn.Module):
    """The CoLIE implicit network (official ``INF``): two SIREN branches, concatenated, then an output branch.

    Untrained it is meaningless: ``colie_enhance`` fits one per image. Layers are created in the
    official order, so a given seed gives the official initialization.

    Args:
        window: Context window size (``window²`` patch features per pixel).
        hidden_dim: Hidden channels (256 in the paper); each input branch outputs half.
        num_layers, add_layer: Depth parameters of the official ``INF`` (``1 <= add_layer <= num_layers - 2``).

    Shapes:
        ``forward(patches [..., window²], coords [..., 2])`` → residual illumination ``[..., 1]`` in ``(0, 1)``.
    """

    def __init__(self, window: int = 7, hidden_dim: int = 256, num_layers: int = 4, add_layer: int = 2) -> None:
        super().__init__()
        self.window = window
        patch_layers   = [_SirenLayer(window**2, hidden_dim, is_first=True)]
        spatial_layers = [_SirenLayer(2, hidden_dim, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_layers.append(_SirenLayer(hidden_dim, hidden_dim))
            spatial_layers.append(_SirenLayer(hidden_dim, hidden_dim))
        patch_layers.append(_SirenLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(_SirenLayer(hidden_dim, hidden_dim // 2))
        output_layers = [_SirenLayer(hidden_dim, hidden_dim) for _ in range(add_layer, num_layers - 1)]
        output_layers.append(_SirenLayer(hidden_dim, 1, is_last=True))
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

    def forward(self, patches: Tensor, coords: Tensor) -> Tensor:
        out: Tensor = self.output_net(torch.cat([self.patch_net(patches), self.spatial_net(coords)], dim=-1))
        return out


# --- Loss ---

class CoLIELoss(nn.Module):
    """CoLIE objective on the low-resolution Value ``v`` and illumination ``i = v + residual``.

    ``α·MSE(i, v) + β·TV(i) + γ·mean((avgpool16(i)^½ - L)²) + δ·mean(v / (i + 1e-4))``: fidelity,
    smoothness (the Zero-DCE total variation), exposure, and sparsity of the enhanced Value. With
    ``reduction="mean"`` the value equals the official loss.

    Args:
        alpha, beta, gamma, delta: Term weights (paper: 1, 20, 8, 5).
        exposure: The "optimally-intense threshold" ``L`` (lower is brighter).
        reduction: ``"mean"``, ``"sum"``, or ``"none"`` (per-sample ``[B]``).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 20.0, gamma: float = 8.0, delta: float = 5.0,
                 exposure: float = 0.5, reduction: Reduction = "mean") -> None:
        super().__init__()
        check_reduction(reduction)
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.exposure  = exposure
        self.reduction = reduction
        self.tv        = IlluminationSmoothnessLoss(reduction="none")  # Zero-DCE's L_TV

    def forward(self, illumination: Tensor, value: Tensor) -> Tensor:
        """Args: ``illumination`` and ``value``, both ``[B, 1, h, w]`` with ``h, w >= 16``."""
        fidelity = (illumination - value).pow(2).flatten(1).mean(1)
        smooth   = self.tv(illumination)
        exposure = (F.avg_pool2d(illumination, 16).sqrt() - self.exposure).pow(2).flatten(1).mean(1)
        sparsity = (value / (illumination + 1e-4)).flatten(1).mean(1)
        total = self.alpha * fidelity + self.beta * smooth + self.gamma * exposure + self.delta * sparsity
        return reduce_loss(total, self.reduction)


# --- Guided filter and color ---

def _box_sum(x: Tensor, r: int) -> Tensor:
    """Sum over a ``(2r+1)²`` window truncated at the borders (official cumsum box filter)."""
    def diff(c: Tensor, dim: int) -> Tensor:
        n = c.shape[dim]
        left   = c.narrow(dim, r, r + 1)
        middle = c.narrow(dim, 2 * r + 1, n - 2 * r - 1) - c.narrow(dim, 0, n - 2 * r - 1)
        right  = c.narrow(dim, n - 1, 1) - c.narrow(dim, n - 2 * r - 1, r)
        return torch.cat([left, middle, right], dim=dim)
    return diff(diff(x.cumsum(dim=2), 2).cumsum(dim=3), 3)


def fast_guided_filter(lr_guide: Tensor, lr_target: Tensor, hr_guide: Tensor, r: int = 1, eps: float = 1e-8) -> Tensor:
    """Fit ``target ≈ A·guide + b`` locally at low resolution; apply the upsampled ``A, b`` to ``hr_guide``.

    Same computation as the official ``FastGuidedFilter`` (cumsum box filter; bilinear,
    ``align_corners=True`` upsampling of ``A`` and ``b``).

    Args:
        lr_guide, lr_target: ``[B, C, h, w]`` with ``h, w > 2r + 1``.
        hr_guide: ``[B, C, H, W]``.
        r: Box radius.
        eps: Regularization of the local variance.
    """
    h, w = lr_guide.shape[-2:]
    if min(h, w) <= 2 * r + 1:
        raise ValueError(f"low-resolution size {h}×{w} must exceed 2r + 1 = {2 * r + 1}")
    n = _box_sum(lr_guide.new_ones(1, 1, h, w), r)
    mean_x = _box_sum(lr_guide, r) / n
    mean_y = _box_sum(lr_target, r) / n
    cov_xy = _box_sum(lr_guide * lr_target, r) / n - mean_x * mean_y
    var_x  = _box_sum(lr_guide * lr_guide, r) / n - mean_x * mean_x
    a = cov_xy / (var_x + eps)
    b = mean_y - a * mean_x
    size = hr_guide.shape[-2:]
    a_hr = F.interpolate(a, size, mode="bilinear", align_corners=True)
    b_hr = F.interpolate(b, size, mode="bilinear", align_corners=True)
    return a_hr * hr_guide + b_hr


def replace_value(rgb: Tensor, value: Tensor) -> Tensor:
    """Set the HSV Value (``max`` over channels) of ``rgb [B, 3, H, W]`` to ``value [B, 1, H, W]``, keeping hue and saturation.

    RGB is linear in V for fixed H and S, so this scales the pixel by ``value / V``; black pixels
    (no hue or saturation) become gray ``value``. Equals an RGB→HSV→RGB round trip.
    """
    v = rgb.amax(dim=1, keepdim=True)
    pos = v > 0
    return torch.where(pos, rgb * (value / torch.where(pos, v, 1.0)), value.expand_as(rgb))


# --- Per-image fitting ---

def _patches(v: Tensor, window: int) -> Tensor:
    """``[1, 1, h, w]`` → ``[h, w, window²]``: reflect-padded neighborhoods, row-major (official ``get_patches``)."""
    p = window // 2
    padded = F.pad(v, (p, p, p, p), mode="reflect") if p else v
    cols = F.unfold(padded, window)  # [1, window², h·w]
    return cols[0].T.reshape(*v.shape[-2:], window**2)


def _coords(n: int, device: torch.device) -> Tensor:
    """``[n, n, 2]`` grid with ``coords[i, j] = (j, i) / (n - 1)`` (official ``get_coords``)."""
    lin = torch.linspace(0, 1, n, dtype=torch.float64).float().to(device)
    xs, ys = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([xs, ys], dim=-1)


def fit_illumination(
    value       : Tensor,
    *,
    window      : int   = 7,
    hidden_dim  : int   = 256,
    epochs      : int   = 100,
    lr          : float = 1e-5,
    weight_decay: float = 3e-4,
    loss        : CoLIELoss | None = None,
    seed        : int | None = None,
) -> Tensor:
    """Fit a fresh ``CoLIE`` to each low-resolution Value map and return its illumination ``V + residual``.

    Args:
        value: ``[B, 1, n, n]`` low-resolution Value (square, ``n >= 16``).
        window, hidden_dim: Network shape.
        epochs, lr, weight_decay: Adam steps and settings (official: 100, 1e-5, 3e-4).
        loss: Objective; defaults to ``CoLIELoss()``.
        seed: Seed for the network initialization (the global RNG is left untouched).

    Returns:
        ``[B, 1, n, n]`` illumination (detached).
    """
    n = value.shape[-1]
    if value.ndim != 4 or value.shape[1] != 1 or value.shape[-2] != n or n < 16:
        raise ValueError(f"value must be [B, 1, n, n] with n >= 16, got shape {tuple(value.shape)}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    loss_fn = loss or CoLIELoss()
    out = []
    with torch.inference_mode(False), torch.enable_grad():
        coords = _coords(n, value.device)  # created here: inference tensors can't be saved for backward
        for k in range(value.shape[0]):
            v = value[k : k + 1].clone()  # clone: inference tensors can't be saved for backward
            patches = _patches(v, window)
            with torch.random.fork_rng(devices=[], enabled=seed is not None):
                if seed is not None:
                    torch.manual_seed(seed)
                net = CoLIE(window=window, hidden_dim=hidden_dim)  # CPU init: same weights on any device
            net = net.to(device=v.device, dtype=v.dtype).train()
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
            for _ in range(epochs):
                optimizer.zero_grad()
                illumination = net(patches, coords).view(1, 1, n, n) + v
                loss_fn(illumination, v).backward()
                optimizer.step()
            with torch.no_grad():
                out.append(net(patches, coords).view(1, 1, n, n) + v)
    return torch.cat(out)


def _peak(x: Tensor) -> Tensor:
    """Per-image maximum ``[B, 1, 1, 1]`` (at least 1e-12)."""
    return x.flatten(1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)


def colie_enhance(
    image        : Tensor,
    *,
    window       : int   = 7,
    down_size    : int   = 256,
    hidden_dim   : int   = 256,
    epochs       : int   = 100,
    lr           : float = 1e-5,
    weight_decay : float = 3e-4,
    exposure     : float = 0.5,
    alpha        : float = 1.0,
    beta         : float = 20.0,
    gamma        : float = 8.0,
    delta        : float = 5.0,
    rescale_input : bool = True,
    rescale_output: bool = True,
    seed         : int | None = None,
) -> Tensor:
    """Enhance each image with CoLIE (works under ``torch.inference_mode``).

    Args:
        image: ``[B, 3, H, W]`` or ``[3, H, W]`` RGB in ``[0, 1]``.
        window, down_size, hidden_dim, epochs, lr, weight_decay: See ``fit_illumination``;
            ``down_size`` is the side of the square working resolution (>= 16).
        exposure, alpha, beta, gamma, delta: ``CoLIELoss`` parameters (``exposure`` is ``L``).
        rescale_input: Divide each image by its maximum first (official ``get_image``).
        rescale_output: Divide each result by its maximum (official script); else clamp to ``[0, 1]``.
        seed: Seed for each network's initialization.

    Returns:
        Enhanced images, same shape as ``image``.
    """
    if image.ndim not in (3, 4) or image.shape[-3] != 3:
        raise ValueError(f"image must be [B, 3, H, W] or [3, H, W], got shape {tuple(image.shape)}")
    batch = image if image.ndim == 4 else image[None]
    if rescale_input:
        batch = batch / _peak(batch)
    value    = batch.amax(dim=1, keepdim=True)
    value_lr = F.interpolate(value, size=(down_size, down_size))  # nearest, as the official code
    loss = CoLIELoss(alpha, beta, gamma, delta, exposure)
    illumination = fit_illumination(value_lr, window=window, hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                                    weight_decay=weight_decay, loss=loss, seed=seed)
    with torch.no_grad():
        enhanced_lr = value_lr / (illumination + 1e-4)
        enhanced_v  = fast_guided_filter(value_lr, enhanced_lr, value).clamp(0, 1)
        out = replace_value(batch, enhanced_v)
        out = out / _peak(out) if rescale_output else out.clamp(0, 1)
    return out if image.ndim == 4 else out[0]
