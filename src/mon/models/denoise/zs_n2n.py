"""Zero-Shot Noise2Noise (ZS-N2N): denoise one image by training a tiny network on it.

Each noisy image is split into two half-resolution images by averaging diagonal pixel pairs
(``pair_downsample``); with pixel-independent noise they are two noisy views of nearly the same
scene, so a network trained to map one to the other learns to remove the noise (Noise2Noise),
plus a consistency term. The network (two 3×3 convolutions and one 1×1, 48 channels) predicts
the noise; the result is ``clamp(x - f(x), 0, 1)``. No training data and no released weights:
``zs_n2n_denoise`` fits a fresh network to every image (paper defaults: Adam, lr 1e-3, 3000
steps, learning rate halved every 1000).

Cost: each step runs the network on the image and both half-resolution views, forward and
backward (~45 GFLOP for 400×600): 3000 steps took 3.5–6 min per image on an Apple M1 GPU (MPS),
about 5× longer on its CPU. Use fewer iterations for previews.

The noise must be pixel-independent at the network's input resolution: never resize the noisy
image first (interpolation correlates neighboring noise values; ``mon_old``'s predictor resized
to 512 px).

Differences from ``mon_old`` (``models/restore/denoise/zs_n2n``), which otherwise matched the
paper's code:

- The saved initial weights were ``state_dict()`` tensors that share storage with the live
  weights, so ``reset_weights=True`` did nothing: every image after the first started from the
  previous image's trained network. ``zs_n2n_denoise`` builds a new network per image.
- A batch was fitted with one shared network (the loss averaged over the batch); now one
  network per image.
- ``forward`` without fitting returned an untrained network's output; fitting is now explicit.
- Dropped IZS-N2N (no official code to verify against), ``add_noise``, ``forward_train``,
  ``benchmark``, and the ``device`` argument.

Examples:
    >>> import torch
    >>> g = torch.Generator().manual_seed(0)
    >>> clean = torch.linspace(0.2, 0.8, 32).expand(1, 3, 32, 32)
    >>> noisy = (clean + 0.1 * torch.randn(1, 3, 32, 32, generator=g)).clamp(0, 1)
    >>> denoised = zs_n2n_denoise(noisy, iterations=200, seed=0)
    >>> bool((denoised - clean).pow(2).mean() < (noisy - clean).pow(2).mean())
    True

References:
    - Mansour, Heckel, "Zero-Shot Noise2Noise: Efficient Image Denoising without any Data",
      CVPR 2023. https://arxiv.org/abs/2303.11253
    - Official code (Colab): https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b
"""

__all__ = [
    "ZSN2N",
    "ZSN2NLoss",
    "pair_downsample",
    "zs_n2n_denoise",
]

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss


def pair_downsample(image: Tensor) -> tuple[Tensor, Tensor]:
    """Split ``[..., H, W]`` into two ``[..., H//2, W//2]`` images from the diagonals of each 2×2 block.

    The first averages the top-right and bottom-left pixels, the second the top-left and
    bottom-right (the paper's ``filter1`` and ``filter2``). An odd last row or column is dropped.

    Raises:
        ValueError: If ``H`` or ``W`` is smaller than 2.
    """
    h, w = image.shape[-2:]
    if h < 2 or w < 2:
        raise ValueError(f"image must be at least 2×2, got {h}×{w}")
    x = image[..., : h - h % 2, : w - w % 2]
    tl, tr = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    bl, br = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (tr + bl) / 2, (tl + br) / 2


class ZSN2N(nn.Module):
    """The ZS-N2N noise predictor: conv3×3 → LeakyReLU → conv3×3 → LeakyReLU → conv1×1.

    Untrained it is meaningless: fit it to one image (``zs_n2n_denoise``, or ``ZSN2NLoss`` in
    your own loop). Parameter names match the official code.

    Args:
        channels: Image channels.
        hidden_dim: Feature channels (48 in the paper).

    Shapes:
        ``forward(noisy [B, C, H, W] in [0, 1])`` → ``{"restored": clamp(noisy - noise, 0, 1),
        "noise": [B, C, H, W]}``.
    """

    def __init__(self, channels: int = 3, hidden_dim: int = 48) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, channels, 1)

    def predict_noise(self, noisy: Tensor) -> Tensor:
        """The predicted noise ``f(noisy)`` (unclipped)."""
        y = F.leaky_relu(self.conv1(noisy), 0.2)
        y = F.leaky_relu(self.conv2(y), 0.2)
        out: Tensor = self.conv3(y)
        return out

    def forward(self, noisy: Tensor) -> dict[str, Tensor]:
        noise = self.predict_noise(noisy)
        return {"restored": (noisy - noise).clamp(0, 1), "noise": noise}


class ZSN2NLoss(nn.Module):
    """The ZS-N2N objective: residual (Noise2Noise between the pair) + consistency, both MSE.

    ``loss(model, noisy)`` needs the model itself: it is applied to the downsampled pair and to
    the full image. With ``reduction="mean"`` the value equals the official loss.

    Args:
        reduction: ``"mean"``, ``"sum"``, or ``"none"`` (per-sample ``[B]``).
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        check_reduction(reduction)
        self.reduction = reduction

    def forward(self, model: ZSN2N, noisy: Tensor) -> Tensor:
        def mse(a: Tensor, b: Tensor) -> Tensor:
            return (a - b).pow(2).flatten(1).mean(dim=1)

        noisy1, noisy2 = pair_downsample(noisy)
        pred1 = noisy1 - model.predict_noise(noisy1)
        pred2 = noisy2 - model.predict_noise(noisy2)
        residual = (mse(noisy1, pred2) + mse(noisy2, pred1)) / 2
        denoised1, denoised2 = pair_downsample(noisy - model.predict_noise(noisy))
        consistency = (mse(pred1, denoised1) + mse(pred2, denoised2)) / 2
        return reduce_loss(residual + consistency, self.reduction)


def zs_n2n_denoise(
    image     : Tensor,
    *,
    iterations: int   = 3000,
    lr        : float = 1e-3,
    step_size : int   = 1000,
    gamma     : float = 0.5,
    hidden_dim: int   = 48,
    seed      : int | None = None,
) -> Tensor:
    """Denoise each image by fitting a fresh ``ZSN2N`` to it (works under ``torch.inference_mode``).

    Args:
        image: ``[B, C, H, W]`` or ``[C, H, W]`` float in ``[0, 1]`` at native resolution.
        iterations: Adam steps per image.
        lr: Initial learning rate.
        step_size, gamma: Multiply the learning rate by ``gamma`` every ``step_size`` steps.
        hidden_dim: Network width.
        seed: Seed for the network initialization (each image starts from the same weights; the
            global RNG is left untouched); ``None`` uses the global RNG.

    Returns:
        The denoised images, same shape as ``image``, in ``[0, 1]``.

    Raises:
        ValueError: For other shapes or ``iterations < 1``.
    """
    if image.ndim not in (3, 4):
        raise ValueError(f"image must be [B, C, H, W] or [C, H, W], got shape {tuple(image.shape)}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    batch = image if image.ndim == 4 else image[None]
    loss_fn = ZSN2NLoss()
    outputs = []
    with torch.inference_mode(False), torch.enable_grad():
        for i in range(batch.shape[0]):
            noisy = batch[i : i + 1].clone()  # clone: inference tensors can't be saved for backward
            with torch.random.fork_rng(devices=[], enabled=seed is not None):
                if seed is not None:
                    torch.manual_seed(seed)
                model = ZSN2N(noisy.shape[1], hidden_dim)  # initialized on CPU: same weights on any device
            model = model.to(device=noisy.device, dtype=noisy.dtype).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            for _ in range(iterations):
                loss = loss_fn(model, noisy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            with torch.no_grad():
                outputs.append(model.eval()(noisy)["restored"])
    out = torch.cat(outputs)
    return out if image.ndim == 4 else out[0]
