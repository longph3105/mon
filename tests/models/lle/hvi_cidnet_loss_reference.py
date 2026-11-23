"""Reference values for the HVI-CIDNet loss parity test (no ``mon`` imports).

``OFFICIAL_LOSS`` in ``test_hvi_cidnet.py`` comes from the official loss
(https://github.com/Fediory/HVI-CIDNet @ ``eb43d7d``, MIT): ``loss/losses.py`` ``L1Loss``, ``SSIM``,
``EdgeLoss``, ``PerceptualLoss`` (VGG-19 ``conv1_2``/``conv2_2``/``conv3_4``/``conv4_4``, MSE) built as
``train.py``'s ``init_loss`` with the ``data/options.py`` defaults (L1 1, SSIM 0.5, edge 50, perceptual
0.01, HVI 1), combined as ``train.py``'s training step on ``net/HVI_transform.py``'s ``RGB_HVI.HVIT``
(``density_k`` = ``DENSITY_K``), on deterministic images, with the gradient statistics of the prediction and
of ``density_k``. ``Tensor.cuda`` is patched to a no-op (the code hard-codes CUDA).

The VGG-19 is torchvision's architecture with seeded random weights (``torch.manual_seed(VGG_SEED)``, then
``vgg19()``), as ``HVICIDNetLoss(pretrained_backbone=False)`` builds it after the same seed: the parity
covers how features are taken and compared, not the ImageNet weights (a 548 MB download).

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/models/lle/hvi_cidnet_loss_reference.py <checkout>
"""

import pprint
import sys
from pathlib import Path
from typing import Any

import torch

STATS_INDICES = (0, 7, 101, 523, -1)
SHAPE = (2, 3, 24, 32)
VGG_SEED, DENSITY_K = 0, 0.2


def field(phase: float, low: float, high: float) -> torch.Tensor:
    """Deterministic ``[B, 3, H, W]`` RGB values in ``[low, high]`` (no RNG)."""
    b, c, h, w = SHAPE
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = torch.empty(b, c, h, w, dtype=torch.float64)
    for n in range(b):
        for k in range(c):
            wave = torch.sin(0.37 * (k + 1) * i + 0.23 * (n + 1) * j + phase) * torch.cos(0.13 * j - 0.29 * i + k)
            out[n, k] = low + (high - low) * (1 + wave) / 2
    return out.float()


def inputs() -> dict[str, torch.Tensor]:
    """A prediction and its target (both RGB in ``[0, 1]``)."""
    return {"pred": field(0.0, 0.05, 0.9), "target": field(1.5, 0.1, 1.0)}


def stats(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().double().flatten()
    return {"sum": flat.sum().item(), "sumsq": (flat**2).sum().item(),
            "values": [flat[k % flat.numel()].item() for k in STATS_INDICES]}


def main(checkout: str) -> None:
    sys.path.insert(0, str(Path(checkout).resolve()))
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    from torchvision.models import vgg as tv_vgg

    original = tv_vgg.vgg19

    def seeded_vgg19(pretrained: bool = False, **kwargs: Any) -> torch.nn.Module:  # no ImageNet download
        torch.manual_seed(VGG_SEED)
        return original(weights=None)

    tv_vgg.vgg19 = seeded_vgg19
    from loss.losses import SSIM, EdgeLoss, L1Loss, PerceptualLoss  # official code
    from net.HVI_transform import RGB_HVI  # official code

    L1_loss = L1Loss(loss_weight=1.0, reduction="mean")
    D_loss  = SSIM(weight=0.5)
    E_loss  = EdgeLoss(loss_weight=50.0)
    P_loss  = PerceptualLoss({"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1}, perceptual_weight=1.0,
                             criterion="mse")
    P_weight, HVI_weight = 1e-2, 1.0
    trans = RGB_HVI()
    with torch.no_grad():
        trans.density_k.fill_(DENSITY_K)
    x = inputs()
    output_rgb, gt_rgb = x["pred"].requires_grad_(True), x["target"]
    output_hvi, gt_hvi = trans.HVIT(output_rgb), trans.HVIT(gt_rgb)
    loss_hvi = (L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi)
                + P_weight * P_loss(output_hvi, gt_hvi)[0])
    loss_rgb = (L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb)
                + P_weight * P_loss(output_rgb, gt_rgb)[0])
    loss = loss_rgb + HVI_weight * loss_hvi
    g_pred, g_k = torch.autograd.grad(loss, [output_rgb, trans.density_k])
    result = {"loss": loss.item(), "loss_rgb": loss_rgb.item(), "loss_hvi": loss_hvi.item(),
              "grad_pred": stats(g_pred), "grad_density_k": g_k.item()}
    print(f"# Generated from the official HVI-CIDNet loss with torch {torch.__version__}.")
    print("OFFICIAL_LOSS: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
