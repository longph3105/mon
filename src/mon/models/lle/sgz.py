"""SGZ: semantic-guided zero-shot low-light enhancement (Zero-DCE++ network, semantic training loss).

At inference SGZ is exactly the Zero-DCE++ network (the official ``enhance_net_nopool`` with
``conv_type="dsc"``): depthwise-separable layers estimate one curve map, optionally on the input
downsampled by ``scale_factor``, applied 8 times. Parameter names match, so ``SGZ`` reuses
``ZeroDCEPP`` and only adds its weights. What makes SGZ different is training: Zero-DCE's
zero-reference losses plus a semantic loss from a segmentation network (not ported).

The official test script uses ``scale_factor=12`` and crops each image to a multiple of 12 first;
here any size works (the curve is upsampled to the input size, ``align_corners=True`` as the
official ``UpsamplingBilinear2d``), so ``mon predict`` doesn't crop. The release records 12 in
``meta["test_scale_factor"]``; pass ``model_args.scale_factor=12`` to use it (default 1, as
``ZeroDCEPP``).

Published numbers: the paper's LOL score (Table 6: PSNR 20.60, SSIM 0.79; Zero-DCE 20.38 / 0.78) states
no protocol. On LOL-v1 ``eval15`` (``mon evaluate``, 8-bit outputs, 2026-11-19) the release gives
16.04 / 0.679 / LPIPS 0.335 at ``scale_factor=12`` (15.85 / 0.675 at 1); with ``gt_mean`` PSNR is 20.37
(Zero-DCE 21.06), close to the paper's, but SSIM stays ≈ 0.67, so the protocol remains unknown.
Complexity (Table 7, 1200×900): the paper's 0.120 G "FLOPs" for SGZ and 84.99 G for Zero-DCE are
thop-style MACs; ``mon benchmark`` counts 0.076 and 85.5 GMACs (convolutions only).

Differences from ``mon_old`` (``models/enhance/lle/sgz``): its ``forward_step`` inverted the
downsampling test (downsampled only when ``scale_factor == 1``, a no-op, and upsampled an
undownsampled curve otherwise, which crashed); the zoo file was named ``sgz_lol_v1.pt`` although the
weights were trained on Zero-DCE's SICE training images (renamed ``sgz_sice_me.pt``).

Examples:
    >>> import torch
    >>> out = SGZ(scale_factor=12).eval()(torch.rand(1, 3, 48, 60))
    >>> out["enhanced"].shape, out["curve"].shape
    (torch.Size([1, 3, 48, 60]), torch.Size([1, 3, 48, 60]))
    >>> from mon.models.registry import load
    >>> model = load("sgz", scale_factor=12)  # official test setting; needs DATA_DIR  # doctest: +SKIP

References:
    - Zheng, Gupta, "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement",
      WACV 2022 Workshops. https://arxiv.org/abs/2110.00970
    - https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement (MIT;
      ``modeling/model.py``, ``test.py``)
"""

__all__ = [
    "SGZ",
    "SGZ_Weights",
]

from typing import ClassVar

from mon.core.weights import Weights, WeightsEnum
from mon.models.lle.zero_dce import ZeroDCEPP

_REPO = "https://raw.githubusercontent.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement"


class SGZ_Weights(WeightsEnum):
    """Released SGZ weights (the official test script runs them with ``scale_factor=12``)."""

    SICE_ME = Weights(
        path   = "lle/sgz/sgz_sice_me.pt",
        name   = "SGZ, official release (Zero-DCE's SICE training images)",
        url    = f"{_REPO}/6e1c840885a06f6937754e31cd31df6693e51099/weight/Epoch99.pth",
        sha256 = "54d87f6f3a2b5ee495abec7e714afcdf60eaa8d15370dd500792cd4f45f442f1",
        meta   = {"source": "official weight/Epoch99.pth", "train": "sice_me", "test_scale_factor": 12},
    )
    DEFAULT = SICE_ME


class SGZ(ZeroDCEPP):
    """SGZ (WACV 2022): the Zero-DCE++ network with SGZ's released weights; arguments as ``ZeroDCEPP``."""

    weights_enum: ClassVar[type[WeightsEnum]] = SGZ_Weights
