"""SCI and SCI++: self-calibrated illumination learning for low-light enhancement.

A tiny network estimates the illumination ``i`` (clamped to ``[1e-4, 1]``) and the result is the
Retinex reflectance ``clamp(x / i, 0, 1)``. During training, a self-calibration module refines
the input over several weight-sharing stages so that the first stage alone learns to do the
job; at test time only that first stage runs (3 conv layers for SCI), as in the official test
scripts. SCI++ (TPAMI) adds a second illumination branch and alternates two optimizers
(``SCIPP_RECIPE``).

Parameter names match the official code (including its weight sharing: the same block is
appended to ``blocks`` several times), so the released checkpoints load with ``strict=True``.

Differences from ``mon_old`` (``models/enhance/lle/sci``), whose inference matched:

- ``forward_train`` (SCI) divided the *updated* input by the illumination and accumulated
  ``x = x + att`` across stages; the official loop uses the original input in both places.
- ``forward_train`` (SCI++) reused ``i`` as the loop variable, so the illumination became
  ``0 + att + hb(att)`` and an int was stacked (crash); it also ran 3 extra stages instead of 2.
- Weights paths pointed at the old zoo layout; ``benchmark``, ``verbose``, and weights loading
  in ``__init__`` were removed (use ``mon.models.load``).

The training loss ``SCILoss`` follows the official code, including a quirk: its RGB→YCbCr
conversion reshapes an NCHW tensor with ``view(-1, 3)``, which mixes three horizontally
neighboring values of one channel instead of a pixel's R, G, B. The released weights were trained
with it; ``official_ycbcr=False`` uses a per-pixel conversion.

Examples:
    >>> import torch
    >>> out = SCI().eval()(torch.rand(1, 3, 37, 50))
    >>> out["enhanced"].shape, out["illumination"].shape
    (torch.Size([1, 3, 37, 50]), torch.Size([1, 3, 37, 50]))
    >>> from mon.models.registry import load
    >>> model = load("sci", "easy")  # also "medium" (default), "difficult"; needs DATA_DIR  # doctest: +SKIP

References:
    - Ma, Ma, Liu, Fan, Luo, "Toward Fast, Flexible, and Robust Low-Light Image Enhancement",
      CVPR 2022. https://arxiv.org/abs/2204.10137
    - Ma, Ma, Fan, Liu, Luo, "Learning with Nested Scene Modeling and Cooperative Architecture
      Search for Low-Light Vision", IEEE TPAMI 2023 (SCI++). https://arxiv.org/abs/2112.04719
    - https://github.com/vis-opt-group/SCI (MIT; ``CVPR/`` and ``TPAMI/``)
"""

__all__ = [
    "SCI",
    "SCI_RECIPE",
    "SCILoss",
    "SCIPP",
    "SCIPP_RECIPE",
    "SCIPP_Weights",
    "SCI_Weights",
]

from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mon.core.weights import Weights, WeightsEnum
from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss
from mon.nn.training import StepOutput, TrainRecipe

_REPO = "https://raw.githubusercontent.com/vis-opt-group/SCI"


# --- Weights ---

class SCI_Weights(WeightsEnum):
    """Released SCI weights (CVPR). The authors note they are not the models behind the paper's tables."""

    EASY = Weights(
        path   = "lle/sci/sci_easy.pt",
        name   = "SCI easy (official release, trained mainly on MIT-Adobe FiveK)",
        url    = f"{_REPO}/b2bfef399c715836ca87f1f96c7e8676794c4e74/CVPR/weights/easy.pt",
        sha256 = "b17326a26a93bf187ecbab217f9d3c57467a8b56020d4d9ff600ba257bbffb36",
    )
    MEDIUM = Weights(
        path   = "lle/sci/sci_medium.pt",
        name   = "SCI medium (official release, trained mainly on LOL + LSRW)",
        url    = f"{_REPO}/b2bfef399c715836ca87f1f96c7e8676794c4e74/CVPR/weights/medium.pt",
        sha256 = "4ff1e7e66e8f18114f790e5085ca4c936f0858e329e7ad1cb65b8aa09a9c6916",
    )
    DIFFICULT = Weights(
        path   = "lle/sci/sci_difficult.pt",
        name   = "SCI difficult (official release, trained mainly on DarkFace)",
        url    = f"{_REPO}/b2bfef399c715836ca87f1f96c7e8676794c4e74/CVPR/weights/difficult.pt",
        sha256 = "657fa1f280cae1035df0d5cbe628e91f2d0f28660df608a3fa3f7e9a71da5a13",
    )
    DEFAULT = MEDIUM  # the official test script's default


class SCIPP_Weights(WeightsEnum):
    """Released SCI++ weights (TPAMI ``weights_1_3500.pt``)."""

    TPAMI = Weights(
        path   = "lle/sci++/sci++_1_3500.pt",
        name   = "SCI++ (official release, weights_1_3500)",
        url    = f"{_REPO}/ac4b5dee2157c575a20dbf847abbbb21045adf6a/TPAMI/weights/weights_1_3500.pt",
        sha256 = "067d41fdfb0b11611944d15f30b53cac54db8206bd57ee27fe4e7ef02a55a4d8",
    )
    DEFAULT = TPAMI


# --- Networks (structure and names as in the official code) ---

def _conv(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, 1, 1)


def _official_init(module: nn.Module) -> None:
    """Official SCI (CVPR) training init: conv N(0, 0.02) with zero bias, batch-norm weight N(1, 0.02)."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)


class _Residual(nn.Module):
    """``in_conv`` → residual ``blocks`` → ``out_conv`` (the official networks' shared skeleton)."""

    in_conv : nn.Sequential
    blocks  : nn.ModuleList
    out_conv: nn.Sequential

    def features(self, x: Tensor) -> Tensor:
        fea = self.in_conv(x)
        for block in self.blocks:
            fea = fea + block(fea)
        out: Tensor = self.out_conv(fea)
        return out


class _Enhance(_Residual):
    """SCI illumination estimator (``EnhanceNetwork``); ``blocks`` repeat one shared ``conv``."""

    def __init__(self, layers: int = 1, channels: int = 3) -> None:
        super().__init__()
        self.in_conv  = nn.Sequential(_conv(3, channels), nn.ReLU())
        self.conv     = nn.Sequential(_conv(channels, channels), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks   = nn.ModuleList([self.conv] * layers)
        self.out_conv = nn.Sequential(_conv(channels, 3), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return (self.features(x) + x).clamp(1e-4, 1)


class _Calibrate(_Residual):
    """SCI self-calibration module (``CalibrateNetwork``); ``blocks`` repeat one shared ``convs``."""

    def __init__(self, layers: int = 3, channels: int = 16) -> None:
        super().__init__()
        self.in_conv  = nn.Sequential(_conv(3, channels), nn.BatchNorm2d(channels), nn.ReLU())
        self.convs    = nn.Sequential(_conv(channels, channels), nn.BatchNorm2d(channels), nn.ReLU(),
                                      _conv(channels, channels), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks   = nn.ModuleList([self.convs] * layers)
        self.out_conv = nn.Sequential(_conv(channels, 3), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return x - self.features(x)


class _EnhanceA(_Residual):
    """SCI++ illumination branch ``ha`` (``EnhanceNetwork_Ha``: no batch norm, separate blocks)."""

    def __init__(self, layers: int = 1, channels: int = 3) -> None:
        super().__init__()
        self.in_conv  = nn.Sequential(_conv(3, channels), nn.ReLU())
        self.blocks   = nn.ModuleList([nn.Sequential(_conv(channels, channels), nn.ReLU()) for _ in range(layers)])
        self.out_conv = nn.Sequential(_conv(channels, 3), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return (self.features(x) + x).clamp(1e-4, 1)


class _EnhanceB(_Residual):
    """SCI++ residual illumination branch ``hb`` (``EnhanceNetwork_Hb``: linear output)."""

    def __init__(self, layers: int = 3, channels: int = 16) -> None:
        super().__init__()
        self.in_conv  = nn.Sequential(_conv(3, channels), nn.ReLU())
        self.blocks   = nn.ModuleList([nn.Sequential(_conv(channels, channels), nn.ReLU()) for _ in range(layers)])
        self.out_conv = nn.Sequential(_conv(channels, 3))

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class _CalibratePP(_Residual):
    """SCI++ self-calibration module (TPAMI ``CalibrateNetwork``: no batch norm, ``x + f(x)``)."""

    def __init__(self, layers: int = 3, channels: int = 16) -> None:
        super().__init__()
        self.in_conv  = nn.Sequential(_conv(3, channels), nn.ReLU())
        self.blocks   = nn.ModuleList([
            nn.Sequential(_conv(channels, channels), nn.ReLU(), _conv(channels, channels), nn.ReLU())
            for _ in range(layers)
        ])
        self.out_conv = nn.Sequential(_conv(channels, 3))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.features(x)


def _reflectance(image: Tensor, illumination: Tensor) -> Tensor:
    return (image / illumination).clamp(0, 1)


# --- Models ---

class SCI(nn.Module):
    """SCI (CVPR 2022). ``forward`` runs the test-time model (illumination estimator only).

    Args:
        stages: Self-calibrated stages used in training (``forward_stages``).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W],
        "illumination": [B, 3, H, W]}``; any ``H, W``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = SCI_Weights

    def __init__(self, stages: int = 3) -> None:
        super().__init__()
        self.stages    = stages
        self.enhance   = _Enhance(layers=1, channels=3)
        self.calibrate = _Calibrate(layers=3, channels=16)
        self.apply(_official_init)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        illumination = self.enhance(image)
        return {"enhanced": _reflectance(image, illumination), "illumination": illumination}

    def forward_stages(self, image: Tensor) -> dict[str, list[Tensor]]:
        """Training forward (official ``Network.forward``): per-stage inputs, illuminations, results, |att|.

        The official objective is ``sum(SCILoss()(inputs[k], illuminations[k]) for k in stages)``.
        """
        inputs, illuminations, enhanced, attention = [], [], [], []
        x = image
        for _ in range(self.stages):
            inputs.append(x)
            i = self.enhance(x)
            r = _reflectance(image, i)
            att = self.calibrate(r)
            x = image + att
            illuminations.append(i)
            enhanced.append(r)
            attention.append(att.abs())
        return {"inputs": inputs, "illuminations": illuminations, "enhanced": enhanced, "attention": attention}


class SCIPP(nn.Module):
    """SCI++ (TPAMI). ``forward`` runs the test-time model (branch ``ha`` only).

    Args:
        stages: Stages used in training (``forward_stages``); the first is ``ha`` alone.

    Shapes:
        As ``SCI``.
    """

    weights_enum: ClassVar[type[WeightsEnum]] = SCIPP_Weights

    def __init__(self, stages: int = 3) -> None:
        super().__init__()
        self.stages    = stages
        self.ha        = _EnhanceA(layers=1, channels=3)
        self.hb        = _EnhanceB(layers=3, channels=16)
        self.calibrate = _CalibratePP(layers=3, channels=16)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        illumination = self.ha(image)
        return {"enhanced": _reflectance(image, illumination), "illumination": illumination}

    def forward_stages(self, image: Tensor) -> dict[str, list[Tensor]]:
        """Training forward (official TPAMI ``Network.forward``).

        The official objective alternates two Adam optimizers: ``ha`` on
        ``SCILoss()(inputs[0], illuminations[0])``, and ``hb`` + ``calibrate`` on
        ``Σ_k≥1 L1(illuminations[0], illuminations[k]) + 0.1·SCILoss()(inputs[0], illuminations[k])``.
        """
        i = self.ha(image)
        r = _reflectance(image, i)
        inputs, illuminations, enhanced, attention = [image], [i], [r], []
        for _ in range(1, self.stages):
            inputs.append(i)
            att = self.calibrate(r)
            i = i + att + self.hb(att)
            r = _reflectance(image, i)
            illuminations.append(i)
            enhanced.append(r)
            attention.append(att.abs())
        return {"inputs": inputs, "illuminations": illuminations, "enhanced": enhanced, "attention": attention}


# --- Loss ---

_YCBCR = ((0.257, -0.148, 0.439), (0.564, -0.291, -0.368), (0.098, 0.439, -0.071))  # rows R, G, B → Y, Cb, Cr
_YCBCR_BIAS = (16 / 255, 128 / 255, 128 / 255)
# The 24 official neighbor terms are 12 offsets, each counted twice (a-b and b-a give equal values).
_OFFSETS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (2, -1), (1, 2), (1, -2), (2, 2), (2, -2))


def _rgb_to_ycbcr(image: Tensor, official: bool) -> Tensor:
    mat  = torch.tensor(_YCBCR, dtype=torch.float32, device=image.device)
    bias = torch.tensor(_YCBCR_BIAS, dtype=torch.float32, device=image.device)
    x = image.float()
    if official:  # view(-1, 3) on NCHW: groups of three values along W within one channel
        return (x.contiguous().view(-1, 3) @ mat + bias).view(x.shape)
    return torch.einsum("bchw,ck->bkhw", x, mat) + bias.view(1, 3, 1, 1)


def _pairs(x: Tensor, dy: int, dx: int) -> tuple[Tensor, Tensor]:
    """Overlapping views ``x[p]`` and ``x[p + (dy, dx)]`` for every valid pixel ``p`` (``dy >= 0``)."""
    h, w = x.shape[-2:]
    cols_a, cols_b = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    return x[..., : h - dy, cols_a], x[..., dy:, cols_b]


class SCILoss(nn.Module):
    """The SCI training objective: ``1.5·MSE(illumination, input) + smoothness(input, illumination)``.

    The smoothness term weights L1 differences of the illumination between each pixel and its 24
    neighbors within distance 2 by ``exp(-‖ΔYCbCr(input)‖² / (2·10²))`` (edge-aware). Shared by SCI
    and SCI++ (identical official code).

    Args:
        official_ycbcr: Reproduce the official RGB→YCbCr reshaping quirk (see the module docstring).
        reduction: ``"mean"`` (equals the official value), ``"sum"``, or ``"none"`` (per-sample ``[B]``).
    """

    def __init__(self, official_ycbcr: bool = True, reduction: Reduction = "mean") -> None:
        super().__init__()
        check_reduction(reduction)
        self.official_ycbcr = official_ycbcr
        self.reduction      = reduction
        self.sigma          = 10.0

    def smoothness(self, image: Tensor, illumination: Tensor) -> Tensor:
        """Per-sample edge-aware smoothness ``[B]`` of ``illumination`` guided by ``image``."""
        guide = _rgb_to_ycbcr(image, self.official_ycbcr)
        scale = -1.0 / (2 * self.sigma**2)
        total = illumination.new_zeros(illumination.shape[0])
        for dy, dx in _OFFSETS:
            ga, gb = _pairs(guide, dy, dx)
            ia, ib = _pairs(illumination, dy, dx)
            weight = torch.exp((ga - gb).pow(2).sum(dim=1, keepdim=True) * scale)
            total = total + 2 * (weight * (ia - ib).abs().sum(dim=1, keepdim=True)).flatten(1).mean(dim=1)
        return total

    def forward(self, image: Tensor, illumination: Tensor) -> Tensor:
        fidelity = (illumination - image).pow(2).flatten(1).mean(dim=1)
        return reduce_loss(1.5 * fidelity + self.smoothness(image, illumination), self.reduction)


# --- Training recipe ---

def _train_step(model: nn.Module, loss: nn.Module, batch: dict[str, Tensor]) -> StepOutput:
    assert isinstance(model, SCI)
    stages = model.forward_stages(batch["image"])
    terms: dict[str, Tensor] = {f"stage{k + 1}": loss(x, i)
                                for k, (x, i) in enumerate(zip(stages["inputs"], stages["illuminations"]))}
    return {"loss": torch.stack(list(terms.values())).sum(), **terms}


SCI_RECIPE = TrainRecipe(
    loss     = SCILoss,
    step     = _train_step,
    defaults = {
        "batch_size": 1, "epochs": 1000, "seed": 2,
        "optimizer": {"name": "adam", "lr": 3e-4, "betas": [0.9, 0.999], "weight_decay": 3e-4}, "grad_clip": 5.0,
    },
    source   = "vis-opt-group/SCI CVPR/train.py (f6f88fd): full images (its crop is commented out), batch 1",
    notes    = "the official script names no training set ('Your train dataset'); pass data=…",
)
"""Official SCI (CVPR) training: Adam 3e-4 (weight decay 3e-4), clip 5, batch 1, full images, 1000 epochs, seed 2.

The three released weights share this recipe and differ in training data, which the authors give
only as "mainly" (``CVPR/README.md``) without image lists, so none is exactly reproducible. The
closest ``data=`` ids: ``easy`` → ``mit_5k`` (MIT-Adobe FiveK), ``medium`` → ``lol_v1`` and
``lsrw_huawei``/``lsrw_nikon`` (LOL + LSRW; one id per run), ``difficult`` → ``darkface``. The
paper's tables use 500 random MIT and 500 random LSRW training pairs (the authors' subsets are
on Google Drive, not in ``mon``)."""


def _scipp_step(model: nn.Module, loss: nn.Module, batch: dict[str, Tensor]) -> StepOutput:
    assert isinstance(model, SCIPP)
    stages = model.forward_stages(batch["image"])
    image, illum = stages["inputs"][0], stages["illuminations"]
    first = loss(image, illum[0])
    later = {f"stage{k + 1}": F.l1_loss(illum[0], illum[k]) + 0.1 * loss(image, illum[k]) for k in range(1, len(illum))}
    return {"loss_ha": first, "loss_hb": torch.stack(list(later.values())).sum(), "stage1": first, **later}


def _scipp_schedule(step: int) -> str | None:
    """The official alternation as it acts: ``ha`` at steps ≡ 1–6 (mod 10), ``hb`` + ``calibrate`` at ≡ 8, 9."""
    r = step % 10
    if 1 <= r <= 6:
        return "ha"
    return "hb" if r in (8, 9) else None


SCIPP_RECIPE = TrainRecipe(
    loss       = SCILoss,
    step       = _scipp_step,
    defaults   = {
        "batch_size": 1, "epochs": 3000, "seed": 2, "shuffle": False,
        "optimizer": {"name": "adam", "lr": 3e-4, "betas": [0.9, 0.999], "weight_decay": 3e-4}, "grad_clip": None,
    },
    source     = "vis-opt-group/SCI TPAMI/train.py (f6f88fd): two Adam optimizers alternated by global step, "
                 "full images in file order, batch 1, 3000 epochs, seed 2",
    notes      = "the official script names no training set; its clip_grad_norm_ runs after the optimizer step "
                 "(no effect), so the recipe has none; steps ≡ 7, 0 (mod 10) update nothing (see SCIPP_RECIPE)",
    optimizers = {"ha": ("ha",), "hb": ("hb", "calibrate")},
    schedule   = _scipp_schedule,
)
"""Official SCI++ (TPAMI) training: Adam 3e-4 (weight decay 3e-4) for ``ha``, and another for ``hb`` +
``calibrate``; batch 1, full images in file order, 3000 epochs, seed 2, no init (PyTorch defaults).

Objectives: ``ha`` minimizes ``SCILoss(x, i₁)``; ``hb`` + ``calibrate`` minimize
``Σ_k≥2 L1(i₁, i_k) + 0.1·SCILoss(x, i_k)``. The official loop picks ``ha`` when ``total_step % 10 < 7``,
but it sets ``requires_grad`` after the forward pass, so each step's graph has the previous step's
flags: the first step of each phase (steps ≡ 7 and ≡ 0 mod 10) builds no graph to the parameters it
updates, and with PyTorch ≥ 2.0 (``zero_grad`` sets ``None``, Adam skips them) changes nothing. The
schedule reproduces that effective alternation: 6 ``ha`` updates, 2 ``hb`` + ``calibrate`` updates and
2 empty steps per 10 (``tests/runners/test_trainer_parity.py``). The released ``weights_1_3500`` was
saved at epoch 1, step 3500, so its training set had 1750–3499 images; the official README names
none (``mon_old`` trained it on LSRW)."""
