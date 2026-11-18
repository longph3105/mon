"""CLODE: low-light enhancement as a neural ODE over a continuous curve-adjustment time.

Zero-DCE applies its quadratic curve ``x + A·(x² − x)`` a fixed number of times; CLODE makes the
iteration continuous: ``dx/dt = A(t, x̃)·(x² − x)`` with ``x̃ = x − D(x)`` (``D``: a small residual
denoiser), integrated from ``0`` to ``T`` with an adaptive Dormand–Prince solver (``dopri5``,
``rtol = atol = 1e-5``). The output is the final state, denoised and clamped: ``clamp(x_T −
D(x_T), 0, 1)``. ``T`` sets the exposure: training and the official inference use ``T = 3``;
larger brightens, negative darkens.

The ODE state has nine channels: the image, then the total variation of ``A`` and the
self-supervised denoising loss of ``D`` (two scalars broadcast to three channels each),
integrated over time as training regularizers. They are part of the state, so they also enter the
solver's error estimate and its step sizes. Because both scalars and the error norm are taken over
the whole batch, **outputs depend on the other images in the batch**: run one image per batch
for results that match per-image inference.

Parameter names match the official code (``odefunc.*``, and the same modules again under
``odeblock.odefunc.*``), so the zoo checkpoints load with ``strict=True``. The official release
files also hold modules that ``forward`` never uses (``conv1``, ``e_conv1``–``e_conv7``,
``condition``, ``cond_brightness``, ``norm``): load those with ``strict=False``, as the official
``inference.py`` does.

Differences from the official code (``network/conv_node.py``, ``inference.py``):

- Uses PyPI ``torchdiffeq`` (MIT); the official code vendors a git snapshot between 0.2.3 and
  0.2.4 whose ``dopri5`` path matches 0.2.5 (no ``min_step``/``max_step``, finite steps). In
  float64 the outputs are identical for ``T > 0`` and differ by up to 4e-9 for ``T < 0``.
- Without gradients, or with ``adjoint=False``, integrates with ``odeint`` instead of
  ``odeint_adjoint``; their forward passes are identical.
- Unused modules, the ``device``/``img_size``/``augment_dim``/``time_dependent`` arguments, the
  ever-growing ``pred_t`` list, and ``add_noise`` were removed. ``curve_map`` is returned raw
  (``tanh``, in ``[-1, 1]``); the official code min–max normalizes it over the batch for display.
- The paper loss is not ported: the official repository has no training script
  (``main_experiment.py`` is referenced but missing), and its ``losses.py`` differs from the loss
  that ``mon_old`` trains with.

Differences from ``mon_old`` (``models/enhance/lle/clode``), whose inference matched the official
code: its predictor config used ``T = 2`` (official: 3); weights loading in ``__init__``,
``forward_patch`` (use ``mon.nn.tiling.tiled_forward``), and ``benchmark`` were removed.

Examples:
    >>> import torch
    >>> out = CLODE(t_end=1.0).eval()(torch.rand(1, 3, 12, 16))
    >>> sorted(out), out["enhanced"].shape
    (['curve_map', 'enhanced', 'noise', 'state'], torch.Size([1, 3, 12, 16]))
    >>> from mon.models.registry import load
    >>> model = load("clode")  # official "universal" weights; needs DATA_DIR  # doctest: +SKIP

References:
    - Jung, Kim, Kim, "Continuous Exposure Learning for Low-light Image Enhancement using Neural
      ODEs", ICLR 2025. https://openreview.net/forum?id=Mn2qgIcIPS
    - https://github.com/dgjung0220/CLODE (no license stated; read, not copied; commit on ``main``
      pushed 2025-03-29)
    - https://github.com/rtqichen/torchdiffeq (MIT)
"""

__all__ = [
    "CLODE",
    "CLODE_Weights",
]

from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint

from mon.core.weights import Weights, WeightsEnum

# The official Drive files (folder 1E1Oi89TJeZIL4pz7d4p-D_Yq1rAG4Uhc) hold 98 tensors; the zoo files
# keep the 48 that ``forward`` uses, unchanged (checked tensor by tensor).
_DRIVE = "https://drive.google.com/drive/folders/1E1Oi89TJeZIL4pz7d4p-D_Yq1rAG4Uhc"


class CLODE_Weights(WeightsEnum):
    """Released CLODE weights, with the modules ``forward`` never uses removed."""

    LOL_V1 = Weights(
        path   = "lle/clode/clode_lol_v1.pt",
        name   = "CLODE, official release (lowlight.pth), unused modules removed",
        sha256 = "157dab6ca391c43337d336aaa3e8c2f3d1086235251d38da63ef391645bbb4c6",
        meta   = {"source": f"official lowlight.pth ({_DRIVE})", "train": "lol_v1",
                  "source_sha256": "6222a24aa2daf6b4ba886fa37c41190d777ff0e691be114d815704de6481c937"},
    )
    SICE_ME = Weights(
        path   = "lle/clode/clode_sice_me.pt",
        name   = "CLODE, official release (sice.pth), unused modules removed",
        sha256 = "0c7635fa0ad3bce24c0a18d5d01d69f983d14f52c6608e75fb0e2c79498a879f",
        meta   = {"source": f"official sice.pth ({_DRIVE})", "train": "sice_me",
                  "source_sha256": "f3894dbf11bbe32630ecd0d7e929f811e5df26ee0a47b58c00f4315663f8fc19"},
    )
    UNIVERSAL = Weights(
        path   = "lle/clode/clode_universal.pt",
        name   = "CLODE, official release (universal.pth), unused modules removed",
        sha256 = "a170bb22c1ac183f4309b48ed15ec664c7202bacbf45926eff3ea68ac9b7c8c6",
        meta   = {"source": f"official universal.pth ({_DRIVE}); the official inference.py default",
                  "train": "mixed data",
                  "source_sha256": "d228895f13715e897bfa65bb8bd4bdea4011df0d7a5e49cd9725882722ea5b0b"},
    )
    DEFAULT = UNIVERSAL


# --- Layers ---

class _Conv2dTime(nn.Conv2d):
    """Convolution with the time ``t`` prepended as a constant input channel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels + 1, out_channels, kernel_size, padding=kernel_size // 2,
                         padding_mode="reflect")

    def forward(self, t: Tensor, x: Tensor) -> Tensor:  # type: ignore[override]
        t_img = torch.ones_like(x[:, :1]) * t
        return super().forward(torch.cat([t_img, x], 1))


class _Denoiser(nn.Module):
    """Noise estimator ``D`` (``x − D(x)`` is the denoised image): two 3×3 convs, then a 1×1."""

    def __init__(self, channels: int = 3, embed_channels: int = 48) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, channels, 1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv3(self.act(self.conv2(self.act(self.conv1(x)))))
        return out


def _pair_downsample(image: Tensor) -> tuple[Tensor, Tensor]:
    """The two diagonal-average half-resolution views used by Zero-Shot Noise2Noise."""
    c  = image.shape[1]
    f1 = image.new_tensor([[[[0.0, 0.5], [0.5, 0.0]]]]).repeat(c, 1, 1, 1)
    f2 = image.new_tensor([[[[0.5, 0.0], [0.0, 0.5]]]]).repeat(c, 1, 1, 1)
    return F.conv2d(image, f1, stride=2, groups=c), F.conv2d(image, f2, stride=2, groups=c)


def _total_variation(x: Tensor) -> Tensor:
    """Zero-DCE's illumination smoothness term (squared differences; a batch-mean scalar)."""
    b, _, h, w = x.shape
    h_tv = (x[:, :, 1:, :] - x[:, :, :h - 1, :]).pow(2).sum()
    w_tv = (x[:, :, :, 1:] - x[:, :, :, :w - 1]).pow(2).sum()
    return 2 * (h_tv / ((h - 1) * w) + w_tv / (h * (w - 1))) / b


class _EnhanceFunc(nn.Module):
    """ODE right-hand side ``f(t, state)`` for the state ``[image, ∫TV(A), ∫L_denoise]``."""

    def __init__(self, num_filters: int = 32) -> None:
        super().__init__()
        nf = num_filters
        self.norm32    = nn.GroupNorm(1, nf)
        self.norm64    = nn.GroupNorm(1, nf * 2)
        self.up_conv   = _Conv2dTime(6, nf, 3)
        self.down_conv = _Conv2dTime(nf, 3, 3)
        self.conv_3_1  = _Conv2dTime(nf, nf, 3)
        self.conv_5_1  = _Conv2dTime(nf, nf, 5)
        self.conv_3_2  = _Conv2dTime(nf * 2, nf * 2, 3)
        self.conv_5_2  = _Conv2dTime(nf * 2, nf * 2, 5)
        self.confusion = _Conv2dTime(nf * 4, nf, 1)
        self.denoise   = _Denoiser(3)
        self.relu      = nn.ReLU(inplace=True)
        self.nfe       = 0                          # function evaluations in the last solve
        self.last_curve_map: Tensor | None = None   # A at the last evaluation

    def denoise_loss(self, noisy: Tensor) -> Tensor:
        """Zero-Shot Noise2Noise residual + consistency loss of ``self.denoise`` (batch mean)."""
        noisy1, noisy2 = _pair_downsample(noisy)
        pred1    = noisy1 - self.denoise(noisy1)
        pred2    = noisy2 - self.denoise(noisy2)
        loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
        denoised1, denoised2 = _pair_downsample(noisy - self.denoise(noisy))
        loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
        return loss_res + loss_cons

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        self.nfe += 1
        x       = state[:, :3]
        l_noise = self.denoise_loss(x)
        p_x     = x - self.denoise(x)
        h       = torch.cat([p_x, 1 - p_x], 1)

        h   = self.relu(self.norm32(self.up_conv(t, h)))
        h   = torch.cat([self.relu(self.norm32(self.conv_3_1(t, h))), self.relu(self.norm32(self.conv_5_1(t, h)))], 1)
        h   = torch.cat([self.relu(self.norm64(self.conv_3_2(t, h))), self.relu(self.norm64(self.conv_5_2(t, h)))], 1)
        h   = self.relu(self.norm32(self.confusion(t, h)))
        a   = torch.tanh(self.down_conv(t, h))

        self.last_curve_map = a
        dx  = a * (x.pow(2) - x)
        return torch.cat([dx, torch.ones_like(a) * _total_variation(a), torch.ones_like(a) * l_noise], 1)


class _ODEBlock(nn.Module):
    """Holds the ODE function again, as the official ``ODEBlock`` does (checkpoint layout)."""

    def __init__(self, odefunc: _EnhanceFunc) -> None:
        super().__init__()
        self.odefunc = odefunc


# --- Model ---

class CLODE(nn.Module):
    """CLODE network: integrate the curve-adjustment ODE from 0 to ``t_end``, then denoise.

    Args:
        t_end: Final integration time ``T`` (official: 3, as in training). Larger brightens; a
            negative value darkens; zero is invalid. ``forward`` can override it per call.
        num_filters: Feature channels of the curve estimator (32 in the paper).
        tol: Relative and absolute tolerance of the ``dopri5`` solver (official: 1e-5).
        adjoint: Backpropagate with the adjoint method (constant memory; official: True).
        max_num_steps: Solver step limit; exceeding it raises (official: 1000).

    Shapes:
        ``forward(image [B, 3, H, W] in [0, 1])`` → ``{"enhanced": [B, 3, H, W] in [0, 1],
        "curve_map": [B, 3, H, W] in [-1, 1] (A at the solver's last evaluation), "noise":
        [B, 3, H, W] (D(x_T)), "state": [B, 9, H, W] (x_T, ∫TV, ∫L_denoise; for training)}``;
        ``H, W >= 3``. Outputs depend on the whole batch (see the module docstring).
    """

    weights_enum: ClassVar[type[WeightsEnum]] = CLODE_Weights

    def __init__(
        self,
        t_end        : float = 3.0,
        num_filters  : int   = 32,
        tol          : float = 1e-5,
        adjoint      : bool  = True,
        max_num_steps: int   = 1000,
    ) -> None:
        super().__init__()
        self.t_end         = t_end
        self.tol           = tol
        self.adjoint       = adjoint
        self.max_num_steps = max_num_steps
        self.odefunc       = _EnhanceFunc(num_filters)
        self.odeblock      = _ODEBlock(self.odefunc)

    def extra_repr(self) -> str:
        return f"t_end={self.t_end}, tol={self.tol}, adjoint={self.adjoint}, max_num_steps={self.max_num_steps}"

    def forward(self, image: Tensor, t_end: float | None = None) -> dict[str, Tensor]:
        t_end = self.t_end if t_end is None else t_end
        if t_end == 0:
            raise ValueError("t_end must be non-zero (the solver needs distinct start and end times)")
        t     = torch.tensor([0.0, t_end], dtype=image.dtype, device=image.device)
        state = torch.cat([image, torch.zeros_like(image), torch.zeros_like(image)], 1)
        solve = odeint_adjoint if self.adjoint and torch.is_grad_enabled() else odeint
        self.odefunc.nfe = 0
        final = solve(self.odefunc, state, t, rtol=self.tol, atol=self.tol, method="dopri5",
                      options={"max_num_steps": self.max_num_steps})[-1]
        assert self.odefunc.last_curve_map is not None
        x_t   = final[:, :3]
        noise = self.odefunc.denoise(x_t)
        return {
            "enhanced" : torch.clamp(x_t - noise, 0, 1),
            "curve_map": self.odefunc.last_curve_map,
            "noise"    : noise,
            "state"    : final,
        }
