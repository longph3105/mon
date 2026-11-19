"""NRQM: Ma et al.'s no-reference quality metric for super-resolution (higher is better, about 0–10).

Each image is converted to luma rounded to 8-bit levels (pyiqa's ``nrqm`` default; ``color_space="ycbcr"``
for the PIRM perceptual index) and described by three groups of statistics, each scored by its own
random-forest regressor; a linear model combines the three scores:

- local frequency (18): GGD shape, coefficient of variation, and orientation variance of the DCT of
  overlapping 7×7 blocks (stride 3), pooled over the worst 10% and all blocks, on a 3-level Gaussian pyramid;
- global frequency (45): GGD shapes of divisively normalized subbands of a 2-scale, 6-orientation complex
  steerable pyramid, and SSIM contrast-structure terms across scales and orientations;
- spatial (75): singular values of the distinct 5×5 patches at the three pyramid levels.

Ported from pyiqa's ``nrqm_arch.py`` (by reading; pyiqa is a non-commercial reference only), which re-implements
the official MATLAB code with a PyTorch complex steerable pyramid (Fourier domain, from
``tomrunia/PyTorchSteerablePyramid``, MIT). Its numerics are kept where they change scores: block positions
follow TensorFlow "same" padding (MATLAB ``blkproc`` pads differently), the worst-10% pools take
``ceil(0.1 n) + 1`` / ``n - floor(0.9 n)`` values, the GGD shape grids are float32 ``arange``, the pyramid's
filter window and masks are rounded to float32, and resizing runs in float32 (pyiqa's ``imresize`` always casts).
GGD fits look the shape up with ``searchsorted`` on the monotone ratio table (pyiqa: dense ``argmin``).

The regressors are the official ``model.mat`` (MIT, https://github.com/chaoma99/sr-metric; byte-identical to
pyiqa's ``NRQM_model.mat``), 3 forests of 2000 trees. mon has no SciPy, so the file is converted once to
``zoo_dir() / "metrics/nrqm/nrqm_model.npz"`` (only the arrays prediction needs)::

    uv run --with scipy python -m mon.metrics.nrqm convert path/to/model.mat

References:
    - C. Ma, C.-Y. Yang, X. Yang, M.-H. Yang, "Learning a No-Reference Quality Metric for Single-Image
      Super-Resolution," CVIU 158, 2017. https://arxiv.org/abs/1612.05890
"""

__all__ = [
    "NRQM",
    "NRQM_Weights",
    "convert_nrqm_model",
    "nrqm_features",
]

import hashlib
import importlib
import math
import sys
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mon.core.path import Path, zoo_dir
from mon.core.weights import Weights, WeightsEnum, WeightsLike, resolve_weights
from mon.metrics.base import Metric, MetricMode, YColorSpace, crop_border, rgb_to_y, to_8bit_levels
from mon.metrics.fidelity import gaussian_window
from mon.metrics.niqe import matlab_resize_matrix

_MODEL_URL    = "https://github.com/chaoma99/sr-metric/raw/master/model.mat"
_MODEL_SHA256 = "4142385df9f7f3e91647673f6400aa4da567539ebfc92bfb8b97486df5122d31"
_CONVERT      = "uv run --with scipy python -m mon.metrics.nrqm convert <model.mat>"


class NRQM_Weights(WeightsEnum):
    """The official regressors (``linear`` [4]; per forest ``ldau``, ``rdau``, ``upper``, ``avnode``, ``mbest``)."""

    MA2017 = Weights(
        path = "metrics/nrqm/nrqm_model.npz",
        name = "NRQM regressors (official model.mat, converted)",
        meta = {"source": _MODEL_URL, "source_sha256": _MODEL_SHA256, "license": "MIT", "convert": _CONVERT},
    )
    DEFAULT = MA2017


# --- Small helpers ---

def _f32_window(size: int, sigma: float) -> Tensor:
    """MATLAB ``fspecial('gaussian')`` rounded to float32, as pyiqa's (float64 out)."""
    return gaussian_window(size, sigma).float().double()


def _resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    """MATLAB bicubic ``imresize(x, size)`` of ``[..., H, W]``, computed in float32 as pyiqa's (float64 out)."""
    h, w = x.shape[-2:]
    if (h, w) == size:
        return x
    m_h = matlab_resize_matrix(h, out_len=size[0]).float().to(x.device)
    m_w = matlab_resize_matrix(w, out_len=size[1]).float().to(x.device)
    return (m_h @ x.float() @ m_w.T).double()


def _ggd_grid(start: float, device: torch.device) -> tuple[Tensor, Tensor]:
    """pyiqa's float32 shape grid ``arange(start, 10.001, 0.001)`` and its GGD ratio table (float64, decreasing)."""
    grid = torch.arange(start, 10 + 0.001, 0.001, dtype=torch.float32).double().to(device)
    ratio = torch.exp(torch.lgamma(1 / grid) + torch.lgamma(3 / grid) - 2 * torch.lgamma(2 / grid))
    return grid, ratio


def _nearest(table: Tensor, values: Tensor) -> Tensor:
    """Index of the ``table`` entry closest to each value (first on ties), for a strictly decreasing table."""
    rev = table.flip(0)                                      # increasing
    pos = torch.searchsorted(rev, values.contiguous()).clamp(1, len(rev) - 1)
    lo, hi = pos - 1, pos                                    # rev[lo] <= v <= rev[hi] (clamped at the ends)
    d_lo, d_hi = (values - rev[lo]).abs(), (rev[hi] - values).abs()
    # In table order hi comes first (smaller index), so it wins ties, as argmin does.
    rev_idx = torch.where(d_hi <= d_lo, hi, lo)
    return len(table) - 1 - rev_idx


def _gamma_gen_gauss(x: Tensor) -> Tensor:
    """GGD shape of each row of ``[N, n]`` (official ``gama_gen_gauss``)."""
    grid, ratio = _ggd_grid(0.03, x.device)
    mean_abs = (x - x.mean(dim=-1, keepdim=True)).abs().mean(dim=-1).square()
    rho = x.var(dim=-1, unbiased=True) / (mean_abs + 1e-7)
    return grid[_nearest(ratio, rho)]


def _coeff_var(x: Tensor, eps: float = 1e-7) -> Tensor:
    """``std(|x|) / (mean(|x|) + eps)`` of each row (official ``coeff_var_gen_gauss``)."""
    a = x.abs()
    return a.std(dim=-1, unbiased=True) / (a.mean(dim=-1) + eps)


def _dct_matrix(n: int) -> Tensor:
    """Orthonormal DCT-II matrix (MATLAB ``dct2(X) = C @ X @ C.T``)."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None, :]
    c = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2 / n)
    c[0] /= math.sqrt(2)
    return c


def _same_pad(x: Tensor, kernel: int, stride: int) -> Tensor:
    """TensorFlow "same" zero padding of ``[B, C, H, W]`` for a ``kernel`` / ``stride`` sliding window."""
    h, w = x.shape[-2:]
    ph = (math.ceil(h / stride) - 1) * stride + kernel - h
    pw = (math.ceil(w / stride) - 1) * stride + kernel - w
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def _gauss_pyramid(x: Tensor, levels: int = 2) -> list[Tensor]:
    """``[x, …]``: blur with ``fspecial('gaussian', 3)`` (σ 0.5, zero padding), keep every second pixel."""
    window = _f32_window(3, 0.5).to(x.device)[None, None]
    pyr = [x]
    for _ in range(levels):
        x = F.conv2d(F.pad(x, (1, 1, 1, 1)), window)[:, :, 1::2, 1::2]
        pyr.append(x)
    return pyr


# --- Local frequency (DCT) features ---

def _pool_low(sorted_vals: Tensor) -> Tensor:
    return sorted_vals[:, : math.ceil(0.1 * sorted_vals.shape[1]) + 1].mean(dim=1)


def _pool_high(sorted_vals: Tensor) -> Tensor:
    return sorted_vals[:, math.floor(0.9 * sorted_vals.shape[1]):].mean(dim=1)


def _block_dct(img: Tensor) -> Tensor:
    """``[B, 1, H, W]`` in ``[0, 1]`` → ``[B, 6]`` DCT statistics of 7×7 blocks at stride 3."""
    b = img.shape[0]
    cols = F.unfold(_same_pad(img, 7, 3), 7, stride=3)                  # [B, 49, L]
    blocks = cols.transpose(1, 2).reshape(b, -1, 7, 7)
    c = _dct_matrix(7).to(img.device)
    d = c @ blocks @ c.T                                                # [B, L, 7, 7]
    n = d.shape[1]
    ac = d.reshape(b, n, 49)[:, :, 1:]

    gamma = torch.sort(_gamma_gen_gauss(ac.reshape(b * n, 48)).reshape(b, n), dim=1).values
    cv    = torch.sort(_coeff_var(ac), dim=1).values
    ori1  = torch.cat([d[..., 0, 1:], d[..., 1, 2:], d[..., 2, 4:], d[..., 3, 5:]], dim=-1)
    ori2  = torch.cat([d[..., 1, 1:2], d[..., 2, 2:4], d[..., 3, 2:5], d[..., 4, 3:], d[..., 5, 4:], d[..., 6, 4:]],
                      dim=-1)
    ori3  = torch.cat([d[..., 1:, 0], d[..., 2:, 1], d[..., 4:, 2], d[..., 5:, 3]], dim=-1)
    ori   = torch.stack([_coeff_var(o, 1e-8) for o in (ori1, ori2, ori3)], dim=-1).var(dim=-1, unbiased=True)
    ori   = torch.sort(ori, dim=1).values
    return torch.stack([_pool_low(gamma), gamma.mean(1), _pool_high(cv), cv.mean(1), _pool_high(ori), ori.mean(1)],
                       dim=1)


# --- Global frequency (steerable pyramid) features ---

def _prepare_grid(m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-(m // 2) / (m / 2), (m // 2) / (m / 2) - (1 - m % 2) * 2 / m, num=m)
    y = np.linspace(-(n // 2) / (n / 2), (n // 2) / (n / 2) - (1 - n % 2) * 2 / n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m // 2][n // 2] = rad[m // 2][n // 2 - 1]
    return np.log2(rad), angle


def _rcos(width: float, position: float) -> tuple[np.ndarray, np.ndarray]:
    n = 256
    x = np.pi * np.arange(-n - 1, 2) / 2 / n
    y = np.cos(x) ** 2
    y[0], y[n + 2] = y[1], y[n + 1]
    return position + 2 * width / np.pi * (x + np.pi / 4), y


def _mask(values: np.ndarray, ys: np.ndarray, xs: np.ndarray, device: torch.device) -> Tensor:
    """``interp(values; xs → ys)`` as a float32-rounded float64 ``[1, H, W]`` mask (pyiqa's ``pointOp(...).float()``)."""
    out = np.interp(values.flatten(), xs, ys).reshape(values.shape)
    return torch.from_numpy(out).float().double().to(device)[None]


def _steerable_pyramid(im: Tensor, height: int = 2, nbands: int = 6) -> tuple[Tensor, list[list[Tensor]]]:
    """Complex steerable pyramid of ``[B, H, W]``: ``(highpass, [real bands per scale])`` (pyiqa ``SCFpyr``)."""
    h, w = im.shape[-2:]
    if height > int(np.floor(np.log2(min(h, w))) - 2):
        raise ValueError(f"NRQM's steerable pyramid needs a larger image than {h}×{w}")
    device = im.device
    log_rad, angle = _prepare_grid(h, w)
    xr, yr = _rcos(1, -0.5)
    yr = np.sqrt(yr)
    dft = torch.fft.fftshift(torch.fft.fft2(im), dim=(-2, -1))
    lodft = dft * _mask(log_rad, np.sqrt(1 - yr**2), xr, device)
    hi0 = torch.fft.ifft2(torch.fft.ifftshift(dft * _mask(log_rad, yr, xr, device), dim=(-2, -1))).real

    lut = 1024
    xcosn = np.pi * np.arange(-(2 * lut + 1), lut + 2) / lut
    alpha = (xcosn + np.pi) % (2 * np.pi) - np.pi
    order = nbands - 1
    const = 2 ** (2 * order) * math.factorial(order) ** 2 / (nbands * math.factorial(2 * order))
    ycosn = 2 * np.sqrt(const) * np.cos(xcosn) ** order * (np.abs(alpha) < np.pi / 2)
    fact = np.power(complex(0, -1), order)                  # numpy's value, tiny real part included

    scales: list[list[Tensor]] = []
    for _ in range(height):
        xr = xr - 1                                          # log2(scale_factor = 2)
        himask = _mask(log_rad, yr, xr, device)
        bands = []
        for b in range(nbands):
            banddft = lodft * _mask(angle, ycosn, xcosn + np.pi * b / nbands, device) * himask
            re, im_ = banddft.real, banddft.imag
            banddft = torch.complex(fact.real * re - fact.imag * im_, fact.real * im_ + fact.imag * re)
            bands.append(torch.fft.ifft2(torch.fft.ifftshift(banddft, dim=(-2, -1))).real)
        scales.append(bands)
        dims = np.array(lodft.shape[-2:])
        start = (np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)).astype(int)
        end = (start + np.ceil((dims - 0.5) / 2)).astype(int)
        log_rad = log_rad[start[0]:end[0], start[1]:end[1]]
        angle = angle[start[0]:end[0], start[1]:end[1]]
        lodft = lodft[..., start[0]:end[0], start[1]:end[1]] * _mask(log_rad, np.abs(np.sqrt(1 - yr**2)), xr, device)
    return hi0, scales


def _normalize_subbands(bands: list[Tensor], num_scales: int = 2, num_bands: int = 6) -> list[Tensor]:
    """Divisive normalization of each band by its 3×3 + parent + sibling neighborhood (official ``norm_sender_normalized``)."""
    border, guardband, eps = 1, 16, 1e-12
    out = []
    for si in range(num_scales):
        for bi in range(num_bands):
            idx = si * num_bands + bi
            band = bands[idx]
            n = 9
            tmp = F.unfold(band.unsqueeze(1), 3).transpose(1, 2)            # [B, L, 9]
            b, hw = tmp.shape[:2]
            if idx + num_bands < len(bands):
                parent = _resize(bands[idx + num_bands], tuple(band.shape[-2:]))  # type: ignore[arg-type]
                tmp = torch.cat((tmp, parent[:, border:-border, border:-border].reshape(b, hw, 1)), dim=-1)
                n += 1
            for ni in range(num_bands):
                if ni != bi:
                    nb = bands[si * num_bands + ni][:, border:-border, border:-border].reshape(b, hw, 1)
                    tmp = torch.cat((tmp, nb), dim=-1)
            c_x = tmp.transpose(1, 2) @ tmp / hw
            lam, q = torch.linalg.eigh(c_x)
            pos = lam * (lam > 0)
            pos_sum = pos.sum(dim=1, keepdim=True)
            lam = pos * lam.sum(dim=1, keepdim=True) / (pos_sum + (pos_sum == 0).to(lam.dtype))
            c_x = q @ torch.diag_embed(lam) @ q.transpose(1, 2)
            center = band[:, border:-border, border:-border]
            hh, ww = center.shape[-2:]
            o_c = center.reshape(b, hw)
            o_c = o_c - o_c.mean(dim=1, keepdim=True)
            y = torch.linalg.solve(c_x.transpose(1, 2), tmp.transpose(1, 2)).transpose(1, 2) * tmp / n
            z = y.sum(dim=2).sqrt()
            nonzero = z != 0
            g = (o_c * nonzero / (z * nonzero + eps)).reshape(b, hh, ww)
            gb = guardband // 2**si
            g = g[:, gb:-gb, gb:-gb]
            out.append(g - g.mean(dim=(1, 2), keepdim=True))
    return out


def _gauss_1d(size: int, sigma: float) -> Tensor:
    """1-D factor of ``fspecial('gaussian', size, sigma)`` (whose entries are all above MATLAB's cutoff)."""
    r = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-r.square() / (2 * sigma**2))
    return g / g.sum()


def _ssim_cs(x: Tensor, y: Tensor, g: Tensor) -> Tensor:
    """Mean SSIM contrast-structure term of ``[B, H, W]`` pairs (data range 255, valid 11×11 window, clipped at 0).

    The Gaussian window is applied as two 1-D passes (pyiqa: one float32-rounded 2-D window; 1e-8 apart).
    """
    gh, gw = g.view(1, 1, -1, 1), g.view(1, 1, 1, -1)
    stack = torch.stack([x, y, x * x, y * y, x * y], dim=1).flatten(0, 1).unsqueeze(1)   # [5B, 1, H, W]
    filtered = F.conv2d(F.conv2d(stack, gh), gw)
    mu1, mu2, xx, yy, xy = filtered.reshape(-1, 5, *filtered.shape[1:]).unbind(1)
    c2 = (0.03 * 255) ** 2
    s1, s2, s12 = xx - mu1.square(), yy - mu2.square(), xy - mu1 * mu2
    return F.relu((2 * s12 + c2) / (s1 + s2 + c2)).mean(dim=(1, 2, 3))


def _global_gsm(levels: Tensor) -> Tensor:
    """``[B, 1, H, W]`` in ``[0, 255]`` → ``[B, 45]`` steerable-pyramid statistics."""
    b = levels.shape[0]
    hi0, scales = _steerable_pyramid(levels[:, 0])
    bands = scales[0] + scales[1]
    subbands = _normalize_subbands(bands)
    window = _gauss_1d(11, 1.5).to(levels.device)
    feats = [_gamma_gen_gauss(sb.reshape(b, -1)) for sb in subbands]
    feats += [_gamma_gen_gauss(torch.cat((subbands[i].reshape(b, -1), subbands[i + 6].reshape(b, -1)), dim=1))
              for i in range(6)]
    feats += [_ssim_cs(_resize(band, tuple(hi0.shape[-2:])), hi0, window) for band in bands]  # type: ignore[arg-type]
    feats += [_ssim_cs(subbands[i], subbands[j], window) for i in range(6) for j in range(i + 1, 6)]
    return torch.stack(feats, dim=1)


# --- Spatial (SVD) features ---

def _svd_features(img: Tensor) -> Tensor:
    """``[B, 1, H, W]`` → ``[B, 25]`` singular values of the distinct 5×5 patches (zero-padded right and bottom)."""
    h, w = img.shape[-2:]
    x = F.pad(img, (0, (-w) % 5, 0, (-h) % 5))
    cols = F.unfold(x, 5, stride=5).transpose(1, 2)                     # [B, P, 25]
    values: Tensor = torch.linalg.svdvals(cols)
    return values


def nrqm_features(levels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The three NRQM feature groups of ``[B, 1, H, W]`` float64 gray levels in ``[0, 255]``: ``[B, 18], [B, 45], [B, 75]``."""
    pyr = _gauss_pyramid(levels / 255)
    f1 = torch.cat([_block_dct(im) for im in pyr], dim=1)
    f2 = _global_gsm(levels)
    f3 = torch.cat([_svd_features(im) for im in pyr], dim=1)
    return f1, f2, f3


# --- Regression ---

def _forest_predict(x: Tensor, forest: Sequence[Tensor]) -> Tensor:
    """Mean prediction of a randomForest-MATLAB regression forest for ``[B, D]`` features → ``[B]``."""
    ldau, rdau, upper, avnode, mbest = forest                           # each [nodes, trees]
    b, trees = x.shape[0], ldau.shape[1]
    tree = torch.arange(trees, device=x.device).expand(b, trees)
    node = torch.zeros(b, trees, dtype=torch.long, device=x.device)
    for _ in range(ldau.shape[0]):
        left = ldau[node, tree]
        active = left != 0                                              # terminal nodes have no children
        if not active.any():
            break
        attr = (mbest[node, tree] - 1).clamp(min=0)
        go_left = x.gather(1, attr) <= upper[node, tree]
        nxt = torch.where(go_left, left, rdau[node, tree]) - 1
        node = torch.where(active, nxt, node)
    return avnode[node, tree].mean(dim=1)


class NRQM(Metric):
    """NRQM score per image (higher is better, about 0–10). Runs on any device; the forests are 3 × 2000 trees.

    Args:
        crop_border: Pixels cropped on every side first.
        color_space: Gray channel: ``"yiq"`` (luma, pyiqa's default) or ``"ycbcr"`` (BT.601 Y, as PIRM's PI).
        weights: Regressors: ``"default"`` (``NRQM_Weights.MA2017``, the converted ``.npz``), a member, or a path.

    Raises:
        FileNotFoundError: If the converted model is missing (the message gives the convert command).
        ValueError: (on call) If an image is too small for the steerable pyramid or the normalization guard
            band (needs roughly ``H, W >= 70``).
    """

    mode         : ClassVar[MetricMode] = "nr"
    higher_better: ClassVar[bool] = True

    def __init__(self, *, crop_border: int = 0, color_space: YColorSpace = "yiq",
                 weights: WeightsLike = "default") -> None:
        super().__init__()
        if color_space not in ("yiq", "ycbcr"):
            raise ValueError(f"color_space must be 'yiq' or 'ycbcr', got {color_space!r}")
        self.crop_border = crop_border
        self.color_space = color_space
        spec = resolve_weights(weights, NRQM_Weights)
        if spec is None:
            raise ValueError("NRQM needs its regressors; weights=None is not supported")
        try:
            path = spec.resolve()
        except FileNotFoundError as err:
            raise FileNotFoundError(f"{err}. Convert the official model ({_MODEL_URL}) once: {_CONVERT}") from None
        with np.load(path, allow_pickle=False) as params:
            self.linear = torch.from_numpy(params["linear"]).double()
            self.forests = [
                [torch.from_numpy(params[f"rf{i}_{k}"]).long() for k in ("ldau", "rdau")]
                + [torch.from_numpy(params[f"rf{i}_{k}"]).double() for k in ("upper", "avnode")]
                + [torch.from_numpy(params[f"rf{i}_mbest"]).long()]
                for i in range(3)
            ]

    def extra_repr(self) -> str:
        return f"crop_border={self.crop_border}" + ("" if self.color_space == "yiq" else f", color_space={self.color_space!r}")

    def forward(self, pred: Tensor) -> Tensor:
        if pred.ndim != 4 or pred.shape[1] not in (1, 3) or not pred.is_floating_point():
            raise ValueError(f"pred must be a float [B, 1 or 3, H, W] batch, got {tuple(pred.shape)} {pred.dtype}")
        levels = to_8bit_levels(rgb_to_y(crop_border(pred, self.crop_border).double(), self.color_space))
        h, w = levels.shape[-2:]
        if min(h, w) < 70:
            raise ValueError(f"NRQM needs H, W >= 70, got {h}×{w}")
        device = levels.device
        scores = [torch.ones(levels.shape[0], dtype=torch.float64, device=device)]
        for feats, forest in zip(nrqm_features(levels), self.forests):
            scores.append(_forest_predict(feats, [t.to(device) for t in forest]))
        out: Tensor = torch.stack(scores, dim=1) @ self.linear.to(device)
        return out


# --- Model conversion (run once; needs SciPy, which mon doesn't depend on) ---

def convert_nrqm_model(mat_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Convert the official ``model.mat`` into the ``.npz`` NRQM loads; return its path.

    Keeps ``linear`` and, per forest, ``lDau``, ``rDau``, ``upper``, ``avnode``, ``mbest`` cut to the largest
    tree's node count. Needs SciPy (``uv run --with scipy``).

    Raises:
        ValueError: If the file isn't the official model (SHA-256 mismatch).
    """
    mat_path = Path(mat_path)
    digest = hashlib.sha256(mat_path.read_bytes()).hexdigest()
    if digest != _MODEL_SHA256:
        raise ValueError(f"{mat_path} has SHA-256 {digest}, expected the official model.mat ({_MODEL_SHA256})")
    sio: Any = importlib.import_module("scipy.io")
    model = sio.loadmat(str(mat_path))["model"]
    arrays: dict[str, np.ndarray] = {"linear": np.ravel(model["linear"][0, 0]).astype(np.float64)}
    for i in range(3):
        rf = model["rf"][0, 0][0, i][0, 0]
        nodes = int(np.max(rf["ndtree"]))
        for key, name in (("ldau", "lDau"), ("rdau", "rDau"), ("upper", "upper"), ("avnode", "avnode"),
                          ("mbest", "mbest")):
            arrays[f"rf{i}_{key}"] = np.ascontiguousarray(rf[name][:nodes])
    out = Path(out_path) if out_path is not None else zoo_dir() / NRQM_Weights.MA2017.weights.path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)  # type: ignore[arg-type]
    return out


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or sys.argv[1] != "convert":
        print(f"usage: {_CONVERT} [out.npz]", file=sys.stderr)
        raise SystemExit(2)
    print(convert_nrqm_model(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None))
