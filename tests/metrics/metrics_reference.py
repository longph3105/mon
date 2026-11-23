"""Reference inputs and scores for the metric parity tests (no ``mon`` imports).

``REFERENCE`` in ``test_metrics.py`` comes from third-party implementations run on the inputs below:

- pyiqa 0.1.16 (https://github.com/chaofengc/IQA-PyTorch, PolyForm Noncommercial: a reference only,
  never a dependency): ``psnr``, ``psnry``, ``ssim``, ``ssimc``, ``lpips``, ``lpips-vgg``, ``niqe``;
- scikit-image: ``peak_signal_noise_ratio`` and ``structural_similarity`` (Gaussian, σ 1.5, population
  covariance), on the RGB 8-bit images;
- the official ``lpips`` 0.1.4 package (``normalize=True``), AlexNet and VGG;
- BasicSR's ``basicsr/metrics/niqe.py`` (``calculate_niqe``, ``convert_to="gray"``) from a checkout.

``lol_v1`` repeats pyiqa on natural images: LOL-v1 test pairs ``REAL_STEMS`` (input as ``pred``, target as
``target``; NIQE on both), read with OpenCV from ``<lol_v1>/test/{images,targets}/<stem>.png``.

Running it executes downloaded third-party code: get the owner's go-ahead first. pyiqa and lpips download
weights into ``TORCH_HOME``; point it at a scratch directory. Regenerate with:

    TORCH_HOME=<scratch> uv run --no-project --python 3.12 --with pyiqa==0.1.16 --with lpips==0.1.4 \\
        --with scikit-image --with opencv-python-headless python -I tests/metrics/metrics_reference.py \\
        <BasicSR checkout> <DATASETS_DIR>/lol_v1
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

_fmt_spec = importlib.util.spec_from_file_location(
    "reference_format", Path(__file__).parents[1] / "reference_format.py",
)
assert _fmt_spec is not None and _fmt_spec.loader is not None
_fmt = importlib.util.module_from_spec(_fmt_spec)
_fmt_spec.loader.exec_module(_fmt)
format_literal = _fmt.format_literal

SIZE = (2, 3, 200, 296)
REAL_STEMS = ("111", "146", "179")  # NIQE needs whole 96×96 blocks: 2×3 of them, with a remainder to crop


def _hash_noise(shape: tuple[int, ...], salt: float) -> torch.Tensor:
    """Deterministic pseudo-noise in [0, 1) (no RNG: same values on every platform and torch version)."""
    idx = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return torch.frac(torch.sin(idx * 12.9898 + salt * 78.233) * 43758.5453)


def reference_images() -> tuple[torch.Tensor, torch.Tensor]:
    """``(pred, target)``: ``SIZE`` float32 RGB in [0, 1] on exact 8-bit levels, as decoded files are.

    ``target`` is a smooth scene with texture; ``pred`` a darker, noisier, channel-shifted version of it.
    """
    b, c, h, w = SIZE
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    target = torch.empty(SIZE, dtype=torch.float64)
    for n in range(b):
        for k in range(c):
            wave = torch.sin(0.05 * (k + 1) * i + 0.031 * (n + 1) * j + k) * torch.cos(0.017 * j - 0.043 * i)
            target[n, k] = 0.45 + 0.3 * wave + 0.12 * torch.sin(0.6 * i + 0.9 * j + n)
    target = (target + 0.08 * (_hash_noise(SIZE, 1.0) - 0.5)).clamp(0, 1)
    tint = torch.tensor([0.02, -0.01, 0.03]).view(1, 3, 1, 1)
    pred = 0.8 * target.pow(1.15) + 0.06 * (_hash_noise(SIZE, 2.0) - 0.5) + tint

    def quant(t: torch.Tensor) -> torch.Tensor:
        return (t.clamp(0, 1) * 255).round() / 255

    return quant(pred).float(), quant(target).float()


def _basicsr_niqe(checkout: str) -> Any:
    """``calculate_niqe`` from a BasicSR checkout without running ``basicsr/__init__.py`` (it imports everything)."""
    root = Path(checkout).resolve() / "basicsr"
    for name, sub in (("basicsr", ""), ("basicsr.utils", "utils"), ("basicsr.metrics", "metrics")):
        mod = ModuleType(name)
        mod.__path__ = [str(root / sub)]
        sys.modules[name] = mod
    from basicsr.utils.color_util import bgr2ycbcr  # official code
    sys.modules["basicsr.utils"].bgr2ycbcr = bgr2ycbcr  # type: ignore[attr-defined]
    from basicsr.metrics.niqe import calculate_niqe  # official code
    return calculate_niqe


def read_rgb(path: Path) -> torch.Tensor:
    """``[1, 3, H, W]`` float32 in [0, 1] from an 8-bit image file (OpenCV, BGR → RGB)."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def main(checkout: str, lol_v1: str) -> None:
    import lpips
    import pyiqa
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    pred, target = reference_images()
    result: dict[str, Any] = {"pyiqa": {}}
    for name in ("psnr", "psnry", "ssim", "ssimc", "lpips", "lpips-vgg", "niqe"):
        metric = pyiqa.create_metric(name, device="cpu", as_loss=False)
        with torch.no_grad():
            score = metric(pred) if name == "niqe" else metric(pred, target)
        result["pyiqa"][name] = [float(v) for v in score.flatten()]

    p8 = (pred.double() * 255).round().numpy().transpose(0, 2, 3, 1)
    t8 = (target.double() * 255).round().numpy().transpose(0, 2, 3, 1)
    pairs = list(zip(p8, t8, strict=True))
    result["skimage"] = {
        "psnr": [float(peak_signal_noise_ratio(t, p, data_range=255)) for p, t in pairs],
        "ssim": [float(structural_similarity(t, p, data_range=255, channel_axis=2, gaussian_weights=True,
                                             sigma=1.5, use_sample_covariance=False)) for p, t in pairs],
    }
    result["lpips"] = {}
    for net in ("alex", "vgg"):
        model = lpips.LPIPS(net=net, version="0.1", verbose=False).eval()
        with torch.no_grad():
            result["lpips"][net] = [float(v) for v in model(pred, target, normalize=True).flatten()]

    root = Path(lol_v1)
    result["lol_v1"] = {}
    names = ("psnr", "psnry", "ssim", "ssimc", "lpips", "lpips-vgg", "niqe")
    metrics = {n: pyiqa.create_metric(n, device="cpu", as_loss=False) for n in names}
    for stem in REAL_STEMS:
        low = read_rgb(root / "test" / "images" / f"{stem}.png")
        high = read_rgb(root / "test" / "targets" / f"{stem}.png")
        with torch.no_grad():
            scores = {n: float(m(low, high)) for n, m in metrics.items() if n != "niqe"}
            scores["niqe_pred"]   = float(metrics["niqe"](low))
            scores["niqe_target"] = float(metrics["niqe"](high))
        result["lol_v1"][stem] = scores

    calculate_niqe = _basicsr_niqe(checkout)
    # BasicSR expects BGR HWC in [0, 255]; "gray" = cv2 BGR2GRAY (luma), then rounding, as MATLAB.
    result["basicsr_niqe_gray"] = [float(calculate_niqe(p[..., ::-1].copy(), 0, input_order="HWC", convert_to="gray"))
                                   for p in p8]

    print(f"# Generated with pyiqa {pyiqa.__version__}, lpips 0.1.4, torch {torch.__version__}, BasicSR checkout.")
    print(format_literal("REFERENCE", result))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
