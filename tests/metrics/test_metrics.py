"""Tests for ``mon.metrics``: properties, the registry, and parity with pyiqa, scikit-image, lpips and BasicSR.

``REFERENCE`` comes from ``tests/metrics/metrics_reference.py`` (see its docstring). Expected gaps, all from
reference float32 steps (ours are float64, as MATLAB):

- pyiqa PSNR: float32 MSE (~1e-6 dB); scikit-image PSNR: our ``eps = 1e-8`` (2e-6 dB here);
- pyiqa luma SSIM: pyiqa rounds a float32 luma (a few pixels near .5 land on the other level) and uses a
  float32 Gaussian window (3e-7 on real images); scikit-image RGB SSIM: exact;
- NIQE: same float32 luma rounding in pyiqa (most of the gap) and float32 resizing in BasicSR (≤ 0.1%).
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.core.path import datasets_dir
from mon.data.io import read_image
from mon.metrics import LPIPS, METRICS, NIQE, PSNR, SSIM, gaussian_window, matlab_resize_matrix, psnr, ssim
from mon.metrics.base import crop_border, rgb_to_luma, to_8bit_levels

_spec = importlib.util.spec_from_file_location("metrics_reference", Path(__file__).with_name("metrics_reference.py"))
assert _spec is not None and _spec.loader is not None
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_images, REAL_STEMS = _ref.reference_images, _ref.REAL_STEMS

# Generated (tests/metrics/metrics_reference.py) with pyiqa 0.1.16, lpips 0.1.4, torch 2.15.0+cu132, BasicSR checkout.
REFERENCE: dict = {'pyiqa': {'psnr': [16.68532371520996, 16.69416618347168],
           'psnry': [16.63145637512207, 16.62883758544922],
           'ssim': [0.8673559398609013, 0.8680106816189251],
           'ssimc': [0.8203270265195526, 0.8210184466865551],
           'lpips': [0.068680040538311, 0.08534465730190277],
           'lpips-vgg': [0.15297581255435944, 0.1487678736448288],
           'niqe': [548.1413555523488, 540.8499010486305]},
 'skimage': {'psnr': [16.685325743791843, 16.694167641078753], 'ssim': [0.820327025115919, 0.8210184452442199]},
 'lpips': {'alex': [0.068680040538311, 0.08534465730190277], 'vgg': [0.15297581255435944, 0.1487678736448288]},
 'lol_v1': {'111': {'psnr': 5.6122307777404785,
                    'psnry': 5.604128837585449,
                    'ssim': 0.19705209756606873,
                    'ssimc': 0.1913987794618394,
                    'lpips': 0.5379783511161804,
                    'lpips-vgg': 0.3945755958557129,
                    'niqe_pred': 8.65203596190973,
                    'niqe_target': 4.468111856844746},
            '146': {'psnr': 6.629547595977783,
                    'psnry': 6.591195583343506,
                    'ssim': 0.26062749356538095,
                    'ssimc': 0.2621304370184836,
                    'lpips': 0.38978832960128784,
                    'lpips-vgg': 0.31864967942237854,
                    'niqe_pred': 6.788621405480913,
                    'niqe_target': 6.227620277730135},
            '179': {'psnr': 12.377197265625,
                    'psnry': 13.64020824432373,
                    'ssim': 0.43589527527872357,
                    'ssimc': 0.4352048323566422,
                    'lpips': 0.3401371240615845,
                    'lpips-vgg': 0.21591687202453613,
                    'niqe_pred': 4.984233121553272,
                    'niqe_target': 3.9365726199087168}},
 'basicsr_niqe_gray': [548.0347261782667, 540.6891405434175]}


def _close(got: torch.Tensor, want: list[float], *, abs: float = 0, rel: float = 0) -> None:
    assert got.tolist() == pytest.approx(want, abs=abs, rel=rel)


# --- Properties ---

def test_identical_images():
    x = torch.rand(2, 3, 32, 32)
    assert psnr(x, x).tolist() == pytest.approx([80.0, 80.0])  # eps = 1e-8, as pyiqa
    assert ssim(x, x).tolist() == [1.0, 1.0] and ssim(x, x, y_channel=False).tolist() == [1.0, 1.0]
    assert LPIPS(weights=None, pretrained_backbone=False)(x, x).tolist() == [0.0, 0.0]


def test_psnr_known_value_and_options():
    x = torch.zeros(1, 3, 10, 10)
    y = x.clone()
    y[..., 0, :] = 1.0  # only the first row differs
    assert psnr(x, y).item() == pytest.approx(10.0, abs=1e-6)              # MSE 0.1
    assert psnr(x, y, border=1).item() == pytest.approx(80.0)              # cropped away
    assert psnr(x, y, y_channel=True).item() == pytest.approx(10.0, abs=1e-6)  # gray stays gray


def test_luma_and_levels():
    assert rgb_to_luma(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)).item() == pytest.approx(0.299)
    g = torch.rand(1, 1, 4, 4)
    assert rgb_to_luma(g) is g
    levels = torch.arange(256, dtype=torch.float64)
    assert torch.equal(to_8bit_levels((levels / 255).float()), levels)  # decoded 8-bit files round-trip
    with pytest.raises(ValueError, match="leaves nothing"):
        crop_border(torch.zeros(1, 1, 4, 4), 2)


@pytest.mark.parametrize(("pred", "target", "match"), [
    (torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 9), "shapes differ"),
    (torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), "must be"),
    (torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8), "must be"),
])
def test_bad_inputs(pred: torch.Tensor, target: torch.Tensor, match: str):
    with pytest.raises(ValueError, match=match):
        psnr(pred, target)


def test_ssim_window_and_size_checks():
    w = gaussian_window()
    assert w.shape == (11, 11) and w.sum().item() == pytest.approx(1.0) and w[5, 5] == w.max()
    with pytest.raises(ValueError, match="smaller than the 11"):
        ssim(torch.zeros(1, 1, 10, 30), torch.zeros(1, 1, 10, 30))


def test_ssim_downsample_matches_pooled_input():
    x, y = torch.rand(1, 1, 512, 512), torch.rand(1, 1, 512, 512)
    pool = torch.nn.functional.avg_pool2d
    # factor round(512 / 256) = 2; pooling 8-bit levels equals pooling the images first only up to rounding.
    assert ssim(x, y, downsample=True).item() != ssim(x, y).item()
    assert ssim(pool(x, 2), pool(y, 2)).item() == pytest.approx(ssim(x, y, downsample=True).item(), abs=5e-3)


def test_matlab_resize_matrix():
    m = matlab_resize_matrix(10, 0.5)
    assert m.shape == (5, 10)
    assert torch.allclose(m.sum(dim=1), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(m, m.flip(0, 1))  # symmetric kernel, symmetric grid
    up = matlab_resize_matrix(4, 2.0, antialiasing=False)
    assert up.shape == (8, 4) and torch.allclose(up.sum(dim=1), torch.ones(8, dtype=torch.float64))


def test_niqe_needs_two_blocks():
    from mon.metrics.niqe import _niqe_one

    with pytest.raises(ValueError, match="at least two 96"):
        _niqe_one(torch.zeros(96, 100, dtype=torch.float64), torch.zeros(36), torch.eye(36), gaussian_window(7, 7 / 6)[None, None])


# --- Registry ---

def test_registry_metadata_matches_classes():
    assert METRICS.names() == ["lpips", "niqe", "psnr", "ssim"]
    for entry in METRICS.entries():
        assert entry.meta["mode"] == entry.obj.mode
        assert entry.meta["higher_better"] == entry.obj.higher_better


def test_repr_and_build():
    assert repr(METRICS.build("ssim", y_channel=False)) == "SSIM(y_channel=False, crop_border=0, downsample=False)"


# --- Parity on the synthetic pair ---

def test_psnr_parity():
    pred, target = reference_images()
    _close(PSNR()(pred, target), REFERENCE["pyiqa"]["psnr"], abs=1e-5)
    _close(PSNR()(pred, target), REFERENCE["skimage"]["psnr"], abs=1e-5)
    _close(PSNR(y_channel=True)(pred, target), REFERENCE["pyiqa"]["psnry"], abs=1e-5)


def test_ssim_parity():
    pred, target = reference_images()
    _close(SSIM(y_channel=False)(pred, target), REFERENCE["skimage"]["ssim"], abs=1e-10)
    _close(SSIM(y_channel=False)(pred, target), REFERENCE["pyiqa"]["ssimc"], abs=1e-8)
    _close(SSIM()(pred, target), REFERENCE["pyiqa"]["ssim"], abs=1e-5)


@pytest.mark.slow  # downloads the LPIPS layers and torchvision's AlexNet/VGG-16 (~600 MB) once
@pytest.mark.parametrize(("net", "key"), [("alex", "lpips"), ("vgg", "lpips-vgg")])
def test_lpips_parity(net: str, key: str):
    pred, target = reference_images()
    got = LPIPS(net)(pred, target)  # type: ignore[arg-type]
    _close(got, REFERENCE["lpips"][net], abs=1e-6)
    _close(got, REFERENCE["pyiqa"][key], abs=1e-6)


@pytest.mark.slow  # downloads the pristine model (12 kB) once
def test_niqe_parity_and_window():
    import numpy as np

    from mon.metrics.niqe import NIQE_Weights

    pred, _ = reference_images()
    got = NIQE()(pred)
    _close(got, REFERENCE["pyiqa"]["niqe"], rel=1e-3)
    _close(got, REFERENCE["basicsr_niqe_gray"], rel=1e-3)
    with np.load(NIQE_Weights.PRISTINE.weights.resolve()) as params:
        assert np.allclose(params["gaussian_window"], gaussian_window(7, 7 / 6).numpy(), atol=1e-12)


# --- Parity on natural images (LOL-v1 test pairs) ---

def _lol_pair(stem: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        root = datasets_dir() / "lol_v1" / "test"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    load = lambda p: read_image(p).float()[None] / 255  # noqa: E731
    return load(root / "images" / f"{stem}.png"), load(root / "targets" / f"{stem}.png")


@pytest.mark.slow
@pytest.mark.parametrize("stem", REAL_STEMS)
def test_parity_on_lol_v1(stem: str):
    low, high = _lol_pair(stem)
    want = REFERENCE["lol_v1"][stem]
    assert PSNR()(low, high).item() == pytest.approx(want["psnr"], abs=1e-5)
    assert PSNR(y_channel=True)(low, high).item() == pytest.approx(want["psnry"], abs=1e-5)
    assert SSIM()(low, high).item() == pytest.approx(want["ssim"], abs=1e-5)
    assert SSIM(y_channel=False)(low, high).item() == pytest.approx(want["ssimc"], abs=1e-6)
    assert LPIPS()(low, high).item() == pytest.approx(want["lpips"], abs=1e-6)
    assert LPIPS("vgg")(low, high).item() == pytest.approx(want["lpips-vgg"], abs=1e-6)
    niqe = NIQE()
    assert niqe(low).item() == pytest.approx(want["niqe_pred"], rel=2e-3)
    assert niqe(high).item() == pytest.approx(want["niqe_target"], rel=2e-3)
    assert math.isfinite(niqe(torch.cat([low, high])).sum().item())
