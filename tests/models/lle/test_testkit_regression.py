"""Fast regression of every LLE model with released weights on ``testkit_lle`` (100 pairs at 256×256).

Mean PSNR of 8-bit outputs vs targets, measured 2026-11-19 on CUDA without TF32 (``regression_device``); CPU
matches to 1e-4 dB except FLOL (2.3e-2: numerically sensitive, see ``test_flol.py``). These check that the code
doesn't drift; they are not published numbers (those are the ``full`` tests on the whole datasets).
"""

import pytest
import torch

from mon.core.path import datasets_dir
from mon.core.weights import resolve_weights
from mon.models.registry import MODELS, load

CASES = [  # (model, weights, mean PSNR, tolerance)
    ("zero_dce",   "default",   16.4971, 5e-3),
    ("zero_dce++", "default",   16.2052, 5e-3),
    ("sci",        "easy",      11.5024, 5e-3),
    ("sci",        "medium",    13.7053, 5e-3),
    ("sci",        "difficult", 14.6500, 5e-3),
    ("sci++",      "default",   13.5350, 5e-3),
    ("pairlie",    "default",   16.7572, 5e-3),
    ("dccnet",     "default",   21.8438, 5e-3),
    ("hvi_cidnet", "default",   22.4841, 5e-3),
    ("retinexnet", "default",   16.2308, 5e-3),
    ("flol",       "default",   19.1995, 3e-2),
    ("ruas",       "lol_v1",    12.2862, 5e-3),
    ("ruas",       "upe",       11.5522, 5e-3),
    ("ruas",       "darkface",  12.0249, 5e-3),
    pytest.param("clode", "default", 16.8303, 1e-2, marks=pytest.mark.gpu),  # adaptive ODE solver: minutes on CPU
]


@pytest.fixture(scope="module")
def testkit_lle() -> tuple[torch.Tensor, torch.Tensor]:
    try:
        root = datasets_dir() / "testkit" / "lle"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine (unzip archives/datasets/testkit.zip into datasets/)")
    from mon.data.dataset import DATASETS

    ds = DATASETS.build("testkit_lle", "test")
    images  = torch.stack([ds[i]["image"] for i in range(len(ds))]).float() / 255
    targets = torch.stack([ds[i]["target"] for i in range(len(ds))]).float() / 255
    return images, targets


@pytest.mark.slow
@pytest.mark.parametrize(("name", "weights", "expected", "tol"), CASES)
def test_testkit_psnr(name: str, weights: str, expected: float, tol: float,
                      testkit_lle: tuple[torch.Tensor, torch.Tensor], regression_device: torch.device):
    if name == "clode" and regression_device.type != "cuda":
        pytest.skip("needs CUDA")
    enum = MODELS.get(name).weights_enum  # type: ignore[attr-defined]
    spec = resolve_weights(weights, enum)
    assert spec is not None
    try:
        spec.resolve(download=False)
    except (FileNotFoundError, RuntimeError):
        pytest.skip("released weights not in the zoo")
    key = str(spec.meta.get("output_key", "enhanced"))  # the checkpoint's official output
    images, targets = testkit_lle
    model = load(name, weights, device=regression_device)
    with torch.no_grad():
        out = torch.cat([model(images[i:i + 8].to(regression_device))[key].clamp(0, 1).cpu()
                         for i in range(0, len(images), 8)])
    y = (out * 255).round() / 255
    psnr = 10 * torch.log10(1 / ((y - targets) ** 2).mean(dim=(1, 2, 3)))
    assert len(psnr) == 100
    assert psnr.mean().item() == pytest.approx(expected, abs=tol)
