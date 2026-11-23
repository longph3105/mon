"""Tests for ``mon.models.lle.zero_dce``.

Parity uses reference values (``OFFICIAL``) computed with the released weights on
``reference_input()``; ``zero_dce_reference.py`` documents how to regenerate them.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mon.core.weights import WeightsEnum
from mon.models.lle.zero_dce import (
    ZeroDCE, ZeroDCELoss, ZeroDCEPP, ZeroDCEPP_Weights, ZeroDCE_Weights, apply_curves,
)
from mon.nn.losses.zero_reference import (
    ColorConstancyLoss, ExposureControlLoss, IlluminationSmoothnessLoss, SpatialConsistencyLoss,
)
from mon.nn.tiling import tiled_forward

_spec = importlib.util.spec_from_file_location("zero_dce_reference", Path(__file__).with_name("zero_dce_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# Reference outputs and state-dict shapes: the output of ``zero_dce_reference.py``. Generated from mon_old
# (torch 2.14) and, identically, from the official model.py files (torch 2.15, 2026-11-18); both loaded the
# released checkpoints with strict=True.
# Generated from mon_old with torch 2.14.0.
OFFICIAL: dict = {
    "zero_dce_state": {
        "e_conv1.weight": [32, 3, 3, 3],
        "e_conv1.bias": [32],
        "e_conv2.weight": [32, 32, 3, 3],
        "e_conv2.bias": [32],
        "e_conv3.weight": [32, 32, 3, 3],
        "e_conv3.bias": [32],
        "e_conv4.weight": [32, 32, 3, 3],
        "e_conv4.bias": [32],
        "e_conv5.weight": [32, 64, 3, 3],
        "e_conv5.bias": [32],
        "e_conv6.weight": [32, 64, 3, 3],
        "e_conv6.bias": [32],
        "e_conv7.weight": [24, 64, 3, 3],
        "e_conv7.bias": [24],
    },
    "zero_dce": {
        "enhanced": {
            "shape": [2, 3, 24, 36],
            "sum": 2545.1896739825606,
            "sumsq": 1407.2926001745436,
            "values": [
                0.3514877259731293,
                0.5605628490447998,
                0.17432542145252228,
                0.4480573236942291,
                0.5827096700668335,
                0.5191973447799683,
                0.6226564049720764,
            ],
        },
        "curve": {
            "shape": [2, 24, 24, 36],
            "sum": -7490.1650299578905,
            "sumsq": 1358.2244093124204,
            "values": [
                -0.13253331184387207,
                -0.16230224072933197,
                -0.1840895265340805,
                -0.1811741441488266,
                -0.18804442882537842,
                -0.21062670648097992,
                -0.1615048497915268,
            ],
        },
    },
    "zero_dce++_state": {
        "e_conv1.depth_conv.weight": [3, 1, 3, 3],
        "e_conv1.depth_conv.bias": [3],
        "e_conv1.point_conv.weight": [32, 3, 1, 1],
        "e_conv1.point_conv.bias": [32],
        "e_conv2.depth_conv.weight": [32, 1, 3, 3],
        "e_conv2.depth_conv.bias": [32],
        "e_conv2.point_conv.weight": [32, 32, 1, 1],
        "e_conv2.point_conv.bias": [32],
        "e_conv3.depth_conv.weight": [32, 1, 3, 3],
        "e_conv3.depth_conv.bias": [32],
        "e_conv3.point_conv.weight": [32, 32, 1, 1],
        "e_conv3.point_conv.bias": [32],
        "e_conv4.depth_conv.weight": [32, 1, 3, 3],
        "e_conv4.depth_conv.bias": [32],
        "e_conv4.point_conv.weight": [32, 32, 1, 1],
        "e_conv4.point_conv.bias": [32],
        "e_conv5.depth_conv.weight": [64, 1, 3, 3],
        "e_conv5.depth_conv.bias": [64],
        "e_conv5.point_conv.weight": [32, 64, 1, 1],
        "e_conv5.point_conv.bias": [32],
        "e_conv6.depth_conv.weight": [64, 1, 3, 3],
        "e_conv6.depth_conv.bias": [64],
        "e_conv6.point_conv.weight": [32, 64, 1, 1],
        "e_conv6.point_conv.bias": [32],
        "e_conv7.depth_conv.weight": [64, 1, 3, 3],
        "e_conv7.depth_conv.bias": [64],
        "e_conv7.point_conv.weight": [3, 64, 1, 1],
        "e_conv7.point_conv.bias": [3],
    },
    "zero_dce++_x1": {
        "enhanced": {
            "shape": [2, 3, 24, 36],
            "sum": 2800.162372507155,
            "sumsq": 1690.8033572851841,
            "values": [
                0.4794083535671234,
                0.6184284687042236,
                0.19538265466690063,
                0.46712175011634827,
                0.6103564500808716,
                0.6126052737236023,
                0.7267330288887024,
            ],
        },
        "curve": {
            "shape": [2, 3, 24, 36],
            "sum": -1088.234554052353,
            "sumsq": 230.63185521689485,
            "values": [
                -0.20095093548297882,
                -0.1888299584388733,
                -0.20050407946109772,
                -0.18757757544517517,
                -0.19681775569915771,
                -0.24221017956733704,
                -0.23120297491550446,
            ],
        },
    },
    "zero_dce++_x12": {
        "enhanced": {
            "shape": [2, 3, 24, 36],
            "sum": 2952.6146163269877,
            "sumsq": 1863.3860186689012,
            "values": [
                0.553777277469635,
                0.6925702095031738,
                0.19524380564689636,
                0.511510968208313,
                0.6300845742225647,
                0.6127456426620483,
                0.7580097913742065,
            ],
        },
        "curve": {
            "shape": [2, 3, 24, 36],
            "sum": -1177.8005364239216,
            "sumsq": 269.61276863250623,
            "values": [
                -0.23968997597694397,
                -0.22916050255298615,
                -0.20037809014320374,
                -0.21073047816753387,
                -0.20722538232803345,
                -0.24228443205356598,
                -0.2504587471485138,
            ],
        },
    },
}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-5) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(enum: type[WeightsEnum]):
    """The released checkpoint's state dict, or skip if it isn't on disk (no download in tests)."""
    try:
        return enum.DEFAULT.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity with the official code ---

@pytest.mark.parametrize(("model", "key"), [(ZeroDCE(), "zero_dce_state"), (ZeroDCEPP(), "zero_dce++_state")])
def test_state_dict_matches_official_names_and_shapes(model, key: str):
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == OFFICIAL[key]


def test_zero_dce_matches_official_with_released_weights():
    model = ZeroDCE().eval()
    model.load_state_dict(released(ZeroDCE_Weights))
    with torch.no_grad():
        out = model(reference_input())
    assert_matches(out["enhanced"], OFFICIAL["zero_dce"]["enhanced"])
    assert_matches(out["curve"], OFFICIAL["zero_dce"]["curve"])


@pytest.mark.parametrize("scale", [1, 12])
def test_zero_dce_pp_matches_official_with_released_weights(scale: int):
    model = ZeroDCEPP(scale_factor=scale).eval()
    model.load_state_dict(released(ZeroDCEPP_Weights))
    with torch.no_grad():
        out = model(reference_input())
    assert_matches(out["enhanced"], OFFICIAL[f"zero_dce++_x{scale}"]["enhanced"])
    assert_matches(out["curve"], OFFICIAL[f"zero_dce++_x{scale}"]["curve"])


# --- Behavior ---

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize(("size", "scale"), [((17, 23), 1), ((30, 45), 4), ((25, 37), 12), ((12, 12), 12)])
def test_zero_dce_pp_any_size(size: tuple[int, int], scale: int):
    out = ZeroDCEPP(scale_factor=scale)(torch.rand(2, 3, *size))
    assert out["enhanced"].shape == (2, 3, *size) and out["curve"].shape == (2, 3, *size)


def test_zero_dce_pp_upsampling_equals_official_when_size_divides():
    curve = torch.rand(1, 3, 2, 3)
    official = torch.nn.UpsamplingBilinear2d(scale_factor=12)(curve)
    assert torch.equal(F.interpolate(curve, size=(24, 36), mode="bilinear", align_corners=True), official)


def test_zero_dce_pp_rejects_bad_scale_and_tiny_inputs():
    with pytest.raises(ValueError, match="scale_factor must be >= 1"):
        ZeroDCEPP(scale_factor=0.5)
    with pytest.raises(ValueError, match="smaller than scale_factor"):
        ZeroDCEPP(scale_factor=12)(torch.rand(1, 3, 8, 40))


@pytest.mark.parametrize("model", [ZeroDCE(), ZeroDCEPP(), ZeroDCE(num_iters=4, hidden_dim=8), ZeroDCE(channels=1)])
def test_shapes_and_output_range(model):
    c = model.channels
    x = torch.rand(2, c, 19, 26)
    out = model(x)
    assert out["enhanced"].shape == x.shape
    assert out["curve"].shape[1] == (c * model.num_iters if isinstance(model, ZeroDCE) else c)
    assert out["curve"].abs().max() <= 1
    assert out["enhanced"].min() >= 0 and out["enhanced"].max() <= 1


def test_apply_curves_formula_and_range():
    x = torch.linspace(0, 1, 11).view(1, 1, 1, 11)
    assert torch.allclose(apply_curves(x, [torch.full_like(x, 1.0)]), x**2)
    assert torch.allclose(apply_curves(x, [torch.full_like(x, -1.0)]), 2 * x - x**2)
    assert torch.equal(apply_curves(x, []), x)
    for r in (-1.0, -0.3, 0.7, 1.0):  # any r in [-1, 1] keeps [0, 1]
        y = apply_curves(x, [torch.full_like(x, r)] * 8)
        assert y.min() >= 0 and y.max() <= 1


def test_batch_samples_are_independent():
    model = ZeroDCEPP(scale_factor=4).eval()
    x = torch.rand(3, 3, 32, 32)
    with torch.no_grad():
        batched, single = model(x)["enhanced"], model(x[1:2])["enhanced"]
    assert torch.allclose(batched[1:2], single, atol=1e-6)


def test_initialization_follows_official_code():
    zdce, zdcepp = ZeroDCE(), ZeroDCEPP()
    assert 0.015 < zdce.e_conv2.weight.std().item() < 0.025  # N(0, 0.02)
    default = torch.nn.Conv2d(32, 32, 1)  # Zero-DCE++ keeps PyTorch's default (kaiming uniform)
    assert zdcepp.e_conv2.point_conv.weight.abs().max() <= default.weight.abs().max() * 1.01


def test_gradients_reach_every_parameter():
    for model in (ZeroDCE(), ZeroDCEPP(scale_factor=2)):
        x = torch.rand(2, 3, 16, 16)
        out = model(x)
        ZeroDCELoss()(out["enhanced"], out["curve"], x).backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_tiled_forward_matches_full_image_for_zero_dce():
    # Zero-DCE is fully convolutional with a 7-layer (15 px) receptive field: tiles with enough overlap
    # reproduce the full-image output away from tile seams.
    model = ZeroDCE().eval()
    x = torch.rand(1, 3, 64, 80)
    with torch.no_grad():
        full = model(x)
        tiled = tiled_forward(model, x, tile=48, overlap=32)
    assert set(tiled) == {"enhanced", "curve"}
    assert torch.allclose(tiled["enhanced"], full["enhanced"], atol=2e-2)


# --- Loss ---

def test_loss_matches_weighted_official_terms():
    x = torch.rand(4, 3, 32, 32) * 0.4
    out = ZeroDCE()(x)
    enhanced, curve = out["enhanced"], out["curve"]
    expected = (SpatialConsistencyLoss()(enhanced, x) + 10 * ExposureControlLoss()(enhanced)
                + 5 * ColorConstancyLoss()(enhanced) + 200 * IlluminationSmoothnessLoss()(curve))
    loss = ZeroDCELoss()
    assert torch.allclose(loss(enhanced, curve, x), expected)
    terms = loss.terms(enhanced, curve, x)
    assert set(terms) == {"spatial", "exposure", "color", "smoothness"}
    assert torch.allclose(sum(terms.values()), expected)
    pp = ZeroDCELoss(tv_weight=1600)(enhanced, curve, x)
    assert torch.allclose(pp - expected, 1400 * IlluminationSmoothnessLoss()(curve))


def test_loss_reductions():
    x = torch.rand(3, 3, 32, 32) * 0.4
    out = ZeroDCEPP()(x)
    per_sample = ZeroDCELoss(reduction="none")(out["enhanced"], out["curve"], x)
    assert per_sample.shape == (3,)
    assert torch.allclose(ZeroDCELoss()(out["enhanced"], out["curve"], x), per_sample.mean())
    assert torch.allclose(ZeroDCELoss(reduction="sum")(out["enhanced"], out["curve"], x), per_sample.sum())
    with pytest.raises(ValueError, match="reduction"):
        ZeroDCELoss(reduction="max")  # type: ignore[arg-type]


# --- Weights ---

@pytest.mark.parametrize(("enum", "path"), [
    (ZeroDCE_Weights, "lle/zero_dce/zero_dce_sice_me.pt"),
    (ZeroDCEPP_Weights, "lle/zero_dce++/zero_dce++_sice_me.pt"),
])
def test_weights_enums(enum: type[WeightsEnum], path: str):
    w = enum.DEFAULT.weights
    assert enum.DEFAULT is enum.SICE_ME and w.path == path
    assert w.url and w.url.startswith("https://raw.githubusercontent.com/Li-Chongyi/") and len(w.sha256 or "") == 64
    assert ZeroDCE.weights_enum is ZeroDCE_Weights and ZeroDCEPP.weights_enum is ZeroDCEPP_Weights


def test_released_weights_on_disk_match_sha256():
    from mon.core.fileio import sha256sum
    for enum in (ZeroDCE_Weights, ZeroDCEPP_Weights):
        try:
            path = enum.DEFAULT.weights.resolve(download=False)
        except FileNotFoundError:
            pytest.skip("released weights not available")
        assert sha256sum(path) == enum.DEFAULT.weights.sha256


# --- Real data (LOL-v1 test split, released weights) ---

@pytest.mark.full
@pytest.mark.parametrize(("name", "kwargs", "expected_psnr"), [
    ("zero_dce", {}, 14.861),
    ("zero_dce++", {}, 15.142),
    ("zero_dce++", {"scale_factor": 12}, 15.348),  # full 400×600: no crop to a multiple of 12 needed
])
def test_lol_v1_psnr_regression(name: str, kwargs: dict, expected_psnr: float, regression_device: torch.device):
    """Mean PSNR of 8-bit outputs vs targets; values measured 2026-11-18, outputs identical to mon_old."""
    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv1
    from mon.models.registry import MODELS, load
    try:
        root = datasets_dir() / LOLv1.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    ds = LOLv1("test")
    released(MODELS.get(name).weights_enum)  # skip without the weights
    model = load(name, **kwargs, device=regression_device)
    psnrs = []
    with torch.no_grad():
        for sample in ds:
            x      = (sample["image"].float() / 255).to(regression_device)
            target = (sample["target"].float() / 255).to(regression_device)
            y = (model(x[None])["enhanced"][0].clamp(0, 1) * 255).round() / 255
            psnrs.append(10 * torch.log10(1 / F.mse_loss(y, target)).item())
    assert len(psnrs) == 15
    assert sum(psnrs) / len(psnrs) == pytest.approx(expected_psnr, abs=5e-3)
