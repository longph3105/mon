"""Tests for ``mon.models.lle.retinexnet``: parity with the PyTorch reimplementation and checkpoint loading."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.models.lle.retinexnet import RetinexNet, RetinexNet_Weights
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location(
    "retinexnet_reference", Path(__file__).with_name("retinexnet_reference.py"),
)
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# The reimplementation's DecomNet/RelightNet (aasharma90/RetinexNet_PyTorch 2267510), CPU, per checkpoint file.
# Output of retinexnet_reference.py.
# Generated from the reimplementation with torch 2.15.0+cu132.
REIMPL: dict = {
    "retinexnet_lol_v1_psnr.pt": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 2126.9225109517574,
            "sumsq": 1287.967523730904,
            "values": [
                0.7155070304870605,
                0.7669087052345276,
                0.3448498249053955,
                0.09731335937976837,
                0.5952599048614502,
                0.5071680545806885,
                0.8495768904685974,
            ],
        },
        "reflectance": {
            "shape": [2, 3, 21, 30],
            "sum": 1059.445309203118,
            "sumsq": 319.93193236098585,
            "values": [
                0.3688491880893707,
                0.3843621611595154,
                0.17197474837303162,
                0.05002909526228905,
                0.30347004532814026,
                0.25253427028656006,
                0.4388158321380615,
            ],
        },
        "illumination": {
            "shape": [2, 1, 21, 30],
            "sum": 977.600623190403,
            "sumsq": 758.5008967781566,
            "values": [
                0.7757745981216431,
                0.7716432213783264,
                0.7747400403022766,
                0.7786819934844971,
                0.7758656740188599,
                0.7771843671798706,
                0.7742763757705688,
            ],
        },
        "illumination_adjusted": {
            "shape": [2, 1, 21, 30],
            "sum": 2530.2866376638412,
            "sumsq": 5082.735741142367,
            "values": [
                1.9398362636566162,
                1.9952762126922607,
                2.005235195159912,
                1.945135235786438,
                2.0199925899505615,
                2.0083138942718506,
                1.9360671043395996,
            ],
        },
    },
    "retinexnet_lol_v1_ssim.pt": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 1840.3364009931684,
            "sumsq": 963.2821097859448,
            "values": [
                0.6311901807785034,
                0.6696088910102844,
                0.29859575629234314,
                0.08605816960334778,
                0.5221180319786072,
                0.44127875566482544,
                0.7279831171035767,
            ],
        },
        "reflectance": {
            "shape": [2, 3, 21, 30],
            "sum": 1059.445309203118,
            "sumsq": 319.93193236098585,
            "values": [
                0.3688491880893707,
                0.3843621611595154,
                0.17197474837303162,
                0.05002909526228905,
                0.30347004532814026,
                0.25253427028656006,
                0.4388158321380615,
            ],
        },
        "illumination": {
            "shape": [2, 1, 21, 30],
            "sum": 977.600623190403,
            "sumsq": 758.5008967781566,
            "values": [
                0.7757745981216431,
                0.7716432213783264,
                0.7747400403022766,
                0.7786819934844971,
                0.7758656740188599,
                0.7771843671798706,
                0.7742763757705688,
            ],
        },
        "illumination_adjusted": {
            "shape": [2, 1, 21, 30],
            "sum": 2190.7770924568176,
            "sumsq": 3810.085570109095,
            "values": [
                1.7112419605255127,
                1.7421300411224365,
                1.7362767457962036,
                1.7201623916625977,
                1.7340213060379028,
                1.747401475906372,
                1.658971905708313,
            ],
        },
    },
    "retinexnet_reimpl_9200.pt": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 1999.5950404703617,
            "sumsq": 1165.3012192794708,
            "values": [
                0.4650900661945343,
                0.6927893161773682,
                0.5222093462944031,
                0.10476148128509521,
                0.5430830717086792,
                0.22838924825191498,
                0.7682365775108337,
            ],
        },
        "reflectance": {
            "shape": [2, 3, 21, 30],
            "sum": 2179.377967707813,
            "sumsq": 1384.4648369272695,
            "values": [
                0.4863593876361847,
                0.7438653111457825,
                0.5752995014190674,
                0.12078671157360077,
                0.5921008586883545,
                0.2471202313899994,
                0.8426348567008972,
            ],
        },
        "illumination": {
            "shape": [2, 1, 21, 30],
            "sum": 384.2318042740226,
            "sumsq": 122.63844369082742,
            "values": [
                0.45521652698516846,
                0.4158090353012085,
                0.2597203552722931,
                0.10289920121431351,
                0.30234089493751526,
                0.3249206244945526,
                0.322892427444458,
            ],
        },
        "illumination_adjusted": {
            "shape": [2, 1, 21, 30],
            "sum": 1156.000565648079,
            "sumsq": 1061.4597412435508,
            "values": [
                0.956268310546875,
                0.9313369989395142,
                0.9077173471450806,
                0.8673262000083923,
                0.9179599285125732,
                0.9242029786109924,
                0.9117075800895691,
            ],
        },
    },
}

ZOO = {
    "retinexnet_lol_v1_psnr.pt": RetinexNet_Weights.LOL_V1_PSNR,
    "retinexnet_lol_v1_ssim.pt": RetinexNet_Weights.LOL_V1_SSIM,
    "retinexnet_reimpl_9200.pt": RetinexNet_Weights.REIMPL_9200,
}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(member: RetinexNet_Weights) -> dict:
    try:
        return member.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"weights not available: {err}")


# --- Parity ---

@pytest.mark.parametrize("name", list(REIMPL))
def test_checkpoints_load_strictly_and_match_the_reimplementation(name):
    model = RetinexNet().eval()
    model.load_state_dict(released(ZOO[name]))
    with torch.no_grad():
        out = model(reference_input())
    for key in ("enhanced", "reflectance", "illumination", "illumination_adjusted"):
        assert_matches(out[key], REIMPL[name][key])


def test_psnr_and_ssim_checkpoints_share_the_decomposition_net():
    a, b = released(RetinexNet_Weights.LOL_V1_PSNR), released(RetinexNet_Weights.LOL_V1_SSIM)
    assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("decom_net."))
    assert not all(torch.equal(a[k], b[k]) for k in a if k.startswith("enhance_net."))


# --- Forward ---

def test_forward_definition_and_any_size():
    torch.manual_seed(0)
    model = RetinexNet(channels=8).eval()
    with torch.no_grad():
        for h, w in ((2, 2), (13, 7), (21, 30)):
            out = model(torch.rand(1, 3, h, w))
            assert out["enhanced"].shape == (1, 3, h, w) and out["illumination"].shape == (1, 1, h, w)
            assert torch.allclose(out["enhanced"], out["reflectance"] * out["illumination_adjusted"])
            assert out["reflectance"].min() > 0 and out["reflectance"].max() < 1


def test_state_dict_layout():
    keys = list(RetinexNet().state_dict())
    assert len(keys) == 32 and keys[0] == "decom_net.net1_conv0.weight" and keys[-1] == "enhance_net.net2_output.bias"


def test_batch_independence():
    torch.manual_seed(0)
    model = RetinexNet().eval()
    x = torch.rand(3, 3, 16, 20) * 0.4
    with torch.no_grad():
        assert torch.allclose(model(x)["enhanced"][1:2], model(x[1:2])["enhanced"], atol=1e-5)


# --- Registry ---

def test_registry_and_default_weights():
    assert MODELS.get("retinexnet") is RetinexNet and RetinexNet_Weights.DEFAULT is RetinexNet_Weights.REIMPL_9200
    assert isinstance(load("retinexnet", weights=None), RetinexNet)


# --- LOL-v1 ---

@pytest.mark.full
@pytest.mark.parametrize(("member", "expected"), [
    (RetinexNet_Weights.LOL_V1_PSNR, 12.780),
    (RetinexNet_Weights.LOL_V1_SSIM, 11.579),
    (RetinexNet_Weights.REIMPL_9200, 17.610),
])
def test_lol_v1_psnr_regression(member: RetinexNet_Weights, expected: float, regression_device: torch.device):
    """Clamp, truncate to 8 bit (the reimplementation saves ``clip(x·255).astype(uint8)``); measured 2026-11-18, CPU.
    The mon_old checkpoints score far below the reimplementation's release."""
    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv1
    try:
        root = datasets_dir() / LOLv1.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = RetinexNet().eval().to(regression_device)
    model.load_state_dict(released(member))
    psnrs = []
    with torch.no_grad():
        for sample in LOLv1("test"):
            x      = (sample["image"].float()[None] / 255).to(regression_device)
            target = (sample["target"].double()).to(regression_device)
            y8 = (model(x)["enhanced"][0].clamp(0, 1) * 255).to(torch.uint8).double()
            psnrs.append(10 * math.log10(255.0**2 / ((y8 - target) ** 2).mean().item()))
    assert sum(psnrs) / len(psnrs) == pytest.approx(expected, abs=5e-3)


# --- Loss ---

_lspec = importlib.util.spec_from_file_location("retinexnet_loss_reference",
                                                Path(__file__).with_name("retinexnet_loss_reference.py"))
assert _lspec and _lspec.loader
_lref = importlib.util.module_from_spec(_lspec)
_lspec.loader.exec_module(_lref)

# Generated from the reimplementation's loss with torch 2.15.0+cu132.
REIMPL_LOSS: dict = {
    "decomposition": {
        "loss": 0.4895414113998413,
        "grads": {
            "r_low": {
                "sum": -0.026732932978120516,
                "sumsq": 7.945507092618043e-05,
                "values": [
                    0.00039959067362360656,
                    -0.00016224960563704371,
                    0.00024204149667639285,
                    -0.00020793740986846387,
                    -0.00030000272090546787,
                ],
            },
            "l_low": {
                "sum": -0.09480029676342383,
                "sumsq": 0.0004401383099496252,
                "values": [
                    0.001042630523443222,
                    -0.0005344994715414941,
                    0.0009607065003365278,
                    -0.0009972148109227419,
                    0.00021038981503807008,
                ],
            },
            "r_high": {
                "sum": -0.2645087370474357,
                "sumsq": 0.00031388858254397716,
                "values": [
                    -0.00031991576543077826,
                    -0.00039405617280863225,
                    0.0006429197965189815,
                    -0.00032160134287551045,
                    -0.0004441533819772303,
                ],
            },
            "l_high": {
                "sum": -0.10831197957304539,
                "sumsq": 0.0002843863148785529,
                "values": [
                    -0.0009967256337404251,
                    4.6518107410520315e-05,
                    0.0007750748191028833,
                    -0.0008666767971590161,
                    5.2435731049627066e-05,
                ],
            },
        },
    },
    "relight": {
        "loss": 0.4995548725128174,
        "grads": {
            "r_low": {
                "sum": -0.8122088516975055,
                "sumsq": 0.0028138926302911683,
                "values": [
                    -0.00039552271482534707,
                    -6.913233664818108e-05,
                    -0.00034892535768449306,
                    -0.0002905079163610935,
                    -0.0009256251505576074,
                ],
            },
            "l_adjusted": {
                "sum": -0.3983304318244336,
                "sumsq": 0.007170822620490723,
                "values": [
                    -0.0012364319991320372,
                    -0.0009686285047791898,
                    -0.0008533927612006664,
                    -0.004197234287858009,
                    0.004401967860758305,
                ],
            },
        },
    },
}


def test_loss_matches_the_reimplementation():
    from mon.models.lle.retinexnet import RetinexNetLoss

    raw = _lref.inputs()
    x = {k: v.requires_grad_(True) for k, v in raw.items() if k not in ("low", "high")}
    out = RetinexNetLoss()(
        raw["low"], raw["high"],
        {"reflectance": x["r_low"], "illumination": x["l_low"], "illumination_adjusted": x["l_adjusted"]},
        {"reflectance": x["r_high"], "illumination": x["l_high"]},
    )
    for name, want in REIMPL_LOSS.items():
        assert out[name].item() == pytest.approx(want["loss"], rel=1e-6), name
        grads = torch.autograd.grad(out[name], [x[n] for n in want["grads"]], retain_graph=True)
        for (n, w), g in zip(want["grads"].items(), grads, strict=True):
            got = _lref.stats(g)
            assert got["sum"] == pytest.approx(w["sum"], rel=1e-5, abs=1e-9), (name, n)
            assert got["sumsq"] == pytest.approx(w["sumsq"], rel=1e-5), (name, n)
            assert got["values"] == pytest.approx(w["values"], rel=1e-5, abs=1e-9), (name, n)


def _naive_gradient(x: torch.Tensor, direction: str) -> torch.Tensor:
    """|x[i, j] − x[i, j−1]| (x) or |x[i, j] − x[i−1, j]| (y) on an [H+1, W+1] grid, zeros outside."""
    b, c, h, w = x.shape
    out = torch.zeros(b, c, h + 1, w + 1, dtype=x.dtype)

    def at(i: int, j: int) -> torch.Tensor:
        return x[..., i, j] if 0 <= i < h and 0 <= j < w else torch.zeros(b, c, dtype=x.dtype)

    for i in range(h + 1):
        for j in range(w + 1):
            out[..., i, j] = (at(i, j) - (at(i, j - 1) if direction == "x" else at(i - 1, j))).abs()
    return out


def _naive_smoothness(illum: torch.Tensor, refl: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * refl[:, 0:1] + 0.587 * refl[:, 1:2] + 0.114 * refl[:, 2:3]
    total = torch.zeros(illum.shape[0], dtype=illum.dtype)
    for d in ("x", "y"):
        g = _naive_gradient(gray, d)
        padded = torch.nn.functional.pad(g, (1, 1, 1, 1))
        avg = sum(padded[..., di : di + g.shape[-2], dj : dj + g.shape[-1]] for di in range(3) for dj in range(3)) / 9
        total = total + (_naive_gradient(illum, d) * torch.exp(-10 * avg)).flatten(1).mean(1)
    return total


def test_loss_smoothness_matches_the_described_padding():
    from mon.models.lle.retinexnet import _gradient, _smoothness

    torch.manual_seed(0)
    illum, refl = torch.rand(2, 1, 5, 6, dtype=torch.float64), torch.rand(2, 3, 5, 6, dtype=torch.float64)
    assert torch.allclose(_gradient(illum, "x"), _naive_gradient(illum, "x"))
    assert torch.allclose(_gradient(illum, "y"), _naive_gradient(illum, "y"))
    assert torch.allclose(_smoothness(illum, refl), _naive_smoothness(illum, refl))


def test_loss_decomposition_and_relight_values():
    from mon.models.lle.retinexnet import RetinexNetLoss, _smoothness

    torch.manual_seed(1)
    low, high = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    r_l, r_h = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    l_l, l_h, l_adj = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)

    def l1(a, b):  # noqa: ANN001, ANN202
        return (a - b).abs().flatten(1).mean(1)

    expected = (l1(r_l * l_l, low) + l1(r_h * l_h, high) + 0.001 * (l1(r_h * l_l, low) + l1(r_l * l_h, high))
                + 0.1 * (_smoothness(l_l, r_l) + _smoothness(l_h, r_h)) + 0.01 * l1(r_l, r_h))
    loss = RetinexNetLoss(reduction="none")
    assert torch.allclose(loss.decomposition(low, high, r_l, l_l, r_h, l_h), expected)
    assert torch.allclose(loss.relight(high, r_l, l_adj), l1(r_l * l_adj, high) + 3 * _smoothness(l_adj, r_l))
    out = RetinexNetLoss()(low, high, {"reflectance": r_l, "illumination": l_l, "illumination_adjusted": l_adj},
                           {"reflectance": r_h, "illumination": l_h})
    assert out["decomposition"].item() == pytest.approx(expected.mean().item(), rel=1e-6)


def test_loss_equal_reflectance_term_stops_the_gradient_to_the_normal_light_image():
    from mon.models.lle.retinexnet import RetinexNetLoss

    low = high = torch.rand(1, 3, 6, 6)
    r_l, r_h = torch.rand(1, 3, 6, 6, requires_grad=True), torch.rand(1, 3, 6, 6, requires_grad=True)
    ones = torch.ones(1, 1, 6, 6)
    loss = RetinexNetLoss()
    full = loss.decomposition(low, high, r_l, ones, r_h, ones)
    without = full - 0.01 * (r_l - r_h).abs().mean()  # subtracting the term with r_h *not* detached
    g_full, = torch.autograd.grad(full, r_h, retain_graph=True)
    g_without, = torch.autograd.grad(without, r_h)
    assert not torch.allclose(g_full, g_without)  # the detached term contributes no gradient to r_h ...
    ref_loss = loss.decomposition(low, high, r_l, ones, r_h, ones) - 0.01 * (r_l - r_h.detach()).abs().mean()
    g_ref, = torch.autograd.grad(ref_loss, r_h)
    assert torch.allclose(g_full, g_ref)  # ... so removing it leaves the r_h gradient unchanged


def test_loss_trains_the_model_end_to_end():
    from mon.models.lle.retinexnet import RetinexNetLoss

    model, loss = RetinexNet(), RetinexNetLoss()
    low, high = torch.rand(1, 3, 12, 12) * 0.2, torch.rand(1, 3, 12, 12)
    out_low, out_high = model(low), model(high)
    terms = loss(low, high, out_low, out_high)
    (terms["decomposition"] + terms["relight"]).backward()
    assert model.decom_net.net1_conv0.weight.grad is not None and model.enhance_net.net2_output.weight.grad is not None
