"""Tests for ``mon.models.lle.sci``: parity with the official code (via mon_old), training stages, loss."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mon.core.weights import WeightsEnum
from mon.models.lle.sci import SCI, SCILoss, SCIPP, SCIPP_Weights, SCI_Weights
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("sci_reference", Path(__file__).with_name("sci_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

ENUM_MEMBER = {"sci_easy": SCI_Weights.EASY, "sci_medium": SCI_Weights.MEDIUM,
               "sci_difficult": SCI_Weights.DIFFICULT, "sci++_tpami": SCIPP_Weights.TPAMI}

# mon_old outputs (its inference networks and loss are copies of the official code) with the released weights
# on reference_input(). Output of ``sci_reference.py``.
# Generated from mon_old with torch 2.14.0.
MON_OLD: dict = {'sci_easy': {'enhanced': {'shape': [2, 3, 23, 31],
                           'sum': 1683.4741559252143,
                           'sumsq': 788.005694931388,
                           'values': [0.38416799902915955,
                                      0.5299792885780334,
                                      0.22763559222221375,
                                      0.35356730222702026,
                                      0.4418177008628845,
                                      0.740907609462738,
                                      0.10121969878673553]},
              'illumination': {'shape': [2, 3, 23, 31],
                               'sum': 2072.8705643564463,
                               'sumsq': 1023.4449324801278,
                               'values': [0.4268965721130371,
                                          0.4787779748439789,
                                          0.5551254749298096,
                                          0.5141807794570923,
                                          0.5099813938140869,
                                          0.6004779934883118,
                                          0.4045306444168091]}},
 'sci_medium': {'enhanced': {'shape': [2, 3, 23, 31],
                             'sum': 2816.865302953869,
                             'sumsq': 2097.481230303018,
                             'values': [0.43561363220214844,
                                        0.8268312215805054,
                                        0.5295080542564392,
                                        0.6906942129135132,
                                        0.8026677966117859,
                                        0.9625951051712036,
                                        0.13666479289531708]},
                'illumination': {'shape': [2, 3, 23, 31],
                                 'sum': 1229.8519018813968,
                                 'sumsq': 381.3234739977704,
                                 'values': [0.37648043036460876,
                                            0.30688536167144775,
                                            0.23864853382110596,
                                            0.26320981979370117,
                                            0.28071239590644836,
                                            0.4621867835521698,
                                            0.2996124029159546]}},
 'loss': {'total': 4.31812047958374, 'smoothness': 4.298910140991211},
 'sci_difficult': {'enhanced': {'shape': [2, 3, 23, 31],
                                'sum': 2467.9352274294943,
                                'sumsq': 1606.7773511957753,
                                'values': [0.46739107370376587,
                                           0.8501536846160889,
                                           0.388293981552124,
                                           0.5797471404075623,
                                           0.6227001547813416,
                                           0.9349529147148132,
                                           0.24343319237232208]},
                   'illumination': {'shape': [2, 3, 23, 31],
                                    'sum': 1392.1553357467055,
                                    'sumsq': 479.550883087752,
                                    'values': [0.3508839011192322,
                                               0.2984665036201477,
                                               0.32543981075286865,
                                               0.3135806918144226,
                                               0.3618415594100952,
                                               0.47585147619247437,
                                               0.16820412874221802]}},
 'sci++_tpami': {'enhanced': {'shape': [2, 3, 23, 31],
                              'sum': 2771.3345605656505,
                              'sumsq': 1940.6126612082949,
                              'values': [0.6476444005966187,
                                         0.7460941672325134,
                                         0.6277298927307129,
                                         0.7067926526069641,
                                         0.7569718360900879,
                                         0.8805832862854004,
                                         0.14194414019584656]},
                 'illumination': {'shape': [2, 3, 23, 31],
                                  'sum': 1223.5883500650525,
                                  'sumsq': 385.66394756894533,
                                  'values': [0.2532253861427307,
                                             0.34009432792663574,
                                             0.20130683481693268,
                                             0.25721475481987,
                                             0.29765811562538147,
                                             0.5052318572998047,
                                             0.2884688973426819]}}}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(member: WeightsEnum) -> dict:
    try:
        return member.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

@pytest.mark.parametrize("name", list(ENUM_MEMBER))
def test_released_weights_load_strictly_and_match_official(name: str):
    member = ENUM_MEMBER[name]
    model = (SCIPP if name.startswith("sci++") else SCI)().eval()
    model.load_state_dict(released(member))  # strict: same names, including the shared blocks
    with torch.no_grad():
        out = model(reference_input())
    assert_matches(out["enhanced"], MON_OLD[name]["enhanced"])
    assert_matches(out["illumination"], MON_OLD[name]["illumination"])


def test_loss_matches_official():
    model = SCI().eval()
    model.load_state_dict(released(SCI_Weights.MEDIUM))
    x = reference_input()
    with torch.no_grad():
        illumination = model(x)["illumination"]
    loss = SCILoss()
    assert loss(x, illumination).item() == pytest.approx(MON_OLD["loss"]["total"], rel=1e-6)
    assert loss.smoothness(x, illumination).mean().item() == pytest.approx(MON_OLD["loss"]["smoothness"], rel=1e-6)


# --- Structure follows the official code ---

def test_state_dict_layout_and_weight_sharing():
    sci, scipp = SCI(), SCIPP()
    assert len(sci.state_dict()) == 83 and len(scipp.state_dict()) == 32  # as the released checkpoints
    assert sci.enhance.blocks[0] is sci.enhance.conv
    assert all(block is sci.calibrate.convs for block in sci.calibrate.blocks)  # official: one module, 3×
    assert len({id(b) for b in scipp.hb.blocks}) == 3  # SCI++ blocks are separate
    assert {k.split(".")[0] for k in scipp.state_dict()} == {"ha", "hb", "calibrate"}


def test_initialization():
    sci = SCI()
    convs = [m for m in sci.modules() if isinstance(m, torch.nn.Conv2d)]
    assert all(m.bias is not None and torch.all(m.bias == 0) for m in convs)
    assert 0.01 < torch.cat([m.weight.flatten() for m in convs]).std().item() < 0.03  # N(0, 0.02)
    scipp_convs = [m for m in SCIPP().modules() if isinstance(m, torch.nn.Conv2d)]
    assert any(m.bias is not None and m.bias.abs().sum() > 0 for m in scipp_convs)  # PyTorch default init


# --- Training stages follow the official loops ---

def test_sci_stages_use_the_original_input():
    torch.manual_seed(0)
    model = SCI(stages=3).train()
    x = torch.rand(2, 3, 16, 20) * 0.4
    out = model.forward_stages(x)
    # Official CVPR loop, written out: every stage divides the *original* input and calibrates from it.
    x_op = x
    for k in range(3):
        assert torch.allclose(out["inputs"][k], x_op)
        i = model.enhance(x_op)
        r = (x / i).clamp(0, 1)
        att = model.calibrate(r)
        assert torch.allclose(out["illuminations"][k], i) and torch.allclose(out["enhanced"][k], r)
        assert torch.allclose(out["attention"][k], att.abs())
        x_op = x + att  # mon_old accumulated x_op + att
    total = sum(SCILoss()(out["inputs"][k], out["illuminations"][k]) for k in range(3))
    total.backward()  # the official objective trains both modules
    assert model.enhance.in_conv[0].weight.grad is not None and model.calibrate.in_conv[0].weight.grad is not None


def test_scipp_stages_follow_tpami_loop():
    torch.manual_seed(0)
    model = SCIPP(stages=3)
    x = torch.rand(1, 3, 16, 20) * 0.4
    out = model.forward_stages(x)
    assert [len(out[k]) for k in ("inputs", "illuminations", "enhanced", "attention")] == [3, 3, 3, 2]
    i = model.ha(x)
    r = (x / i).clamp(0, 1)
    for k in range(1, 3):
        assert torch.allclose(out["inputs"][k], i)
        att = model.calibrate(r)
        i = i + att + model.hb(att)
        r = (x / i).clamp(0, 1)
        assert torch.allclose(out["illuminations"][k], i) and torch.allclose(out["enhanced"][k], r)


def test_test_time_output_is_first_stage():
    torch.manual_seed(0)
    for model in (SCI().eval(), SCIPP().eval()):
        x = torch.rand(1, 3, 12, 14)
        with torch.no_grad():
            out, stages = model(x), model.forward_stages(x)
        assert torch.equal(out["enhanced"], stages["enhanced"][0])
        assert torch.equal(out["illumination"], stages["illuminations"][0])


# --- Loss ---

def official_smoothness(image: torch.Tensor, illumination: torch.Tensor, official: bool) -> torch.Tensor:
    """The 24 official terms written out from their definition (weights from the YCbCr guide)."""
    mat = torch.tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]])
    bias = torch.tensor([16.0, 128.0, 128.0]) / 255
    if official:
        g = (image.contiguous().view(-1, 3) @ mat + bias).view(image.shape)
    else:
        g = (image.permute(0, 2, 3, 1) @ mat + bias).permute(0, 3, 1, 2)
    total = torch.zeros(())
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if (dy, dx) == (0, 0):
                continue
            h, w = image.shape[-2:]
            ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
            xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
            weight = torch.exp(-(g[..., ys, xs] - g[..., yd, xd]).pow(2).sum(1, keepdim=True) / 200)
            total = total + (weight * (illumination[..., ys, xs] - illumination[..., yd, xd]).abs().sum(1, keepdim=True)).mean()
    return total


@pytest.mark.parametrize("official", [True, False])
def test_smoothness_equals_all_24_neighbor_terms(official: bool):
    x, i = torch.rand(2, 3, 13, 17), torch.rand(2, 3, 13, 17)
    expected = official_smoothness(x, i, official)
    assert torch.allclose(SCILoss(official_ycbcr=official).smoothness(x, i).mean(), expected, rtol=1e-5)


def test_ycbcr_quirk_changes_the_loss_and_reductions():
    x, i = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    assert not torch.allclose(SCILoss()(x, i), SCILoss(official_ycbcr=False)(x, i))
    per_sample = SCILoss(reduction="none")(x, i)
    assert per_sample.shape == (2,) and torch.allclose(per_sample.mean(), SCILoss()(x, i))
    assert torch.allclose(per_sample[1], SCILoss()(x[1:], i[1:]))
    assert torch.allclose(1.5 * F.mse_loss(i, x) + SCILoss().smoothness(x, i).mean(), SCILoss()(x, i))
    with pytest.raises(ValueError, match="reduction"):
        SCILoss(reduction="max")  # type: ignore[arg-type]


# --- Behavior ---

@pytest.mark.parametrize("cls", [SCI, SCIPP])
def test_shapes_range_and_batch_independence(cls: type):
    torch.manual_seed(0)
    model = cls().eval()
    x = torch.rand(3, 3, 19, 26)
    with torch.no_grad():
        out, single = model(x), model(x[1:2])
    assert out["enhanced"].shape == out["illumination"].shape == x.shape
    assert out["enhanced"].min() >= 0 and out["enhanced"].max() <= 1
    assert out["illumination"].min() >= 1e-4 and out["illumination"].max() <= 1
    assert torch.allclose(out["enhanced"][1:2], single["enhanced"], atol=1e-6)


# --- Weights and registry ---

@pytest.mark.parametrize(("member", "path"), [
    (SCI_Weights.EASY, "lle/sci/sci_easy.pt"),
    (SCI_Weights.MEDIUM, "lle/sci/sci_medium.pt"),
    (SCI_Weights.DIFFICULT, "lle/sci/sci_difficult.pt"),
    (SCIPP_Weights.TPAMI, "lle/sci++/sci++_1_3500.pt"),
])
def test_weights_enums(member: WeightsEnum, path: str):
    w = member.weights
    assert w.path == path and len(w.sha256 or "") == 64
    assert w.url and w.url.startswith("https://raw.githubusercontent.com/vis-opt-group/SCI/")
    assert SCI_Weights.DEFAULT is SCI_Weights.MEDIUM and SCIPP_Weights.DEFAULT is SCIPP_Weights.TPAMI


def test_released_weights_on_disk_match_sha256():
    from mon.core.fileio import sha256sum
    for member in ENUM_MEMBER.values():
        try:
            path = member.weights.resolve(download=False)
        except FileNotFoundError:
            pytest.skip("released weights not available")
        assert sha256sum(path) == member.weights.sha256


def test_registry():
    assert {"sci", "sci++"} <= set(MODELS.names(tasks="lle", arch="sci"))
    assert MODELS.get("SCI") is SCI and MODELS.get("sci++") is SCIPP
    released(SCI_Weights.EASY)
    model = load("sci", "easy")
    assert isinstance(model, SCI) and not model.training


# --- Real data (LOL-v1 test split, released weights) ---

@pytest.mark.full
@pytest.mark.parametrize(("name", "weights", "expected_psnr"), [
    ("sci", "easy", 9.627),
    ("sci", "medium", 14.854),
    ("sci", "difficult", 13.869),
    ("sci++", "default", 13.913),
])
def test_lol_v1_psnr_regression(name: str, weights: str, expected_psnr: float, regression_device: torch.device):
    """Mean PSNR of 8-bit outputs vs targets; measured 2026-11-18 (outputs identical to mon_old)."""
    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv1
    try:
        root = datasets_dir() / LOLv1.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = MODELS.get(name)
    from mon.core.weights import resolve_weights
    spec = resolve_weights(weights, model.weights_enum)
    assert spec is not None
    try:
        spec.resolve(download=False)
    except FileNotFoundError:
        pytest.skip("released weights not available")
    net = load(name, weights, device=regression_device)
    psnrs = []
    with torch.no_grad():
        for sample in LOLv1("test"):
            x      = (sample["image"].float() / 255).to(regression_device)
            target = (sample["target"].float() / 255).to(regression_device)
            y = (net(x[None])["enhanced"][0] * 255).round() / 255
            psnrs.append(10 * torch.log10(1 / F.mse_loss(y, target)).item())
    assert sum(psnrs) / len(psnrs) == pytest.approx(expected_psnr, abs=5e-3)
