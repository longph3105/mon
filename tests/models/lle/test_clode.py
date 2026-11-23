"""Tests for ``mon.models.lle.clode``: parity with the official inference and the ODE wiring."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.models.lle.clode import CLODE, CLODE_Weights
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("clode_reference", Path(__file__).with_name("clode_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# Official network/conv_node.py (dgjung0220/CLODE b12ffae, vendored torchdiffeq) with the released
# Drive weights; keys "<official file stem>@<T>". Output of clode_reference.py, in float64 (see its
# docstring: float32 outputs, `curve_map` most, depend on the platform's rounding).
# Generated from the official code with torch 2.15.0, float64.
OFFICIAL: dict = {
    "universal@3.0": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 2190.1510933390778,
            "sumsq": 1370.1474308307966,
            "values": [
                0.5792392488421144,
                0.7399810256345729,
                0.6727623891861024,
                0.0730736846338896,
                0.7032401653508025,
                0.2674061992282947,
                0.6774096900894692,
            ],
        },
        "curve_map": {
            "shape": [2, 3, 21, 30],
            "sum": -2670.55411519477,
            "sumsq": 1902.5447056264484,
            "values": [
                -0.6234511528209086,
                -0.6241539903322245,
                -0.6204684141111688,
                -0.6479250212873661,
                -0.7917774981778825,
                -0.62703703765058,
                -0.7112307679837002,
            ],
        },
        "noise": {
            "shape": [2, 3, 21, 30],
            "sum": -2.6792030302957732,
            "sumsq": 4.903326678502933,
            "values": [
                0.00930043893095385,
                -0.011684260815951473,
                -0.05471757959973594,
                0.00909893760280801,
                -0.015417473076133772,
                0.03502766836418959,
                -0.0027906511424058006,
            ],
        },
        "state": {
            "shape": [2, 9, 21, 30],
            "sum": 2205.848827083502,
            "sumsq": 1366.8739077085033,
            "values": [
                0.5885396877730683,
                0.7282967648186215,
                0.6180448095863665,
                0.0821726222366976,
                0.6878226922746687,
                0.0002748960942211521,
                0.004618433279069326,
            ],
        },
        "nfe": 74,
    },
    "universal@-1.4": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 210.28484457587192,
            "sumsq": 15.274729065145928,
            "values": [
                0.09500873200628863,
                0.12139101336406653,
                0.05962297175701221,
                0.0035119512144662613,
                0.07380054264601361,
                0.004619452014029711,
                0.08948532754662172,
            ],
        },
        "curve_map": {
            "shape": [2, 3, 21, 30],
            "sum": -3648.739272630566,
            "sumsq": 3526.8660955494006,
            "values": [
                -0.9806017260201277,
                -0.9790067701670436,
                -0.9783580407258624,
                -0.9821451726504468,
                -0.9215864362460342,
                -0.9815203201315178,
                -0.9989075503356082,
            ],
        },
        "noise": {
            "shape": [2, 3, 21, 30],
            "sum": -0.4866753177321374,
            "sumsq": 0.3866658866880056,
            "values": [
                -0.05009654351537959,
                -0.04041919676214335,
                -0.009365105639739447,
                -0.0006452044006177799,
                -0.0025234590347537615,
                0.008275161516538,
                -0.016196692232123777,
            ],
        },
        "state": {
            "shape": [2, 9, 21, 30],
            "sum": 206.8522761276049,
            "sumsq": 14.798749558817835,
            "values": [
                0.04491218849090905,
                0.08097181660192318,
                0.05025786611727276,
                0.0028667468138484815,
                0.07127708361125985,
                -1.319041312714169e-05,
                -0.0005894932036558955,
            ],
        },
        "nfe": 26,
    },
    "lowlight@3.0": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 2583.018373993409,
            "sumsq": 1815.571381355731,
            "values": [
                0.6788081857281884,
                0.7799411801670519,
                0.7483197713321501,
                0.2196467856508225,
                0.7806866547573553,
                0.4548517386729037,
                0.7389723866992407,
            ],
        },
        "curve_map": {
            "shape": [2, 3, 21, 30],
            "sum": -3327.2173792644408,
            "sumsq": 2937.766109620059,
            "values": [
                -0.858703108619604,
                -0.8851969752520658,
                -0.8695497174600857,
                -0.8793440612661179,
                -0.9376626585566894,
                -0.8764114827737545,
                -0.8179403386053075,
            ],
        },
        "noise": {
            "shape": [2, 3, 21, 30],
            "sum": 31.741184371667813,
            "sumsq": 15.940655513815896,
            "values": [
                0.03452785167382912,
                0.048282919858888176,
                -0.01270413386376871,
                -0.08092070439529209,
                0.008729819411864579,
                -0.028798917609535072,
                0.02515778686518448,
            ],
        },
        "state": {
            "shape": [2, 9, 21, 30],
            "sum": 2701.8759260352463,
            "sumsq": 1903.0445474031503,
            "values": [
                0.7133360374020175,
                0.8282241000259402,
                0.7356156374683814,
                0.1387260812555304,
                0.7894164741692199,
                0.0017680198345773471,
                0.021278638279223504,
            ],
        },
        "nfe": 50,
    },
    "sice@3.0": {
        "enhanced": {
            "shape": [2, 3, 21, 30],
            "sum": 2258.2113566531916,
            "sumsq": 1437.452621224896,
            "values": [
                0.5915356224506693,
                0.7520007834084533,
                0.635215838601159,
                0.09158790790862185,
                0.7022743011580587,
                0.2946222961794554,
                0.695647233365308,
            ],
        },
        "curve_map": {
            "shape": [2, 3, 21, 30],
            "sum": -1300.8324255038044,
            "sumsq": 660.2281121474175,
            "values": [
                -0.1386119207726251,
                -0.14150899382365636,
                -0.15441374594256793,
                -0.13321800058673405,
                -0.21382015364021895,
                -0.14712552386267003,
                -0.6671824467346468,
            ],
        },
        "noise": {
            "shape": [2, 3, 21, 30],
            "sum": 3.5191845297368776,
            "sumsq": 1.8829886088262668,
            "values": [
                0.016383083017736028,
                -0.009504445001055968,
                0.0020382556482533023,
                -0.002964996243764304,
                -4.379827063812093e-05,
                0.018302497490911094,
                0.0003171181084658109,
            ],
        },
        "state": {
            "shape": [2, 9, 21, 30],
            "sum": 2272.721631548984,
            "sumsq": 1454.1967435179513,
            "values": [
                0.6079187054684053,
                0.7424963384073974,
                0.6372540942494124,
                0.08862291166485756,
                0.7022305028874205,
                8.80873685365641e-05,
                0.002819608495499415,
            ],
        },
        "nfe": 62,
    },
}

# Official file stem → zoo weights (same tensors; the zoo files drop unused modules).
ZOO = {"universal": CLODE_Weights.UNIVERSAL, "lowlight": CLODE_Weights.LOL_V1, "sice": CLODE_Weights.SICE_ME}


# Parity runs in float64, where mon and the official code agree to 0 except backward integration
# (`universal@-1.4`: 3.7e-9; PyPI torchdiffeq vs the official snapshot). In float32, rounding moves
# the adaptive steps: `curve_map` (A at the solver's last evaluation) then differs by ~1e-3 across
# platforms, so float32 literals can't be portable.
ATOL = 1e-8


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = ATOL) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(member: CLODE_Weights) -> dict:
    try:
        return member.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

@pytest.mark.parametrize("case", list(OFFICIAL))
def test_released_weights_load_strictly_and_match_official(case):
    stem, t_end = case.split("@")
    model = CLODE().eval()
    model.load_state_dict(released(ZOO[stem]))
    model = model.double()
    with torch.no_grad():
        out = model(reference_input().double(), t_end=float(t_end))
    expected = OFFICIAL[case]
    assert model.odefunc.nfe == expected["nfe"]
    for key in ("enhanced", "curve_map", "noise", "state"):
        assert out[key].dtype == torch.float64
        assert_matches(out[key], expected[key])


def test_released_weights_brighten_with_positive_and_darken_with_negative_end_time():
    model = CLODE().eval()
    model.load_state_dict(released(CLODE_Weights.UNIVERSAL))
    x = reference_input()
    with torch.no_grad():
        darker, brighter = model(x, t_end=-1.4)["enhanced"], model(x)["enhanced"]
    assert darker.mean() < x.mean() < brighter.mean()


def test_state_dict_layout_shares_the_ode_function():
    model = CLODE()
    keys  = list(model.state_dict())
    assert len(keys) == 48 and sum(k.startswith("odeblock.odefunc.") for k in keys) == 24
    assert model.odeblock.odefunc is model.odefunc
    assert len(list(model.parameters())) == 24  # registered twice, stored once


# --- Forward ---

def test_end_time_argument_and_override():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 8, 9) * 0.3
    model = CLODE(t_end=0.5).eval()
    with torch.no_grad():
        a = model(x)
        b = CLODE.forward(model, x, t_end=0.5)
        darker = model(x, t_end=-0.5)
        with pytest.raises(ValueError, match="non-zero"):
            model(x, t_end=0.0)
    assert torch.equal(a["enhanced"], b["enhanced"]) and not torch.equal(a["enhanced"], darker["enhanced"])
    assert a["enhanced"].min() >= 0 and a["enhanced"].max() <= 1
    assert a["curve_map"].abs().max() <= 1 and a["state"].shape == (1, 9, 8, 9)


def test_smallest_input():
    with torch.no_grad():
        assert CLODE(t_end=0.2).eval()(torch.rand(1, 3, 3, 3))["enhanced"].shape == (1, 3, 3, 3)


def test_adjoint_gradients_match_direct_backprop():
    # The adjoint solves a backward ODE; direct backprop differentiates the solver steps. Both converge
    # to the same gradient as the tolerance shrinks (relative gap ~1% at the default 1e-5).
    torch.manual_seed(0)
    x = torch.rand(1, 3, 6, 7) * 0.3
    grads = []
    for adjoint in (True, False):
        torch.manual_seed(1)
        model = CLODE(t_end=0.5, adjoint=adjoint, tol=1e-8)
        model(x)["state"][:, :3].mean().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0
    assert (grads[0] - grads[1]).norm() < 5e-3 * grads[1].norm()


# --- Registry ---

def test_registry_and_default_weights():
    assert MODELS.get("clode") is CLODE
    assert CLODE_Weights.DEFAULT is CLODE_Weights.UNIVERSAL
    model = load("clode", weights=None, t_end=2.0)
    assert isinstance(model, CLODE) and model.t_end == 2.0 and not model.training


# --- LOL-v1 ---

@pytest.mark.full
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("member", "expected_psnr"), [(CLODE_Weights.LOL_V1, 19.602), (CLODE_Weights.UNIVERSAL, 19.162)],
)
def test_lol_v1_psnr_regression(member: CLODE_Weights, expected_psnr: float):
    """Mean PSNR of 8-bit outputs vs targets at T 3, one image per batch, CUDA; measured 2026-11-18 (README: 19.61
    for the LOL weights). CPU takes over 10 minutes per weights file."""
    from torch.nn import functional as F

    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv1
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        root = datasets_dir() / LOLv1.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = CLODE().eval().cuda()
    model.load_state_dict(released(member))
    psnrs = []
    with torch.no_grad():
        for sample in LOLv1("test"):
            x, target = (sample["image"].float() / 255).cuda(), (sample["target"].float() / 255).cuda()
            y = (model(x[None])["enhanced"][0] * 255).round() / 255
            psnrs.append(10 * torch.log10(1 / F.mse_loss(y, target)).item())
    assert sum(psnrs) / len(psnrs) == pytest.approx(expected_psnr, abs=1e-2)
