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
# Drive weights; keys "<official file stem>@<T>". Output of clode_reference.py.
# Generated from the official code with torch 2.15.0+cu132.
OFFICIAL: dict = {'universal@3.0': {'enhanced': {'shape': [2, 3, 21, 30],
                                'sum': 2190.1509495452046,
                                'sumsq': 1370.1471968553046,
                                'values': [0.5792397260665894,
                                           0.7399806380271912,
                                           0.6727626919746399,
                                           0.0730738490819931,
                                           0.703239381313324,
                                           0.2674058675765991,
                                           0.6774089336395264]},
                   'curve_map': {'shape': [2, 3, 21, 30],
                                 'sum': -2669.7992346286774,
                                 'sumsq': 1901.4497831991423,
                                 'values': [-0.6233017444610596,
                                            -0.6240113973617554,
                                            -0.620258092880249,
                                            -0.6475529670715332,
                                            -0.7914838790893555,
                                            -0.6269564032554626,
                                            -0.7110368013381958]},
                   'noise': {'shape': [2, 3, 21, 30],
                             'sum': -2.679198322370212,
                             'sumsq': 4.903323960184862,
                             'values': [0.009300494566559792,
                                        -0.011684215627610683,
                                        -0.054717402905225754,
                                        0.00909910723567009,
                                        -0.015417474322021008,
                                        0.035027846693992615,
                                        -0.002790787722915411]},
                   'state': {'shape': [2, 9, 21, 30],
                             'sum': 2205.848753176513,
                             'sumsq': 1366.8736756534172,
                             'values': [0.5885401964187622,
                                        0.7282963991165161,
                                        0.6180452704429626,
                                        0.08217296004295349,
                                        0.6878219246864319,
                                        0.0002749133564066142,
                                        0.004618432838469744]},
                   'nfe': 74},
 'universal@-1.4': {'enhanced': {'shape': [2, 3, 21, 30],
                                 'sum': 210.28470827895217,
                                 'sumsq': 15.274709192682778,
                                 'values': [0.09500864148139954,
                                            0.12139067053794861,
                                            0.05962279438972473,
                                            0.00351192569360137,
                                            0.0738007128238678,
                                            0.004619402810931206,
                                            0.08948532491922379]},
                    'curve_map': {'shape': [2, 3, 21, 30],
                                  'sum': -3648.797670543194,
                                  'sumsq': 3526.970338994254,
                                  'values': [-0.9805891513824463,
                                             -0.9789930582046509,
                                             -0.9783408641815186,
                                             -0.9821199774742126,
                                             -0.921665370464325,
                                             -0.9814969301223755,
                                             -0.9989078044891357]},
                    'noise': {'shape': [2, 3, 21, 30],
                              'sum': -0.48666401479999877,
                              'sumsq': 0.386665475704881,
                              'values': [-0.0500965341925621,
                                         -0.04041904956102371,
                                         -0.009365088306367397,
                                         -0.0006451982771977782,
                                         -0.0025234806817024946,
                                         0.008275164291262627,
                                         -0.016196683049201965]},
                    'state': {'shape': [2, 9, 21, 30],
                              'sum': 206.85214450161584,
                              'sumsq': 14.798734020812923,
                              'values': [0.044912103563547134,
                                         0.0809716209769249,
                                         0.05025770515203476,
                                         0.00286672729998827,
                                         0.07127723097801208,
                                         -1.3192160622566007e-05,
                                         -0.0005894924397580326]},
                    'nfe': 26},
 'lowlight@3.0': {'enhanced': {'shape': [2, 3, 21, 30],
                               'sum': 2583.018384218216,
                               'sumsq': 1815.5713857615501,
                               'values': [0.678808331489563,
                                          0.7799410820007324,
                                          0.7483197450637817,
                                          0.21964684128761292,
                                          0.7806867361068726,
                                          0.45485177636146545,
                                          0.7389723658561707]},
                  'curve_map': {'shape': [2, 3, 21, 30],
                                'sum': -3327.2369483709335,
                                'sumsq': 2937.8008941393846,
                                'values': [-0.8587154150009155,
                                           -0.885205864906311,
                                           -0.8695612549781799,
                                           -0.8793541789054871,
                                           -0.9376674294471741,
                                           -0.8764219880104065,
                                           -0.8179428577423096]},
                  'noise': {'shape': [2, 3, 21, 30],
                            'sum': 31.74118975106876,
                            'sumsq': 15.94065534016545,
                            'values': [0.034527841955423355,
                                       0.04828290268778801,
                                       -0.012704131193459034,
                                       -0.0809207335114479,
                                       0.008729676716029644,
                                       -0.02879893034696579,
                                       0.02515772357583046]},
                  'state': {'shape': [2, 9, 21, 30],
                            'sum': 2701.8759238780476,
                            'sumsq': 1903.0445568766372,
                            'values': [0.713336169719696,
                                       0.8282240033149719,
                                       0.735615611076355,
                                       0.1387261003255844,
                                       0.7894164323806763,
                                       0.0017680168384686112,
                                       0.021278636530041695]},
                  'nfe': 50},
 'sice@3.0': {'enhanced': {'shape': [2, 3, 21, 30],
                           'sum': 2258.210348073393,
                           'sumsq': 1437.4513586132612,
                           'values': [0.5915351510047913,
                                      0.7519991397857666,
                                      0.6352137923240662,
                                      0.09158748388290405,
                                      0.7022733092308044,
                                      0.29462194442749023,
                                      0.6956461668014526]},
              'curve_map': {'shape': [2, 3, 21, 30],
                            'sum': -1301.0178555250168,
                            'sumsq': 660.7502004916469,
                            'values': [-0.13844846189022064,
                                       -0.1413409411907196,
                                       -0.15427550673484802,
                                       -0.1328807771205902,
                                       -0.2137955278158188,
                                       -0.14693017303943634,
                                       -0.6675286889076233]},
              'noise': {'shape': [2, 3, 21, 30],
                        'sum': 3.5192736795476662,
                        'sumsq': 1.8829918555835112,
                        'values': [0.01638312079012394,
                                   -0.009504539892077446,
                                   0.002038012258708477,
                                   -0.002965029329061508,
                                   -4.3870662921108305e-05,
                                   0.01830263063311577,
                                   0.0003171491262037307]},
              'state': {'shape': [2, 9, 21, 30],
                        'sum': 2272.720366441121,
                        'sumsq': 1454.1955627911468,
                        'values': [0.6079182624816895,
                                   0.7424945831298828,
                                   0.6372517943382263,
                                   0.08862245827913284,
                                   0.7022294402122498,
                                   8.800496289040893e-05,
                                   0.0028195998165756464]},
              'nfe': 62}}

# Official file stem → zoo weights (same tensors; the zoo files drop unused modules).
ZOO = {"universal": CLODE_Weights.UNIVERSAL, "lowlight": CLODE_Weights.LOL_V1, "sice": CLODE_Weights.SICE_ME}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
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
    with torch.no_grad():
        out = model(reference_input(), t_end=float(t_end))
    expected = OFFICIAL[case]
    assert model.odefunc.nfe == expected["nfe"]
    for key in ("enhanced", "curve_map", "noise", "state"):
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

@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(("member", "expected_psnr"), [(CLODE_Weights.LOL_V1, 19.602), (CLODE_Weights.UNIVERSAL, 19.162)])
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
