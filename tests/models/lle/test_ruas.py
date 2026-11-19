"""Tests for ``mon.models.lle.ruas``: parity with the official forward pass (via mon_old), structure, outputs."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mon.core.weights import WeightsEnum
from mon.models.lle.ruas import RUAS, RUAS_Weights, _local_max
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("ruas_reference", Path(__file__).with_name("ruas_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

ENUM_MEMBER = {"lol": RUAS_Weights.LOL_V1, "upe": RUAS_Weights.UPE, "dark": RUAS_Weights.DARKFACE}

# mon_old's archived RUAS (a copy of the official forward pass) with the released weights on reference_input().
# Output of ``ruas_reference.py``.
# Generated from mon_old's archived RUAS with torch 2.15.0+cu132.
MON_OLD: dict = {'lol': {'enhanced': {'shape': [2, 3, 23, 31],
                      'sum': 3878.820615634322,
                      'sumsq': 3786.462397420183,
                      'values': [1.1323288679122925,
                                 1.0339823961257935,
                                 0.9130187630653381,
                                 1.0121941566467285,
                                 0.9902706742286682,
                                 1.0503833293914795,
                                 0.37495335936546326]},
         'pre_denoise': {'shape': [2, 3, 23, 31],
                         'sum': 3622.669194728136,
                         'sumsq': 3356.8675529194998,
                         'values': [1.0, 1.0, 0.7809017300605774, 1.0, 1.0, 1.0, 0.25444895029067993]},
         'illumination': {'shape': [2, 3, 23, 31],
                          'sum': 697.3889406174421,
                          'sumsq': 113.71166291876011,
                          'values': [0.1617269068956375,
                                     0.16188234090805054,
                                     0.16182102262973785,
                                     0.16188891232013702,
                                     0.16637031733989716,
                                     0.16096071898937225,
                                     0.16092213988304138]}},
 'upe': {'enhanced': {'shape': [2, 3, 23, 31],
                      'sum': 2606.3005480766296,
                      'sumsq': 1878.019702686151,
                      'values': [0.650302529335022,
                                 0.8088586330413818,
                                 0.4992123246192932,
                                 0.5689408183097839,
                                 0.6322524547576904,
                                 1.0445318222045898,
                                 0.2111048698425293]},
         'pre_denoise': {'shape': [2, 3, 23, 31],
                         'sum': 2398.185796139762,
                         'sumsq': 1647.8081179644705,
                         'values': [0.5060448050498962,
                                    0.7798833847045898,
                                    0.3897578716278076,
                                    0.5590572953224182,
                                    0.6087177395820618,
                                    1.0,
                                    0.11942171305418015]},
         'illumination': {'shape': [2, 3, 23, 31],
                          'sum': 1478.5554690063,
                          'sumsq': 512.5205339007255,
                          'values': [0.32408198714256287,
                                     0.32535940408706665,
                                     0.3242174983024597,
                                     0.3251858353614807,
                                     0.37015315890312195,
                                     0.34206143021583557,
                                     0.34287288784980774]}},
 'dark': {'enhanced': {'shape': [2, 3, 23, 31],
                       'sum': 3929.689506098628,
                       'sumsq': 3851.510394737899,
                       'values': [1.1303216218948364,
                                  1.0307753086090088,
                                  0.9734677672386169,
                                  1.0288499593734741,
                                  1.0040245056152344,
                                  1.0304234027862549,
                                  0.3982071280479431]},
          'pre_denoise': {'shape': [2, 3, 23, 31],
                          'sum': 3724.759777225554,
                          'sumsq': 3502.3289466208653,
                          'values': [1.0, 1.0, 0.8724069595336914, 1.0, 1.0, 1.0, 0.28655505180358887]},
          'illumination': {'shape': [2, 3, 23, 31],
                           'sum': 617.6152129620314,
                           'sumsq': 89.17082284672948,
                           'values': [0.1449621319770813,
                                      0.14504002034664154,
                                      0.14484789967536926,
                                      0.14504320919513702,
                                      0.1452423483133316,
                                      0.14281097054481506,
                                      0.14289215207099915]}}}


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
    model = RUAS().eval()
    model.load_state_dict(released(ENUM_MEMBER[name]))  # strict: official parameter names
    with torch.no_grad():
        out = model(reference_input())
    for key in ("enhanced", "pre_denoise", "illumination"):
        assert_matches(out[key], MON_OLD[name][key])


def test_official_output_per_checkpoint():
    # Official test.py: lol saves u_list[-1] (denoised), upe and dark save u_list[-2].
    assert RUAS_Weights.LOL_V1.weights.meta["output_key"] == "enhanced"
    assert RUAS_Weights.UPE.weights.meta["output_key"] == "pre_denoise"
    assert RUAS_Weights.DARKFACE.weights.meta["output_key"] == "pre_denoise"
    assert RUAS_Weights.DEFAULT is RUAS_Weights.UPE  # official test.py default


# --- Structure follows the official code ---

def test_state_dict_matches_released_layout():
    model = RUAS()
    assert len(model.state_dict()) == 76 and sum(p.numel() for p in model.parameters()) == 3438
    keys = set(model.state_dict())
    assert "enhance_net.iems.2.cell.c5.weight" in keys and "denoise_net.activate.0.weight" in keys
    assert not any(".c1_d." in k for k in keys if k.startswith("enhance_net"))  # IEM c1_d is skip_connect


def test_local_max_matches_official():
    x = torch.randn(2, 3, 7, 9)
    padded = F.pad(x, (1, 1, 1, 1))[:, :, 1:, 1:]  # official max_operation, written out
    official = torch.max(padded[:, :, :-1, :], padded[:, :, 1:, :])
    official = torch.max(official[:, :, :, :-1], official[:, :, :, 1:])
    assert torch.equal(_local_max(x), official)


def test_outputs_and_ranges():
    torch.manual_seed(0)
    model = RUAS().eval()
    x = torch.rand(1, 3, 17, 22) * 0.3
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out["enhanced"], out["pre_denoise"] - out["noise"])
    assert out["pre_denoise"].min() >= 0 and out["pre_denoise"].max() <= 1
    assert out["illumination"].min() >= 0.001 and out["illumination"].max() <= 1
    with pytest.raises(ValueError, match="iterations"):
        RUAS(iterations=0)


def test_registry_and_load():
    assert MODELS.entry("ruas").meta["arch"] == "ruas"
    try:
        model = load("ruas", "lol_v1")
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")
    assert isinstance(model, RUAS) and not model.training
