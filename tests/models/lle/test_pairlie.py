"""Tests for ``mon.models.lle.pairlie``: parity with the official inference (via mon_old) and the loss."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mon.models.lle.pairlie import PairLIE, PairLIELoss, PairLIE_Weights
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("pairlie_reference", Path(__file__).with_name("pairlie_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# mon_old (= official net.py + eval.py) with the released weights, alpha 0.2. Output of pairlie_reference.py.
# Generated from mon_old with torch 2.14.0.
MON_OLD: dict = {'enhanced': {'shape': [2, 3, 21, 30],
              'sum': 2177.326342459768,
              'sumsq': 1340.5246323417364,
              'values': [0.6605066061019897,
                         0.7529559135437012,
                         0.2953943908214569,
                         0.049221061170101166,
                         0.7186455130577087,
                         0.36007630825042725,
                         0.7367973327636719]},
 'illumination': {'shape': [2, 1, 21, 30],
                  'sum': 282.288801163435,
                  'sumsq': 63.48227042011075,
                  'values': [0.24904072284698486,
                             0.24397210776805878,
                             0.23133531212806702,
                             0.18076105415821075,
                             0.21267202496528625,
                             0.23214657604694366,
                             0.22337090969085693]},
 'reflectance': {'shape': [2, 3, 21, 30],
                 'sum': 2935.8350493982434,
                 'sumsq': 2436.112038208018,
                 'values': [0.8722140789031982,
                            0.9983929991722107,
                            0.3958711624145508,
                            0.06929948180913925,
                            0.9652169942855835,
                            0.4822165369987488,
                            0.9943578839302063]},
 'denoised': {'shape': [2, 3, 21, 30],
              'sum': 681.2776468256488,
              'sumsq': 147.1830011448701,
              'values': [0.2515251636505127,
                         0.3690585494041443,
                         0.17692765593528748,
                         0.010042750276625156,
                         0.22169427573680878,
                         0.047633565962314606,
                         0.29188743233680725]}}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released() -> dict:
    try:
        return PairLIE_Weights.DEFAULT.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

def test_released_weights_load_strictly_and_match_official():
    model = PairLIE().eval()
    model.load_state_dict(released())
    with torch.no_grad():
        out = model(reference_input())
    for key in ("enhanced", "illumination", "reflectance", "denoised"):
        assert_matches(out[key], MON_OLD[key])


def test_state_dict_layout():
    keys = list(PairLIE().state_dict())
    assert len(keys) == 30 and keys[0] == "L_net.L_net.1.weight" and keys[-1] == "N_net.N_net.13.bias"


# --- Forward ---

def test_forward_definition_and_alpha():
    torch.manual_seed(0)
    model = PairLIE(alpha=0.5).eval()
    x = torch.rand(2, 3, 13, 18)
    with torch.no_grad():
        out = model(x)
        denoised = model.N_net(x)
    assert torch.allclose(out["denoised"], denoised)
    assert torch.allclose(out["enhanced"], out["illumination"] ** 0.5 * out["reflectance"])
    assert out["illumination"].shape == (2, 1, 13, 18) and 0 < out["illumination"].min()
    assert out["enhanced"].min() >= 0 and out["enhanced"].max() <= 1
    model.alpha = 1.0
    with torch.no_grad():
        assert torch.allclose(model(x)["enhanced"], out["illumination"] * out["reflectance"])


def test_batch_independence_and_small_sizes():
    torch.manual_seed(0)
    model = PairLIE(channels=8).eval()
    x = torch.rand(3, 3, 9, 7)
    with torch.no_grad():
        assert torch.allclose(model(x)["enhanced"][1:2], model(x[1:2])["enhanced"], atol=1e-6)
        assert model(torch.rand(1, 3, 2, 2))["enhanced"].shape == (1, 3, 2, 2)


# --- Loss ---

_lspec = importlib.util.spec_from_file_location("pairlie_loss_reference",
                                                Path(__file__).with_name("pairlie_loss_reference.py"))
assert _lspec and _lspec.loader
_lref = importlib.util.module_from_spec(_lspec)
_lspec.loader.exec_module(_lref)

# Generated from the official PairLIE loss with torch 2.15.0+cu132.
OFFICIAL_LOSS: dict = {'loss': 18.790653228759766,
 'grads': {'L1': {'sum': 0.5139898679153703,
                  'sumsq': 0.012062491491188137,
                  'values': [0.004054706078022718,
                             0.004035044461488724,
                             -0.0005480038234964013,
                             0.004172489047050476,
                             0.006129732821136713]},
           'R1': {'sum': 0.09827967794262804,
                  'sumsq': 0.0021969887858123045,
                  'values': [0.002759053837507963,
                             0.0005522723658941686,
                             -0.002682353137061,
                             -9.511085227131844e-05,
                             -0.00038150936597958207]},
           'X1': {'sum': -7.987538380592014,
                  'sumsq': 26.088069385284882,
                  'values': [0.017147010192275047,
                             -0.1663847267627716,
                             0.25991079211235046,
                             0.027788715437054634,
                             -0.1604386568069458]},
           'R2': {'sum': -0.0032117450953421667,
                  'sumsq': 0.00045329615720123744,
                  'values': [-0.0011273467680439353,
                             -7.143455150071532e-05,
                             0.0005196553538553417,
                             0.00033498930861242115,
                             0.0004636509402189404]}}}


def test_loss_matches_official():
    x = {k: v.requires_grad_(True) for k, v in _lref.inputs().items()}
    out1 = {"illumination": x["L1"], "reflectance": x["R1"], "denoised": x["X1"]}
    loss = PairLIELoss()(x["im1"], out1, {"reflectance": x["R2"]})
    assert loss.item() == pytest.approx(OFFICIAL_LOSS["loss"], rel=1e-6)
    names = ["L1", "R1", "X1", "R2"]
    for name, grad in zip(names, torch.autograd.grad(loss, [x[n] for n in names]), strict=True):
        want, got = OFFICIAL_LOSS["grads"][name], _lref.stats(grad)
        assert got["sum"] == pytest.approx(want["sum"], rel=1e-5, abs=1e-9), name
        assert got["sumsq"] == pytest.approx(want["sumsq"], rel=1e-5), name
        assert got["values"] == pytest.approx(want["values"], rel=1e-5, abs=1e-9), name


def test_loss_follows_official_definition():
    torch.manual_seed(0)
    model = PairLIE(channels=8)
    im1, im2 = torch.rand(2, 3, 16, 20) * 0.3, torch.rand(2, 3, 16, 20) * 0.5
    out1, out2 = model(im1), model(im2)
    l1, r1, x1, r2 = out1["illumination"], out1["reflectance"], out1["denoised"], out2["reflectance"]
    mse = F.mse_loss
    tv = (l1[:, :, 2:, :] - l1[:, :, :-2, :]).abs().mean() + (l1[:, :, :, 2:] - l1[:, :, :, :-2]).abs().mean()
    expected = (mse(r1, r2)
                + mse(l1 * r1, x1) + mse(r1, x1 / l1.detach()) + mse(l1, im1.max(1, keepdim=True)[0]) + tv
                + 500 * mse(im1, x1))
    loss = PairLIELoss()(im1, out1, out2)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_loss_reductions_and_stop_gradient():
    torch.manual_seed(0)
    model = PairLIE(channels=8)
    im1, im2 = torch.rand(3, 3, 12, 12) * 0.3, torch.rand(3, 3, 12, 12) * 0.5
    out1, out2 = model(im1), model(im2)
    per_sample = PairLIELoss(reduction="none")(im1, out1, out2)
    assert per_sample.shape == (3,) and torch.allclose(per_sample.mean(), PairLIELoss()(im1, out1, out2))
    with pytest.raises(ValueError, match="reduction"):
        PairLIELoss(reduction="max")  # type: ignore[arg-type]


def test_reflectance_term_stops_the_illumination_gradient():
    # Official MSE(R, X / L.detach()): that term must not push gradients into L.
    torch.manual_seed(0)
    im1 = torch.rand(1, 3, 10, 12) * 0.4
    illu = (torch.rand(1, 1, 10, 12) * 0.8 + 0.1).requires_grad_(True)
    refl, den, refl2 = torch.rand(1, 3, 10, 12), torch.rand(1, 3, 10, 12) * 0.4, torch.rand(1, 3, 10, 12)
    out1 = {"illumination": illu, "reflectance": refl, "denoised": den}
    (grad,) = torch.autograd.grad(PairLIELoss()(im1, out1, {"reflectance": refl2}), illu)
    tv = (illu[..., 2:, :] - illu[..., :-2, :]).abs().mean() + (illu[..., :, 2:] - illu[..., :, :-2]).abs().mean()
    without = F.mse_loss(illu * refl, den) + F.mse_loss(illu, im1.amax(1, keepdim=True)) + tv
    (expected,) = torch.autograd.grad(without, illu)
    assert torch.allclose(grad, expected)


# --- Weights and registry ---

def test_weights_enum():
    w = PairLIE_Weights.DEFAULT.weights
    assert PairLIE_Weights.DEFAULT is PairLIE_Weights.SICE and w.path == "lle/pairlie/pairlie_sice.pt"
    assert w.url and w.url.startswith("https://raw.githubusercontent.com/zhenqifu/PairLIE/") and len(w.sha256 or "") == 64


def test_released_weights_on_disk_match_sha256():
    from mon.core.fileio import sha256sum
    try:
        path = PairLIE_Weights.DEFAULT.weights.resolve(download=False)
    except FileNotFoundError:
        pytest.skip("released weights not available")
    assert sha256sum(path) == PairLIE_Weights.DEFAULT.weights.sha256


def test_registry():
    assert MODELS.get("pairlie") is PairLIE and "pairlie" in MODELS.names(tasks="lle")
    released()
    model = load("pairlie", alpha=0.14)
    assert isinstance(model, PairLIE) and model.alpha == 0.14 and not model.training


# --- Real data (LOL-v1 test split, released weights) ---

@pytest.mark.full
@pytest.mark.parametrize(("alpha", "expected_psnr"), [(0.2, 18.498), (0.14, 19.521)])
def test_lol_v1_psnr_regression(alpha: float, expected_psnr: float, regression_device: torch.device):
    """Mean PSNR of 8-bit outputs vs targets; measured 2026-11-18 (outputs identical to mon_old)."""
    from mon.core.path import datasets_dir
    from mon.data.zoo.lol import LOLv1
    try:
        root = datasets_dir() / LOLv1.folder
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    model = PairLIE(alpha=alpha).eval().to(regression_device)
    model.load_state_dict(released())
    psnrs = []
    with torch.no_grad():
        for sample in LOLv1("test"):
            x      = (sample["image"].float() / 255).to(regression_device)
            target = (sample["target"].float() / 255).to(regression_device)
            y = (model(x[None])["enhanced"][0].clamp(0, 1) * 255).round() / 255
            psnrs.append(10 * torch.log10(1 / F.mse_loss(y, target)).item())
    assert sum(psnrs) / len(psnrs) == pytest.approx(expected_psnr, abs=5e-3)
