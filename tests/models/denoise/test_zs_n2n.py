"""Tests for ``mon.models.denoise.zs_n2n``: the paper's loss, parity with mon_old, and behavior."""

import importlib.util
from pathlib import Path

import pytest
import torch

from mon.models.denoise.zs_n2n import ZSN2N, ZSN2NLoss, pair_downsample, zs_n2n_denoise
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("zs_n2n_reference", Path(__file__).with_name("zs_n2n_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats, ITERATIONS = _ref.reference_input, _ref.stats, _ref.ITERATIONS

# mon_old ZS_N2N (which matches the official Colab code) after torch.manual_seed(0): initial weights, loss
# at initialization, and the result after ITERATIONS steps. Output of ``zs_n2n_reference.py``.
# Generated from mon_old with torch 2.14.0.
MON_OLD: dict = {
    "init": {
        "conv1.weight": {
            "shape": [48, 3, 3, 3],
            "sum": -2.6454449654193013,
            "sumsq": 16.643728956873254,
            "values": [
                -0.0014408392598852515,
                0.15259164571762085,
                0.12967631220817566,
                -0.13940973579883575,
                -0.016361592337489128,
                7.951641600811854e-05,
                -0.16071386635303497,
            ],
        },
        "conv1.bias": {
            "shape": [48],
            "sum": -0.28599515464156866,
            "sumsq": 0.547764820199541,
            "values": [
                0.07237076759338379,
                0.15575577318668365,
                0.004155340604484081,
                0.052989497780799866,
                0.08634032309055328,
                -0.07147350907325745,
                0.1680828034877777,
            ],
        },
        "conv2.weight": {
            "shape": [48, 48, 3, 3],
            "sum": 3.909176804175786,
            "sumsq": 15.956705877904623,
            "values": [
                -0.01740807294845581,
                0.046124618500471115,
                0.014728046953678131,
                0.01498213317245245,
                -0.0013804677873849869,
                -0.010636536404490471,
                -0.0004677554243244231,
            ],
        },
        "conv2.bias": {
            "shape": [48],
            "sum": -0.01747287204489112,
            "sumsq": 0.03407639322171445,
            "values": [
                0.04034576937556267,
                0.03568185493350029,
                0.024484826251864433,
                0.025472765788435936,
                -0.029922237619757652,
                0.032278575003147125,
                -0.032000582665205,
            ],
        },
        "conv3.weight": {
            "shape": [3, 48, 1, 1],
            "sum": 0.9201678648241796,
            "sumsq": 0.9637687576163187,
            "values": [
                -0.032582737505435944,
                -0.11975255608558655,
                -0.11248122900724411,
                0.07708027213811874,
                0.04276395961642265,
                -0.026587743312120438,
                0.09782354533672333,
            ],
        },
        "conv3.bias": {
            "shape": [3],
            "sum": -0.14217806607484818,
            "sumsq": 0.04982165781764741,
            "values": [
                -0.1300467699766159,
                -0.13419821858406067,
                0.1220669224858284,
                -0.13419821858406067,
                -0.13419821858406067,
                -0.13419821858406067,
                0.1220669224858284,
            ],
        },
    },
    "loss0": 0.031736522912979126,
    "restored": {
        "shape": [1, 3, 41, 53],
        "sum": 2373.649613466114,
        "sumsq": 1005.6648424138863,
        "values": [
            0.268196165561676,
            0.48514002561569214,
            0.6633163094520569,
            0.5113349556922913,
            0.09503906965255737,
            0.28912797570228577,
            0.0,
        ],
    },
}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = actual.numel()
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


# --- Pair downsampler and loss follow the paper ---

def test_pair_downsample_averages_diagonals_and_drops_odd_edges():
    x = torch.arange(1.0, 31.0).view(1, 1, 5, 6)  # odd height: last row dropped
    d1, d2 = pair_downsample(x)
    assert d1.shape == d2.shape == (1, 1, 2, 3)
    assert d1[0, 0, 0, 0] == (x[0, 0, 0, 1] + x[0, 0, 1, 0]) / 2  # filter1 [[0, .5], [.5, 0]]
    assert d2[0, 0, 0, 0] == (x[0, 0, 0, 0] + x[0, 0, 1, 1]) / 2  # filter2 [[.5, 0], [0, .5]]

    def conv(k):
        return torch.nn.functional.conv2d(x[..., :4, :], torch.tensor(k).view(1, 1, 2, 2), stride=2)

    assert torch.equal(d1, conv([0.0, 0.5, 0.5, 0.0])) and torch.equal(d2, conv([0.5, 0.0, 0.0, 0.5]))
    with pytest.raises(ValueError, match="at least 2"):
        pair_downsample(torch.zeros(1, 1, 1, 8))


def test_loss_follows_the_paper():
    torch.manual_seed(1)
    model = ZSN2N()
    y = torch.rand(2, 3, 18, 22)
    f = model.predict_noise
    y1, y2 = pair_downsample(y)
    mse = torch.nn.functional.mse_loss
    res = 0.5 * (mse(y1, y2 - f(y2)) + mse(y2, y1 - f(y1)))
    d1, d2 = pair_downsample(y - f(y))
    cons = 0.5 * (mse(y1 - f(y1), d1) + mse(y2 - f(y2), d2))
    assert torch.allclose(ZSN2NLoss()(model, y), res + cons)
    per_sample = ZSN2NLoss(reduction="none")(model, y)
    assert per_sample.shape == (2,) and torch.allclose(per_sample[1], ZSN2NLoss()(model, y[1:]))
    with pytest.raises(ValueError, match="reduction"):
        ZSN2NLoss(reduction="max")  # type: ignore[arg-type]


# --- Parity with mon_old ---

def test_initialization_and_loss_match_mon_old():
    torch.manual_seed(0)
    model = ZSN2N()
    for name, value in model.state_dict().items():
        assert_matches(value, MON_OLD["init"][name], atol=1e-12)  # same RNG use: identical weights
    assert ZSN2NLoss()(model, reference_input()).item() == pytest.approx(MON_OLD["loss0"], rel=1e-6)


def test_fitting_matches_mon_old():
    out = zs_n2n_denoise(reference_input(), iterations=ITERATIONS, seed=0)
    assert_matches(out, MON_OLD["restored"], atol=1e-5)


# --- Behavior ---

def noisy_ramp(seed: int, h: int = 32, w: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    clean = torch.linspace(0.2, 0.8, w).expand(1, 3, h, w).clone()
    return clean, (clean + 0.1 * torch.randn(clean.shape, generator=g)).clamp(0, 1)


def test_denoising_reduces_error():
    clean, noisy = noisy_ramp(0)
    out = zs_n2n_denoise(noisy, iterations=300, seed=0)
    assert (out - clean).pow(2).mean() < 0.5 * (noisy - clean).pow(2).mean()
    assert out.min() >= 0 and out.max() <= 1


def test_each_image_gets_a_fresh_network():
    _, a = noisy_ramp(1)
    _, b = noisy_ramp(2)
    batched = zs_n2n_denoise(torch.cat([a, b]), iterations=30, seed=0)
    assert torch.allclose(batched[0], zs_n2n_denoise(a, iterations=30, seed=0)[0], atol=1e-6)
    assert torch.allclose(batched[1], zs_n2n_denoise(b, iterations=30, seed=0)[0], atol=1e-6)  # not b after a


def test_seed_is_reproducible_and_leaves_global_rng_alone():
    _, noisy = noisy_ramp(3)
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    first = zs_n2n_denoise(noisy, iterations=20, seed=7)
    assert torch.equal(torch.rand(1), expected_next)
    assert torch.equal(zs_n2n_denoise(noisy, iterations=20, seed=7), first)
    assert not torch.equal(zs_n2n_denoise(noisy, iterations=20, seed=8), first)


def test_works_under_inference_mode_and_for_unbatched_input():
    _, noisy = noisy_ramp(4, h=17, w=23)  # odd size
    with torch.inference_mode():
        out = zs_n2n_denoise(noisy[0], iterations=5, seed=0)
    assert out.shape == (3, 17, 23)


def test_learning_rate_schedule(monkeypatch: pytest.MonkeyPatch):
    seen = []
    step = torch.optim.Adam.step

    def spy(self, *args, **kwargs):
        seen.append(self.param_groups[0]["lr"])
        return step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", spy)
    zs_n2n_denoise(noisy_ramp(5, 8, 8)[1], iterations=7, lr=0.01, step_size=3, gamma=0.5, seed=0)
    assert seen == pytest.approx([0.01] * 3 + [0.005] * 3 + [0.0025])


@pytest.mark.parametrize(("x", "kwargs", "match"), [
    (torch.rand(3, 8), {}, r"\[B, C, H, W\]"),
    (torch.rand(1, 3, 8, 8), {"iterations": 0}, "iterations must be >= 1"),
])
def test_errors(x: torch.Tensor, kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        zs_n2n_denoise(x, **kwargs)


def test_module_outputs_and_registry():
    model = ZSN2N(channels=1, hidden_dim=8)
    out = model(torch.rand(2, 1, 9, 11))
    assert out["restored"].shape == out["noise"].shape == (2, 1, 9, 11)
    entry = MODELS.entry("zs_n2n")
    assert entry.meta["zero_shot"] is True and entry.obj is ZSN2N
    with pytest.raises(ValueError, match="no pretrained weights"):
        load("zs_n2n")
    assert isinstance(load("zs_n2n", weights=None), ZSN2N)
