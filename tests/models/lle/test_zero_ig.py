"""Tests for ``mon.models.lle.zero_ig``: parity with the official inference, loss, gradients and fitting."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mon.models.lle.zero_ig import ZeroIG, ZeroIG_Weights, ZeroIGLoss, zero_ig_enhance
from mon.models.registry import MODELS, load

_spec = importlib.util.spec_from_file_location("zero_ig_reference", Path(__file__).with_name("zero_ig_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
reference_input, stats = _ref.reference_input, _ref.stats

# Official model.py / loss.py / train.py (Doyle59217/ZeroIG 5af4b8a), CPU. Output of zero_ig_reference.py.
# Generated from the official code with torch 2.15.0+cu132.
OFFICIAL: dict = {
    "inference": {
        "LOL.pt": {
            "enhanced": {
                "shape": [1, 3, 24, 30],
                "sum": 1572.0751309701227,
                "sumsq": 1250.4902248542585,
                "values": [
                    0.6698089838027954,
                    0.9566454887390137,
                    0.6882104873657227,
                    0.14699077606201172,
                    0.8735415935516357,
                    0.9262012243270874,
                    0.7781505584716797,
                ],
            },
            "enhanced_noisy": {
                "shape": [1, 3, 24, 30],
                "sum": 1712.4116000048816,
                "sumsq": 1509.2871435552304,
                "values": [
                    0.6058545708656311,
                    1.0,
                    1.0,
                    0.07531283050775528,
                    1.0,
                    0.9893598556518555,
                    0.9314242005348206,
                ],
            },
        },
        "LSRW-Huawei.pt": {
            "enhanced": {
                "shape": [1, 3, 24, 30],
                "sum": 1398.871166691184,
                "sumsq": 973.1215576304696,
                "values": [
                    0.5567468404769897,
                    0.7515758275985718,
                    0.8662751317024231,
                    0.19582709670066833,
                    0.814521312713623,
                    0.6805074214935303,
                    0.8177321553230286,
                ],
            },
            "enhanced_noisy": {
                "shape": [1, 3, 24, 30],
                "sum": 1490.811042908579,
                "sumsq": 1178.4986412703242,
                "values": [
                    0.5055021047592163,
                    0.9430994391441345,
                    0.8960272073745728,
                    0.05756089836359024,
                    0.8831288814544678,
                    0.8167310357093811,
                    0.864656925201416,
                ],
            },
        },
        "LSRW-Nikon.pt": {
            "enhanced": {
                "shape": [1, 3, 24, 30],
                "sum": 1239.0710333318057,
                "sumsq": 801.0267907430125,
                "values": [
                    0.3836566209793091,
                    0.6733974814414978,
                    0.5972006320953369,
                    0.2665241062641144,
                    0.5560068488121033,
                    0.5604835152626038,
                    0.8325422406196594,
                ],
            },
            "enhanced_noisy": {
                "shape": [1, 3, 24, 30],
                "sum": 1212.4555169455707,
                "sumsq": 778.8733050932852,
                "values": [
                    0.4514715373516083,
                    0.7095491290092468,
                    0.6299106478691101,
                    0.055880289524793625,
                    0.6165976524353027,
                    0.6236698031425476,
                    0.9254141449928284,
                ],
            },
        },
    },
    "loss": {
        "value": 1772.6572265625,
        "grad": {
            "shape": [86572],
            "sum": 186170.90366726468,
            "sumsq": 561282635.0715733,
            "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.542525053024292],
        },
    },
    "loss_eval": {
        "value": 187.08677673339844,
        "grad": {
            "shape": [86572],
            "sum": 26857.19558366407,
            "sumsq": 9505151.297671031,
            "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.43185433745384216],
        },
    },
    "fit": {
        "steps": 5,
        "last_loss": 195.8741455078125,
        "enhanced": {
            "shape": [1, 3, 24, 30],
            "sum": 796.0372074605693,
            "sumsq": 367.81020679047066,
            "values": [
                0.33550238609313965,
                0.5571589469909668,
                0.46118199825286865,
                0.14903052151203156,
                0.3247506022453308,
                0.356512188911438,
                0.4289173483848572,
            ],
        },
    },
}

ZOO = {"LOL.pt": ZeroIG_Weights.LOL_V1, "LSRW-Huawei.pt": ZeroIG_Weights.LSRW_HUAWEI,
       "LSRW-Nikon.pt": ZeroIG_Weights.LSRW_NIKON}


def assert_matches(actual: torch.Tensor, expected: dict, atol: float = 1e-6) -> None:
    got = stats(actual)
    assert got["shape"] == expected["shape"]
    n = math.prod(expected["shape"])
    assert got["sum"] == pytest.approx(expected["sum"], abs=atol * n)
    assert got["sumsq"] == pytest.approx(expected["sumsq"], abs=atol * n)
    assert got["values"] == pytest.approx(expected["values"], abs=atol)


def released(member: ZeroIG_Weights) -> dict:
    try:
        return member.weights.load_state_dict(download=False)
    except FileNotFoundError as err:
        pytest.skip(f"released weights not available: {err}")


# --- Parity ---

@pytest.mark.parametrize("name", list(OFFICIAL["inference"]))
def test_released_weights_load_strictly_and_match_official_inference(name):
    model = ZeroIG().eval()
    model.load_state_dict(released(ZOO[name]))
    with torch.no_grad():
        out = model(reference_input())
    assert_matches(out["enhanced"], OFFICIAL["inference"][name]["enhanced"])
    assert_matches(out["enhanced_noisy"], OFFICIAL["inference"][name]["enhanced_noisy"])


@pytest.mark.parametrize(("mode", "key"), [("train", "loss"), ("eval", "loss_eval")])
def test_loss_and_gradients_match_official(mode, key):
    model = ZeroIG().train(mode == "train")  # eval: how the official training runs after its first epoch
    model.load_state_dict(released(ZeroIG_Weights.LOL_V1))
    loss = ZeroIGLoss()(model, reference_input())
    loss.backward()
    assert loss.item() == pytest.approx(OFFICIAL[key]["value"], rel=1e-6)
    grads = stats(torch.cat([p.grad.flatten() for p in model.parameters()]))
    expected = OFFICIAL[key]["grad"]
    assert grads["shape"] == expected["shape"]
    assert grads["sum"] == pytest.approx(expected["sum"], rel=1e-6)
    assert grads["sumsq"] == pytest.approx(expected["sumsq"], rel=1e-6)
    assert grads["values"] == pytest.approx(expected["values"], rel=1e-5, abs=1e-6)


def test_short_fit_matches_official_training():
    # Same init (seed 2), optimizer and clipping as the official train.py. After a few Adam steps, float32
    # gradient noise (~1e-8 relative) alone moves outputs by ~3e-4, hence the looser tolerance.
    fit = OFFICIAL["fit"]
    out = zero_ig_enhance(reference_input(), iterations=fit["steps"], seed=2)
    assert_matches(out, fit["enhanced"], atol=1e-3)


# --- Network ---

def test_state_dict_layout_and_shared_enhancer_block():
    model = ZeroIG()
    keys = list(model.state_dict())
    assert len(keys) == 44 and sum(k.startswith("enhance.blocks.") for k in keys) == 21
    assert model.enhance.blocks[0] is model.enhance.blocks[2] is model.enhance.conv
    assert sum(p.numel() for p in model.parameters()) == 86572


def test_forward_ranges():
    torch.manual_seed(0)
    model = ZeroIG().eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 12, 16) * 0.3)
    assert all(v.shape == (2, 3, 12, 16) for v in out.values())
    assert out["enhanced"].min() >= 1e-4 and out["enhanced"].max() <= 1
    assert out["illumination"].min() >= 1e-4 and out["illumination"].max() <= 1


# --- Loss ---

def test_loss_reductions():
    torch.manual_seed(0)
    model = ZeroIG().eval()  # eval: BatchNorm doesn't couple the batch
    x = torch.rand(2, 3, 12, 16) * 0.3
    per_sample = ZeroIGLoss(reduction="none")(model, x)
    assert per_sample.shape == (2,)
    assert torch.allclose(ZeroIGLoss()(model, x), per_sample.mean())
    assert torch.allclose(ZeroIGLoss(reduction="sum")(model, x), per_sample.sum())
    assert torch.allclose(per_sample[1], ZeroIGLoss()(model, x[1:2]))


# --- Fitting ---

def test_fit_is_seeded_batched_and_validates():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 12, 16) * 0.3
    a = zero_ig_enhance(x, iterations=2, seed=0)
    b = zero_ig_enhance(x[1], iterations=2, seed=0)
    assert a.shape == x.shape and b.shape == x.shape[1:]
    assert torch.allclose(a[1], b, atol=1e-6)
    with torch.inference_mode():
        assert zero_ig_enhance(x[:1], iterations=1, seed=0).shape == (1, 3, 12, 16)
    with pytest.raises(ValueError, match="even"):
        zero_ig_enhance(torch.rand(1, 3, 13, 16), iterations=1)
    with pytest.raises(ValueError, match="iterations"):
        zero_ig_enhance(x, iterations=0)
    with pytest.raises(ValueError, match="eval_mode_after"):
        zero_ig_enhance(x, iterations=1, eval_mode_after=-1)


def test_fit_eval_mode_after_is_the_official_switch_and_can_be_turned_off():
    x = reference_input()
    official = zero_ig_enhance(x, iterations=3, seed=2)
    assert torch.equal(official, zero_ig_enhance(x, iterations=3, seed=2, eval_mode_after=1))
    train_mode = zero_ig_enhance(x, iterations=3, seed=2, eval_mode_after=None)
    assert torch.equal(train_mode, zero_ig_enhance(x, iterations=3, seed=2, eval_mode_after=3))  # never reached
    assert not torch.allclose(official, train_mode, atol=1e-4)


# --- Registry ---

def test_registry_zero_shot_and_default_weights():
    entry = MODELS.entry("zero_ig")
    assert entry.obj is ZeroIG and entry.meta["zero_shot"] is True
    assert ZeroIG_Weights.DEFAULT is ZeroIG_Weights.LOL_V1
    assert isinstance(load("zero_ig", weights=None), ZeroIG)
