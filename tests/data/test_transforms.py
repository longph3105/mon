"""Tests for ``mon.data.transforms`` (synthetic tensors only)."""

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mon.core.size import Size
from mon.data.structures import SampleMeta
from mon.data.transforms import (
    ApplyTo, MinMaxNormalize, PadToMultiple, ResizeToMultiple, crop_to, pad_to_multiple,
)


def _sample(h: int = 37, w: int = 53) -> dict:
    """Image + target + mask + one box + labels + meta, all ``h × w``."""
    torch.manual_seed(0)
    mask = torch.zeros(h, w, dtype=torch.uint8)
    mask[5:15, 10:30] = 7
    return {
        "image" : tv_tensors.Image(torch.rand(3, h, w)),
        "target": tv_tensors.Image(torch.rand(3, h, w)),
        "mask"  : tv_tensors.Mask(mask),
        "boxes" : tv_tensors.BoundingBoxes(torch.tensor([[10.0, 5.0, 30.0, 15.0]]), format="XYXY", canvas_size=(h, w)),
        "labels": torch.tensor([3]),
        "meta"  : SampleMeta(index=0, stem="x", orig_size=Size(h, w)),
    }


# --- ApplyTo ---

def test_apply_to_changes_only_selected_keys():
    s   = _sample()
    out = ApplyTo(["image"], v2.ColorJitter(brightness=(1.5, 1.5)))(s)
    assert not torch.equal(out["image"], s["image"])
    assert type(out["image"]) is tv_tensors.Image
    for k in ("target", "mask", "boxes", "labels", "meta"):
        assert out[k] is s[k]


def test_apply_to_inside_two_stage_pipeline_keeps_target_aligned():
    s  = _sample()
    tf = v2.Compose([v2.RandomHorizontalFlip(p=1.0), ApplyTo(["image"], v2.ColorJitter(brightness=(1.5, 1.5)))])
    out = tf(s)
    assert torch.equal(out["target"], s["target"].flip(-1))  # flipped, not jittered
    assert torch.equal(out["mask"], s["mask"].flip(-1))
    assert not torch.equal(out["image"], s["image"].flip(-1))


def test_apply_to_shares_random_params_across_keys():
    x   = tv_tensors.Image(torch.rand(3, 8, 8))
    out = ApplyTo(["a", "b"], v2.ColorJitter(0.5, 0.5, 0.5, 0.2))({"a": x, "b": x.clone(), "c": 1})
    assert torch.equal(out["a"], out["b"])
    assert out["c"] == 1


def test_apply_to_missing_key():
    s = _sample()
    with pytest.raises(KeyError, match="depth"):
        ApplyTo(["image", "depth"], v2.Identity())(s)
    out = ApplyTo(["depth"], v2.Identity(), missing_ok=True)(s)
    assert set(out) == set(s)


def test_apply_to_rejects_bad_args():
    with pytest.raises(TypeError, match="not a str"):
        ApplyTo("image", v2.Identity())
    with pytest.raises(ValueError, match="empty"):
        ApplyTo([], v2.Identity())
    with pytest.raises(TypeError, match="dict sample"):
        ApplyTo(["image"], v2.Identity())(torch.rand(3, 4, 4))
    with pytest.raises(TypeError, match="must return a dict"):
        ApplyTo(["image"], lambda d: d["image"])(_sample())


# --- ResizeToMultiple ---

def test_resize_to_multiple_keeps_sample_aligned():
    s   = _sample(37, 53)
    out = ResizeToMultiple(32)(s)
    for k in ("image", "target", "mask"):
        assert out[k].shape[-2:] == (64, 64)
    assert tuple(out["boxes"].canvas_size) == (64, 64)
    assert torch.allclose(out["boxes"], torch.tensor([[10 * 64 / 53, 5 * 64 / 37, 30 * 64 / 53, 15 * 64 / 37]]))
    assert set(out["mask"].unique().tolist()) == {0, 7}  # nearest
    assert out["labels"] is s["labels"] and out["meta"] is s["meta"]


def test_resize_to_multiple_with_target_size():
    out = ResizeToMultiple(32, size=(100, 200))(tv_tensors.Image(torch.rand(3, 10, 10)))
    assert out.shape == (3, 128, 224)
    assert ResizeToMultiple(8, size=16)(tv_tensors.Image(torch.rand(3, 10, 10))).shape == (3, 16, 16)


def test_resize_to_multiple_exact_size_unchanged():
    x = tv_tensors.Image(torch.rand(3, 64, 32))
    assert torch.allclose(ResizeToMultiple(32)(x), x)


def test_resize_to_multiple_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="differ in size"):
        ResizeToMultiple(32)({"a": tv_tensors.Image(torch.rand(3, 8, 8)), "b": tv_tensors.Image(torch.rand(3, 9, 9))})


@pytest.mark.parametrize("divisor", [0, -1, 1.5, True])
def test_bad_divisor(divisor):
    for cls in (ResizeToMultiple, PadToMultiple):
        with pytest.raises(ValueError, match="divisor"):
            cls(divisor)
    with pytest.raises(ValueError, match="divisor"):
        pad_to_multiple(torch.rand(1, 3, 4, 4), divisor)


# --- PadToMultiple / pad_to_multiple / crop_to ---

def test_pad_to_multiple_sample_round_trip():
    s   = _sample(37, 53)
    out = PadToMultiple(32)(s)
    for k in ("image", "target", "mask"):
        assert out[k].shape[-2:] == (64, 64)
        assert torch.equal(crop_to(out[k], s["meta"].orig_size), s[k])
    assert torch.equal(out["boxes"], s["boxes"])  # coordinates unchanged
    assert tuple(out["boxes"].canvas_size) == (64, 64)
    kp = tv_tensors.KeyPoints(torch.tensor([[3.0, 4.0]]), canvas_size=(37, 53))
    out = PadToMultiple(32)({"image": s["image"], "kp": kp})
    assert torch.equal(out["kp"], kp) and tuple(out["kp"].canvas_size) == (64, 64)


def test_pad_to_multiple_reflect_vs_constant():
    x = tv_tensors.Image(torch.arange(1.0, 4.0).reshape(1, 1, 3).expand(1, 3, 3).clone())
    assert PadToMultiple(4)(x)[0, 0].tolist() == [1.0, 2.0, 3.0, 2.0]
    assert PadToMultiple(4, padding_mode="constant", fill=0)(x)[0, 0].tolist() == [1.0, 2.0, 3.0, 0.0]


def test_pad_to_multiple_noop_when_divisible():
    x = tv_tensors.Image(torch.rand(3, 64, 32))
    assert PadToMultiple(32)(x) is x
    b = torch.rand(2, 3, 64, 32)
    y, size = pad_to_multiple(b, 32)
    assert y is b and size == Size(64, 32)


def test_pad_to_multiple_batch_round_trip():
    b       = torch.rand(2, 3, 37, 53)
    y, size = pad_to_multiple(b, 16)
    assert y.shape == (2, 3, 48, 64) and size == Size(37, 53)
    assert torch.equal(crop_to(y, size), b)


def test_crop_to_rejects_larger_size():
    with pytest.raises(ValueError, match="larger"):
        crop_to(torch.rand(3, 8, 8), (9, 8))


# --- MinMaxNormalize ---

def test_min_max_normalize_whole_image_and_per_channel():
    x = tv_tensors.Image(torch.stack([torch.full((2, 2), 1.0), torch.tensor([[1.0, 2.0], [3.0, 5.0]])]))
    whole = MinMaxNormalize()(x)
    assert type(whole) is tv_tensors.Image
    assert whole.min() == 0 and whole.max() == 1
    assert torch.allclose(whole[0], torch.zeros(2, 2))
    per = MinMaxNormalize(per_channel=True)(x)
    assert torch.allclose(per[0], torch.zeros(2, 2))  # constant channel -> 0, not NaN
    assert torch.allclose(per[1], torch.tensor([[0.0, 0.25], [0.5, 1.0]]))


def test_min_max_normalize_uint16_and_batch():
    x = tv_tensors.Image(torch.tensor([[[0, 1000], [30000, 65535]]], dtype=torch.uint16))
    out = MinMaxNormalize()(x)
    assert out.dtype == torch.float32 and out.max() == 1
    b = torch.stack([torch.rand(3, 4, 4), torch.rand(3, 4, 4) * 100])  # per sample in a batch
    out = MinMaxNormalize()(b)
    assert torch.allclose(out.amax(dim=(-3, -2, -1)), torch.ones(2))


def test_min_max_normalize_skips_non_images():
    s   = _sample()
    out = MinMaxNormalize()(s)
    assert out["mask"] is s["mask"] and out["boxes"] is s["boxes"] and out["labels"] is s["labels"]
    assert out["target"].max() == 1  # every Image changes: use ApplyTo to limit it
    out = ApplyTo(["image"], MinMaxNormalize())(s)
    assert out["target"] is s["target"]


def test_min_max_normalize_bad_args():
    with pytest.raises(ValueError, match="eps"):
        MinMaxNormalize(eps=0)
    with pytest.raises(ValueError, match=r"\[..., C, H, W\]"):
        MinMaxNormalize()(torch.rand(4, 4))
