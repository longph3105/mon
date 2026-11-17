"""Tests for ``mon.core.size``."""

import numpy as np
import pytest
import torch

from mon.core.size import Size


def test_basic_properties_and_unpacking():
    s = Size(100, 200)
    assert (s.h, s.w, s.hw, s.wh, s.area) == (100, 200, (100, 200), (200, 100), 20000)
    h, w = s
    assert (h, w) == (100, 200) and s[0] == 100 and s[-1] == 200 and len(s) == 2
    with pytest.raises(IndexError):
        s[2]


def test_frozen_hashable_and_equality():
    assert Size(1, 2) == Size(1, 2) and Size(1, 2) != Size(2, 1)
    assert len({Size(1, 2), Size(1, 2)}) == 1
    with pytest.raises(AttributeError):
        Size(1, 2).height = 3  # type: ignore[misc]


@pytest.mark.parametrize("bad", [(0, 1), (-1, 5), (1.5, 2), (True, 2), ("1", 2)])
def test_validation(bad):
    with pytest.raises(ValueError, match="positive int"):
        Size(*bad)


def test_no_ordering_but_fits_within():
    with pytest.raises(TypeError):
        _ = Size(1, 10) < Size(10, 1)  # type: ignore[operator]
    assert Size(10, 20).fits_within(Size(10, 30))
    assert not Size(1, 10).fits_within(Size(10, 1)) and not Size(10, 1).fits_within(Size(1, 10))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(224, (224, 224)), ((480, 640), (480, 640)), ([32], (32, 32)), (Size(3, 4), (3, 4))],
)
def test_from_any(value, expected):
    assert Size.from_any(value).hw == expected


@pytest.mark.parametrize(("value", "exc"), [((3, 224, 224), ValueError), ((), ValueError), ("224", TypeError), (2.0, TypeError), (None, TypeError)])
def test_from_any_rejects_ambiguous(value, exc):
    with pytest.raises(exc):
        Size.from_any(value)


def test_from_shape_layouts():
    assert Size.from_shape(torch.zeros(2, 3, 32, 48)) == Size(32, 48)
    assert Size.from_shape(torch.Size([3, 32, 48])) == Size(32, 48)
    assert Size.from_shape(np.zeros((32, 48))) == Size(32, 48)                   # grayscale HW, 'chw' = last two
    assert Size.from_shape(np.zeros((32, 48, 3)), layout="hwc") == Size(32, 48)
    assert Size.from_shape((4, 32, 48, 3), layout="hwc") == Size(32, 48)
    with pytest.raises(ValueError, match=">= 3 dims"):
        Size.from_shape((32, 48), layout="hwc")
    with pytest.raises(ValueError, match="layout"):
        Size.from_shape((1, 2), layout="whc")  # type: ignore[arg-type]


def test_round_up():
    assert Size(200, 100).round_up(32) == Size(224, 128)
    assert Size(64, 64).round_up(32) == Size(64, 64)
    with pytest.raises(ValueError):
        Size(1, 1).round_up(0)
