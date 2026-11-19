"""Tests for ``mon.data.zoo.testkit``: helpers, and the built testkit when it is on this machine."""

import numpy as np
import pytest

from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.zoo.testkit import SIZE, TestkitMEF, _evenly, _square


def test_square_resizes_short_side_and_centre_crops():
    img = np.zeros((400, 600, 3), np.uint8)
    img[:, 290:310] = 255  # a centred stripe survives the centre crop
    out = _square(img)
    assert out.shape == (SIZE, SIZE, 3)
    assert out[:, SIZE // 2].min() > 200 and out[:, 0].max() == 0
    assert _square(np.zeros((300, 200, 3), np.uint8)).shape == (SIZE, SIZE, 3)  # portrait


def test_evenly_spaced_selection():
    assert _evenly(list(range(10)), 3) == [0, 4, 9]  # first, middle, last
    assert _evenly([1, 2], 5) == [1, 2]
    assert _evenly(list(range(7)), 1) == [0]


def test_mef_targets_pair_by_scene():
    assert TestkitMEF.pair_key("target", "sice_58/2") == "sice_58"
    assert TestkitMEF.pair_key("image", "sice_58/2") == "sice_58/2"


@pytest.mark.slow
@pytest.mark.parametrize(("name", "count", "keys", "size"), [
    ("testkit_lle", 100, {"image", "target", "meta"}, SIZE),
    ("testkit_lle_unpaired", 100, {"image", "meta"}, SIZE),
    ("testkit_mef", 100, {"image", "target", "meta"}, SIZE),
    ("testkit_classify", 100, {"image", "label", "meta"}, 32),
])
def test_built_testkit(name: str, count: int, keys: set[str], size: int):
    try:
        root = datasets_dir() / "testkit"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    ds = DATASETS.build(name, "test")
    assert len(ds) == count
    for i in range(len(ds)):
        sample = ds[i]
        assert set(sample) == keys and tuple(sample["image"].shape) == (3, size, size)
    if name == "testkit_classify":
        assert sorted(ds.labels) == sorted(list(range(10)) * 10)  # 10 per class
