"""Tests for ``mon.data.zoo.testkit``: helpers, and the built testkit when it is on this machine."""

import numpy as np
import pytest

from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.zoo.testkit import EFFICIENCY_SIZES, PARTS, SIZE, TestkitMEF, _cover, _evenly, _square, build


def test_square_resizes_short_side_and_centre_crops():
    img = np.zeros((400, 600, 3), np.uint8)
    img[:, 290:310] = 255  # a centred stripe survives the centre crop
    out = _square(img)
    assert out.shape == (SIZE, SIZE, 3)
    assert out[:, SIZE // 2].min() > 200 and out[:, 0].max() == 0
    assert _square(np.zeros((300, 200, 3), np.uint8)).shape == (SIZE, SIZE, 3)  # portrait


def test_cover_scales_and_centre_crops():
    img = np.zeros((300, 450, 3), np.uint8)
    img[140:160, 215:235] = 255  # a centred block stays centred
    for w, h in [(256, 256), (400, 200), (900, 600)]:  # down (area), crop, up (bicubic)
        out = _cover(img, w, h)
        assert out.shape == (h, w, 3) and out[h // 2, w // 2].min() > 200 and out[0, 0].max() == 0


def test_efficiency_sizes_and_parts(tmp_path):
    assert EFFICIENCY_SIZES[0] == (256, 256) and EFFICIENCY_SIZES[-1] == (7680, 4320)
    assert all(w >= h for w, h in EFFICIENCY_SIZES) and "efficiency" in PARTS
    with pytest.raises(ValueError, match="unknown testkit parts"):
        build(tmp_path, parts=["nope"])
    (tmp_path / "efficiency").mkdir()
    (tmp_path / "efficiency" / "x").write_text("")
    with pytest.raises(FileExistsError):
        build(tmp_path, parts=["efficiency"])


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


@pytest.mark.slow
def test_built_testkit_efficiency():
    try:
        root = datasets_dir() / "testkit" / "efficiency"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine (build: python -m mon.data.zoo testkit build --part efficiency)")
    ds = DATASETS.build("testkit_efficiency", "test")
    items = dict(ds.items())
    assert all(set(files) == {"image", "target"} for files in items.values())
    stems = sorted(items)
    assert len(stems) == 5 * len(EFFICIENCY_SIZES)
    scenes = {s.split("/")[1] for s in stems}
    assert len(scenes) == 5 and all(f"{w}x{h}/{sc}" in stems for w, h in EFFICIENCY_SIZES for sc in scenes)
    sample = ds[stems.index("600x400/" + sorted(scenes)[0])]
    assert sample["image"].shape == sample["target"].shape == (3, 400, 600)
