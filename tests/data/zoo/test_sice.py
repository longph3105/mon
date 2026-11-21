"""Tests for ``mon.data.zoo.sice``: synthetic folders, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.sice import SICE, SICEME, SICEUnder


def _img(value: int) -> torch.Tensor:
    return torch.full((3, 4, 6), value, dtype=torch.uint8)


# --- Registration ---

@pytest.mark.parametrize(("name", "cls", "splits"), [
    ("sice", SICE, {Split.TRAIN, Split.TEST}),
    ("sice_me", SICEME, {Split.TRAIN, Split.VAL}),
])
def test_registered(name: str, cls: type, splits: set[Split]):
    assert DATASETS.get(name) is cls
    assert DATASETS.entry(name).meta["tasks"] == {Task.LLE}
    assert DATASETS.entry(name).meta["splits"] == splits
    assert cls.folder == name


def test_sice_under_registered():
    assert DATASETS.get("sice_under") is SICEUnder
    assert DATASETS.entry("sice_under").meta["tasks"] == {Task.LLE}
    assert DATASETS.entry("sice_under").meta["splits"] == {Split.TEST}
    assert SICEUnder.folder == "sice"  # lives inside the sice folder, as sice/test_under/


# --- Synthetic folders ---

def test_sice_pairs_each_exposure_with_its_scene_target(tmp_path: StdPath):
    for scene, n in (("1", 3), ("10", 2)):
        for k in range(1, n + 1):
            write_image(_img(k), tmp_path / "train" / "images" / scene / f"{k}.jpg", quality=100)
    write_image(_img(100), tmp_path / "train" / "targets" / "1.png")
    write_image(_img(110), tmp_path / "train" / "targets" / "10.JPG", quality=100)  # mixed case, as upstream
    ds = SICE("train", root=tmp_path)
    assert [ds[i]["meta"].stem for i in range(len(ds))] == ["1/1", "1/2", "1/3", "10/1", "10/2"]
    s = ds[3]
    assert set(s) == {"image", "target", "meta"} and isinstance(s["target"], tv_tensors.Image)
    assert s["meta"].paths["target"].name == "10.JPG"
    assert ds[0]["meta"].paths["target"] == ds[2]["meta"].paths["target"]


def test_sice_missing_scene_target(tmp_path: StdPath):
    write_image(_img(1), tmp_path / "test" / "images" / "5" / "1.png")
    write_image(_img(1), tmp_path / "test" / "images" / "6" / "1.png")
    write_image(_img(2), tmp_path / "test" / "targets" / "5.png")
    with pytest.raises(ValueError, match="target: 1 missing"):
        SICE("test", root=tmp_path)
    assert len(SICE("test", root=tmp_path, on_missing="skip")) == 1


def test_sice_me_train_has_no_targets(tmp_path: StdPath):
    write_image(_img(1), tmp_path / "train" / "images" / "1_1.png")
    write_image(_img(1), tmp_path / "val" / "images" / "1_2.png")
    write_image(_img(9), tmp_path / "val" / "targets" / "1_2.png")
    assert set(SICEME("train", root=tmp_path)[0]) == {"image", "meta"}
    s = SICEME("val", root=tmp_path)[0]
    assert set(s) == {"image", "target", "meta"} and s["meta"].stem == "1_2"
    with pytest.raises(ValueError, match="no 'test' split"):
        SICEME("test", root=tmp_path)


def test_sice_under_pairs_by_scene(tmp_path: StdPath):
    for scene in ("2", "10"):
        write_image(_img(10), tmp_path / "test_under" / "images" / f"{scene}.jpg", quality=100)
        write_image(_img(200), tmp_path / "test_under" / "targets" / f"{scene}.png")
    ds = SICEUnder("test", root=tmp_path)
    assert [ds[i]["meta"].stem for i in range(len(ds))] == ["10", "2"]
    assert set(ds[0]) == {"image", "target", "meta"}
    with pytest.raises(ValueError, match="no 'train' split"):
        SICEUnder("train", root=tmp_path)


# --- Real data (skipped unless the datasets are on this machine) ---

def _real_root(name: str) -> StdPath:
    try:
        root = datasets_dir() / name
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
def test_real_sice_counts():
    root = _real_root("sice")
    for split, n_images, n_scenes in ((Split.TRAIN, 3021, 360), (Split.TEST, 1779, 229)):
        ds = SICE(split, root=root)
        assert len(ds) == n_images
        assert len({k.split("/")[0] for k, _ in ds._items}) == n_scenes
    s = SICE(Split.TEST, root=root)[0]
    assert s["image"].shape == s["target"].shape and s["image"].dtype == torch.uint8


@pytest.mark.slow
def test_real_sice_me_is_part1():
    me_root, sice_root = _real_root("sice_me"), _real_root("sice")
    train, val = SICEME("train", root=me_root), SICEME("val", root=me_root)
    assert (len(train), len(val)) == (2422, 599)
    names = {k for k, _ in train._items} | {k for k, _ in val._items}
    part1 = {k.replace("/", "_") for k, _ in SICE("train", root=sice_root)._items}
    assert names == part1 and not {k for k, _ in train._items} & {k for k, _ in val._items}



@pytest.mark.slow
def test_real_sice_under():
    root = _real_root("sice")
    if not (root / "test_under").is_dir():
        pytest.skip("sice/test_under not built (make_sice.sh)")
    ds = SICEUnder("test", root=root)
    assert len(ds) == 210
    assert {k for k, _ in ds._items} <= {k.split("/")[0] for k, _ in SICE("test", root=root)._items}
    for scene in ("127", "169", "194", "195", "208", "212"):  # upstream targets rotated 90°, fixed in test_under
        s = ds[[k for k, _ in ds._items].index(scene)]
        assert s["image"].shape == s["target"].shape
