"""Tests for ``mon.data.zoo.lol``: synthetic folders, plus checks on the real data when present."""

import hashlib
from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS, paired_layout
from mon.data.io import write_image
from mon.data.zoo.lol import LOLv1, LOLv2Real

CLASSES = {"lol_v1": LOLv1, "lol_v2_real": LOLv2Real}
EXPECTED = {"lol_v1": {Split.TRAIN: 485, Split.TEST: 15}, "lol_v2_real": {Split.TRAIN: 689, Split.TEST: 100}}


def _write_paired(root: StdPath, names: dict[str, list[str]]) -> None:
    for split, stems in names.items():
        for i, stem in enumerate(stems):
            write_image(torch.full((3, 4, 6), i, dtype=torch.uint8), root / split / "images" / f"{stem}.png")
            write_image(torch.full((3, 4, 6), 100 + i, dtype=torch.uint8), root / split / "targets" / f"{stem}.png")


# --- Registration and layout ---

@pytest.mark.parametrize(("name", "cls"), CLASSES.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls
    assert DATASETS.entry(name).meta["tasks"] == {Task.LLE}
    assert DATASETS.entry(name).meta["splits"] == {Split.TRAIN, Split.TEST}
    assert cls.folder == name
    assert name in DATASETS.names(tasks=Task.LLE)


def test_paired_layout():
    layout = paired_layout("train", Split.TEST)
    assert list(layout) == [Split.TRAIN, Split.TEST]
    assert [m.dir for m in layout[Split.TRAIN].values()] == ["train/images", "train/targets"]
    assert list(layout[Split.TEST]) == ["image", "target"]
    with pytest.raises(ValueError, match="at least one"):
        paired_layout()
    with pytest.raises(ValueError, match="duplicate"):
        paired_layout("train", Split.TRAIN)


@pytest.mark.parametrize("cls", CLASSES.values())
def test_loads_synthetic_folder(tmp_path: StdPath, cls: type):
    _write_paired(tmp_path, {"train": ["10", "2"], "test": ["1"]})
    ds = cls("train", root=tmp_path)
    assert len(ds) == 2
    s = ds[0]
    assert set(s) == {"image", "target", "meta"}
    assert isinstance(s["image"], tv_tensors.Image) and isinstance(s["target"], tv_tensors.Image)
    assert s["meta"].stem == "10" and int(s["target"][0, 0, 0]) == 100
    assert len(cls("test", root=tmp_path)) == 1
    with pytest.raises(ValueError, match="no 'val' split"):
        cls("val", root=tmp_path)


# --- Real data (skipped unless the datasets are on this machine) ---

def _real_root(name: str) -> StdPath:
    try:
        root = datasets_dir() / name
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASET_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
@pytest.mark.parametrize("name", CLASSES)
def test_real_counts_and_format(name: str):
    root = _real_root(name)
    for split, n in EXPECTED[name].items():
        ds = CLASSES[name](split, root=root)
        assert len(ds) == n
        s = ds[0]
        assert s["image"].shape == s["target"].shape == (3, 400, 600)
        assert s["image"].dtype == torch.uint8


def _hashes(root: StdPath, split: str) -> set[str]:
    return {hashlib.md5(p.read_bytes()).hexdigest() for p in sorted((root / split).rglob("*.png"))}


@pytest.mark.slow
@pytest.mark.parametrize("name", CLASSES)
def test_real_train_and_test_disjoint(name: str):
    root = _real_root(name)
    assert not _hashes(root, "train") & _hashes(root, "test")
