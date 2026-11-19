"""Tests for ``mon.data.zoo.lsrw``: synthetic folders, plus checks on the real data when present."""

import hashlib
from collections import Counter
from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.lsrw import LSRWHuawei, LSRWNikon

CLASSES = {"lsrw_huawei": LSRWHuawei, "lsrw_nikon": LSRWNikon}
EXPECTED = {"lsrw_huawei": {Split.TRAIN: 2450, Split.TEST: 30}, "lsrw_nikon": {Split.TRAIN: 3150, Split.TEST: 20}}
REPEATED = {"lsrw_huawei": 0, "lsrw_nikon": 47}  # train pairs repeated byte-for-byte under another name


@pytest.mark.parametrize(("name", "cls"), CLASSES.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls
    assert DATASETS.entry(name).meta["tasks"] == {Task.LLE}
    assert DATASETS.entry(name).meta["splits"] == {Split.TRAIN, Split.TEST}
    assert cls.folder == f"lsrw/{name.removeprefix('lsrw_')}"


def test_loads_synthetic_folder(tmp_path: StdPath):
    for sub, value in (("images", 20), ("targets", 150)):
        write_image(torch.full((3, 4, 6), value, dtype=torch.uint8), tmp_path / "test" / sub / "2037.jpg")
    s = LSRWHuawei("test", root=tmp_path)[0]
    assert set(s) == {"image", "target", "meta"} and s["meta"].stem == "2037"
    assert s["image"].float().mean() < s["target"].float().mean()


@pytest.mark.slow
@pytest.mark.parametrize("name", CLASSES)
def test_real_counts_and_repeats(name: str):
    try:
        root = datasets_dir() / "lsrw" / name.removeprefix("lsrw_")
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    for split, n in EXPECTED[name].items():
        ds = CLASSES[name](split, root=root)
        assert len(ds) == n
        if split is Split.TRAIN:
            digests = Counter(hashlib.md5(p["image"].read_bytes()).hexdigest() for _, p in ds.items())
            assert sum(c - 1 for c in digests.values()) == REPEATED[name]
    s = CLASSES[name]("test", root=root)[0]
    assert s["image"].shape == s["target"].shape and s["image"].shape[-1] == 960
