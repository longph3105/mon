"""Tests for ``mon.data.zoo.cifar``: registration, plus 100 decoded samples per split of the real data when present."""

import random
from collections import Counter
from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.zoo.cifar import CIFAR10, CIFAR100

# --- Zoo registration ---

ZOO = {"cifar10": CIFAR10, "cifar100": CIFAR100}


@pytest.mark.parametrize(("name", "cls"), ZOO.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls and cls.folder == name
    assert DATASETS.entry(name).meta["tasks"] == {Task.CLASSIFY}
    assert DATASETS.entry(name).meta["splits"] == {Split.TRAIN, Split.TEST}


# --- Real data: full index (no decoding), 100 decoded samples per split ---

EXPECTED = {  # name: (train, test, classes, channels, side, per-class train counts (min, max))
    "cifar10"      : (50000, 10000, 10, 3, 32, (5000, 5000)),
    "cifar100"     : (50000, 10000, 100, 3, 32, (500, 500)),
}


def _real_root(name: str) -> StdPath:
    try:
        root = datasets_dir() / name
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
@pytest.mark.parametrize("name", EXPECTED)
def test_real_dataset(name: str):
    root = _real_root(name)
    n_train, n_test, n_classes, channels, side, (lo, hi) = EXPECTED[name]
    train, test = ZOO[name]("train", root=root), ZOO[name]("test", root=root)
    assert (len(train), len(test), train.classes.num_classes) == (n_train, n_test, n_classes)
    counts = Counter(train.labels)
    assert len(counts) == n_classes and (min(counts.values()), max(counts.values())) == (lo, hi)
    rng = random.Random(0)
    for ds in (train, test):
        for i in rng.sample(range(len(ds)), 100):
            s = ds[i]
            assert s["image"].shape == (channels, side, side) and s["image"].dtype == torch.uint8
            assert ds.classes.names[int(s["label"])] == s["meta"].paths["image"].parent.name
