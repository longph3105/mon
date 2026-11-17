"""Tests for ``ClassFolderDataset`` and ``mon.data.zoo.classify`` (synthetic + 100-image real-data samples)."""

import random
from collections import Counter
from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS, ClassFolderDataset, classification_layout
from mon.data.io import write_image
from mon.data.loader import create_dataloader
from mon.data.zoo.classify import CIFAR10, CIFAR100, MNIST, FashionMNIST


class Toy(ClassFolderDataset):
    layout = classification_layout("train", "test")


def _write(root: StdPath, classes: list[str], files: dict[str, dict[str, int]], c: int = 3) -> StdPath:
    """``files``: split -> class -> number of images."""
    (root / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(classes) + "\n")
    for split, per_class in files.items():
        for cls, n in per_class.items():
            for i in range(n):
                write_image(torch.full((c, 4, 6), i, dtype=torch.uint8), root / split / cls / f"{i:03d}.png")
    return root


# --- ClassFolderDataset (synthetic) ---

def test_labels_follow_classes_txt_not_folder_order(tmp_path: StdPath):
    root = _write(tmp_path, ["zebra", "ant"], {"train": {"zebra": 2, "ant": 1}, "test": {"ant": 1}})
    ds = Toy("train", root=root)
    assert ds.classes.names == ["zebra", "ant"]
    assert [ds[i]["meta"].stem for i in range(len(ds))] == ["ant/000", "zebra/000", "zebra/001"]
    assert ds.labels == [1, 0, 0]
    s = ds[0]
    assert set(s) == {"image", "label", "meta"}
    assert isinstance(s["image"], tv_tensors.Image) and s["image"].shape == (3, 4, 6)
    assert s["label"].dtype == torch.int64 and s["label"].ndim == 0 and int(s["label"]) == 1
    assert s["meta"].split is Split.TRAIN
    assert len(Toy("test", root=root)) == 1  # class "zebra" absent from test: allowed


def test_collate_stacks_labels(tmp_path: StdPath):
    root = _write(tmp_path, ["a", "b"], {"train": {"a": 2, "b": 2}})
    batch = next(iter(create_dataloader(Toy("train", root=root), batch_size=4)))
    assert batch["image"].shape == (4, 3, 4, 6) and batch["label"].tolist() == [0, 0, 1, 1]


def test_gray_mode_and_hidden_files(tmp_path: StdPath):
    class Gray(Toy):
        image_mode = "gray"

    root = _write(tmp_path, ["a"], {"train": {"a": 1}})
    write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), root / "train" / "a" / "._000.png")
    (root / "train" / ".ipynb_checkpoints").mkdir()
    ds = Gray("train", root=root)
    assert len(ds) == 1 and ds[0]["image"].shape == (1, 4, 6)


def test_errors(tmp_path: StdPath):
    root = _write(tmp_path / "d", ["a"], {"train": {"a": 1, "b": 1}})
    with pytest.raises(ValueError, match=r"not in classes.txt: \['b'\]"):
        Toy("train", root=root)
    with pytest.raises(ValueError, match="no 'val' split"):
        Toy("val", root=root)
    with pytest.raises(FileNotFoundError, match="split folder"):
        Toy("test", root=root)
    (root / "classes.txt").unlink()
    with pytest.raises(FileNotFoundError, match="classes.txt"):
        Toy("train", root=root)
    root2 = _write(tmp_path / "e", ["a", "a"], {"train": {"a": 1}})
    with pytest.raises(ValueError):
        Toy("train", root=root2)  # duplicate class name
    root3 = _write(tmp_path / "f", ["a"], {"train": {}})
    (root3 / "train").mkdir()
    with pytest.raises(ValueError, match="no images"):
        Toy("train", root=root3)
    with pytest.raises(ValueError, match="no `folder`"):
        Toy("train")


def test_classification_layout():
    assert classification_layout("train", Split.TEST) == {Split.TRAIN: "train", Split.TEST: "test"}
    with pytest.raises(ValueError, match="duplicate"):
        classification_layout("train", "train")
    with pytest.raises(ValueError, match="at least one"):
        classification_layout()


# --- Zoo registration ---

ZOO = {"cifar10": CIFAR10, "cifar100": CIFAR100, "mnist": MNIST, "fashion_mnist": FashionMNIST}


@pytest.mark.parametrize(("name", "cls"), ZOO.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls and cls.folder == name
    assert DATASETS.entry(name).meta["tasks"] == {Task.CLASSIFY}
    assert DATASETS.entry(name).meta["splits"] == {Split.TRAIN, Split.TEST}


# --- Real data: full index (no decoding), 100 decoded samples per split ---

EXPECTED = {  # name: (train, test, classes, channels, side, per-class train counts (min, max))
    "cifar10"      : (50000, 10000, 10, 3, 32, (5000, 5000)),
    "cifar100"     : (50000, 10000, 100, 3, 32, (500, 500)),
    "mnist"        : (60000, 10000, 10, 1, 28, (5421, 6742)),
    "fashion_mnist": (60000, 10000, 10, 1, 28, (6000, 6000)),
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
