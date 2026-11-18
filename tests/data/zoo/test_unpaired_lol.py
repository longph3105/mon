"""Tests for ``mon.data.zoo.unpaired_lol``: synthetic folders, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS, unpaired_layout
from mon.data.io import write_image
from mon.data.zoo.unpaired_lol import UNPAIRED_LOL_SETS

EXPECTED = {"dicm": 69, "fusion": 16, "lime": 10, "mef": 17, "npe": 8, "vv": 24}


# --- Registration and layout ---

@pytest.mark.parametrize(("name", "cls"), UNPAIRED_LOL_SETS.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls
    assert DATASETS.entry(name).meta["tasks"] == {Task.LLE}
    assert DATASETS.entry(name).meta["splits"] == {Split.TEST}
    assert cls.folder == f"unpaired_lol/{name}"


def test_unpaired_layout():
    layout = unpaired_layout("val", Split.TEST)
    assert list(layout) == [Split.VAL, Split.TEST]
    assert list(layout[Split.TEST]) == ["image"] and layout[Split.TEST]["image"].dir == "test/images"
    with pytest.raises(ValueError, match="at least one"):
        unpaired_layout()
    with pytest.raises(ValueError, match="duplicate"):
        unpaired_layout("test", Split.TEST)


@pytest.mark.parametrize("cls", UNPAIRED_LOL_SETS.values())
def test_loads_synthetic_folder(tmp_path: StdPath, cls: type):
    for i, name in enumerate(("b.png", "a b.jpg")):  # spaces occur upstream (NPE "night fall.JPG")
        write_image(torch.full((3, 4, 6), i, dtype=torch.uint8), tmp_path / "test" / "images" / name, quality=100)
    ds = cls("test", root=tmp_path)
    assert len(ds) == 2 and ds[0]["meta"].stem == "a b"
    s = ds[1]
    assert set(s) == {"image", "meta"} and isinstance(s["image"], tv_tensors.Image)
    with pytest.raises(ValueError, match="no 'train' split"):
        cls("train", root=tmp_path)


# --- Real data (skipped unless the dataset is on this machine) ---

def _real_root(name: str) -> StdPath:
    try:
        root = datasets_dir() / "unpaired_lol" / name
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
@pytest.mark.parametrize("name", EXPECTED)
def test_real_counts_and_rgb(name: str):
    ds = UNPAIRED_LOL_SETS[name]("test", root=_real_root(name))
    assert len(ds) == EXPECTED[name]
    for i in range(len(ds)):  # every file decodes to 3-channel uint8, incl. Fusion's BMPs named .jpg
        image = ds[i]["image"]
        assert image.dtype == torch.uint8 and image.shape[0] == 3
