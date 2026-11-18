"""``items()`` on ``FolderDataset`` and ``ImageFolder``: stems and files without reading images."""

from pathlib import Path as StdPath

import pytest
import torch

from mon.data.dataset import FolderDataset, ImageFolder, paired_layout
from mon.data.io import write_image


class _Paired(FolderDataset):
    folder = "paired"
    layout = paired_layout("test")


def _img(path: StdPath) -> None:
    write_image(torch.zeros(3, 4, 4, dtype=torch.uint8), path)


def test_folder_dataset_items(tmp_path: StdPath):
    for name in ("a", "sub/b"):
        _img(tmp_path / "test" / "images" / f"{name}.png")
        _img(tmp_path / "test" / "targets" / f"{name}.png")
    ds = _Paired("test", root=tmp_path)
    items = ds.items()
    assert [stem for stem, _ in items] == [ds[i]["meta"].stem for i in range(len(ds))]
    stem, files = items[1]
    assert set(files) == {"image", "target"} and files["target"] == tmp_path / "test" / "targets" / "sub" / "b.png"
    with pytest.raises(TypeError):
        files["image"] = files["target"]  # type: ignore[index]  # read-only view


def test_image_folder_items(tmp_path: StdPath):
    _img(tmp_path / "x.png")
    _img(tmp_path / "night" / "y.jpg")
    assert [(s, f["image"].name) for s, f in ImageFolder(tmp_path).items()] == [("night/y", "y.jpg"), ("x", "x.png")]
