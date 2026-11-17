"""Tests for ``mon.data.dataset.ImageFolder`` and hidden-file skipping (synthetic files)."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mon.core.enums import Split
from mon.core.size import Size
from mon.data.dataset import FolderDataset, ImageFolder, paired_layout
from mon.data.io import write_image
from mon.data.loader import create_dataloader
from mon.data.transforms import PadToMultiple


def _img(h: int = 4, w: int = 6, value: int = 0, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.full((3, h, w), value, dtype=dtype)


@pytest.fixture
def folder(tmp_path: StdPath) -> StdPath:
    write_image(_img(value=1), tmp_path / "b.png")
    write_image(_img(value=2), tmp_path / "a.jpg", quality=100)
    write_image(_img(8, 10, value=3), tmp_path / "night" / "c.png")
    write_image(_img(), tmp_path / "._b.png")                   # macOS AppleDouble
    write_image(_img(), tmp_path / ".ipynb_checkpoints" / "x.png")
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def test_folder_sorted_recursive_skips_hidden_and_non_images(folder: StdPath):
    ds = ImageFolder(folder)
    assert len(ds) == 3 and ds.root == folder
    stems = [ds[i]["meta"].stem for i in range(len(ds))]
    assert stems == ["a", "b", "night/c"]
    s = ds[2]
    assert set(s) == {"image", "meta"}
    assert isinstance(s["image"], tv_tensors.Image) and s["image"].shape == (3, 8, 10)
    meta = s["meta"]
    assert (meta.index, meta.split, meta.orig_size) == (2, Split.PREDICT, Size(8, 10))
    assert meta.paths["image"] == folder / "night" / "c.png"
    assert "3 images" in repr(ds)


def test_folder_non_recursive_and_exts(folder: StdPath):
    assert [s for s, _ in ImageFolder(folder, recursive=False)._items] == ["a", "b"]
    assert [s for s, _ in ImageFolder(folder, exts=["PNG"])._items] == ["b", "night/c"]
    with pytest.raises(ValueError, match="no images"):
        ImageFolder(folder, exts=[".tif"])


def test_single_file_and_file_list(folder: StdPath):
    ds = ImageFolder(folder / "night" / "c.png")
    assert len(ds) == 1 and ds.root is None and ds[0]["meta"].stem == "c"
    ds = ImageFolder([folder / "b.png", folder / "a.jpg"])                 # given order kept
    assert [ds[i]["meta"].stem for i in range(2)] == ["b", "a"]
    assert int(ds[0]["image"][0, 0, 0]) == 1


def test_errors(folder: StdPath, tmp_path: StdPath):
    with pytest.raises(FileNotFoundError):
        ImageFolder(folder / "missing.png")
    with pytest.raises(FileNotFoundError):
        ImageFolder([folder / "b.png", folder / "missing.png"])
    with pytest.raises(ValueError, match="not a readable image"):
        ImageFolder(folder / "notes.txt")
    write_image(_img(), folder / "night" / "b.png")
    with pytest.raises(ValueError, match="share the stem 'b'"):
        ImageFolder([folder / "b.png", folder / "night" / "b.png"])
    write_image(_img(), folder / "b.jpg")
    with pytest.raises(ValueError, match="share the stem 'b'"):
        ImageFolder(folder)                                                  # b.png and b.jpg
    with pytest.raises(ValueError, match="empty"):
        ImageFolder([])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        ImageFolder(tmp_path / "empty")
    with pytest.raises(ValueError, match="mode"):
        ImageFolder(folder, mode="bgr")  # type: ignore[arg-type]


def test_modes_and_16_bit(tmp_path: StdPath):
    write_image(_img(value=40000, dtype=torch.uint16), tmp_path / "d.png")
    assert ImageFolder(tmp_path)[0]["image"].dtype == torch.uint16
    assert ImageFolder(tmp_path, mode="gray")[0]["image"].shape == (1, 4, 6)


def test_transform_and_loader(folder: StdPath):
    tf = v2.Compose([PadToMultiple(8), v2.ToDtype(torch.float32, scale=True)])
    ds = ImageFolder(folder, transform=tf)
    s = ds[2]
    assert s["image"].shape == (3, 8, 16) and s["meta"].orig_size == Size(8, 10)
    batch = next(iter(create_dataloader(ImageFolder(folder, recursive=False), batch_size=2)))
    assert batch["image"].shape == (2, 3, 4, 6) and [m.stem for m in batch["meta"]] == ["a", "b"]


def test_folder_dataset_skips_hidden_files(tmp_path: StdPath):
    class Paired(FolderDataset):
        layout = paired_layout("test")

    for sub in ("images", "targets"):
        write_image(_img(), tmp_path / "test" / sub / "1.png")
    write_image(_img(), tmp_path / "test" / "images" / "._1.png")  # would be an unpaired, undecodable input
    assert len(Paired("test", root=tmp_path)) == 1
