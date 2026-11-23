"""Tests for ``mon.data.dataset`` (synthetic folders in ``tmp_path``)."""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mon.core.constants import DATA_DIR_ENV, DATASETS_DIR_ENV
from mon.core.enums import Split, Task
from mon.core.size import Size
from mon.data.dataset import DATASETS, FolderDataset, Modality, detection_layout, register_dataset
from mon.data.io import write_image
from mon.data.structures import ClassList, SampleMeta


def img(h: int = 8, w: int = 12, value: int = 0, c: int = 3, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.full((c, h, w), value, dtype=dtype)


class Paired(FolderDataset):
    layout = {
        Split.TRAIN: {"image": Modality("train/low"), "target": Modality("train/high")},
        Split.TEST : {"image": Modality("test/low"), "target": Modality("test/high")},
    }


@pytest.fixture
def paired_root(tmp_path: StdPath) -> StdPath:
    root = tmp_path / "LOL"
    write_image(img(value=10), root / "train/low/a.png")
    write_image(img(value=20), root / "train/high/a.jpg", quality=100)     # different extension: still paired
    write_image(img(value=30), root / "train/low/sub/b.png")
    write_image(img(value=40), root / "train/high/sub/b.png")
    write_image(img(value=50), root / "test/low/c.png")
    write_image(img(value=60), root / "test/high/c.png")
    return root


# --- Pairing and samples ---

def test_pairs_by_relative_stem_across_extensions(paired_root: StdPath):
    ds = Paired("train", root=paired_root)
    assert len(ds) == 2 and ds.split is Split.TRAIN
    s = ds[1]
    assert set(s) == {"image", "target", "meta"}
    assert isinstance(s["image"], tv_tensors.Image) and s["image"].shape == (3, 8, 12)
    assert int(s["image"][0, 0, 0]) == 30 and int(s["target"][0, 0, 0]) == 40
    meta = s["meta"]
    assert isinstance(meta, SampleMeta)
    assert (meta.index, meta.stem, meta.split, meta.orig_size) == (1, "sub/b", Split.TRAIN, Size(8, 12))
    assert meta.paths["target"].name == "b.png" and meta.paths["image"].parent.name == "sub"
    assert ds[0]["meta"].paths["target"].suffix == ".jpg"
    assert "2 samples" in repr(ds) and Paired.splits() == {Split.TRAIN, Split.TEST}


def test_modality_name_inside_root_path_doesnt_break_pairing(tmp_path: StdPath):
    """mon_old paired by replacing '/low/' with '/high/' in the full path; a root containing 'low' broke it."""
    root = tmp_path / "low" / "low_light" / "LOL"
    write_image(img(value=1), root / "train/low/a.png")
    write_image(img(value=2), root / "train/high/a.png")
    s = Paired("train", root=root)[0]
    assert int(s["image"][0, 0, 0]) == 1 and int(s["target"][0, 0, 0]) == 2


def test_order_is_sorted_and_deterministic(paired_root: StdPath):
    write_image(img(), paired_root / "train/low/0.png")
    write_image(img(), paired_root / "train/high/0.png")
    assert [Paired("train", root=paired_root)[i]["meta"].stem for i in range(3)] == ["0", "a", "sub/b"]


def test_transform_gets_whole_sample(paired_root: StdPath):
    ds = Paired("train", root=paired_root, transform=v2.Compose([v2.RandomHorizontalFlip(p=1.0), v2.Resize((4, 6))]))
    s = ds[0]
    assert s["image"].shape == s["target"].shape == (3, 4, 6)
    assert s["meta"].orig_size == Size(8, 12)


def test_postprocess_runs_before_transform(paired_root: StdPath):
    class Shifted(Paired):
        def postprocess(self, sample):
            assert sample["image"].shape == (3, 8, 12)  # as read from disk: transform hasn't run
            sample["target"] = tv_tensors.wrap(sample["target"] - 5, like=sample["target"])
            return sample

    resize = v2.Resize((4, 6))
    ds = Shifted("train", root=paired_root, transform=resize)
    s = ds[0]
    assert s["image"].shape == (3, 4, 6) and int(s["target"][0, 0, 0]) in (14, 15, 16)  # 20 - 5 (JPEG)
    assert ds.transform is resize


def test_default_root_from_datasets_dir(paired_root: StdPath, monkeypatch: pytest.MonkeyPatch):
    class WithFolder(Paired):
        folder = "enhance/LOL"

    target = paired_root.parent / "datasets" / "enhance" / "LOL"
    target.parent.mkdir(parents=True)
    paired_root.rename(target)
    monkeypatch.setenv(DATASETS_DIR_ENV, str(paired_root.parent / "datasets"))
    assert len(WithFolder("test")) == 1
    monkeypatch.delenv(DATASETS_DIR_ENV)
    monkeypatch.setenv(DATA_DIR_ENV, str(paired_root.parent))
    assert WithFolder("test").root == target


# --- Missing and duplicate files ---

def test_duplicate_stem_in_one_modality_raises(paired_root: StdPath):
    write_image(img(), paired_root / "train/low/a.jpg")
    with pytest.raises(ValueError, match="two files share the stem 'a'"):
        Paired("train", root=paired_root)


def test_missing_required_counterpart(paired_root: StdPath, caplog: pytest.LogCaptureFixture):
    (paired_root / "train/high/sub/b.png").unlink()
    with pytest.raises(ValueError, match=r"missing for 1 sample.*target: 1 missing \(e.g. \['sub/b'\]\)"):
        Paired("train", root=paired_root)
    with caplog.at_level(logging.WARNING, logger="mon.data.dataset"):
        ds = Paired("train", root=paired_root, on_missing="skip")
    assert len(ds) == 1 and "skipping 1 incomplete" in caplog.text


def test_extra_files_in_secondary_are_ignored(paired_root: StdPath):
    write_image(img(), paired_root / "train/high/orphan.png")
    assert len(Paired("train", root=paired_root)) == 2


class WithOptional(FolderDataset):
    layout = {Split.TRAIN: {
        "image": Modality("img"),
        "depth": Modality("depth", kind="depth", required=False),
        "label": Modality("labels", kind="yolo", required=False),
    }}


def test_optional_modalities(tmp_path: StdPath):
    write_image(img(h=10, w=20), tmp_path / "img/a.png")
    write_image(img(h=10, w=20), tmp_path / "img/b.png")
    write_image(img(c=1, h=10, w=20, value=5000, dtype=torch.uint16), tmp_path / "depth/a.png")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels/a.txt").write_text("2 0.5 0.5 0.2 0.4\n")
    ds = WithOptional("train", root=tmp_path)
    a, b = ds[0], ds[1]
    assert a["depth"].dtype == torch.uint16 and a["depth"].shape == (1, 10, 20)
    assert int(a["depth"].float().max()) == 5000  # torch lacks max() for uint16
    assert a["boxes"].tolist() == [[8.0, 3.0, 12.0, 7.0]] and a["labels"].tolist() == [2]
    assert "depth" not in b and b["boxes"].shape == (0, 4) and b["boxes"].canvas_size == (10, 20)


def test_optional_dir_absent(tmp_path: StdPath):
    write_image(img(), tmp_path / "img/a.png")
    s = WithOptional("train", root=tmp_path)[0]
    assert "depth" not in s and s["labels"].shape == (0,)


# --- Kinds ---

def test_mask_kind(tmp_path: StdPath):
    class Seg(FolderDataset):
        layout = {Split.TRAIN: {"image": Modality("images"), "mask": Modality("masks", kind="mask")}}

    write_image(img(), tmp_path / "images/a.jpg")
    m = torch.zeros(1, 8, 12, dtype=torch.uint8)
    m[0, :, 6:] = 7
    write_image(m, tmp_path / "masks/a.png")
    s = Seg("train", root=tmp_path)[0]
    assert isinstance(s["mask"], tv_tensors.Mask) and s["mask"].shape == (8, 12)
    assert sorted(s["mask"].unique().tolist()) == [0, 7]
    out = v2.Resize((5, 7))(s)
    assert set(out["mask"].unique().tolist()) <= {0, 7}  # nearest: no invented class ids


def test_voc_kind(tmp_path: StdPath):
    class Det(FolderDataset):
        classes = ClassList.from_names(["cat", "dog"])
        layout = {Split.TRAIN: {"image": Modality("JPEGImages"), "ann": Modality("Annotations", kind="voc")}}

    write_image(img(h=40, w=60), tmp_path / "JPEGImages/a.jpg")
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "Annotations/a.xml").write_text(
        "<annotation><size><width>60</width><height>40</height></size><object><name>dog</name>"
        "<bndbox><xmin>11</xmin><ymin>1</ymin><xmax>30</xmax><ymax>20</ymax></bndbox></object></annotation>")
    s = Det("train", root=tmp_path)[0]
    assert s["labels"].tolist() == [1] and s["boxes"].tolist() == [[10.0, 0.0, 30.0, 20.0]]
    assert s["difficult"].tolist() == [False] and s["boxes"].canvas_size == (40, 60)


def test_custom_exts(tmp_path: StdPath):
    class OnlyPng(FolderDataset):
        layout = {Split.TRAIN: {"image": Modality("i", exts=frozenset({"PNG"}))}}

    write_image(img(), tmp_path / "i/a.png")
    write_image(img(), tmp_path / "i/b.jpg")
    assert [OnlyPng("train", root=tmp_path)[0]["meta"].stem] == ["a"] and len(OnlyPng("train", root=tmp_path)) == 1


# --- Validation ---

def _make(layout, **cls_attrs) -> type[FolderDataset]:
    return type("Tmp", (FolderDataset,), {"layout": {Split.TRAIN: layout}, **cls_attrs})


@pytest.mark.parametrize(
    ("layout", "attrs", "match"),
    [
        ({}, {}, "is empty"),
        ({"label": Modality("l", kind="yolo")}, {}, "primary"),
        ({"image": Modality("i", required=False)}, {}, "primary"),
        (
            {"image": Modality("i"), "a": Modality("a", kind="yolo"), "b": Modality("b", kind="voc")},
            {"classes": ClassList()},
            "at most one box",
        ),
        ({"image": Modality("i"), "ann": Modality("a", kind="voc")}, {}, "need `classes`"),
        ({"image": Modality("i"), "labels": Modality("l")}, {}, "reserved"),
    ],
)
def test_layout_validation(tmp_path: StdPath, layout, attrs, match: str):
    with pytest.raises(ValueError, match=match):
        _make(layout, **attrs)("train", root=tmp_path)


def test_argument_and_filesystem_errors(paired_root: StdPath, tmp_path: StdPath):
    with pytest.raises(ValueError, match="has no 'val' split; available"):
        Paired("val", root=paired_root)
    with pytest.raises(ValueError):
        Paired("bogus", root=paired_root)
    with pytest.raises(ValueError, match="on_missing"):
        Paired("train", root=paired_root, on_missing="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="defines no `folder`"):
        Paired("train")
    with pytest.raises(FileNotFoundError, match="root not found"):
        Paired("train", root=tmp_path / "nope")
    (tmp_path / "empty" / "train" / "low").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="'target' directory not found"):
        Paired("train", root=tmp_path / "empty")
    (tmp_path / "empty" / "train" / "high").mkdir()
    with pytest.raises(ValueError, match="no 'image' files found"):
        Paired("train", root=tmp_path / "empty")
    with pytest.raises(ValueError, match="unknown modality kind"):
        Modality("x", kind="audio")  # type: ignore[arg-type]


# --- Registry ---

def test_register_dataset_derives_splits():
    @register_dataset("tmp_paired_test", tasks={Task.LLE})
    class TmpPaired(Paired):
        ...

    try:
        entry = DATASETS.entry("tmp_paired_test")
        assert entry.obj is TmpPaired and entry.meta["splits"] == {Split.TRAIN, Split.TEST}
        assert "tmp_paired_test" in DATASETS.names(tasks=Task.LLE, splits=Split.TEST)
        assert "tmp_paired_test" not in DATASETS.names(splits=Split.VAL)
    finally:
        DATASETS.remove("tmp_paired_test")
    with pytest.raises(TypeError):
        register_dataset("x", tasks=[])(object)  # type: ignore[arg-type]


def test_zoo_loads_only_on_first_lookup():
    code = textwrap.dedent("""
        import sys, time
        import mon.data.dataset as d
        before = "mon.data.zoo" in sys.modules
        t = time.perf_counter(); d.DATASETS.names(); ms = (time.perf_counter() - t) * 1000
        print(before, "mon.data.zoo" in sys.modules, ms < 500)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "True", "True"]


def test_detection_layout():
    layout = detection_layout("train", Split.TEST)
    assert list(layout) == [Split.TRAIN, Split.TEST]
    assert [(m.dir, m.kind) for m in layout[Split.TEST].values()] == [("test/images", "image"), ("test/labels", "yolo")]
    with pytest.raises(ValueError, match="at least one"):
        detection_layout()
    with pytest.raises(ValueError, match="duplicate"):
        detection_layout("train", Split.TRAIN)
