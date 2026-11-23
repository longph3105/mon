"""Tests for ``mon.data.structures``."""

import json
from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mon.core.enums import Split
from mon.core.path import Path
from mon.core.size import Size
from mon.data.labels import read_coco
from mon.data.structures import ClassDef, ClassList, SampleMeta

# Real COCO raw ids around the gap at 12 (no category 12 exists in COCO).
COCO_CATS = {1: "person", 2: "bicycle", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 90: "toothbrush"}


# --- ClassDef ---

def test_classdef_defaults_and_validation():
    c = ClassDef("car", 3)
    assert c.color is not None and len(c.color) == 3 and all(0 <= v <= 255 for v in c.color)
    assert ClassDef("car", 3).color == c.color  # deterministic
    assert ClassDef("car", 4).color != c.color
    assert ClassDef("x", 0, color=[1, 2, 3]).color == (1, 2, 3)  # list (YAML) → tuple
    for bad in [
        {"name": "", "id": 0}, {"name": "a", "id": -1}, {"name": "a", "id": True},
        {"name": "a", "id": 0, "source_id": -2}, {"name": "a", "id": 0, "color": (1, 2)},
        {"name": "a", "id": 0, "color": (0, 0, 256)},
    ]:
        with pytest.raises(ValueError):
            ClassDef(**bad)


# --- ClassList ---

def test_classlist_orders_by_id_and_lookups():
    cl = ClassList([ClassDef("dog", 1), {"name": "cat", "id": 0}])
    assert cl.names == ["cat", "dog"] and cl.ids == [0, 1] and len(cl) == cl.num_classes == 2
    assert cl["dog"].id == 1 and cl.by_id(0).name == "cat"
    assert "cat" in cl and "cow" not in cl and cl["cat"] in cl and ClassDef("cat", 5) not in cl
    assert [c.name for c in cl] == ["cat", "dog"] and cl.is_contiguous
    assert len(cl.colors) == 2
    assert not ClassList.from_names(["a", "b"], start=1).is_contiguous


@pytest.mark.parametrize(
    ("classes", "match"),
    [
        ([ClassDef("a", 0), ClassDef("a", 1)], "duplicate name 'a'"),
        ([ClassDef("a", 0), ClassDef("b", 0)], "duplicate id 0"),
        ([ClassDef("a", 0, source_id=5), ClassDef("b", 1, source_id=5)], "duplicate source_id 5"),
    ],
)
def test_classlist_rejects_duplicates(classes, match: str):
    with pytest.raises(ValueError, match=match):
        ClassList(classes)


def test_classlist_lookup_errors():
    cl = ClassList.from_names(["bicycle", "car"])
    with pytest.raises(KeyError, match="did you mean 'bicycle'"):
        cl["bycicle"]
    with pytest.raises(KeyError, match="ids are"):
        cl.by_id(7)
    with pytest.raises(TypeError, match="by_id"):
        cl[0]  # type: ignore[index]


def test_classlist_equality_hash_repr():
    a, b = ClassList.from_names(["x", "y"]), ClassList.from_names(["x", "y"])
    assert a == b and hash(a) == hash(b) and a != ClassList.from_names(["y", "x"])
    assert repr(ClassList.from_names(list("abcdefg"))) == "ClassList(7 classes: a, b, c, d, e, …)"


def test_from_coco_fixes_the_gap_bug():
    cl = ClassList.from_coco(COCO_CATS)
    assert cl.is_contiguous and cl["stop sign"].id == 3 and cl["stop sign"].source_id == 13
    assert cl["toothbrush"].id == 5 and cl["toothbrush"].source_id == 90
    bg = ClassList.from_coco(COCO_CATS, background=True)
    assert bg.by_id(0).name == "background" and bg["person"].id == 1 and bg["toothbrush"].id == 6


def test_encode():
    cl = ClassList.from_coco(COCO_CATS)
    assert cl.encode([13, 1, 90, 11]).tolist() == [3, 0, 5, 2]
    assert cl.encode(torch.zeros(0, dtype=torch.int64)).shape == (0,)
    with pytest.raises(KeyError, match=r"\[12\]"):
        cl.encode([1, 12])               # 12 is COCO's gap
    with pytest.raises(KeyError, match="1000000000"):
        cl.encode([10**9])               # no giant lookup table
    with pytest.raises(KeyError, match=r"\[-1\]"):
        cl.encode([-1])
    with pytest.raises(ValueError, match="source_id"):
        ClassList.from_names(["a"]).encode([0])


def test_encode_coco_target(tmp_path: StdPath):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "height": 10, "width": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 13, "bbox": [1, 1, 2, 2]},
                        {"id": 2, "image_id": 1, "category_id": 90, "bbox": [3, 3, 2, 2]}],
        "categories": [{"id": k, "name": v} for k, v in COCO_CATS.items()],
    }))
    coco = read_coco(f)
    cl = ClassList.from_coco(coco.categories)
    assert cl.encode(coco.target("a.jpg")["labels"]).tolist() == [3, 5]


def test_yaml_round_trip(tmp_path: StdPath):
    cl = ClassList([ClassDef("cat", 0, color=(1, 2, 3), source_id=17), ClassDef("dog", 1)])
    out = cl.save(tmp_path / "classes.yaml")
    assert isinstance(out, Path)
    back = ClassList.load(out)
    assert back == cl and back["cat"].color == (1, 2, 3) and back["dog"].source_id is None
    (tmp_path / "bad.yaml").write_text("names: [a, b]\n")
    with pytest.raises(ValueError, match="'classes' list"):
        ClassList.load(tmp_path / "bad.yaml")


# --- SampleMeta ---

def test_sample_meta_passes_through_v2():
    meta = SampleMeta(
        index=3, stem="0001", split=Split.TRAIN, paths={"image": Path("a/0001.png")}, orig_size=Size(8, 10),
    )
    sample = {"image": tv_tensors.Image(torch.zeros(3, 8, 10, dtype=torch.uint8)), "meta": meta}
    out = v2.Compose([v2.RandomHorizontalFlip(p=1.0), v2.Resize((4, 5))])(sample)
    assert out["meta"] is meta and out["image"].shape == (3, 4, 5)
    assert meta.orig_size == Size(8, 10)  # still the pre-transform size
    with pytest.raises(AttributeError):
        meta.index = 1  # type: ignore[misc]
