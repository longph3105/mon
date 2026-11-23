"""Tests for ``mon.data.labels`` (synthetic label files only)."""

import json
import logging
from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mon.core.size import Size
from mon.data.labels import CocoAnnotations, read_coco, read_voc, read_yolo, write_yolo

H, W = 100, 200


def boxes_of(target) -> list[list[float]]:
    return [[round(v, 3) for v in row] for row in target["boxes"].tolist()]


# --- YOLO ---

def test_yolo_reads_to_absolute_xyxy(tmp_path: StdPath):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.4\n\n3 0.1 0.2 0.1 0.2\n")
    t = read_yolo(f, (H, W))
    assert isinstance(t["boxes"], tv_tensors.BoundingBoxes)
    assert t["boxes"].format == tv_tensors.BoundingBoxFormat.XYXY and t["boxes"].canvas_size == (H, W)
    assert t["boxes"].dtype == torch.float32 and t["labels"].dtype == torch.int64
    assert boxes_of(t) == [[80.0, 30.0, 120.0, 70.0], [10.0, 10.0, 30.0, 30.0]]
    assert t["labels"].tolist() == [0, 3] and "scores" not in t


def test_yolo_scores_column_and_float_class(tmp_path: StdPath):
    f = tmp_path / "p.txt"
    f.write_text("1.0 0.5 0.5 0.1 0.1 0.87\n2 0.2 0.2 0.1 0.1 0.5\n")
    t = read_yolo(f, Size(H, W))
    assert t["labels"].tolist() == [1, 2] and t["scores"].tolist() == pytest.approx([0.87, 0.5])


def test_yolo_missing_and_empty(tmp_path: StdPath):
    t = read_yolo(tmp_path / "none.txt", (H, W))
    assert t["boxes"].shape == (0, 4) and t["labels"].shape == (0,) and t["boxes"].canvas_size == (H, W)
    with pytest.raises(FileNotFoundError):
        read_yolo(tmp_path / "none.txt", (H, W), missing_ok=False)
    (tmp_path / "empty.txt").write_text("\n  \n")
    assert read_yolo(tmp_path / "empty.txt", (H, W))["boxes"].shape == (0, 4)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("0 0.5 0.5 0.2", "expected 5 or 6 columns"),
        ("0 0.5 0.5 0.2 0.4 0.9 7", "expected 5 or 6 columns"),
        ("car 0.5 0.5 0.2 0.4", "not a number"),
        ("1.5 0.5 0.5 0.2 0.4", "non-negative integer"),
        ("-1 0.5 0.5 0.2 0.4", "non-negative integer"),
        ("0 0.5 x 0.2 0.4", "non-numeric"),
        ("0 0.5 nan 0.2 0.4", "non-finite"),
        ("0 0.5 0.5 0 0.4", "positive"),
        ("0 0.5 0.5 0.2 -0.4", "positive"),
    ],
)
def test_yolo_malformed_raises_with_location(tmp_path: StdPath, line: str, match: str):
    f = tmp_path / "bad.txt"
    f.write_text(f"0 0.5 0.5 0.1 0.1\n{line}\n")
    with pytest.raises(ValueError, match=match) as exc:
        read_yolo(f, (H, W))
    assert f"{f}:2" in str(exc.value)  # file and line number


def test_yolo_mixed_columns_raises(tmp_path: StdPath):
    f = tmp_path / "mix.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1 0.9\n")
    with pytest.raises(ValueError, match="mixes 5- and 6-column"):
        read_yolo(f, (H, W))


def test_yolo_round_trip(tmp_path: StdPath):
    src = tmp_path / "in.txt"
    src.write_text("0 0.5 0.5 0.2 0.4\n4 0.25 0.75 0.1 0.2\n")
    t = read_yolo(src, (H, W))
    out = write_yolo(tmp_path / "sub" / "out.txt", t["boxes"], t["labels"])
    assert out.read_text() == "0 0.500000 0.500000 0.200000 0.400000\n4 0.250000 0.750000 0.100000 0.200000\n"
    back = read_yolo(out, (H, W))
    assert torch.allclose(back["boxes"], t["boxes"]) and torch.equal(back["labels"], t["labels"])


def test_write_yolo_scores_formats_and_errors(tmp_path: StdPath):
    cxcywh = tv_tensors.BoundingBoxes([[100, 50, 40, 20]], format="CXCYWH", canvas_size=(H, W))
    out = write_yolo(tmp_path / "s.txt", cxcywh, [7], [0.5], precision=3)
    assert out.read_text() == "7 0.500 0.500 0.200 0.200 0.500\n"
    empty = tv_tensors.BoundingBoxes(torch.zeros(0, 4), format="XYXY", canvas_size=(H, W))
    assert write_yolo(tmp_path / "e.txt", empty, []).read_text() == ""
    with pytest.raises(ValueError, match="2 labels"):
        write_yolo(tmp_path / "x.txt", cxcywh, [1, 2])
    with pytest.raises(ValueError, match="BoundingBoxes"):
        write_yolo(tmp_path / "x.txt", torch.zeros(1, 4), [1])  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        write_yolo(out, cxcywh, [7], overwrite=False)


def test_yolo_target_works_with_v2(tmp_path: StdPath):
    f = tmp_path / "a.txt"
    f.write_text("0 0.25 0.5 0.1 0.2\n")
    sample = {"image": tv_tensors.Image(torch.zeros(3, H, W, dtype=torch.uint8)), **read_yolo(f, (H, W))}
    out = v2.RandomHorizontalFlip(p=1.0)(sample)
    assert boxes_of(out) == [[140.0, 40.0, 160.0, 60.0]] and out["labels"].tolist() == [0]


# --- VOC ---

VOC = """<annotation>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object><name>dog</name><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>50</xmax><ymax>40</ymax></bndbox></object>
  <object><name>cat</name><difficult>1</difficult>
    <bndbox><xmin>101</xmin><ymin>21</ymin><xmax>200</xmax><ymax>100</ymax></bndbox></object>
</annotation>"""


def test_voc_reads_and_converts_to_zero_based(tmp_path: StdPath):
    f = tmp_path / "a.xml"
    f.write_text(VOC.format(w=W, h=H))
    t = read_voc(f, ["cat", "dog"])
    assert t["boxes"].canvas_size == (H, W)
    assert boxes_of(t) == [[0.0, 0.0, 50.0, 40.0], [100.0, 20.0, 200.0, 100.0]]  # full-image extent = [0, W]
    assert t["labels"].tolist() == [1, 0] and t["difficult"].tolist() == [False, True]
    assert read_voc(f, {"cat": 5, "dog": 9})["labels"].tolist() == [9, 5]


def test_voc_unknown_class_raises_instead_of_class_zero(tmp_path: StdPath):
    f = tmp_path / "a.xml"
    f.write_text(VOC.format(w=W, h=H))
    with pytest.raises(ValueError, match="unknown class 'dog'"):
        read_voc(f, ["cat"])


@pytest.mark.parametrize(
    ("xml", "match"),
    [
        ("<annotation><size><width>2</width>", "invalid XML"),
        (
            "<annotation><object><name>cat</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>"
            "</bndbox></object></annotation>",
            "no <size>",
        ),
        (VOC.format(w=W, h=H).replace("<xmin>1</xmin>", "<xmin>0</xmin>"), "invalid box"),          # VOC is 1-based
        (VOC.format(w=W, h=H).replace("<xmax>50</xmax>", "<xmax>a</xmax>"), "bad coordinate"),
        (VOC.format(w=W, h=H).replace("<ymax>40</ymax>", ""), "missing <ymax>"),
        (VOC.format(w=W, h=H).replace("<xmax>50</xmax>", "<xmax>0.5</xmax>"), "invalid box"),
    ],
)
def test_voc_malformed(tmp_path: StdPath, xml: str, match: str):
    f = tmp_path / "bad.xml"
    f.write_text(xml)
    with pytest.raises(ValueError, match=match):
        read_voc(f, ["cat", "dog"])


def test_voc_canvas_override_missing_and_empty(tmp_path: StdPath):
    f = tmp_path / "a.xml"
    f.write_text(VOC.format(w=W, h=H))
    assert read_voc(f, ["cat", "dog"], canvas_size=(10, 20))["boxes"].canvas_size == (10, 20)
    with pytest.raises(FileNotFoundError):
        read_voc(tmp_path / "none.xml", ["cat"])
    with pytest.raises(ValueError, match="needs canvas_size"):
        read_voc(tmp_path / "none.xml", ["cat"], missing_ok=True)
    t = read_voc(tmp_path / "none.xml", ["cat"], missing_ok=True, canvas_size=(H, W))
    assert t["boxes"].shape == (0, 4) and t["difficult"].shape == (0,)
    (tmp_path / "e.xml").write_text("<annotation><size><width>4</width><height>3</height></size></annotation>")
    assert read_voc(tmp_path / "e.xml", ["cat"])["boxes"].canvas_size == (3, 4)


# --- COCO ---

def coco_file(tmp_path: StdPath, **overrides) -> StdPath:
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "height": H, "width": W},
            {"id": 2, "file_name": "b.jpg", "height": 50, "width": 60},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 18, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 1},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}],
    }
    data.update(overrides)
    f = tmp_path / "instances.json"
    f.write_text(json.dumps(data))
    return f


def test_coco_reads_targets(tmp_path: StdPath):
    coco = read_coco(coco_file(tmp_path))
    assert isinstance(coco, CocoAnnotations) and len(coco) == 2 and "a.jpg" in coco and "z.jpg" not in coco
    assert coco.categories == {1: "person", 18: "dog"} and coco.images["b.jpg"] == Size(50, 60)
    t = coco.target("a.jpg")
    assert boxes_of(t) == [[10.0, 20.0, 40.0, 60.0], [0.0, 0.0, 5.0, 5.0]] and t["boxes"].canvas_size == (H, W)
    assert t["labels"].tolist() == [18, 1] and t["iscrowd"].tolist() == [False, True]
    empty = coco.target("b.jpg")  # image without annotations
    assert empty["boxes"].shape == (0, 4) and empty["boxes"].canvas_size == (50, 60)
    with pytest.raises(KeyError):
        coco.target("z.jpg")


def test_coco_skips_degenerate_boxes_with_warning(tmp_path: StdPath, caplog: pytest.LogCaptureFixture):
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 5]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 3, 3]},
    ]
    with caplog.at_level(logging.WARNING, logger="mon.data.labels"):
        coco = read_coco(coco_file(tmp_path, annotations=anns))
    assert coco.num_skipped == 1 and "skipped 1 annotation" in caplog.text
    assert len(coco.target("a.jpg")["labels"]) == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"annotations": [{"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 1, 1]}]}, "unknown image_id 9"),
        (
            {"annotations": [{"id": 1, "image_id": 1, "category_id": 99, "bbox": [0, 0, 1, 1]}]},
            "unknown category_id 99",
        ),
        ({"images": [{"id": 1, "file_name": "a.jpg", "height": 1, "width": 1}] * 2}, "duplicate image"),
    ],
)
def test_coco_malformed(tmp_path: StdPath, overrides: dict, match: str):
    with pytest.raises(ValueError, match=match):
        read_coco(coco_file(tmp_path, **overrides))


def test_coco_not_a_coco_file(tmp_path: StdPath):
    f = tmp_path / "x.json"
    f.write_text("[1, 2]")
    with pytest.raises(ValueError, match="expected a COCO dict"):
        read_coco(f)
    with pytest.raises(FileNotFoundError):
        read_coco(tmp_path / "missing.json")
