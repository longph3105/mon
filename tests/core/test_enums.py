"""Tests for ``mon.core.enums``."""

import pytest

import mon.core
import mon.core.enums as enums
from mon.core.enums import AliasStrEnum, Split, Task

with pytest.warns(DeprecationWarning):
    RunMode, Strategy, BBoxFormat = enums.RunMode, enums.Strategy, enums.BBoxFormat


class Color(AliasStrEnum):
    RED  = "red", "r", "rouge"
    BLUE = "blue"


# --- AliasStrEnum ---

def test_primary_value_and_str_behavior():
    assert Color.RED.value == "red"
    assert Color.RED == "red" and isinstance(Color.RED, str)
    assert str(Color.RED) == "red" and f"{Color.BLUE}" == "blue"


def test_iteration_lists_primary_members_only():
    assert [c.value for c in Color] == ["red", "blue"]
    assert len(Color) == 2


@pytest.mark.parametrize("value", ["red", "r", "rouge", "RED", " Rouge ", Color.RED])
def test_lookup_by_value_alias_and_case(value: str):
    assert Color(value) is Color.RED


@pytest.mark.parametrize("value", [None, "", "green", 1])
def test_unknown_value_raises_with_choices(value: object):
    expected = r"is not a valid Color; expected one of: 'red' \(aliases: 'r', 'rouge'\), 'blue'$"
    with pytest.raises(ValueError, match=expected):
        Color(value)  # type: ignore[arg-type]


def test_name_lookup_still_works():
    assert Color["RED"] is Color.RED


# --- Concrete enums ---

@pytest.mark.parametrize(
    ("enum", "value", "member"),
    [
        (Task, "low_light_enhancement", Task.LLE),
        (Task, "LLE", Task.LLE),
        (Task, "object_detection", Task.DETECT),
        (Task, "super_resolution", Task.SUPER_RES),
        (Split, "val", Split.VAL),
        (RunMode, "metric", RunMode.METRIC),
        (Strategy, "patching", Strategy.PATCH),
        (BBoxFormat, "coco", BBoxFormat.XYWH),
        (BBoxFormat, "yolo", BBoxFormat.CXCYWHN),
        (BBoxFormat, "voc", BBoxFormat.XYXY),
    ],
)
def test_aliases(enum: type[AliasStrEnum], value: str, member: AliasStrEnum):
    assert enum(value) is member


@pytest.mark.parametrize("enum", [Task, Split, RunMode, Strategy, BBoxFormat])
def test_no_implicit_default(enum: type[AliasStrEnum]):
    for value in (None, "default"):
        with pytest.raises(ValueError):
            enum(value)


def test_bbox_format_has_only_formats():
    assert [f.value for f in BBoxFormat] == ["xyxy", "xywh", "cxcywhn"]


def test_values_are_unique_within_each_enum():
    for enum in (Task, Split, RunMode, Strategy, BBoxFormat):
        keys = list(enum._value2member_map_)
        assert len(keys) == len(set(k.lower() for k in keys)), enum.__name__


@pytest.mark.parametrize("name", ["RunMode", "Strategy", "BBoxFormat"])
def test_deprecated_enums_warn_and_keep_their_names(name: str):
    assert name not in enums.__all__ and name not in dir(mon.core)
    with pytest.warns(DeprecationWarning, match=f"{name} is deprecated"):
        cls = getattr(enums, name)
    with pytest.warns(DeprecationWarning, match=name):
        assert getattr(mon.core, name) is cls
    assert cls.__name__ == name
