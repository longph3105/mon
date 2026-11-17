"""Tests for ``mon.core.utils``."""

import numpy as np
import pytest

from mon.core.utils import has_value, is_int_like, is_list_of, to_list, truncate_string


# --- has_value ---

@pytest.mark.parametrize("value", [None, "", "   ", "None", "none", " NULL ", "\tnull\n"])
def test_has_value_false(value: object):
    assert not has_value(value)


@pytest.mark.parametrize("value", ["images", "0", "nan", 0, 0.0, False, [], {}, "Nonexistent"])
def test_has_value_true(value: object):
    assert has_value(value)


# --- is_int_like ---

@pytest.mark.parametrize("value", [0, -3, 10**20, "3", " -3 ", "+7", np.int64(5), np.uint8(1)])
def test_is_int_like_true(value: object):
    assert is_int_like(value)


@pytest.mark.parametrize("value", [3.0, 3.7, "3.0", "3.7", "", "abc", True, False, None, [1], np.float32(1)])
def test_is_int_like_false(value: object):
    assert not is_int_like(value)


# --- is_list_of ---

def test_is_list_of():
    assert is_list_of([], int)
    assert is_list_of([1, 2], int)
    assert is_list_of([1, 2.0], (int, float))
    assert is_list_of([True], int)  # bool is a subclass of int
    assert not is_list_of([1, "a"], int)
    assert not is_list_of((1, 2), int)
    assert not is_list_of("ab", str)


# --- to_list ---

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("a, b,,c ", ["a", "b", "c"]),
        ("C:/data", ["C:/data"]),
        ("http://a.com/x", ["http://a.com/x"]),
        ([1, 2], [1, 2]),
        ((1, 2), [1, 2]),
        (5, [5]),
        ({"a": 1}, [{"a": 1}]),
    ],
)
def test_to_list(value: object, expected: list):
    assert to_list(value) == expected


def test_to_list_returns_new_list():
    src = [1, 2]
    out = to_list(src)
    assert out == src and out is not src


def test_to_list_multiple_separators_and_cast():
    assert to_list("1;2,3", sep=(",", ";"), cast=int) == [1, 2, 3]
    assert to_list("0.5 | 1", sep="|", cast=float) == [0.5, 1.0]
    assert to_list(["1", "2"], cast=int) == [1, 2]
    assert to_list("a::b", sep="::") == ["a", "b"]


def test_to_list_array_is_wrapped():
    arr = np.array([1, 2])
    out = to_list(arr)
    assert len(out) == 1 and out[0] is arr


@pytest.mark.parametrize("sep", ["", [], [",", ""]])
def test_to_list_bad_separator(sep: object):
    with pytest.raises(ValueError):
        to_list("a,b", sep=sep)  # type: ignore[arg-type]


# --- truncate_string ---

@pytest.mark.parametrize("side", ["left", "middle", "right"])
@pytest.mark.parametrize("max_length", [4, 5, 10, 20, 99])
def test_truncate_string_length_exact(side: str, max_length: int):
    out = truncate_string("x" * 100, max_length, side=side)  # type: ignore[arg-type]
    assert len(out) == max_length


def test_truncate_string_sides():
    s = "abcdefghij"
    assert truncate_string(s, 7, side="left") == "...ghij"
    assert truncate_string(s, 7, side="right") == "abcd..."
    assert truncate_string(s, 7, side="middle") == "ab...ij"
    assert truncate_string(s, 8, side="middle") == "abc...ij"
    assert truncate_string(s, 10) == s
    assert truncate_string(s, 5, ellipsis="…") == "ab…ij"


def test_truncate_string_non_str_input():
    assert truncate_string(12345, 4, ellipsis="~") == "12~5"


def test_truncate_string_errors():
    with pytest.raises(ValueError):
        truncate_string("abcdefghij", 3)
    with pytest.raises(ValueError):
        truncate_string("abc", 10, side="top")  # type: ignore[arg-type]
