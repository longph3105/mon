"""General-purpose helpers for values, strings, and lists. Standard library only."""

__all__ = [
    "has_value",
    "is_int_like",
    "is_list_of",
    "to_list",
    "truncate_string",
]

import numbers
from collections.abc import Callable, Iterable
from typing import Any, Literal, overload

_NONE_STRINGS = frozenset({"", "none", "null"})


# --- Validation ---

def has_value(value: object) -> bool:
    """Check that ``value`` is set: not ``None`` and not a blank or "none"-like string.

    Useful for config fields that may arrive as ``None``, ``""``, ``"None"``, or
    ``"null"`` from YAML or the CLI. Non-string values such as ``0``, ``False``, and
    ``[]`` count as set.

    Examples:
        >>> has_value("images"), has_value(0), has_value(" None "), has_value(None)
        (True, True, False, False)
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _NONE_STRINGS
    return True


def is_int_like(value: object) -> bool:
    """Check that ``value`` is an integer or a string spelling one.

    ``bool`` and floats (even ``3.0``) are rejected; NumPy integer scalars are accepted.

    Examples:
        >>> is_int_like(3), is_int_like(" -3 "), is_int_like(3.0), is_int_like("3.0"), is_int_like(True)
        (True, True, False, False, False)
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, str):
        try:
            int(value.strip())
        except ValueError:
            return False
        return True
    return False


def is_list_of(value: object, type_: type | tuple[type, ...]) -> bool:
    """Check that ``value`` is a ``list`` whose items are all instances of ``type_``.

    An empty list passes. As with ``isinstance``, ``bool`` items count as ``int``.

    Examples:
        >>> is_list_of([1, 2], int), is_list_of([1, "a"], int), is_list_of((1,), int)
        (True, False, False)
    """
    return isinstance(value, list) and all(isinstance(v, type_) for v in value)


# --- Conversion ---

@overload
def to_list(value: Any, sep: str | Iterable[str] = ...) -> list[Any]: ...
@overload
def to_list[T](value: Any, sep: str | Iterable[str] = ..., *, cast: Callable[[Any], T]) -> list[T]: ...
def to_list(
    value: Any,
    sep: str | Iterable[str] = ",",
    *,
    cast: Callable[[Any], Any] | None = None,
) -> list[Any]:
    """Normalize ``value`` to a new list, optionally casting each item.

    - ``None`` → ``[]``.
    - ``str`` → split on the literal separator(s), items stripped, empty items dropped.
    - ``list``, ``tuple``, ``set``, ``frozenset`` → their items (set order is arbitrary).
    - Anything else (dicts, arrays, tensors, …) → ``[value]``.

    Args:
        value: Input to normalize.
        sep: Literal separator, or several of them, used only for strings.
        cast: Applied to each item, e.g. ``int`` or ``float``.

    Examples:
        >>> to_list("a, b,,c")
        ['a', 'b', 'c']
        >>> to_list("1;2,3", sep=(",", ";"), cast=int)
        [1, 2, 3]
        >>> to_list("C:/data")
        ['C:/data']
    """
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        seps = [sep] if isinstance(sep, str) else list(sep)
        if not seps or any(not s for s in seps):
            raise ValueError(f"separators must be non-empty strings, got {sep!r}")
        parts = [value]
        for s in seps:
            parts = [piece for part in parts for piece in part.split(s)]
        items = [p.strip() for p in parts if p.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]
    return [cast(v) for v in items] if cast is not None else items


# --- Formatting ---

def truncate_string(
    value: object,
    max_length: int = 80,
    side: Literal["left", "middle", "right"] = "middle",
    ellipsis: str = "...",
) -> str:
    """Shorten ``str(value)`` to at most ``max_length`` characters for display.

    Args:
        value: Object to render and shorten.
        max_length: Maximum length of the result, ellipsis included.
        side: Where the text is cut: ``"left"`` keeps the end (good for paths),
            ``"right"`` keeps the start, ``"middle"`` keeps both ends.
        ellipsis: Marker inserted where text was removed.

    Raises:
        ValueError: If ``max_length`` can't fit the ellipsis plus one character, or
            ``side`` is invalid.

    Examples:
        >>> truncate_string("/data/datasets/coco/images/train2017", 20, side="left")
        '...images/train2017'
        >>> truncate_string("abcdefghij", 7)
        'ab...ij'
    """
    text = str(value)
    if side not in ("left", "middle", "right"):
        raise ValueError(f"side must be 'left', 'middle', or 'right', got {side!r}")
    if len(text) <= max_length:
        return text
    keep = max_length - len(ellipsis)
    if keep < 1:
        raise ValueError(
            f"max_length={max_length} is too small for ellipsis {ellipsis!r} plus one character"
        )
    if side == "left":
        return ellipsis + text[len(text) - keep:]
    if side == "right":
        return text[:keep] + ellipsis
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + ellipsis + text[len(text) - tail:]
