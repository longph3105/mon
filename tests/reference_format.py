"""Format a reference script's result as a Python literal for a test file (no ``mon`` imports).

The ``*_reference.py`` scripts print ``NAME: dict = {...}`` to paste into their tests. This writes it in
PEP 8's hanging-indent style, as black/ruff format it: a value that fits on the line stays on it; a longer
one opens its bracket at the line end, puts each item on its own line one level deeper (with a trailing
comma) and closes the bracket on a line of its own. Floats keep their ``repr`` (exact round trip).

The scripts run with ``python -I``, so they load it by path (see any ``*_reference.py``).
"""

from typing import Any

WIDTH = 118
INDENT = "    "


def _string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _flat(obj: Any) -> str:
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{_flat(k)}: {_flat(v)}" for k, v in obj.items()) + "}"
    if isinstance(obj, list):
        return "[" + ", ".join(_flat(v) for v in obj) + "]"
    if isinstance(obj, tuple):
        return "(" + ", ".join(_flat(v) for v in obj) + ("," if len(obj) == 1 else "") + ")"
    if isinstance(obj, str):
        return _string(obj)
    if isinstance(obj, (bool, int, float)) or obj is None:
        return repr(obj)
    raise TypeError(f"can't format {type(obj).__name__} as a literal")


def _format(obj: Any, level: int, used: int, tail: int) -> str:
    flat = _flat(obj)
    if used + len(flat) + tail <= WIDTH or not isinstance(obj, (dict, list, tuple)) or not obj:
        return flat
    inner = INDENT * (level + 1)
    if isinstance(obj, dict):
        items = [f"{_flat(k)}: " for k in obj]
        values = list(obj.values())
    else:
        items = [""] * len(obj)
        values = list(obj)
    lines = [
        f"{inner}{key}{_format(v, level + 1, len(inner) + len(key), 1)},"
        for key, v in zip(items, values, strict=True)
    ]
    brackets = {dict: "{}", list: "[]", tuple: "()"}[type(obj)]
    return brackets[0] + "\n" + "\n".join(lines) + "\n" + INDENT * level + brackets[1]


def format_literal(name: str, obj: Any) -> str:
    """``NAME: dict = {...}`` (the annotation follows ``obj``'s type), hanging-indented with trailing commas."""
    head = f"{name}: {type(obj).__name__} = "
    return head + _format(obj, 0, len(head), 0)
