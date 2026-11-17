"""Shared type aliases for annotations.

Named ``type_aliases`` (not ``typing``/``types``) so it never shadows the standard library.
Unions are plain assignments, so ``isinstance(x, TensorOrArray)`` works.
"""

__all__ = [
    "Float3",
    "Int2",
    "Int3",
    "OneOrPair",
    "TensorOrArray",
]

from numpy import ndarray
from torch import Tensor

Int2  = tuple[int, int]
"""Two ints, e.g. ``(H, W)`` or ``(x, y)``."""

Int3  = tuple[int, int, int]
"""Three ints, e.g. an RGB color ``(R, G, B)`` in ``[0, 255]``."""

Float3 = tuple[float, float, float]
"""Three floats, e.g. per-channel mean/std."""

TensorOrArray = Tensor | ndarray
"""A PyTorch tensor or a NumPy array."""

type OneOrPair[T] = T | tuple[T, T]
"""A single value or a pair, e.g. ``kernel_size: OneOrPair[int]`` (``3`` or ``(3, 5)``)."""
