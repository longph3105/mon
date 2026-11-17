"""Unambiguous (height, width) image size."""

__all__ = ["Size"]

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self


@dataclass(frozen=True, slots=True)
class Size:
    """Image size with named ``height``/``width`` (positive ints).

    Index ``[0]``/``[1]`` or unpack (``h, w = size``) in ``(H, W)`` order. There is no
    ``<`` ordering (sizes are only partially ordered); use ``fits_within``.

    Examples:
        >>> s = Size(height=100, width=200)
        >>> s.hw, s.wh, s.area, tuple(s)
        ((100, 200), (200, 100), 20000, (100, 200))
    """

    height: int
    width : int

    def __post_init__(self) -> None:
        for name in ("height", "width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

    # --- Container ---

    def __getitem__(self, index: int) -> int:
        return (self.height, self.width)[index]

    def __iter__(self) -> Iterator[int]:
        return iter((self.height, self.width))

    def __len__(self) -> int:
        return 2

    # --- Properties ---

    @property
    def h(self) -> int:
        """Height."""
        return self.height

    @property
    def w(self) -> int:
        """Width."""
        return self.width

    @property
    def hw(self) -> tuple[int, int]:
        """``(height, width)``: PyTorch/NumPy order."""
        return self.height, self.width

    @property
    def wh(self) -> tuple[int, int]:
        """``(width, height)``: OpenCV/PIL order."""
        return self.width, self.height

    @property
    def area(self) -> int:
        """``height * width``."""
        return self.height * self.width

    # --- Operations ---

    def fits_within(self, other: "Size") -> bool:
        """Whether both dimensions are ``<=`` those of ``other``."""
        return self.height <= other.height and self.width <= other.width

    def round_up(self, divisor: int) -> Self:
        """Round both dimensions up to a multiple of ``divisor`` (e.g. a network stride).

        Examples:
            >>> Size(200, 100).round_up(32)
            Size(height=224, width=128)
        """
        if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
            raise ValueError(f"divisor must be a positive int, got {divisor!r}")
        return type(self)(math.ceil(self.height / divisor) * divisor, math.ceil(self.width / divisor) * divisor)

    # --- Creation ---

    @classmethod
    def from_any(cls, value: "Size | int | Sequence[int]") -> Self:
        """Build from a ``Size``, a single int (square), or ``(h, w)`` / ``(s,)``.

        Use ``from_shape`` for tensor/array shapes; longer sequences are rejected rather
        than guessed.

        Raises:
            TypeError: If ``value`` has an unsupported type.
            ValueError: If a sequence doesn't have 1 or 2 items, or values aren't positive ints.
        """
        if isinstance(value, Size):
            return value if type(value) is cls else cls(value.height, value.width)
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value, value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 1:
                return cls(value[0], value[0])
            if len(value) == 2:
                return cls(value[0], value[1])
            raise ValueError(f"expected (h, w) or (s,), got {len(value)} values: {tuple(value)}; use Size.from_shape for shapes")
        raise TypeError(f"cannot build Size from {type(value).__name__}")

    @classmethod
    def from_shape(cls, value: Any, layout: Literal["chw", "hwc"] = "chw") -> Self:
        """Read ``(H, W)`` from a shape or an object with ``.shape`` (tensor, array, PIL-free).

        Args:
            value: A shape (``torch.Size``, tuple) or an object with a ``.shape``.
            layout: ``"chw"``: H, W are the last two dims (``[..., C, H, W]`` or ``[H, W]``).
                ``"hwc"``: H, W are dims -3 and -2 (``[..., H, W, C]``).

        Examples:
            >>> Size.from_shape((8, 3, 224, 320))
            Size(height=224, width=320)
            >>> Size.from_shape((480, 640, 3), layout="hwc")
            Size(height=480, width=640)
        """
        shape = tuple(int(d) for d in getattr(value, "shape", value))
        if layout == "chw":
            if len(shape) < 2:
                raise ValueError(f"'chw' layout needs >= 2 dims, got shape {shape}")
            return cls(shape[-2], shape[-1])
        if layout == "hwc":
            if len(shape) < 3:
                raise ValueError(f"'hwc' layout needs >= 3 dims, got shape {shape}")
            return cls(shape[-3], shape[-2])
        raise ValueError(f"layout must be 'chw' or 'hwc', got {layout!r}")
