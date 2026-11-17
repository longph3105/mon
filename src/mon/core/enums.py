"""Shared string enums (tasks, splits, run modes, strategies, box formats).

Members compare equal to their primary value (``Split.TRAIN == "train"``). Lookup by
value also accepts aliases and is case-insensitive (``Task("Low_Light_Enhancement")``).
Unknown values raise ``ValueError``; there is no implicit default member.
"""

__all__ = [
    "AliasStrEnum",
    "BBoxFormat",
    "RunMode",
    "Split",
    "Strategy",
    "Task",
]

from enum import StrEnum
from typing import Self


# --- Base ---

class AliasStrEnum(StrEnum):
    """``StrEnum`` whose members may list extra accepted values (aliases).

    Declare ``NAME = "primary", "alias1", ...``. Only the primary value is the member's
    ``value`` and appears when iterating; aliases are accepted by ``Cls(value)``.

    Examples:
        >>> class Color(AliasStrEnum):
        ...     RED = "red", "r"
        >>> Color("R") is Color.RED, str(Color.RED), [c.value for c in Color]
        (True, 'red', ['red'])
    """

    def __new__(cls, value: str, *aliases: str) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        for alias in aliases:
            member._add_value_alias_(alias)
        return member

    @classmethod
    def _missing_(cls, value: object) -> Self:
        if isinstance(value, str):
            key = value.strip().lower()
            for candidate, member in cls._value2member_map_.items():
                if isinstance(candidate, str) and candidate.lower() == key:
                    return member  # type: ignore[return-value]
        raise ValueError(f"{value!r} is not a valid {cls.__name__}; expected one of: {cls._describe()}")

    @classmethod
    def _describe(cls) -> str:
        """List members as ``'primary' (aliases: 'a', 'b')`` for error messages."""
        parts = []
        for member in cls:
            aliases = [k for k, m in cls._value2member_map_.items() if m is member and k != member.value]
            alias_str = f" (aliases: {', '.join(repr(a) for a in aliases)})" if aliases else ""
            parts.append(f"{member.value!r}{alias_str}")
        return ", ".join(parts)


# --- ML Workflow ---

class Split(AliasStrEnum):
    """Dataset split."""

    TRAIN   = "train"
    VAL     = "val"
    TEST    = "test"
    PREDICT = "predict"


class RunMode(AliasStrEnum):
    """What a run does."""

    TRAIN   = "train"
    PREDICT = "predict"
    METRIC  = "metric"


class Strategy(AliasStrEnum):
    """How inputs larger or smaller than the model size are handled at inference."""

    NATIVE = "native"                # Use the input as-is.
    RESIZE = "resize", "resizing"    # Resize to the model size.
    PATCH  = "patch", "patching"     # Split into patches and stitch outputs.


class Task(AliasStrEnum):
    """Machine-learning task a model or dataset targets."""

    # --- General ---
    BENCHMARK = "benchmark"
    BACKBONE  = "backbone"

    # --- Low-Level: Restoration ---
    DEBLUR    = "deblur", "deblurring"
    DEHAZE    = "dehaze", "dehazing"
    DEMOIRE   = "demoire", "demoireing"
    DEMOSAIC  = "demosaic", "demosaicing"
    DENOISE   = "denoise", "denoising"
    DERAIN    = "derain", "deraining"
    DESNOW    = "desnow", "desnowing"
    INPAINT   = "inpaint", "inpainting"
    SUPER_RES = "super_res", "super_resolution"

    # --- Low-Level: Enhancement ---
    LLE            = "lle", "low_light_enhancement"
    MEF            = "mef", "multi_exposure_fusion"
    STYLE_TRANSFER = "style_transfer"

    # --- Mid-Level ---
    SEGMENT      = "segment", "segmentation"
    MONODEPTH    = "monodepth", "monocular_depth_estimation"
    BGSUBTRACT   = "bgsubtract", "background_subtraction"
    OPTICAL_FLOW = "optical_flow"
    TRACK        = "track", "object_tracking"
    POSE         = "pose", "pose_estimation"

    # --- High-Level ---
    CLASSIFY = "classify", "classification"
    DETECT   = "detect", "object_detection"


# --- Data ---

class BBoxFormat(AliasStrEnum):
    """Bounding-box coordinate layout."""

    XYXY    = "xyxy", "voc"            # [x1, y1, x2, y2], pixels.
    XYWH    = "xywh", "coco"           # [x, y, w, h], top-left corner, pixels.
    CXCYWHN = "cxcywhn", "yolo"        # [cx, cy, w, h], normalized to [0, 1].
