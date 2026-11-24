"""Sample transforms missing from torchvision v2.

Key selection (``ApplyTo``), sizing to a stride multiple, and min-max scaling. Everything
else (flips, crops, ``Resize``, ``Normalize``, ``ColorJitter``, …) is plain
``torchvision.transforms.v2``. Pipelines run in two stages (see ``CONVENTIONS.md``, Data): geometric
transforms on the whole sample, photometric ones only on input keys via ``ApplyTo``::

    v2.Compose([
        v2.RandomResizedCrop(256),
        v2.RandomHorizontalFlip(),
        ApplyTo(["image"], v2.ColorJitter(0.2, 0.2)),
        v2.SanitizeBoundingBoxes(),
    ])

Sizing for networks with a stride: train with ``ResizeToMultiple`` (then ``ClampPixels``: bicubic
resizing overshoots ``[0, 1]``); at inference, pad with
``PadToMultiple`` (or ``pad_to_multiple`` on a batch) and crop predictions back with
``crop_to``. Padding is exact; resizing in and back out resamples the output twice.

References:
    - Custom v2 transforms: https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html
    - Pad-and-crop inference: https://github.com/caiyuanhao1998/Retinexformer (reflect-pads test inputs)
"""

__all__ = [
    "ApplyTo",
    "ClampPixels",
    "MinMaxNormalize",
    "PadToMultiple",
    "ResizeToMultiple",
    "crop_to",
    "pad_to_multiple",
]

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from mon.core.size import Size

type PaddingMode = Literal["constant", "edge", "reflect", "symmetric"]

# tv_tensors that carry pixels and define the sample's spatial size.
_SPATIAL = (tv_tensors.Image, tv_tensors.Video, tv_tensors.Mask)


# --- Key selection ---

class ApplyTo(torch.nn.Module):
    """Run ``transform`` on some keys of a dict sample; leave the other keys untouched.

    The selected keys go to ``transform`` in one call, so random transforms use the same
    parameters for all of them (e.g. every exposure of a multi-exposure input).

    Args:
        keys: Sample keys to transform.
        transform: Any callable taking and returning a dict (usually a v2 transform).
        missing_ok: Skip keys missing from the sample (optional modalities) instead of raising.

    Examples:
        >>> sample = {"image": tv_tensors.Image(torch.rand(3, 4, 4)), "target": tv_tensors.Image(torch.rand(3, 4, 4))}
        >>> out = ApplyTo(["image"], v2.ColorJitter(brightness=0.5))(sample)
        >>> out["target"] is sample["target"]
        True
    """

    def __init__(self, keys: Sequence[str], transform: Callable[..., Any], *, missing_ok: bool = False) -> None:
        super().__init__()
        if isinstance(keys, str):
            raise TypeError(f"keys must be a sequence of str, not a str: use [{keys!r}]")
        if not keys:
            raise ValueError("keys must not be empty")
        self.keys       = tuple(keys)
        self.transform  = transform
        self.missing_ok = missing_ok

    def forward(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(sample, Mapping):
            raise TypeError(f"ApplyTo expects a dict sample, got {type(sample).__name__}")
        missing = [k for k in self.keys if k not in sample]
        if missing and not self.missing_ok:
            raise KeyError(
                f"sample has no key(s) {missing}; available: {sorted(sample)} (pass missing_ok=True to skip)",
            )
        selected = {k: sample[k] for k in self.keys if k in sample}
        out      = dict(sample)
        if selected:
            result = self.transform(selected)
            if not isinstance(result, Mapping) or set(result) != set(selected):
                raise TypeError(f"transform must return a dict with keys {sorted(selected)}")
            out.update(result)
        return out

    def extra_repr(self) -> str:
        return f"keys={list(self.keys)}, missing_ok={self.missing_ok}"


# --- Value range ---

class ClampPixels(v2.Transform):  # type: ignore[misc]  # torchvision is untyped
    """Clamp float images and videos to ``[low, high]``; masks, boxes and integer tensors pass through.

    Bicubic (and antialiased) resizing overshoots near edges, so a float image in ``[0, 1]`` comes
    out with values slightly below 0 and above 1. Run this after a resize to restore the range.

    Args:
        low, high: Range bounds (``low < high``).

    Examples:
        >>> x = tv_tensors.Image(torch.tensor([[[-0.1, 0.5, 1.2]]]))
        >>> ClampPixels()(x).tolist()
        [[[0.0, 0.5, 1.0]]]
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        if not low < high:
            raise ValueError(f"need low < high, got low={low}, high={high}")
        self.low  = low
        self.high = high

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        pixels = isinstance(inpt, tv_tensors.Image | tv_tensors.Video) or type(inpt) is torch.Tensor
        if not pixels or not inpt.is_floating_point():
            return inpt
        out = inpt.clamp(self.low, self.high)
        return tv_tensors.wrap(out, like=inpt) if isinstance(inpt, tv_tensors.TVTensor) else out

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}"


# --- Sizing to a stride multiple ---

def _spatial_size(flat_inputs: list[Any]) -> Size:
    """The common ``(H, W)`` of the image-like inputs; plain tensors count if no tv_tensor does."""
    found = [x for x in flat_inputs if isinstance(x, _SPATIAL)]
    if not found:
        found = [x for x in flat_inputs if type(x) is torch.Tensor][:1]
    if not found:
        raise TypeError("no image, video, or mask in the sample to take the size from")
    sizes = {tuple(F.get_size(x)) for x in found}
    if len(sizes) > 1:
        raise ValueError(f"image-like inputs differ in size: {sorted(sizes)}")
    h, w = sizes.pop()
    return Size(h, w)


class ResizeToMultiple(v2.Transform):  # type: ignore[misc]  # torchvision is untyped
    """Resize so height and width are multiples of ``divisor`` (rounded up). For training.

    Masks use nearest interpolation; boxes and keypoints are rescaled (v2 dispatches per type).
    At inference, prefer ``PadToMultiple``.

    Args:
        divisor: Network stride, e.g. ``32``.
        size: Target ``(H, W)`` or int (square) before rounding; ``None`` keeps the input size.
        interpolation: Interpolation for images and videos.
        antialias: Antialias when downscaling images.

    Examples:
        >>> ResizeToMultiple(32)(tv_tensors.Image(torch.rand(3, 200, 100))).shape
        torch.Size([3, 224, 128])
    """

    def __init__(
        self,
        divisor      : int,
        size         : Size | int | Sequence[int] | None = None,
        *,
        interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        antialias    : bool = True,
    ) -> None:
        super().__init__()
        Size(1, 1).round_up(divisor)  # validates divisor
        self.divisor       = divisor
        self.size          = None if size is None else Size.from_any(size)
        self.interpolation = interpolation
        self.antialias     = antialias

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        base = self.size if self.size is not None else _spatial_size(flat_inputs)
        return {"size": base.round_up(self.divisor)}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        size: Size = params["size"]
        return F.resize(inpt, [size.height, size.width], interpolation=self.interpolation, antialias=self.antialias)


class PadToMultiple(v2.Transform):  # type: ignore[misc]  # torchvision is untyped
    """Pad bottom and right so height and width are multiples of ``divisor``. For inference.

    Box and keypoint coordinates don't change; their canvas grows (reflected content gets
    no boxes, which is why this is for inference, not training). Undo with
    ``crop_to(prediction, meta.orig_size)``. For a batch, use ``pad_to_multiple``.

    Args:
        divisor: Network stride, e.g. ``32``.
        fill: Value for ``padding_mode="constant"``.
        padding_mode: ``"reflect"`` (default; no artificial borders), ``"constant"``,
            ``"edge"``, or ``"symmetric"``. ``"reflect"`` needs the pad smaller than the image.

    Examples:
        >>> PadToMultiple(32)(tv_tensors.Image(torch.rand(3, 200, 100))).shape
        torch.Size([3, 224, 128])
    """

    def __init__(self, divisor: int, *, fill: float = 0, padding_mode: PaddingMode = "reflect") -> None:
        super().__init__()
        Size(1, 1).round_up(divisor)  # validates divisor
        self.divisor      = divisor
        self.fill         = fill
        self.padding_mode = padding_mode

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        size   = _spatial_size(flat_inputs)
        padded = size.round_up(self.divisor)
        return {"padding": [0, 0, padded.width - size.width, padded.height - size.height]}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not any(params["padding"]):
            return inpt
        if isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.KeyPoints)):
            # Bottom/right padding moves no coordinates in any mode; v2 accepts only "constant" here.
            return F.pad(inpt, params["padding"], padding_mode="constant")
        return F.pad(inpt, params["padding"], fill=self.fill, padding_mode=self.padding_mode)


def pad_to_multiple(
    x           : torch.Tensor,
    divisor     : int,
    *,
    fill        : float = 0,
    padding_mode: PaddingMode = "reflect",
) -> tuple[torch.Tensor, Size]:
    """Pad a ``[..., H, W]`` tensor (e.g. a ``[B, C, H, W]`` batch) bottom/right to a multiple of ``divisor``.

    Returns:
        The padded tensor and the original size, for ``crop_to``.

    Examples:
        >>> y, size = pad_to_multiple(torch.rand(2, 3, 200, 100), 32)
        >>> y.shape, size
        (torch.Size([2, 3, 224, 128]), Size(height=200, width=100))
        >>> torch.equal(crop_to(y, size), y[..., :200, :100])
        True
    """
    size   = Size.from_shape(x)
    padded = size.round_up(divisor)
    if padded == size:
        return x, size
    pad = [0, 0, padded.width - size.width, padded.height - size.height]
    return F.pad(x, pad, fill=fill, padding_mode=padding_mode), size


def crop_to(x: torch.Tensor, size: Size | int | Sequence[int]) -> torch.Tensor:
    """Crop the top-left ``size`` of a ``[..., H, W]`` tensor: undoes ``PadToMultiple``/``pad_to_multiple``.

    Raises:
        ValueError: If ``x`` is smaller than ``size``.
    """
    target  = Size.from_any(size)
    current = Size.from_shape(x)
    if not target.fits_within(current):
        raise ValueError(f"cannot crop {current} to the larger {target}")
    return x[..., : target.height, : target.width]


# --- Value range ---

class MinMaxNormalize(v2.Transform):  # type: ignore[misc]  # torchvision is untyped
    """Scale each image (or video frame) to ``[0, 1]`` by its own min and max; returns float.

    For data without a fixed range (depth, RAW, HDR). Not an augmentation: deterministic,
    per sample. Masks, boxes, and keypoints pass through; like every v2 photometric op it
    changes *every* ``Image``, so wrap it in ``ApplyTo`` to limit it to some keys. A constant
    image becomes all zeros.

    Args:
        per_channel: Use each channel's own min and max instead of the whole image's.
        eps: Lower bound on ``max - min`` (avoids division by zero).

    Examples:
        >>> x = MinMaxNormalize()(tv_tensors.Image(torch.tensor([[[2.0, 4.0], [6.0, 10.0]]])))
        >>> x.tolist()
        [[[0.0, 0.25], [0.5, 1.0]]]
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, *, per_channel: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.per_channel = per_channel
        self.eps         = eps

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, (tv_tensors.Mask, tv_tensors.BoundingBoxes, tv_tensors.KeyPoints)):
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"expected [..., C, H, W], got shape {tuple(inpt.shape)}")
        x    = inpt.float()  # uint16 supports .float() but no reductions
        dims = (-2, -1) if self.per_channel else (-3, -2, -1)
        lo   = x.amin(dim=dims, keepdim=True)
        hi   = x.amax(dim=dims, keepdim=True)
        out  = (x - lo) / (hi - lo).clamp_min(self.eps)
        return tv_tensors.wrap(out, like=inpt) if isinstance(inpt, tv_tensors.TVTensor) else out
