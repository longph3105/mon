"""Read and write images and videos.

Images are ``[C, H, W]`` ``tv_tensors.Image`` in RGB order, ``uint8`` or ``uint16`` (as
stored). Everything goes through OpenCV (JPEG, PNG, WebP, AVIF, TIFF, BMP, PPM, JPEG 2000,
EXR): torchvision 0.29 deprecated its image codecs and removed its video reader in favor
of TorchCodec, an extra dependency that needs system FFmpeg.

References:
    - tv_tensors: https://pytorch.org/vision/stable/tv_tensors.html
    - torchvision codec deprecation: https://meta-pytorch.org/torchcodec/0.16/generated_examples/migration/torchvision_migration.html
"""

__all__ = [
    "VideoInfo",
    "VideoWriter",
    "iter_video_frames",
    "read_image",
    "to_numpy_hwc",
    "video_info",
    "write_image",
]

import os
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Self

import cv2
import numpy as np
import torch
from torchvision import tv_tensors

from mon.core.constants import RAW_IMAGE_EXTENSIONS
from mon.core.fileio import _atomic_write_bytes, _prepare_target
from mon.core.path import Path
from mon.core.size import Size

type ImageMode = Literal["rgb", "gray", "unchanged"]

_MODES        = ("rgb", "gray", "unchanged")
_UNSUPPORTED  = {".heic": "pillow-heif", ".heif": "pillow-heif"}  # suffix -> package that would read it
_FOURCC = {".mp4": "mp4v", ".mov": "mp4v", ".m4v": "mp4v", ".avi": "MJPG", ".mkv": "MJPG"}


# --- Helpers ---

def _existing_file(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    return p


def _hwc_to_image(arr: np.ndarray) -> tv_tensors.Image:
    """``[H, W, C]`` array → ``[C, H, W]`` ``tv_tensors.Image`` sharing its memory (no copy).

    The result isn't contiguous (channels-last strides); v2 transforms and batching handle
    that, and ``.contiguous()`` gives a copy if an op needs one.
    """
    return tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))


def _to_hwc_bgr(image: torch.Tensor) -> np.ndarray:
    """``[C, H, W]``/``[H, W]`` RGB(A)/gray tensor → OpenCV-ordered ``[H, W, C]`` array."""
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError(f"expected [C, H, W] with C in (1, 3, 4) or [H, W], got shape {tuple(image.shape)}")
    if image.dtype.is_floating_point:
        image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
    if image.dtype not in (torch.uint8, torch.uint16):
        raise TypeError(f"expected uint8, uint16, or float in [0, 1], got {image.dtype}")
    arr = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr[..., 0]


# --- Images ---

def read_image(path: str | os.PathLike[str], mode: ImageMode = "rgb") -> tv_tensors.Image:
    """Read an image as a ``[C, H, W]`` RGB ``tv_tensors.Image``.

    Args:
        path: Image file.
        mode: ``"rgb"`` (3 channels; alpha dropped, gray expanded), ``"gray"`` (1 channel),
            or ``"unchanged"`` (channels as stored, incl. alpha).

    Returns:
        ``uint8`` image, or ``uint16`` for 16-bit files (PNG/TIFF) and float for EXR. It is a
        zero-copy view of the decoded array, so not contiguous; use ``.reshape``, not ``.view``.

    Notes:
        EXIF orientation is applied for ``"rgb"`` and ``"gray"``. OpenCV can't keep alpha and
        apply orientation at once, so ``"unchanged"`` returns pixels as stored.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        NotImplementedError: For camera RAW (needs ``rawpy``) and HEIC/HEIF (needs ``pillow-heif``).
        ValueError: If the file can't be decoded or ``mode`` is invalid.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)}, got {mode!r}")
    p = _existing_file(path)
    suffix = p.suffix.lower()
    if suffix in RAW_IMAGE_EXTENSIONS:
        raise NotImplementedError(f"reading camera RAW ({suffix}) needs rawpy, which isn't a mon dependency")
    if suffix in _UNSUPPORTED:
        raise NotImplementedError(f"reading {suffix} needs {_UNSUPPORTED[suffix]}, which isn't a mon dependency")

    # ANYDEPTH|ANYCOLOR keeps 16-bit data and applies EXIF orientation; UNCHANGED keeps alpha.
    flags = cv2.IMREAD_UNCHANGED if mode == "unchanged" else cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    arr = cv2.imread(str(p), flags)
    if arr is None:
        raise ValueError(f"could not decode {p} with OpenCV")
    if arr.ndim == 2:
        arr = arr[..., None]
    channels = arr.shape[2]
    if channels == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif channels == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    if mode == "rgb":
        arr = np.repeat(arr, 3, axis=2) if channels == 1 else arr[..., :3]
    elif mode == "gray" and channels >= 3:
        arr = cv2.cvtColor(np.ascontiguousarray(arr[..., :3]), cv2.COLOR_RGB2GRAY)[..., None]
    return _hwc_to_image(arr)


def to_numpy_hwc(image: torch.Tensor) -> np.ndarray:
    """Return a ``[C, H, W]`` (or ``[H, W]``) image as an ``[H, W, C]`` RGB NumPy array.

    For CPU tensors this is a view sharing memory with ``image`` (no copy; writes show up in
    both). GPU tensors are copied to the CPU. Channel order stays RGB; OpenCV functions
    that expect BGR need ``cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)``.

    Examples:
        >>> import torch
        >>> to_numpy_hwc(torch.zeros(3, 4, 5, dtype=torch.uint8)).shape
        (4, 5, 3)
    """
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"expected [C, H, W] or [H, W], got shape {tuple(image.shape)}")
    return image.detach().as_subclass(torch.Tensor).cpu().permute(1, 2, 0).numpy()


def write_image(
    image    : torch.Tensor,
    path     : str | os.PathLike[str],
    *,
    quality  : int | None = None,
    overwrite: bool = True,
) -> Path:
    """Atomically write a ``[C, H, W]`` (or ``[H, W]``) RGB(A)/gray image; return the path.

    Args:
        image: ``uint8``, ``uint16`` (PNG/TIFF only), or float in ``[0, 1]`` (→ ``uint8``).
        path: Destination; the format comes from the suffix. Parent dirs are created.
        quality: JPEG/WebP quality in ``[0, 100]`` (OpenCV default if ``None``).
        overwrite: Replace an existing file.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is ``False``.
        ValueError/TypeError: For unsupported shapes, dtypes, or formats.
    """
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    arr = _to_hwc_bgr(image)
    if arr.dtype == np.uint16 and suffix not in (".png", ".tif", ".tiff"):
        raise ValueError(f"uint16 images can only be written as PNG or TIFF, not {suffix!r}")
    params: list[int] = []
    if quality is not None:
        if suffix in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        elif suffix == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    try:
        ok, buf = cv2.imencode(suffix, arr, params)
    except cv2.error as err:
        raise ValueError(f"cannot encode an image as {suffix!r}: {err}") from err
    if not ok:
        raise ValueError(f"OpenCV failed to encode an image as {suffix!r}")
    target = _prepare_target(target, overwrite)
    _atomic_write_bytes(target, buf.tobytes())
    return target


# --- Videos ---

@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Container metadata; ``num_frames`` is the container's count and may be approximate."""

    num_frames: int
    fps       : float
    size      : Size


def _open_capture(path: str | os.PathLike[str]) -> tuple[cv2.VideoCapture, Path]:
    p = _existing_file(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"could not open video {p}")
    return cap, p


def video_info(path: str | os.PathLike[str]) -> VideoInfo:
    """Return frame count, frame rate, and frame size of a video."""
    cap, _ = _open_capture(path)
    try:
        return VideoInfo(
            num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps       =float(cap.get(cv2.CAP_PROP_FPS)),
            size      =Size(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        )
    finally:
        cap.release()


def iter_video_frames(
    path : str | os.PathLike[str],
    *,
    start: int = 0,
    stop : int | None = None,
    step : int = 1,
) -> Iterator[tv_tensors.Image]:
    """Yield frames ``start, start + step, …`` (before ``stop``) as ``uint8`` RGB images.

    Frames are decoded sequentially (skipped frames are grabbed, not seeked), so indices are
    exact even for codecs where seeking isn't.
    """
    if start < 0 or step < 1 or (stop is not None and stop < start):
        raise ValueError(f"invalid frame range start={start}, stop={stop}, step={step}")
    cap, _ = _open_capture(path)
    try:
        index = 0
        while stop is None or index < stop:
            if index < start or (index - start) % step:
                if not cap.grab():
                    return
            else:
                ok, frame = cap.read()
                if not ok:
                    return
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield _hwc_to_image(rgb)
            index += 1
    finally:
        cap.release()


class VideoWriter:
    """Write ``uint8`` RGB frames to a video file (size taken from the first frame).

    The codec comes from the suffix (``.mp4``/``.mov``/``.m4v``: ``mp4v``; ``.avi``/``.mkv``:
    ``MJPG``) unless ``fourcc`` is given. Not atomic: the file is written while streaming.

    Examples:
        >>> with VideoWriter("out.mp4", fps=30) as w:  # doctest: +SKIP
        ...     for frame in frames:
        ...         w.write(frame)
    """

    def __init__(
        self,
        path     : str | os.PathLike[str],
        fps      : float,
        *,
        fourcc   : str | None = None,
        overwrite: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        suffix = self.path.suffix.lower()
        if fourcc is None:
            if suffix not in _FOURCC:
                raise ValueError(f"no default codec for {suffix!r}; pass fourcc= (supported: {sorted(_FOURCC)})")
            fourcc = _FOURCC[suffix]
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.path   = _prepare_target(self.path, overwrite)
        self.fps    = float(fps)
        self.fourcc = fourcc
        self.frames = 0
        self._size  : Size | None = None
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: torch.Tensor) -> None:
        """Append a ``[3, H, W]`` (or ``[1, H, W]``) frame; all frames must share one size."""
        arr = _to_hwc_bgr(frame)
        if arr.dtype != np.uint8:
            raise TypeError(f"video frames must be uint8 (or float in [0, 1]), got {arr.dtype}")
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        size = Size(arr.shape[0], arr.shape[1])
        if self._writer is None:
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter.fourcc(*self.fourcc), self.fps, size.wh)
            if not writer.isOpened():
                raise ValueError(f"could not open a {self.fourcc!r} writer for {self.path}")
            self._writer, self._size = writer, size
        elif size != self._size:
            raise ValueError(f"frame size {size.hw} differs from the first frame {self._size.hw if self._size else None}")
        self._writer.write(arr)
        self.frames += 1

    def close(self) -> None:
        """Finish the file. Safe to call more than once."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()
