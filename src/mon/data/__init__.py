"""Data handling: files on disk → samples → batches.

Covers image/video/label I/O, sample structures (``tv_tensors``), datasets, transforms,
and data loaders. Image-processing algorithms live in ``mon.ops``, drawing in
``mon.plot``, metrics in ``mon.metrics``. Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ======  =================================  ====================================
    Module  Purpose                            Key names
    ======  =================================  ====================================
    io      Read and write images and videos.  read_image, write_image, VideoWriter
    ======  =================================  ====================================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("io",)

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # io
    "VideoInfo"        : "io",
    "VideoWriter"      : "io",
    "iter_video_frames": "io",
    "read_image"       : "io",
    "to_numpy_hwc"     : "io",
    "video_info"       : "io",
    "write_image"      : "io",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "VideoInfo",
    "VideoWriter",
    "iter_video_frames",
    "read_image",
    "to_numpy_hwc",
    "video_info",
    "write_image",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.data.io import (
        VideoInfo, VideoWriter, iter_video_frames, read_image, to_numpy_hwc, video_info, write_image,
    )
