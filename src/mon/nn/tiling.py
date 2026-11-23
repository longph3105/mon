"""Tiled inference for large images, blended with a Hann window.

For images too large for memory: tiles of ``tile × tile`` with ``overlap`` pixels between
neighbors; outputs are blended with a 2-D Hann window so tile seams don't show. Every
pixel is covered (the last tile in each direction is flush with the image edge) and the
window is positive everywhere, so borders are kept. For images that fit, call the model
directly (with ``mon.data.transforms.pad_to_multiple``) instead: tiling changes the
result of models that use global statistics (e.g. global pooling, normalization).

Replaces ``mon_old``'s ``ops/image/patch.py`` patchers, which had two coverage bugs: the
tile loop ran over the unpadded size (bottom/right strips past the last full stride never
processed → black), and the periodic Hann window was 0 on tile edges (image borders → 0).
"""

__all__ = [
    "hann_window_2d",
    "tile_starts",
    "tiled_forward",
]

from collections.abc import Callable, Mapping

import torch
from torch import Tensor

type TileOutput = Tensor | Mapping[str, Tensor]


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``[0, length)`` with ``tile``-long windows every ``stride``.

    The last window ends exactly at ``length``. A ``length`` up to ``tile`` gives ``[0]``.

    Examples:
        >>> tile_starts(600, 512, 384)
        [0, 88]
        >>> tile_starts(1000, 512, 384)
        [0, 384, 488]
        >>> tile_starts(300, 512, 384)
        [0]
    """
    if tile < 1 or stride < 1:
        raise ValueError(f"tile and stride must be >= 1, got tile={tile}, stride={stride}")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def hann_window_2d(
    height: int, width: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32,
) -> Tensor:
    """A ``[1, 1, height, width]`` Hann window, strictly positive (edges are not 0).

    Built from a symmetric Hann window of length ``n + 2`` with its two zero ends dropped.
    """
    def one_d(n: int) -> Tensor:
        return torch.hann_window(n + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    return (one_d(height)[:, None] * one_d(width)[None, :])[None, None]


@torch.no_grad()
def tiled_forward(
    fn     : Callable[[Tensor], TileOutput],
    x      : Tensor,
    *,
    tile   : int = 512,
    overlap: int = 128,
) -> TileOutput:
    """Apply ``fn`` to overlapping tiles of ``x`` and blend the results with a Hann window.

    Args:
        fn: Maps a ``[B, C, h, w]`` tile to a tensor, or a dict of tensors, with the same
            ``h, w`` (channels may differ, e.g. curve maps).
        x: ``[B, C, H, W]`` input. A side not larger than ``tile`` is processed whole.
        tile: Tile side. Must be a multiple of the model's stride if it has one.
        overlap: Pixels shared by neighboring tiles; ``0 <= overlap < tile``.

    Returns:
        Same structure as ``fn``'s output, at ``[B, *, H, W]``, in each output's float dtype.
        Runs under ``torch.no_grad()``: for inference only.

    Raises:
        ValueError: Bad ``tile``/``overlap``, or ``fn`` changes the tile's spatial size.

    Examples:
        >>> x = torch.rand(1, 3, 600, 900)
        >>> torch.allclose(tiled_forward(lambda t: t * 2, x, tile=256, overlap=64), x * 2)
        True
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B, C, H, W], got shape {tuple(x.shape)}")
    if not 0 <= overlap < tile:
        raise ValueError(f"need 0 <= overlap < tile, got overlap={overlap}, tile={tile}")
    _, _, h, w = x.shape
    th, tw  = min(tile, h), min(tile, w)
    stride  = tile - overlap
    ys, xs  = tile_starts(h, tile, stride), tile_starts(w, tile, stride)
    window  = None
    sums: dict[str, Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    weight: Tensor | None   = None
    single  = False

    for y0 in ys:
        for x0 in xs:
            out = fn(x[..., y0:y0 + th, x0:x0 + tw])
            if isinstance(out, Tensor):
                single, outs = True, {"_": out}
            else:
                outs = dict(out)
            for key, t in outs.items():
                if t.shape[-2:] != (th, tw):
                    raise ValueError(f"fn changed the tile size: {(th, tw)} -> {tuple(t.shape[-2:])} (output {key!r})")
                if window is None:
                    window = hann_window_2d(th, tw, device=t.device, dtype=torch.float32)
                    weight = torch.zeros(1, 1, h, w, device=t.device, dtype=torch.float32)
                if key not in sums:
                    sums[key]   = torch.zeros(*t.shape[:2], h, w, device=t.device, dtype=torch.float32)
                    dtypes[key] = t.dtype if t.is_floating_point() else torch.float32
                sums[key][..., y0:y0 + th, x0:x0 + tw] += t.float() * window
            assert weight is not None and window is not None
            weight[..., y0:y0 + th, x0:x0 + tw] += window

    assert weight is not None
    blended = {k: (v / weight).to(dtypes[k]) for k, v in sums.items()}  # blended in float32
    return blended["_"] if single else blended
