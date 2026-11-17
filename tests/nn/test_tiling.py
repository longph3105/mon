"""Tests for ``mon.nn.tiling``."""

import pytest
import torch

from mon.nn.tiling import hann_window_2d, tile_starts, tiled_forward


@pytest.mark.parametrize(("length", "tile", "stride"), [(600, 512, 384), (1000, 512, 384), (512, 512, 384),
                                                        (513, 512, 1), (300, 512, 384), (2049, 256, 192)])
def test_tile_starts_cover_everything(length, tile, stride):
    starts  = tile_starts(length, tile, stride)
    covered = torch.zeros(length, dtype=torch.bool)
    for s in starts:
        covered[s:s + min(tile, length)] = True
    assert covered.all() and starts[0] == 0 and starts == sorted(set(starts))
    assert starts[-1] + min(tile, length) == length


def test_hann_window_is_positive_and_symmetric():
    w = hann_window_2d(5, 8)
    assert w.shape == (1, 1, 5, 8) and (w > 0).all()
    assert torch.allclose(w, w.flip(-1)) and torch.allclose(w, w.flip(-2))


@pytest.mark.parametrize("size", [(600, 900), (512, 512), (100, 700), (37, 53), (1024, 1030)])
def test_pointwise_fn_is_reproduced_exactly(size):
    x = torch.rand(2, 3, *size, generator=torch.Generator().manual_seed(0))
    y = tiled_forward(lambda t: t * 2 + 1, x, tile=256, overlap=64)
    assert isinstance(y, torch.Tensor) and y.shape == x.shape
    assert torch.allclose(y, x * 2 + 1, atol=1e-5)
    # Regressions from mon_old's patcher: borders and the bottom/right strip are kept.
    assert torch.allclose(y[..., 0, :], x[..., 0, :] * 2 + 1, atol=1e-5)
    assert torch.allclose(y[..., -1, -1], x[..., -1, -1] * 2 + 1, atol=1e-5)


def test_dict_outputs_with_other_channel_counts():
    x = torch.rand(1, 3, 300, 400)
    out = tiled_forward(lambda t: {"enhanced": t.clamp(0, 1), "curves": t.repeat(1, 8, 1, 1)}, x, tile=128, overlap=32)
    assert set(out) == {"enhanced", "curves"}
    assert out["curves"].shape == (1, 24, 300, 400) and torch.allclose(out["enhanced"], x, atol=1e-6)


def test_tiles_are_seen_at_most_tile_size_and_count():
    x, seen = torch.rand(1, 3, 600, 900), []
    tiled_forward(lambda t: (seen.append(tuple(t.shape[-2:])), t)[1], x, tile=512, overlap=128)
    assert set(seen) == {(512, 512)} and len(seen) == 2 * 3  # starts [0, 88] × [0, 384, 388]


def test_dtype_and_no_grad():
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    y = tiled_forward(lambda t: (t * 3).half(), x, tile=32, overlap=8)
    assert y.dtype == torch.float16 and not y.requires_grad


def test_errors():
    x = torch.rand(1, 3, 64, 64)
    with pytest.raises(ValueError, match="overlap"):
        tiled_forward(lambda t: t, x, tile=32, overlap=32)
    with pytest.raises(ValueError, match="changed the tile size"):
        tiled_forward(lambda t: t[..., ::2, ::2], x, tile=32, overlap=8)
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        tiled_forward(lambda t: t, torch.rand(3, 8, 8))
    with pytest.raises(ValueError, match="tile and stride"):
        tile_starts(10, 0, 1)
