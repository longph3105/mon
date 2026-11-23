"""Tests for ``mon.data.io`` (synthetic images and videos only)."""

from pathlib import Path as StdPath

import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from mon.core.path import Path
from mon.core.size import Size
from mon.data.io import VideoInfo, VideoWriter, iter_video_frames, read_image, to_numpy_hwc, video_info, write_image


def gradient(c: int = 3, h: int = 24, w: int = 32, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Deterministic test image ``[C, H, W]`` with distinct values per channel."""
    hi = 255 if dtype == torch.uint8 else 65535
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    chans = [(yy * hi), (xx * hi), ((yy + xx) / 2 * hi), torch.full((h, w), hi * 0.5)][:c]
    return torch.stack(chans).round().to(dtype)


# --- read_image / write_image ---

@pytest.mark.parametrize("suffix", [".png", ".bmp", ".tif", ".ppm", ".webp"])
def test_lossless_round_trip_rgb(tmp_path: StdPath, suffix: str):
    img = gradient()
    out = write_image(img, tmp_path / f"a{suffix}", quality=101 if suffix == ".webp" else None)  # >100 = lossless WebP
    assert isinstance(out, Path) and out.is_file()
    back = read_image(out)
    assert isinstance(back, tv_tensors.Image) and back.dtype == torch.uint8
    assert torch.equal(back.as_subclass(torch.Tensor), img)


def test_channel_order_is_rgb(tmp_path: StdPath):
    red = torch.zeros(3, 4, 4, dtype=torch.uint8)
    red[0] = 255
    for suffix in (".png", ".tif", ".jpg"):
        back = read_image(write_image(red, tmp_path / f"red{suffix}"))
        assert back[0].min() > 240 and back[1:].max() < 15, suffix  # tolerance: JPEG is lossy


def test_jpeg_lossy_and_quality(tmp_path: StdPath):
    img = gradient(h=64, w=64)
    hi = read_image(write_image(img, tmp_path / "hi.jpg", quality=98)).float()
    lo = read_image(write_image(img, tmp_path / "lo.jpg", quality=5)).float()
    assert (hi - img.float()).abs().mean() < (lo - img.float()).abs().mean()
    assert (hi - img.float()).abs().mean() < 3


@pytest.mark.parametrize("suffix", [".png", ".tif"])
def test_uint16_round_trip(tmp_path: StdPath, suffix: str):
    img = gradient(dtype=torch.uint16)
    back = read_image(write_image(img, tmp_path / f"deep{suffix}"), mode="unchanged")
    assert back.dtype == torch.uint16
    assert torch.equal(back.as_subclass(torch.Tensor).to(torch.int32), img.to(torch.int32))


def test_uint16_jpeg_rejected(tmp_path: StdPath):
    with pytest.raises(ValueError, match="PNG or TIFF"):
        write_image(gradient(dtype=torch.uint16), tmp_path / "x.jpg")


@pytest.mark.parametrize("suffix", [".png", ".tif"])
def test_modes(tmp_path: StdPath, suffix: str):
    rgba = gradient(c=4)
    path = write_image(rgba, tmp_path / f"rgba{suffix}")
    assert read_image(path, "unchanged").shape == (4, 24, 32)
    rgb = read_image(path, "rgb")
    assert rgb.shape == (3, 24, 32) and torch.equal(rgb.as_subclass(torch.Tensor), rgba[:3])
    gray = read_image(path, "gray")
    assert gray.shape == (1, 24, 32)

    gpath = write_image(gradient(c=1), tmp_path / f"g{suffix}")
    assert read_image(gpath, "unchanged").shape == (1, 24, 32)
    g3 = read_image(gpath, "rgb")
    assert g3.shape == (3, 24, 32) and torch.equal(g3[0], g3[2])


def test_write_accepts_hw_and_float(tmp_path: StdPath):
    hw = gradient(c=1)[0]
    assert read_image(write_image(hw, tmp_path / "hw.png"), "unchanged").shape == (1, 24, 32)
    f = torch.tensor([[[0.0, 0.5, 1.0, 2.0]]]).repeat(3, 1, 1)  # >1 is clamped
    back = read_image(write_image(f, tmp_path / "f.png"))
    assert back[0, 0].tolist() == [0, 128, 255, 255]


@pytest.mark.parametrize(
    ("image", "exc"),
    [
        (torch.zeros(2, 4, 4, dtype=torch.uint8), ValueError),
        (torch.zeros(1, 3, 4, 4), ValueError),
        (torch.zeros(3, 4, 4, dtype=torch.int32), TypeError),
    ],
)
def test_write_rejects_bad_tensors(tmp_path: StdPath, image: torch.Tensor, exc: type[Exception]):
    with pytest.raises(exc):
        write_image(image, tmp_path / "bad.png")


def test_write_overwrite_and_atomic(tmp_path: StdPath):
    p = write_image(gradient(), tmp_path / "sub" / "x.png")
    with pytest.raises(FileExistsError):
        write_image(gradient(), p, overwrite=False)
    with pytest.raises(ValueError):
        write_image(gradient(), tmp_path / "x.unknownext")
    assert not any(f.name.endswith(".tmp") for f in (tmp_path / "sub").iterdir())


def test_read_errors(tmp_path: StdPath):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    (tmp_path / "corrupt.png").write_bytes(b"not a png")
    (tmp_path / "corrupt.tif").write_bytes(b"not a tiff")
    for name in ("corrupt.png", "corrupt.tif"):
        with pytest.raises(ValueError, match="could not decode"):
            read_image(tmp_path / name)
    (tmp_path / "photo.dng").write_bytes(b"\0")
    with pytest.raises(NotImplementedError, match="rawpy"):
        read_image(tmp_path / "photo.dng")
    (tmp_path / "photo.heic").write_bytes(b"\0")
    with pytest.raises(NotImplementedError, match="pillow-heif"):
        read_image(tmp_path / "photo.heic")
    with pytest.raises(ValueError, match="mode"):
        read_image(tmp_path / "corrupt.png", mode="bgr")  # type: ignore[arg-type]


def test_exif_orientation_applied_for_jpeg(tmp_path: StdPath):
    from PIL import Image as PILImage  # pillow comes with torchvision; test-only use
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    arr[:, :10] = 255  # left half white
    img = PILImage.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW when displayed
    img.save(tmp_path / "rot.jpg", exif=exif, quality=95)
    assert read_image(tmp_path / "rot.jpg").shape == (3, 20, 10)  # H and W swapped by EXIF orientation
    assert read_image(tmp_path / "rot.jpg", "gray").shape == (1, 20, 10)
    assert read_image(tmp_path / "rot.jpg", "unchanged").shape == (3, 10, 20)  # documented: stored pixels


def test_avif_round_trip(tmp_path: StdPath):
    back = read_image(write_image(gradient(), tmp_path / "x.avif", quality=95))
    assert back.shape == (3, 24, 32)


# --- Video ---

@pytest.fixture
def video(tmp_path: StdPath) -> StdPath:
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, fps=10) as w:
        for i in range(12):
            frame = torch.full((3, 32, 48), i * 20, dtype=torch.uint8)
            w.write(frame)
        assert w.frames == 12
    return path


def test_video_info(video: StdPath):
    info = video_info(video)
    assert isinstance(info, VideoInfo)
    assert info.size == Size(32, 48) and info.fps == pytest.approx(10, abs=0.1) and info.num_frames == 12


def test_iter_frames_range_exact(video: StdPath):
    frames = list(iter_video_frames(video))
    assert len(frames) == 12 and all(isinstance(f, tv_tensors.Image) and f.shape == (3, 32, 48) for f in frames)
    means = [float(f.float().mean()) for f in iter_video_frames(video, start=2, stop=9, step=3)]
    assert len(means) == 3  # frames 2, 5, 8
    for got, idx in zip(means, (2, 5, 8), strict=True):
        assert got == pytest.approx(idx * 20, abs=4)  # lossy codec


@pytest.mark.parametrize(("start", "stop", "step"), [(-1, None, 1), (0, None, 0), (5, 2, 1)])
def test_iter_frames_bad_range(video: StdPath, start: int, stop: int | None, step: int):
    with pytest.raises(ValueError):
        list(iter_video_frames(video, start=start, stop=stop, step=step))


def test_video_writer_errors(tmp_path: StdPath):
    with pytest.raises(ValueError, match="codec"):
        VideoWriter(tmp_path / "x.gifv", fps=5)
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(tmp_path / "x.mp4", fps=0)
    w = VideoWriter(tmp_path / "x.avi", fps=5)
    w.write(torch.zeros(3, 16, 16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="differs"):
        w.write(torch.zeros(3, 8, 8, dtype=torch.uint8))
    with pytest.raises(TypeError):
        w.write(torch.zeros(3, 16, 16, dtype=torch.uint16))
    w.close()
    w.close()  # idempotent
    with pytest.raises(FileExistsError):
        VideoWriter(tmp_path / "x.avi", fps=5, overwrite=False)


def test_video_errors(tmp_path: StdPath):
    with pytest.raises(FileNotFoundError):
        video_info(tmp_path / "none.mp4")
    (tmp_path / "bad.mp4").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="could not open"):
        video_info(tmp_path / "bad.mp4")


# --- Zero-copy layout ---

def test_read_image_is_zero_copy_view_and_v2_safe(tmp_path: StdPath):
    from torchvision.transforms import v2
    img = read_image(write_image(gradient(h=64, w=80), tmp_path / "v.png"))
    assert not img.is_contiguous()  # channels-last view of the decoded array, no CHW copy
    assert img.reshape(3, -1).shape == (3, 64 * 80)
    t = v2.Compose([
        v2.RandomResizedCrop(32, antialias=True), v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
    ])
    torch.manual_seed(0)
    out_view = t(img)
    torch.manual_seed(0)
    out_copy = t(tv_tensors.Image(img.contiguous()))
    assert torch.equal(out_view, out_copy)


def test_write_accepts_non_contiguous(tmp_path: StdPath):
    img = read_image(write_image(gradient(), tmp_path / "a.png"))
    again = read_image(write_image(img, tmp_path / "b.png"))
    assert torch.equal(again.as_subclass(torch.Tensor), img.as_subclass(torch.Tensor))


def test_to_numpy_hwc_shares_memory():
    img = gradient()
    arr = to_numpy_hwc(img)
    assert arr.shape == (24, 32, 3) and arr.dtype == np.uint8
    assert np.array_equal(arr, img.permute(1, 2, 0).numpy())
    arr[0, 0, 0] = 7
    assert img[0, 0, 0] == 7  # same memory
    assert to_numpy_hwc(img[0]).shape == (24, 32, 1)
    assert to_numpy_hwc(tv_tensors.Image(img)).shape == (24, 32, 3)
    with pytest.raises(ValueError):
        to_numpy_hwc(torch.zeros(1, 3, 4, 4))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_numpy_hwc_copies_from_gpu():
    img = gradient().cuda()
    assert np.array_equal(to_numpy_hwc(img), to_numpy_hwc(img.cpu()))
