"""Tests for ``mon.data.zoo.testkit``: helpers, and the built testkit when it is on this machine."""

import hashlib

import numpy as np
import pytest
import torch
import yaml

from mon.core.enums import Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo import testkit
from mon.data.zoo.testkit import EFFICIENCY_SIZES, PARTS, SIZE, TestkitMEF, _cover, _evenly, _square, build


def test_square_resizes_short_side_and_centre_crops():
    img = np.zeros((400, 600, 3), np.uint8)
    img[:, 290:310] = 255  # a centred stripe survives the centre crop
    out = _square(img)
    assert out.shape == (SIZE, SIZE, 3)
    assert out[:, SIZE // 2].min() > 200 and out[:, 0].max() == 0
    assert _square(np.zeros((300, 200, 3), np.uint8)).shape == (SIZE, SIZE, 3)  # portrait


def test_cover_scales_and_centre_crops():
    img = np.zeros((300, 450, 3), np.uint8)
    img[140:160, 215:235] = 255  # a centred block stays centred
    for w, h in [(256, 256), (400, 200), (900, 600)]:  # down (area), crop, up (bicubic)
        out = _cover(img, w, h)
        assert out.shape == (h, w, 3) and out[h // 2, w // 2].min() > 200 and out[0, 0].max() == 0


def test_task_tags_keep_efficiency_out_of_lle_lists():
    lle = set(DATASETS.names(tasks=Task.LLE))
    assert {"testkit_lle", "testkit_lle_unpaired"} <= lle and "testkit_efficiency" not in lle
    assert "testkit_efficiency" in DATASETS.names(tasks=Task.BENCHMARK)


def test_efficiency_sizes_and_parts(tmp_path):
    assert EFFICIENCY_SIZES[0] == (256, 256) and EFFICIENCY_SIZES[-1] == (7680, 4320)
    assert all(w >= h for w, h in EFFICIENCY_SIZES) and "efficiency" in PARTS
    with pytest.raises(ValueError, match="unknown testkit parts"):
        build(tmp_path, parts=["nope"])
    (tmp_path / "efficiency").mkdir()
    (tmp_path / "efficiency" / "x").write_text("")
    with pytest.raises(FileExistsError):
        build(tmp_path, parts=["efficiency"])


def test_evenly_spaced_selection():
    assert _evenly(list(range(10)), 3) == [0, 4, 9]  # first, middle, last
    assert _evenly([1, 2], 5) == [1, 2]
    assert _evenly(list(range(7)), 1) == [0]


def test_mef_targets_pair_by_scene():
    assert TestkitMEF.pair_key("target", "sice_58/2") == "sice_58"
    assert TestkitMEF.pair_key("image", "sice_58/2") == "sice_58/2"


@pytest.mark.slow
@pytest.mark.parametrize(("name", "count", "keys", "size"), [
    ("testkit_lle", 100, {"image", "target", "meta"}, SIZE),
    ("testkit_lle_unpaired", 100, {"image", "meta"}, SIZE),
    ("testkit_mef", 100, {"image", "target", "meta"}, SIZE),
    ("testkit_classify", 100, {"image", "label", "meta"}, 32),
])
def test_built_testkit(name: str, count: int, keys: set[str], size: int):
    try:
        root = datasets_dir() / "testkit"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    ds = DATASETS.build(name, "test")
    assert len(ds) == count
    for i in range(len(ds)):
        sample = ds[i]
        assert set(sample) == keys and tuple(sample["image"].shape) == (3, size, size)
    if name == "testkit_classify":
        assert sorted(ds.labels) == sorted(list(range(10)) * 10)  # 10 per class


@pytest.mark.slow
def test_built_testkit_efficiency():
    try:
        root = datasets_dir() / "testkit" / "efficiency"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine (build: python -m mon.data.zoo testkit build --part efficiency)")
    ds = DATASETS.build("testkit_efficiency", "test")
    items = dict(ds.items())
    assert all(set(files) == {"image", "target"} for files in items.values())
    stems = sorted(items)
    assert len(stems) == 5 * len(EFFICIENCY_SIZES)
    scenes = {s.split("/")[1] for s in stems}
    assert len(scenes) == 5 and all(f"{w}x{h}/{sc}" in stems for w, h in EFFICIENCY_SIZES for sc in scenes)
    sample = ds[stems.index("600x400/" + sorted(scenes)[0])]
    assert sample["image"].shape == sample["target"].shape == (3, 400, 600)


# --- Builders on synthetic sources ---

_SMALL_SIZES = ((16, 16), (48, 32), (64, 48))  # (W, H); the second to last sets the minimum scene size


def _png(root, rel: str, value: int, h: int = 40, w: int = 60) -> None:
    write_image(torch.full((3, h, w), value, dtype=torch.uint8), root / rel)


@pytest.fixture
def sources(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A ``DATASETS_DIR`` with tiny copies of every source the builders read; efficiency sizes shrunk."""
    root = tmp_path / "datasets"
    for name, split, n in (("lol_v1", "test", 3), ("lol_v2_real", "test", 2), ("sice_me", "val", 2)):
        for i in range(n):
            _png(root, f"{name}/{split}/images/{i:02d}.png", 10 + i)
            _png(root, f"{name}/{split}/targets/{i:02d}.png", 200 + i)
    for name in ("dicm", "fusion", "lime", "mef", "npe", "vv"):
        for i in range(2):
            _png(root, f"unpaired_lol/{name}/test/images/{i}.png", 30 + i)
    for scene in range(1, 7):  # SICE: scenes 1–6; scene 3 has only 3 exposures (no MEF stack)
        for k in range(1, (4 if scene == 3 else 6)):
            _png(root, f"sice/test/images/{scene}/{k}.png", 20 * k)
        _png(root, f"sice/test/targets/{scene}.png", 250)
    classes = [f"c{i}" for i in range(10)]
    (root / "cifar10").mkdir(parents=True)
    (root / "cifar10" / "classes.txt").write_text("\n".join(classes) + "\n")
    for c in classes:
        for i in range(12):
            _png(root, f"cifar10/test/{c}/{i:05d}.png", i, 32, 32)
    monkeypatch.setenv("DATASETS_DIR", str(root))
    monkeypatch.setattr(testkit, "EFFICIENCY_SIZES", _SMALL_SIZES)
    return root


def _files(root, part: str) -> list[str]:
    return sorted(str(p.relative_to(root / part)) for p in (root / part).rglob("*") if p.is_file())


def test_build_every_part_from_sources(sources):
    root = build()
    assert root == sources / "testkit"
    lle = DATASETS.build("testkit_lle", "test")
    assert sorted(s for s, _ in lle.items()) == ["lol_v1_00", "lol_v1_01", "lol_v1_02", "lol_v2_real_00",
                                                 "lol_v2_real_01", "sice_me_00", "sice_me_01"]
    sample = lle[0]
    assert sample["image"].shape == sample["target"].shape == (3, SIZE, SIZE)
    assert int(sample["image"][0, 0, 0]) == 10 and int(sample["target"][0, 0, 0]) == 200  # pair stays aligned
    assert len(DATASETS.build("testkit_lle_unpaired", "test")) == 12
    mef = dict(DATASETS.build("testkit_mef", "test").items())
    scenes = sorted({s.split("/")[0] for s in mef})
    assert scenes == [f"sice_{s}" for s in (1, 2, 4, 5, 6)]  # scene 3: too few exposures
    assert sorted(s for s in mef if s.startswith("sice_1/")) == [f"sice_1/{k}" for k in (1, 2, 3, 4)]  # renumbered
    darkest, brightest = (testkit._read(mef[f"sice_1/{k}"]["image"])[0, 0, 0] for k in (1, 4))
    assert (darkest, brightest) == (20, 100)  # exposures 1 and 5 of 5: the ends of the stack
    eff = dict(DATASETS.build("testkit_efficiency", "test").items())
    assert len(eff) == 5 * len(_SMALL_SIZES) and "48x32/sice_1" in eff
    assert testkit._read(eff["64x48/sice_1"]["target"]).shape == (48, 64, 3)
    classify = DATASETS.build("testkit_classify", "test")
    assert sorted(classify.labels) == sorted(list(range(10)) * 10)
    picked = _files(root, "classify")
    assert "test/c0/cifar10_00009.png" in picked and "test/c0/cifar10_00010.png" not in picked  # the first 10
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    files = {e["file"]: e for e in manifest["files"]}
    on_disk = {f"{p}/{f}" for p in PARTS.values() for f in _files(root, p) if f != "classes.txt"}
    assert set(files) == on_disk and manifest["efficiency_sizes"] == ["16x16", "48x32", "64x48"]
    entry = files["lle/test/images/lol_v1_00.png"]
    assert entry["source"] == "lol_v1/test/images/00.png"
    assert entry["sha256"] == hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest()


def test_part_rebuild_keeps_the_other_manifest_entries(sources):
    root   = build()
    before = yaml.safe_load((root / "manifest.yaml").read_text())["files"]
    with pytest.raises(FileExistsError):
        build(parts=["lle"])
    build(parts=["lle"], overwrite=True)
    after = yaml.safe_load((root / "manifest.yaml").read_text())["files"]
    assert after == before  # same sources, same outputs, same order
    with pytest.raises(FileExistsError):
        build()  # a full build checks the whole root


def test_cli_builds_one_part(sources, capsys: pytest.CaptureFixture[str]):
    from mon.data.zoo.__main__ import main

    assert main(["testkit", "build", "--part", "lle_unpaired", "--out", str(sources / "kit")]) == 0
    assert capsys.readouterr().out.strip().endswith(str(sources / "kit"))  # after the log lines
    assert len(_files(sources / "kit", "lle_unpaired")) == 12
    entries = yaml.safe_load((sources / "kit" / "manifest.yaml").read_text())["files"]
    assert len(entries) == 12 and all(e["file"].startswith("lle_unpaired/") for e in entries)
