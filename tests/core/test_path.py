"""Tests for ``mon.core.path``."""

import pathlib

import pytest

from mon.core.constants import IMAGE_EXTENSIONS, RAW_IMAGE_EXTENSIONS
from mon.core.constants import DATA_DIR_ENV, ZOO_DIR_ENV
from mon.core.path import Path, _normalize_exts, zoo_dir


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> Path:
    """Create ``root/{b.png, a.JPG, c.txt, sub/d.png, sub/deep/e.png, v1.0/}``."""
    root = Path(tmp_path)
    for rel in ["b.png", "a.JPG", "c.txt", "sub/d.png", "sub/deep/e.png"]:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 3)
    (root / "v1.0").mkdir()
    return root


# --- Subclassing ---

def test_derived_paths_keep_type(tree: Path):
    assert type(tree / "b.png") is Path
    assert type((tree / "b.png").parent) is Path
    assert all(type(p) is Path for p in tree.glob("*"))


# --- Helpers ---

def test_normalize_exts_order_case_dedup():
    assert _normalize_exts(["JPG", ".png", "jpg", ("tif", "")]) == (".jpg", ".png", ".tif")


def test_normalize_exts_sorts_sets():
    assert _normalize_exts([{".png", ".bmp", ".jpg"}]) == (".bmp", ".jpg", ".png")


# --- Properties ---

def test_size_and_missing(tree: Path):
    assert (tree / "b.png").size == 3
    assert (tree / "missing.png").size == 0
    assert tree.size == 0


def test_modified_at(tree: Path):
    assert (tree / "b.png").modified_at > 0
    assert (tree / "missing").modified_at == 0.0


def test_fingerprint_changes_with_content(tree: Path):
    f = tree / "b.png"
    before = f.fingerprint
    assert before is not None and before[0] == 3
    f.write_bytes(b"x" * 5)
    assert f.fingerprint != before
    assert (tree / "missing").fingerprint is None
    assert tree.fingerprint is None


# --- Predicates ---

@pytest.mark.parametrize(
    ("name", "exts", "expected"),
    [
        ("a.JPG", ("jpg",), True),
        ("a.jpg", (".JPG",), True),
        ("a.jpg", (["png", "jpg"],), True),
        ("a.jpg", ({"png", "jpeg"},), False),
        ("a.tar.gz", ("gz",), True),
        ("a.tar.gz", ("tar.gz",), False),  # suffix is only the last part
        ("noext", ("",), False),
    ],
)
def test_has_ext(name: str, exts: tuple, expected: bool):
    assert Path(name).has_ext(*exts) is expected


def test_has_ext_exists(tree: Path):
    assert (tree / "b.png").has_ext("png", exists=True)
    assert not (tree / "missing.png").has_ext("png", exists=True)
    assert (tree / "missing.png").has_ext("png")


def test_type_predicates_are_string_only_by_default():
    assert Path("nope/x.webp").is_image_file()
    assert Path("nope/x.NEF").is_raw_image_file()
    assert Path("nope/x.NEF").is_image_file()  # raw is a subset of image
    assert Path("nope/x.mp4").is_video_file()
    assert Path("nope/x.safetensors").is_weights_file()
    assert not Path("nope/x.tar").is_weights_file()
    assert Path("nope/x.yml").is_config_file()
    assert not Path("nope/x.webp").is_image_file(exists=True)


def test_raw_subset_of_image():
    assert RAW_IMAGE_EXTENSIONS <= IMAGE_EXTENSIONS


# --- Navigation ---

@pytest.mark.parametrize(
    ("ext", "expected"),
    [("json", "a/img.json"), (".json", "a/img.json"), ("", "a/img")],
)
def test_sibling(ext: str, expected: str):
    assert Path("a/img.png").sibling(ext) == Path(expected)


def test_find_sibling_order_and_none(tree: Path):
    (tree / "b.jpg").write_bytes(b"")
    stem = tree / "b.json"
    assert stem.find_sibling("jpg", "png") == tree / "b.jpg"
    assert stem.find_sibling("png", "jpg") == tree / "b.png"
    assert stem.find_sibling("bmp") is None


def test_resolve_subdir(tree: Path):
    assert tree.resolve_subdir("sub") == tree / "sub"           # exists: step in
    assert (tree / "sub").resolve_subdir("sub") == tree / "sub"  # already there
    assert tree.resolve_subdir("nope") == tree                   # missing: unchanged


def test_files_sorted_unique_default_pattern(tree: Path):
    assert tree.files() == [tree / "a.JPG", tree / "b.png", tree / "c.txt"]
    assert tree.files("*.png", "b.*") == [tree / "b.png"]  # overlapping patterns, no dupes


def test_files_recursive_and_from_file(tree: Path):
    assert tree.files("*.png", recursive=True) == [
        tree / "b.png", tree / "sub/d.png", tree / "sub/deep/e.png",
    ]
    assert (tree / "c.txt").files("*.png") == [tree / "b.png"]  # searches parent
    assert (tree / "missing").files() == []


def test_subdirs(tree: Path):
    assert tree.subdirs() == [tree / "sub", tree / "v1.0"]
    assert tree.subdirs(recursive=True) == [tree / "sub", tree / "sub/deep", tree / "v1.0"]
    assert (tree / "b.png").subdirs() == []


# --- Filesystem ---

def test_normalize(tree: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tree))
    assert Path("~/sub/../b.png").normalize() == (tree / "b.png").resolve()


def test_ensure_dir(tree: Path):
    d = tree / "new/dir"
    assert d.ensure_dir() == d and d.is_dir()

    f = tree / "out/run/log.txt"
    f.ensure_dir()
    assert f.parent.is_dir() and not f.exists()

    v = tree / "ckpt/v2.0"           # suffix heuristic would treat this as a file
    v.ensure_dir(is_file=False)
    assert v.is_dir()

    (tree / "v1.0").ensure_dir()     # existing dir with a suffix stays a dir
    assert (tree / "v1.0").is_dir()


# --- zoo_dir ---

def test_zoo_dir_prefers_mon_zoo_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setenv(ZOO_DIR_ENV, str(tmp_path / "zoo"))
    monkeypatch.setenv(DATA_DIR_ENV, "/elsewhere")
    assert zoo_dir() == tmp_path / "zoo"
    assert type(zoo_dir()) is Path


def test_zoo_dir_falls_back_to_data_dir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ZOO_DIR_ENV, raising=False)
    monkeypatch.setenv(DATA_DIR_ENV, "~/data")
    monkeypatch.setenv("HOME", "/home/u")
    assert zoo_dir() == Path("/home/u/data/zoo")


def test_zoo_dir_blank_values_count_as_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ZOO_DIR_ENV, "  ")
    monkeypatch.setenv(DATA_DIR_ENV, "/d")
    assert zoo_dir() == Path("/d/zoo")


def test_zoo_dir_unset_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ZOO_DIR_ENV, raising=False)
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match="MON_ZOO_DIR"):
        zoo_dir()


# --- find_weights ---

from mon.core.path import find_weights  # noqa: E402


@pytest.fixture
def zoo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    z = tmp_path / "zoo"
    (z / "yolo").mkdir(parents=True)
    (z / "yolo" / "n.pt").write_bytes(b"")
    monkeypatch.setenv(ZOO_DIR_ENV, str(z))
    return z


def test_find_weights_existing_path_wins(tmp_path: pathlib.Path, zoo: pathlib.Path):
    local = tmp_path / "local.pt"
    local.write_bytes(b"")
    assert find_weights(local) == local
    assert type(find_weights(local)) is Path


def test_find_weights_relative_to_zoo(zoo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.chdir(tmp_path)
    assert find_weights("yolo/n.pt") == zoo / "yolo" / "n.pt"
    assert find_weights("yolo") == zoo / "yolo"  # directories too


def test_find_weights_missing_lists_candidates(zoo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"tried: yolo/x\.pt, .*zoo/yolo/x\.pt"):
        find_weights("yolo/x.pt")


def test_find_weights_absolute_missing_skips_zoo(zoo: pathlib.Path):
    with pytest.raises(FileNotFoundError) as exc:
        find_weights("/nope/w.pt")
    assert "zoo" not in str(exc.value)


def test_find_weights_zoo_unset_is_explained(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ZOO_DIR_ENV, raising=False)
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="MON_ZOO_DIR"):
        find_weights("yolo/n.pt")
