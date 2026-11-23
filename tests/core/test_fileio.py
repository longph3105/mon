"""Tests for ``mon.core.fileio``."""

import hashlib
import os
import pathlib
from pathlib import Path as StdPath
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from mon.core import fileio
from mon.core.fileio import download_file, load_json, load_txt, load_yaml, save_json, save_txt, save_yaml, sha256sum
from mon.core.path import Path


def _no_temp_files(d: StdPath) -> bool:
    return not any(p.name.endswith(".tmp") for p in d.iterdir())


# --- Round trips ---

def test_json_round_trip_and_types(tmp_path: StdPath):
    data = {
        "name"  : "Mon 🐈 café",
        "lr"    : 1e-3,
        "path"  : pathlib.Path("/data/coco"),
        "np"    : np.float32(0.5),
        "arr"   : np.arange(3),
        "tensor": torch.tensor([[1, 2]]),
        "none"  : None,
    }
    out = save_json(data, tmp_path / "a" / "b.json")
    assert isinstance(out, Path) and out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "café" in text and text.endswith("\n")  # ensure_ascii=False, trailing newline
    assert load_json(out) == {
        "name": "Mon 🐈 café", "lr": 1e-3, "path": "/data/coco",
        "np": 0.5, "arr": [0, 1, 2], "tensor": [[1, 2]], "none": None,
    }


def test_json_unsupported_type_leaves_file_untouched(tmp_path: StdPath):
    f = tmp_path / "x.json"
    save_json({"ok": 1}, f)
    with pytest.raises(TypeError, match="not JSON serializable"):
        save_json({"bad": object()}, f)
    assert load_json(f) == {"ok": 1}
    assert _no_temp_files(tmp_path)


def test_yaml_round_trip_keeps_order_unicode_and_paths(tmp_path: StdPath):
    data = {"z": 1, "a": {"root": pathlib.Path("/data"), "name": "ảnh"}, "list": [1, 2]}
    out = save_yaml(data, tmp_path / "cfg.yaml")
    text = out.read_text(encoding="utf-8")
    assert text.index("z:") < text.index("a:")  # sort_keys=False
    assert "ảnh" in text                        # allow_unicode
    assert load_yaml(out) == {"z": 1, "a": {"root": "/data", "name": "ảnh"}, "list": [1, 2]}


def test_yaml_rejects_arbitrary_objects(tmp_path: StdPath):
    with pytest.raises(yaml.representer.RepresenterError):
        save_yaml({"x": object()}, tmp_path / "x.yaml")
    assert not (tmp_path / "x.yaml").exists()


def test_load_yaml_empty_file_is_none(tmp_path: StdPath):
    f = tmp_path / "empty.yml"
    f.write_text("")
    assert load_yaml(f) is None


def test_txt_round_trip(tmp_path: StdPath):
    out = save_txt(["a", " b ", "", "c"], tmp_path / "l.txt")
    assert out.read_text() == "a\n b \n\nc\n"
    assert load_txt(out) == ["a", "b", "", "c"]
    assert load_txt(save_txt("single", tmp_path / "s.txt")) == ["single"]
    assert load_txt(save_txt((s for s in ["x", "y"]), tmp_path / "g.txt")) == ["x", "y"]
    assert save_txt([], tmp_path / "e.txt").read_text() == ""


# --- Behavior ---

@pytest.mark.parametrize(
    ("save", "data", "name"),
    [(save_json, {"a": 1}, "f.json"), (save_yaml, {"a": 1}, "f.yaml"), (save_txt, ["a"], "f.txt")],
)
def test_overwrite_false_raises_and_keeps_file(tmp_path: StdPath, save, data, name: str):
    f = tmp_path / name
    f.write_text("original")
    with pytest.raises(FileExistsError):
        save(data, f, overwrite=False)
    assert f.read_text() == "original"
    save(data, f)  # overwrite=True by default
    assert f.read_text() != "original"


def test_expanduser(tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = save_json([1], "~/sub/x.json")
    assert out == tmp_path / "sub" / "x.json"
    assert load_json("~/sub/x.json") == [1]


def test_permissions_follow_umask(tmp_path: StdPath):
    umask = os.umask(0)
    os.umask(umask)
    out = save_txt("x", tmp_path / "p.txt")
    assert out.stat().st_mode & 0o777 == 0o666 & ~umask


def test_crash_during_replace_keeps_old_file_and_cleans_temp(tmp_path: StdPath):
    f = tmp_path / "state.json"
    save_json({"step": 1}, f)
    with (
        mock.patch.object(fileio.os, "replace", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        save_json({"step": 2}, f)
    assert load_json(f) == {"step": 1}
    assert _no_temp_files(tmp_path)


def test_load_missing_file_raises(tmp_path: StdPath):
    for load in (load_json, load_yaml, load_txt):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "missing")


# --- sha256sum / download_file ---

PAYLOAD = b"pretend these are weights" * 100
DIGEST  = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def source_url(tmp_path: StdPath) -> str:
    src = tmp_path / "remote" / "w.pt"
    src.parent.mkdir()
    src.write_bytes(PAYLOAD)
    return src.as_uri()


def test_sha256sum(tmp_path: StdPath):
    f = tmp_path / "x.bin"
    f.write_bytes(PAYLOAD)
    assert sha256sum(f) == DIGEST
    assert sha256sum(f, chunk_size=7) == DIGEST


def test_download_creates_parents_and_verifies(tmp_path: StdPath, source_url: str):
    out = download_file(source_url, tmp_path / "zoo" / "a" / "w.pt", sha256=DIGEST.upper(), progress=False)
    assert isinstance(out, Path) and out.read_bytes() == PAYLOAD


def test_download_bad_hash_leaves_nothing(tmp_path: StdPath, source_url: str):
    dst = tmp_path / "out" / "w.pt"
    with pytest.raises(RuntimeError, match="invalid hash"):
        download_file(source_url, dst, sha256="0" * 64, progress=False)
    assert not dst.exists()
    assert list(dst.parent.iterdir()) == []


def test_download_keeps_existing_unless_overwrite(tmp_path: StdPath, source_url: str):
    dst = tmp_path / "w.pt"
    dst.write_bytes(b"old")
    assert download_file(source_url, dst, progress=False).read_bytes() == b"old"
    assert download_file(source_url, dst, overwrite=True, progress=False).read_bytes() == PAYLOAD


def test_download_existing_hash_mismatch_raises(tmp_path: StdPath, source_url: str):
    dst = tmp_path / "w.pt"
    dst.write_bytes(b"corrupt partial file")
    with pytest.raises(ValueError, match="doesn't match"):
        download_file(source_url, dst, sha256=DIGEST, progress=False)
    assert download_file(source_url, dst, sha256=DIGEST, overwrite=True, progress=False).read_bytes() == PAYLOAD


@pytest.mark.parametrize("url", ["ftp://x/w.pt", "/local/path/w.pt", "https:/x.com/w.pt", "s3://b/w.pt"])
def test_download_rejects_bad_urls(tmp_path: StdPath, url: str):
    with pytest.raises(ValueError, match="URL"):
        download_file(url, tmp_path / "w.pt")


@pytest.mark.parametrize("sha", ["abc", "z" * 64, "0" * 63])
def test_download_rejects_bad_sha(tmp_path: StdPath, source_url: str, sha: str):
    with pytest.raises(ValueError, match="64 hex"):
        download_file(source_url, tmp_path / "w.pt", sha256=sha)
