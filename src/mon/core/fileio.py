"""Atomic JSON/YAML/text I/O and verified downloads."""

__all__ = [
    "download_file",
    "load_json",
    "load_txt",
    "load_yaml",
    "save_json",
    "save_txt",
    "save_yaml",
    "sha256sum",
]

import hashlib
import json
import os
import pathlib
import re
import urllib.parse
import uuid
from collections.abc import Iterable
from typing import Any

import yaml

from mon.core.path import Path

type PathLike = str | os.PathLike[str]

_URL_SCHEMES = frozenset({"http", "https", "file"})
_SHA256_RE   = re.compile(r"^[0-9a-f]{64}$")


# --- Helpers ---

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory and ``os.replace``.

    Readers never see a partial file, and a crash mid-write leaves any old file intact.
    The temp file is created with mode ``0o666`` so the usual umask applies.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """``_atomic_write_bytes`` for UTF-8 text (no newline translation)."""
    _atomic_write_bytes(path, text.encode("utf-8"))


def _prepare_target(path: PathLike, overwrite: bool) -> Path:
    """Expand ``~``, refuse an existing file unless ``overwrite``, and create parent dirs."""
    target = Path(path).expanduser()
    if not overwrite and target.exists():
        raise FileExistsError(f"{target} exists and overwrite=False")
    return target.ensure_dir(is_file=True)


def _json_default(obj: Any) -> Any:
    """Encode paths as strings and arrays/tensors/NumPy scalars via ``tolist()``."""
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


class _YamlDumper(yaml.SafeDumper):
    """``SafeDumper`` that also writes ``pathlib`` paths as plain strings."""


_YamlDumper.add_multi_representer(
    pathlib.PurePath, lambda dumper, p: dumper.represent_str(os.fspath(p)),
)


# --- Input ---

def load_json(path: PathLike) -> Any:
    """Load a JSON file."""
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: PathLike) -> Any:
    """Load a YAML file with ``yaml.safe_load``. An empty file returns ``None``."""
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_txt(path: PathLike) -> list[str]:
    """Read a text file as a list of lines, each stripped of surrounding whitespace.

    Blank lines are kept (as ``""``) so line numbers still match the file.
    """
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        return [line.strip() for line in f]


# --- Output ---

def save_json(data: Any, path: PathLike, *, indent: int | None = 4, overwrite: bool = True) -> Path:
    """Atomically write ``data`` as UTF-8 JSON and return the path.

    Paths become strings; NumPy arrays/scalars and tensors become lists/numbers.
    Data is serialized before the file is touched, so a failure leaves it unchanged.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is ``False``.
        TypeError: If ``data`` holds a value JSON can't represent.
    """
    text = json.dumps(data, indent=indent, ensure_ascii=False, default=_json_default) + "\n"
    target = _prepare_target(path, overwrite)
    _atomic_write_text(target, text)
    return target


def save_yaml(data: Any, path: PathLike, *, overwrite: bool = True) -> Path:
    """Atomically write ``data`` as UTF-8 YAML (key order kept) and return the path.

    Uses a safe dumper, so only plain Python types and paths are allowed. For OmegaConf
    configs, use ``OmegaConf.save``.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is ``False``.
        yaml.representer.RepresenterError: If ``data`` holds an unsupported type.
    """
    text = yaml.dump(data, Dumper=_YamlDumper, sort_keys=False, allow_unicode=True)
    target = _prepare_target(path, overwrite)
    _atomic_write_text(target, text)
    return target


def save_txt(lines: str | Iterable[str], path: PathLike, *, overwrite: bool = True) -> Path:
    """Atomically write a string or lines (newline-terminated) and return the path.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is ``False``.
    """
    items = [lines] if isinstance(lines, str) else list(lines)
    text = "".join(f"{line}\n" for line in items)
    target = _prepare_target(path, overwrite)
    _atomic_write_text(target, text)
    return target


# --- Download ---

def sha256sum(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(Path(path).expanduser(), "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url      : str,
    dst      : PathLike,
    *,
    sha256   : str | None = None,
    overwrite: bool = False,
    progress : bool = True,
) -> Path:
    """Download ``url`` to ``dst`` and return the path.

    Uses ``torch.hub.download_url_to_file``: data goes to a temp file that is moved into
    place only after it is complete and (if ``sha256`` is given) verified, so ``dst`` is
    never left half-written.

    Args:
        url: ``http``, ``https``, or ``file`` URL.
        dst: Destination file; parent directories are created.
        sha256: Expected full hex SHA-256 digest. Also checked against an existing
            ``dst`` when it is kept.
        overwrite: Re-download even if ``dst`` exists.
        progress: Show a progress bar.

    Raises:
        ValueError: If ``url`` has an unsupported scheme, ``sha256`` is malformed, or an
            existing ``dst`` doesn't match ``sha256``.
        RuntimeError: If the downloaded data doesn't match ``sha256``.
        urllib.error.URLError: If the download fails.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _URL_SCHEMES or (scheme != "file" and not parsed.netloc):
        raise ValueError(f"expected an http(s) URL with a host, or a file URL, got {url!r}")
    if sha256 is not None:
        sha256 = sha256.strip().lower()
        if not _SHA256_RE.match(sha256):
            raise ValueError(f"sha256 must be 64 hex characters, got {sha256!r}")

    target = Path(dst).expanduser()
    if target.exists() and not overwrite:
        if sha256 is not None and sha256sum(target) != sha256:
            raise ValueError(f"{target} exists but its SHA-256 doesn't match; pass overwrite=True to re-download")
        return target

    from torch.hub import download_url_to_file  # imported lazily: torch is slow to import

    target.ensure_dir(is_file=True)
    download_url_to_file(url, os.fspath(target), hash_prefix=sha256, progress=progress)
    return target
