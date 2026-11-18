"""Path subclass plus model-zoo and weights lookup."""

__all__ = ["Path", "datasets_dir", "find_weights", "runs_dir", "zoo_dir"]

import logging
import os
import pathlib
import sys
from collections.abc import Iterable
from typing import Self

from mon.core.constants import (
    CONFIG_EXTENSIONS,
    DATA_DIR_ENV,
    DATASETS_DIR_ENV,
    IMAGE_EXTENSIONS,
    RAW_IMAGE_EXTENSIONS,
    RUNS_DIR_ENV,
    VIDEO_EXTENSIONS,
    WEIGHTS_EXTENSIONS,
    ZOO_DIR_ENV,
)


# --- Helpers ---

def _normalize_exts(exts: Iterable[str | Iterable[str]]) -> tuple[str, ...]:
    """Flatten, lowercase, and dot-prefix extensions, keeping first-seen order.

    Args:
        exts: Extensions such as ``"jpg"``, ``".PNG"``, or iterables of them.

    Returns:
        Unique extensions like ``(".jpg", ".png")``, in input order. Iterables that
        are not sequences (sets) are sorted so the result is deterministic.
    """
    flat: list[str] = []
    for item in exts:
        if isinstance(item, str):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.extend(sorted(item))
    out: dict[str, None] = {}
    for ext in flat:
        ext = ext.strip().lower()
        if ext:
            out[ext if ext.startswith(".") else f".{ext}"] = None
    return tuple(out)


# --- Path ---

class Path(pathlib.Path):
    """``pathlib.Path`` with extension checks, sorted listing, and dir helpers.

    Operations such as ``/``, ``.parent``, and ``glob()`` return ``Path`` (Python ≥ 3.12).
    Methods named ``is_*`` and ``has_*`` only inspect the path string unless
    ``exists=True`` is passed.
    """

    # --- Properties ---

    @property
    def size(self) -> int:
        """File size in bytes, or ``0`` if the path is not a file."""
        return self.stat().st_size if self.is_file() else 0

    @property
    def modified_at(self) -> float:
        """Modification time (POSIX timestamp), or ``0.0`` if the path doesn't exist."""
        return self.stat().st_mtime if self.exists() else 0.0

    @property
    def fingerprint(self) -> tuple[int, int] | None:
        """Cheap change detector ``(size, mtime_ns)``, or ``None`` if not a file.

        Equal fingerprints mean "probably unchanged", not equal content. Use a content
        hash when correctness matters.
        """
        if not self.is_file():
            return None
        st = self.stat()
        return st.st_size, st.st_mtime_ns

    # --- Predicates ---

    def has_ext(self, *exts: str | Iterable[str], exists: bool = False) -> bool:
        """Check the suffix against extensions, case-insensitively.

        Args:
            *exts: Extensions with or without a dot (``"jpg"``, ``".PNG"``), or
                iterables of them.
            exists: Also require the path to be an existing file.

        Examples:
            >>> Path("a/B.JPG").has_ext("jpg", ".png")
            True
        """
        if exists and not self.is_file():
            return False
        return self.suffix.lower() in _normalize_exts(exts)

    def is_image_file(self, exists: bool = False) -> bool:
        """Check for an image extension (``IMAGE_EXTENSIONS``, raw formats included)."""
        return self.has_ext(IMAGE_EXTENSIONS, exists=exists)

    def is_raw_image_file(self, exists: bool = False) -> bool:
        """Check for a camera raw extension (``RAW_IMAGE_EXTENSIONS``)."""
        return self.has_ext(RAW_IMAGE_EXTENSIONS, exists=exists)

    def is_video_file(self, exists: bool = False) -> bool:
        """Check for a video extension (``VIDEO_EXTENSIONS``)."""
        return self.has_ext(VIDEO_EXTENSIONS, exists=exists)

    def is_weights_file(self, exists: bool = False) -> bool:
        """Check for a model weights extension (``WEIGHTS_EXTENSIONS``)."""
        return self.has_ext(WEIGHTS_EXTENSIONS, exists=exists)

    def is_config_file(self, exists: bool = False) -> bool:
        """Check for a config extension (``CONFIG_EXTENSIONS``)."""
        return self.has_ext(CONFIG_EXTENSIONS, exists=exists)

    # --- Navigation ---

    def sibling(self, ext: str) -> Self:
        """Return the path with its suffix replaced; ``ext`` may omit the dot.

        Examples:
            >>> Path("data/img_001.png").sibling("json")
            Path('data/img_001.json')
        """
        ext = ext.strip()
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        return self.with_suffix(ext)

    def find_sibling(self, *exts: str | Iterable[str]) -> Self | None:
        """Return the first existing file with the same stem and one of ``exts``.

        Extensions are tried in the given order (sets are sorted first).

        Examples:
            >>> Path("data/img_001.json").find_sibling("png", "jpg")  # doctest: +SKIP
            Path('data/img_001.png')
        """
        for ext in _normalize_exts(exts):
            candidate = self.with_suffix(ext)
            if candidate.is_file():
                return candidate
        return None

    def resolve_subdir(self, name: str) -> Self:
        """Step into the subdirectory ``name`` if needed and if it exists.

        Returns ``self`` if it already ends with ``name``, ``self / name`` if that is an
        existing directory, and ``self`` otherwise. Lets callers accept either a dataset
        root or one of its subdirectories.

        Examples:
            >>> Path("data/coco").resolve_subdir("images")  # doctest: +SKIP
            Path('data/coco/images')        # if data/coco/images/ exists
        """
        if self.name == name:
            return self
        candidate = self / name
        return candidate if candidate.is_dir() else self

    def files(self, *patterns: str, recursive: bool = False) -> list[Self]:
        """List files matching glob ``patterns`` (default ``"*"``), sorted and unique.

        If ``self`` is a file, its parent directory is searched. Returns ``[]`` if the
        directory doesn't exist.

        Args:
            *patterns: Glob patterns such as ``"*.jpg"``.
            recursive: Also search subdirectories.
        """
        root = self.parent if self.is_file() else self
        if not root.is_dir():
            return []
        glob = root.rglob if recursive else root.glob
        found = {p for pattern in (patterns or ("*",)) for p in glob(pattern) if p.is_file()}
        return sorted(found)

    def subdirs(self, recursive: bool = False) -> list[Self]:
        """List subdirectories, sorted. Returns ``[]`` if ``self`` is not a directory.

        Args:
            recursive: Include nested subdirectories.
        """
        if not self.is_dir():
            return []
        it = self.rglob("*") if recursive else self.iterdir()
        return sorted(p for p in it if p.is_dir())

    # --- Filesystem ---

    def normalize(self) -> Self:
        """Expand ``~`` and resolve to an absolute path with symlinks and ``..`` removed."""
        return self.expanduser().resolve()

    def ensure_dir(self, is_file: bool | None = None) -> Self:
        """Create the directory for this path and return ``self``.

        Args:
            is_file: ``True`` creates the parent directory (``self`` is a file path).
                ``False`` creates ``self`` as a directory. ``None`` uses the existing
                path type, or else guesses "file" when there is a suffix, which is
                wrong for directories like ``v1.0``. Pass it explicitly in that case.
        """
        if is_file is None:
            is_file = self.is_file() or (not self.is_dir() and bool(self.suffix))
        target = self.parent if is_file else self
        target.mkdir(parents=True, exist_ok=True)
        return self


# --- Well-Known Directories ---

# stdlib logging (not mon.core.console) keeps this module light to import.
logger = logging.getLogger(__name__)
_announced: set[tuple[str, str]] = set()


def _resolve_env_dir(primary: str, subdir: str) -> tuple[Path, str] | None:
    """``($primary, "PRIMARY")``, else ``($DATA_DIR/subdir, "DATA_DIR/subdir")``, else ``None``."""
    if value := os.environ.get(primary, "").strip():
        return Path(value).expanduser(), primary
    if value := os.environ.get(DATA_DIR_ENV, "").strip():
        return Path(value).expanduser() / subdir, f"{DATA_DIR_ENV}/{subdir}"
    return None


def _env_dir(primary: str, subdir: str, what: str) -> Path:
    """``$primary``, else ``$DATA_DIR/subdir``; ``~`` expanded; raise if neither is set.

    Logs the resolved directory and its source once per process (INFO), so a stray
    variable from another tool is visible.
    """
    resolved = _resolve_env_dir(primary, subdir)
    if resolved is not None:
        path, source = resolved
        if (what, str(path)) not in _announced:
            _announced.add((what, str(path)))
            logger.info("%s directory: %s (from %s)", what, path, source)
        return path
    raise RuntimeError(
        f"{what} directory is not configured: set {primary}=/path/to/{subdir} "
        f"or {DATA_DIR_ENV}=/path/to/data (uses $DATA_DIR/{subdir})"
    )


def datasets_dir() -> Path:
    """Return the datasets directory: ``$DATASETS_DIR``, then ``$DATA_DIR/datasets``.

    Each dataset lives in a folder named by its id, e.g. ``datasets_dir() / "lol_v1"``.

    Raises:
        RuntimeError: If neither variable is set.
    """
    return _env_dir(DATASETS_DIR_ENV, "datasets", "datasets")


def runs_dir() -> Path:
    """Return the run-output directory: ``$RUNS_DIR``, then ``$DATA_DIR/runs``.

    Runners write under ``runs_dir() / <mode>``, e.g. ``runs_dir() / "predict"``. The
    directory is not required to exist.

    Raises:
        RuntimeError: If neither variable is set.
    """
    return _env_dir(RUNS_DIR_ENV, "runs", "runs")


def zoo_dir() -> Path:
    """Return the pretrained-weights directory from the environment.

    Resolution order: ``$ZOO_DIR``, then ``$DATA_DIR/zoo``. ``~`` is expanded. The
    directory is not required to exist. Read on every call, so env changes apply.

    Raises:
        RuntimeError: If neither variable is set.

    Examples:
        >>> weights = zoo_dir() / "detect/yolo11/yolo11n_coco.pt"  # <task>/<model>/<file>  # doctest: +SKIP
    """
    return _env_dir(ZOO_DIR_ENV, "zoo", "model zoo")


def find_weights(path: str | os.PathLike[str]) -> Path:
    """Locate a weights file or directory: as given, else relative to ``zoo_dir()``.

    Args:
        path: Absolute path, path relative to the working directory, or path relative
            to the model zoo (e.g. ``"framework/ultralytics/yolo11/yolo11n/coco/yolo11n_coco.pt"``).

    Raises:
        FileNotFoundError: If nothing exists at any candidate; the message lists them.

    Examples:
        >>> find_weights("framework/ultralytics/yolo11/yolo11n/coco/yolo11n_coco.pt")  # doctest: +SKIP
        Path('/data/zoo/framework/ultralytics/yolo11/yolo11n/coco/yolo11n_coco.pt')
    """
    given = Path(path).expanduser()
    if given.exists():
        return given

    tried = [str(given)]
    if not given.is_absolute():
        try:
            candidate = zoo_dir() / given
        except RuntimeError as err:
            tried.append(f"<zoo: {err}>")
        else:
            if candidate.exists():
                return candidate
            tried.append(str(candidate))
    raise FileNotFoundError(f"weights not found; tried: {', '.join(tried)}")


# --- Diagnostics: python -m mon.core.path ---

def main() -> int:
    """Print where ``mon`` looks for data, datasets, and weights, and whether those exist.

    Returns ``0`` if all three are configured and exist, else ``1``.
    """
    rows: list[tuple[str, Path | None, str]] = []
    data = os.environ.get(DATA_DIR_ENV, "").strip()
    rows.append(("data", Path(data).expanduser() if data else None, DATA_DIR_ENV if data else f"set {DATA_DIR_ENV}"))
    for label, env, subdir in (("datasets", DATASETS_DIR_ENV, "datasets"), ("zoo", ZOO_DIR_ENV, "zoo")):
        resolved = _resolve_env_dir(env, subdir)
        rows.append((label, *resolved) if resolved else (label, None, f"set {env} or {DATA_DIR_ENV}"))
    ok = True
    for label, path, source in rows:
        if path is None:
            status, ok = "not configured", False
        elif path.is_dir():
            status = "exists"
        else:
            status, ok = "missing", False
        print(f"{label:<9} {str(path) if path else '-':<50} {status:<15} ({source})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
