"""Console output and logging: a shared ``rich`` console and stdlib-logging setup.

Library code only calls ``get_logger(__name__)``. Importing ``mon`` never configures
logging; entry points (scripts, CLIs) call ``setup_logging()`` once.
"""

__all__ = [
    "DEFAULT_QUIET_LOGGERS",
    "console",
    "get_logger",
    "print_dict",
    "setup_logging",
]

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty
from rich.theme import Theme

# --- Console ---

_THEME = Theme({
    "debug"   : "dark_green",
    "info"    : "green",
    "warning" : "yellow",
    "error"   : "bright_red",
    "critical": "bold red",
})

console = Console(theme=_THEME, log_time_format="[%X]", soft_wrap=True)
"""Shared ``rich`` console (stdout) for rules, tables, and progress bars."""


# --- Logging ---

_ROOT_LOGGER   = "mon"
_HANDLER_TAG   = "_mon_handler"
_FILE_FORMAT   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_RANK_ENV_VARS = ("RANK", "SLURM_PROCID", "LOCAL_RANK")

DEFAULT_QUIET_LOGGERS: tuple[str, ...] = (
    "PIL", "fsspec", "matplotlib", "tensorboard", "torch", "urllib3",
)
"""Third-party loggers that ``setup_logging`` lowers to WARNING by default."""

# A library must not emit or configure anything by default (Python logging HOWTO).
logging.getLogger(_ROOT_LOGGER).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return ``logging.getLogger(name)``, defaulting to the ``mon`` logger.

    Use ``get_logger(__name__)`` in library modules; this only fetches a logger and
    never adds handlers or changes levels.
    """
    return logging.getLogger(name or _ROOT_LOGGER)


def _to_level(level: int | str) -> int:
    """Convert a level name (``"info"``) or number to a logging level number."""
    if isinstance(level, int):
        return level
    mapping = logging.getLevelNamesMapping()
    try:
        return mapping[level.strip().upper()]
    except KeyError:
        raise ValueError(f"unknown log level {level!r}; expected one of {sorted(mapping)}") from None


def _process_rank() -> int:
    """Global process rank from ``RANK``/``SLURM_PROCID``/``LOCAL_RANK``; ``0`` if unset."""
    for var in _RANK_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value.isdigit():
            return int(value)
    return 0


def setup_logging(
    level         : int | str = "INFO",
    log_file      : str | os.PathLike[str] | None = None,
    *,
    quiet         : Iterable[str] = DEFAULT_QUIET_LOGGERS,
    rank_zero_only: bool = True,
) -> logging.Logger:
    """Configure the root logger for an entry point (script, CLI, notebook).

    Logs go to ``console`` through ``rich`` and, optionally, to a plain-text file.
    Python ``warnings`` are routed into logging so they reach the file too. Calling
    it again replaces the handlers it added before; other handlers are left alone.

    Args:
        level: Minimum level, as a name (``"DEBUG"``) or number.
        log_file: Append logs to this file, creating parent directories.
        quiet: Logger names to lower to WARNING (noisy third-party libraries).
        rank_zero_only: In multi-process runs (``RANK`` etc. set), non-zero ranks
            print only warnings and above and don't write ``log_file``.

    Returns:
        The root logger.
    """
    lvl  = _to_level(level)
    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(h)
        h.close()

    is_main = _process_rank() == 0 or not rank_zero_only

    stream = RichHandler(console=console, rich_tracebacks=True, show_path=False, log_time_format="[%X]")
    stream.setLevel(lvl if is_main else max(lvl, logging.WARNING))
    handlers: list[logging.Handler] = [stream]

    if log_file is not None and is_main:
        path = os.fspath(log_file)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        file = logging.FileHandler(path, mode="a", encoding="utf-8")
        file.setLevel(lvl)
        file.setFormatter(logging.Formatter(_FILE_FORMAT))
        handlers.append(file)

    for h in handlers:
        setattr(h, _HANDLER_TAG, True)
        root.addHandler(h)
    root.setLevel(lvl)

    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    return root


# --- Printing ---

def print_dict(value: Mapping[Any, Any] | DictConfig, title: str = "") -> None:
    """Pretty-print a mapping or OmegaConf config in a panel on ``console``.

    OmegaConf interpolations are resolved; missing values (``???``) are shown as-is.
    """
    data = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
    pretty = Pretty(data, expand_all=True, indent_guides=True, overflow="fold")
    console.print(Panel(pretty, title=title or None, title_align="left"))
