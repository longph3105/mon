"""Tests for ``mon.core.console``."""

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from rich.logging import RichHandler

import mon.core.console as mc


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Undo ``setup_logging`` side effects so tests stay independent."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    quiet = {n: logging.getLogger(n).level for n in mc.DEFAULT_QUIET_LOGGERS}
    yield
    for h in root.handlers:
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    for n, lvl in quiet.items():
        logging.getLogger(n).setLevel(lvl)
    logging.captureWarnings(False)


def _ours() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, mc._HANDLER_TAG, False)]


@pytest.fixture
def no_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in mc._RANK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --- Import has no side effects ---

def test_import_does_not_configure_logging():
    assert not _ours()
    mon_logger = logging.getLogger("mon")
    assert any(isinstance(h, logging.NullHandler) for h in mon_logger.handlers)
    assert mon_logger.level == logging.NOTSET


def test_get_logger():
    assert mc.get_logger() is logging.getLogger("mon")
    assert mc.get_logger("mon.core.path") is logging.getLogger("mon.core.path")


# --- Levels and ranks ---

@pytest.mark.parametrize(("level", "expected"), [("info", 20), (" DEBUG ", 10), (30, 30)])
def test_to_level(level: int | str, expected: int):
    assert mc._to_level(level) == expected


def test_to_level_invalid():
    with pytest.raises(ValueError, match="unknown log level"):
        mc._to_level("loud")


@pytest.mark.parametrize(
    ("env", "expected"),
    [({}, 0), ({"RANK": "3"}, 3), ({"LOCAL_RANK": "1"}, 1), ({"RANK": "2", "LOCAL_RANK": "0"}, 2), ({"RANK": "x"}, 0)],
)
def test_process_rank(monkeypatch: pytest.MonkeyPatch, no_rank: None, env: dict[str, str], expected: int):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert mc._process_rank() == expected


# --- setup_logging ---

def test_setup_logging_console_and_file(tmp_path: Path, no_rank: None):
    log_file = tmp_path / "run" / "log.txt"
    mc.setup_logging("DEBUG", log_file=log_file)
    kinds = sorted(type(h).__name__ for h in _ours())
    assert kinds == ["FileHandler", "RichHandler"]
    assert logging.getLogger().level == logging.DEBUG

    mc.get_logger("mon.test").debug("hello %s", "file")
    warnings.warn("careful", UserWarning, stacklevel=1)
    for h in _ours():
        h.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "DEBUG    | mon.test | hello file" in text
    assert "careful" in text  # warnings captured into logging


def test_setup_logging_is_idempotent(tmp_path: Path, no_rank: None):
    mc.setup_logging(log_file=tmp_path / "a.txt")
    mc.setup_logging()
    assert [type(h) for h in _ours()] == [RichHandler]


def test_setup_logging_keeps_foreign_handlers(no_rank: None):
    foreign = logging.StreamHandler()
    logging.getLogger().addHandler(foreign)
    mc.setup_logging()
    assert foreign in logging.getLogger().handlers


def test_setup_logging_quiets_libraries(no_rank: None):
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    mc.setup_logging("DEBUG", quiet=["urllib3"])
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_non_zero_rank_is_quiet_and_writes_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_rank: None):
    monkeypatch.setenv("RANK", "1")
    log_file = tmp_path / "log.txt"
    mc.setup_logging("INFO", log_file=log_file)
    (stream,) = _ours()
    assert stream.level == logging.WARNING
    assert not log_file.exists()


def test_non_zero_rank_all_ranks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_rank: None):
    monkeypatch.setenv("RANK", "1")
    mc.setup_logging("INFO", log_file=tmp_path / "log.txt", rank_zero_only=False)
    assert {type(h).__name__: h.level for h in _ours()} == {"RichHandler": 20, "FileHandler": 20}


# --- print_dict ---

def test_print_dict_plain_and_omegaconf():
    cfg = OmegaConf.create({"root": "/data", "out": "${root}/runs", "ckpt": "???"})
    with mc.console.capture() as cap:
        mc.print_dict({"lr": 0.001, "nested": {"depth": 18}}, title="Plain")
        mc.print_dict(cfg, title="Config")
    out = cap.get()
    assert "Plain" in out and "0.001" in out and "depth" in out
    assert "/data/runs" in out and "???" in out
