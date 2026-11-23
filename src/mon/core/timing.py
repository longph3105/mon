"""Wall-clock timers that account for async GPU work."""

__all__ = [
    "Profiler",
    "Timer",
    "synchronize",
]

import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Self

import torch
from rich.table import Table

from mon.core.console import console


def synchronize() -> None:
    """Wait for queued CUDA and MPS work so a clock read reflects finished computation.

    Doesn't initialize CUDA if it hasn't been used yet.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


class Timer:
    """Accumulating interval timer (``time.perf_counter``).

    Args:
        name: Label used in summaries.
        sync: Call ``synchronize()`` at both start and stop, so GPU work queued before
            ``tick`` isn't counted and work queued inside is.

    Examples:
        >>> t = Timer("step", sync=False)
        >>> with t:
        ...     pass
        >>> t.calls, t.total >= 0.0
        (1, True)
    """

    def __init__(self, name: str = "", *, sync: bool = True) -> None:
        self.name   = name
        self.sync   = sync
        self.total  = 0.0
        self.calls  = 0
        self.last   = 0.0
        self._start: float | None = None

    def __repr__(self) -> str:
        return f"Timer({self.name!r}, calls={self.calls}, total={self.total:.6f}s)"

    @property
    def mean(self) -> float:
        """Mean interval in seconds (``0.0`` before the first interval)."""
        return self.total / self.calls if self.calls else 0.0

    def tick(self) -> None:
        """Start an interval."""
        if self.sync:
            synchronize()
        self._start = time.perf_counter()

    def tock(self) -> float:
        """Stop the current interval, add it to the totals, and return its duration (s).

        Raises:
            RuntimeError: If ``tick`` wasn't called first.
        """
        if self._start is None:
            raise RuntimeError(f"Timer {self.name!r}: tock() called without tick()")
        if self.sync:
            synchronize()
        self.last    = time.perf_counter() - self._start
        self.total  += self.last
        self.calls  += 1
        self._start  = None
        return self.last

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self.total, self.calls, self.last, self._start = 0.0, 0, 0.0, None

    def __enter__(self) -> Self:
        self.tick()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        self.tock()


class Profiler:
    """A set of named ``Timer`` stages, e.g. ``preprocess``/``infer``/``postprocess``.

    Examples:
        >>> prof = Profiler(sync=False)
        >>> with prof.stage("infer"):
        ...     pass
        >>> prof["infer"].calls, list(prof.summary())
        (1, ['infer'])
    """

    def __init__(self, *, sync: bool = True) -> None:
        self.sync = sync
        self._timers: dict[str, Timer] = {}

    def __getitem__(self, name: str) -> Timer:
        return self._timers[name]

    def __contains__(self, name: object) -> bool:
        return name in self._timers

    def timer(self, name: str) -> Timer:
        """Return the timer for ``name``, creating it on first use."""
        if name not in self._timers:
            self._timers[name] = Timer(name, sync=self.sync)
        return self._timers[name]

    @contextmanager
    def stage(self, name: str) -> Iterator[Timer]:
        """Time the ``with`` block under ``name`` (stopped even if the block raises)."""
        timer = self.timer(name)
        timer.tick()
        try:
            yield timer
        finally:
            timer.tock()

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage ``{"total", "mean", "calls"}`` in insertion order (seconds)."""
        return {n: {"total": t.total, "mean": t.mean, "calls": float(t.calls)} for n, t in self._timers.items()}

    def print_summary(self, title: str = "Timing") -> None:
        """Print a table of per-stage totals, means (ms), and call counts on ``console``."""
        table = Table(title=title, header_style="bold magenta")
        table.add_column("Stage")
        table.add_column("Total (s)", justify="right")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Calls", justify="right")
        for name, t in self._timers.items():
            table.add_row(name, f"{t.total:.4f}", f"{t.mean * 1000:.2f}", str(t.calls))
        console.print(table)
