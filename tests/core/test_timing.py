"""Tests for ``mon.core.timing``."""

import time

import pytest

from mon.core import timing
from mon.core.timing import Profiler, Timer


def test_timer_accumulates():
    t = Timer("x", sync=False)
    assert t.mean == 0.0
    for _ in range(3):
        t.tick()
        time.sleep(0.01)
        t.tock()
    assert t.calls == 3 and t.total >= 0.03 and t.mean == pytest.approx(t.total / 3)
    assert 0.0 < t.last <= t.total
    t.reset()
    assert (t.total, t.calls, t.last) == (0.0, 0, 0.0)


def test_timer_context_manager_and_errors():
    t = Timer(sync=False)
    with t:
        time.sleep(0.005)
    assert t.calls == 1 and t.last >= 0.005
    with pytest.raises(RuntimeError, match="without tick"):
        t.tock()


@pytest.mark.parametrize("sync", [True, False])
def test_sync_called_on_both_ends(monkeypatch: pytest.MonkeyPatch, sync: bool):
    calls: list[str] = []
    monkeypatch.setattr(timing, "synchronize", lambda: calls.append("sync"))
    t = Timer(sync=sync)
    t.tick()
    t.tock()
    assert calls == (["sync", "sync"] if sync else [])


def test_synchronize_runs_on_this_machine():
    timing.synchronize()  # must not raise (CPU, MPS, or CUDA)


def test_profiler_stages():
    prof = Profiler(sync=False)
    for _ in range(2):
        with prof.stage("pre"):
            pass
        with prof.stage("infer") as t:
            time.sleep(0.005)
    assert t is prof["infer"] and "pre" in prof and "post" not in prof
    summary = prof.summary()
    assert list(summary) == ["pre", "infer"]
    assert summary["infer"]["calls"] == 2 and summary["infer"]["total"] >= 0.01


def test_profiler_stage_stops_on_exception():
    prof = Profiler(sync=False)
    with pytest.raises(ZeroDivisionError), prof.stage("boom"):
        _ = 1 / 0
    assert prof["boom"].calls == 1


def test_print_summary():
    from mon.core.console import console
    prof = Profiler(sync=False)
    with prof.stage("infer"):
        pass
    with console.capture() as cap:
        prof.print_summary("Run")
    out = cap.get()
    assert "Run" in out and "infer" in out and "Mean (ms)" in out
