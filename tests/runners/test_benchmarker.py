"""Tests for ``mon.runners.benchmarker`` and ``mon benchmark``."""

import pytest

from mon.core.fileio import load_yaml
from mon.runners.benchmarker import BenchmarkConfig, benchmark
from mon.runners.cli import main


def test_benchmark_zero_dce():
    r = benchmark(BenchmarkConfig(model="zero_dce", size=[32, 48], device="cpu", latency=False, seed=0))
    assert r["params"] == 79_416 and r["macs"] == 9 * 8800 * 32 * 48 and r["input_shapes"] == [[1, 3, 32, 48]]
    assert r["latency_ms"] is None and r["config"]["seed"] == 0 and r["config"]["device"] == "cpu"


def test_benchmark_rounds_to_input_multiple_and_stacks():
    r = benchmark({"model": "dccnet", "size": [30], "device": "cpu", "latency": False})
    assert r["input_shapes"] == [[1, 3, 32, 32]]  # DCCNet input_multiple = 8
    r = benchmark({"model": "mertens", "size": [16, 16], "stack": 4, "batch_size": 2, "device": "cpu",
                   "latency": False})
    assert r["input_shapes"] == [[2, 4, 3, 16, 16]] and r["params"] == 0


def test_benchmark_validates():
    with pytest.raises(ValueError, match="model is required"):
        benchmark({})
    with pytest.raises(ValueError, match="size"):
        benchmark({"model": "zero_dce", "size": [1, 2, 3]})


def test_cli_writes_output(tmp_path):
    out  = tmp_path / "zero_dce.yaml"
    args = ["benchmark", "model=zero_dce", "size=[16,16]", "device=cpu", "warmup=0", "runs=2", f"output={out}"]
    assert main(args) == 0
    data = load_yaml(out)
    assert data["params"] == 79_416 and len(data["input_shapes"]) == 1 and data["latency_ms"] > 0
    assert main(args) == 2  # exists, no exist_ok
    assert main([*args, "exist_ok=true"]) == 0
