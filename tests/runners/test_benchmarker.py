"""Tests for ``mon.runners.benchmarker`` and ``mon benchmark``."""

import pytest
import torch

from mon.data.io import write_image

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


def _ladder(root, sizes=((40, 24), (64, 64), (24, 40))):
    gen = torch.Generator().manual_seed(0)
    for w, h in sizes:
        for i in range(2):
            write_image(torch.randint(0, 80, (3, h, w), dtype=torch.uint8, generator=gen), root / f"{w}x{h}" / f"{i}.png")
    return root


def test_benchmark_data_groups_by_resolution(tmp_path):
    out = tmp_path / "res.yaml"
    r = benchmark({"model": "dccnet", "data": str(_ladder(tmp_path / "imgs")), "device": "cpu", "warmup": 0,
                   "runs": 1, "seed": 0, "output": str(out)})
    assert [row["size"] for row in r["rows"]] == ["40x24", "24x40", "64x64"]       # by pixel count
    assert all(row["status"] == "ok" and row["images"] == 2 for row in r["rows"])
    assert r["rows"][0]["input"] == [1, 3, 24, 40] and r["rows"][2]["macs"] > r["rows"][0]["macs"] > 0
    assert r["params"] == 13_188_724 and r["rows"][0]["latency_ms"] > 0 and r["rows"][0]["peak_memory"] is None
    assert load_yaml(out)["rows"] == r["rows"]
    assert (tmp_path / "res.csv").read_text().splitlines()[0].startswith("size,images,macs")


def test_benchmark_data_sizes_filter_and_errors(tmp_path, monkeypatch):
    imgs = _ladder(tmp_path / "imgs")
    r = benchmark({"model": "zero_dce", "data": str(imgs), "device": "cpu", "latency": False,
                   "sizes": ["64x64"]})
    assert [row["size"] for row in r["rows"]] == ["64x64"]
    with pytest.raises(ValueError, match="no images"):
        benchmark({"model": "zero_dce", "data": str(imgs), "device": "cpu", "sizes": ["1x1"]})
    with pytest.raises(FileNotFoundError, match="neither"):
        benchmark({"model": "zero_dce", "data": "no_such_dataset", "device": "cpu"})
    with pytest.raises(ValueError, match="exposure stacks"):
        benchmark({"model": "mertens", "data": str(imgs), "device": "cpu"})

    import mon.runners.benchmarker as bm

    real = bm._measure_size

    def flaky(model, images, config, multiple, device):
        if images[0].shape[-1] == 64:
            raise torch.OutOfMemoryError("CUDA out of memory")
        if images[0].shape[-1] == 24:
            raise RuntimeError("input too small\nmore")
        return real(model, images, config, multiple, device)

    monkeypatch.setattr(bm, "_measure_size", flaky)
    r = benchmark({"model": "zero_dce", "data": str(imgs), "device": "cpu", "latency": False})
    assert [row["status"] for row in r["rows"]] == ["ok", "error: input too small", "oom"]
