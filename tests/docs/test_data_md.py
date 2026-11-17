"""Runs every Python block of ``docs/data.md`` in order, against a synthetic ``lol_v1`` folder."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path as StdPath

import pytest
import torch

from mon.data.io import write_image

DOC = StdPath(__file__).resolve().parents[2] / "docs" / "data.md"


def _python_blocks(text: str) -> list[str]:
    return [textwrap.dedent(b) for b in re.findall(r"^[ \t]*```python\n(.*?)^[ \t]*```", text, re.S | re.M)]


def test_doc_has_blocks():
    assert len(_python_blocks(DOC.read_text())) >= 5


@pytest.mark.slow
def test_doc_blocks_run(tmp_path: StdPath):
    root = tmp_path / "datasets" / "lol_v1"
    for split, n in (("train", 3), ("test", 2)):
        for i in range(n):
            for sub in ("images", "targets"):
                write_image(torch.randint(0, 256, (3, 300, 320), dtype=torch.uint8), root / split / sub / f"{i}.png")
    script = tmp_path / "doc.py"
    script.write_text("\n\n".join(_python_blocks(DOC.read_text())))
    env = {k: v for k, v in os.environ.items() if k not in ("DATA_DIR", "ZOO_DIR")}
    env["DATASETS_DIR"] = str(tmp_path / "datasets")
    # Subprocess: the doc registers a dataset, which must not leak into this test session.
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
