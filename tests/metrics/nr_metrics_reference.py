"""Reference scores for the BRISQUE, NRQM, and PI parity tests (no ``mon`` imports).

``NR_REFERENCE`` in ``test_nr_metrics.py`` comes from pyiqa 0.1.16 (https://github.com/chaofengc/IQA-PyTorch,
PolyForm Noncommercial: a reference only, never a dependency): ``brisque``, ``nrqm``, ``pi``, run on the inputs of
``metrics_reference.py`` (the synthetic ``pred``/``target`` batch, and the LOL-v1 test pairs ``REAL_STEMS``).

Running it executes downloaded third-party code: get the owner's go-ahead first. pyiqa downloads its model files
into ``TORCH_HOME`` (its BRISQUE SVR equals piq's release; its NRQM model equals the official ``model.mat``); point
it at a scratch directory. Regenerate with:

    TORCH_HOME=<scratch> uv run --no-project --python 3.12 --with pyiqa==0.1.16 --with opencv-python-headless \\
        python -I tests/metrics/nr_metrics_reference.py <DATASETS_DIR>/lol_v1
"""

import importlib.util
import pprint
import sys
from pathlib import Path
from typing import Any

import torch

METRICS = ("brisque", "nrqm", "pi")


def _reference_module() -> Any:
    spec = importlib.util.spec_from_file_location("metrics_reference", Path(__file__).with_name("metrics_reference.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(lol_v1: str) -> None:
    import cv2
    import pyiqa

    ref = _reference_module()
    pred, target = ref.reference_images()
    real = []
    for stem in ref.REAL_STEMS:
        for sub in ("images", "targets"):
            bgr = cv2.imread(f"{lol_v1}/test/{sub}/{stem}.png", cv2.IMREAD_COLOR)
            real.append(torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float() / 255)
    real_batch = torch.stack(real)  # [stem0 image, stem0 target, stem1 image, ...]

    result: dict[str, Any] = {}
    for name in METRICS:
        metric = pyiqa.create_metric(name, device="cpu", as_loss=False)
        result[name] = {
            "synthetic": [metric(x[None]).item() for x in torch.cat([pred, target])],
            "lol_v1"   : [metric(x[None]).item() for x in real_batch],
        }
    print(f"# Generated with pyiqa {pyiqa.__version__}, torch {torch.__version__}; lol_v1 order: "
          f"{', '.join(f'{s} image, {s} target' for s in ref.REAL_STEMS)}.")
    print("NR_REFERENCE: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
