"""Zero-DCE training-loop parity with the official ``lowlight_train.py`` (see ``zero_dce_train_reference.py``).

Same seeded init, same 8-bit images in file order, the recipe's optimizer and clipping, on CPU: the
runner must reproduce the official per-epoch losses and final weights to float32 rounding.
"""

import csv
import importlib.util
from pathlib import Path as StdPath

import pytest
import torch

from mon.data.io import write_image
from mon.runners.trainer import train

_spec = importlib.util.spec_from_file_location("zero_dce_train_reference", StdPath(__file__).with_name("zero_dce_train_reference.py"))
assert _spec is not None and _spec.loader is not None
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

# Generated from the official Zero-DCE loop with torch 2.15.0+cu132.
OFFICIAL: dict = {'init': {'sum': -1.8475460590249462,
          'sumsq': 32.38603868190064,
          'values': [0.010812884196639061,
                     -0.014962579123675823,
                     0.0007421610062010586,
                     0.018815496936440468,
                     -0.001214824616909027,
                     0.023221846669912338,
                     -0.0062813907861709595]},
 'losses': [2.7027347087860107,
            2.686758518218994,
            2.65739369392395,
            2.642648220062256,
            2.6146130561828613,
            2.600914716720581],
 'epoch_mean': [2.6947466135025024, 2.650020956993103, 2.607763886451721],
 'final': {'sum': -2.3131460917100526,
           'sumsq': 31.3502140374019,
           'values': [0.01041758805513382,
                      -0.015553772449493408,
                      0.00023975536169018596,
                      0.018217317759990692,
                      -0.001798209617845714,
                      0.022624755278229713,
                      -0.006883133668452501]}}


def test_zero_dce_loop_matches_official(tmp_path: StdPath):
    for n, img in enumerate(_ref.reference_images()):
        write_image(img, tmp_path / "train" / f"{n:02d}.png")
    run = train({"model": "zero_dce", "data": str(tmp_path / "train"), "resize": None, "shuffle": False,
                 "epochs": _ref.EPOCHS, "batch_size": _ref.BATCH, "seed": _ref.SEED, "device": "cpu",
                 "num_workers": 0, "run_dir": str(tmp_path / "run")})
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert [float(r["train/loss"]) for r in rows] == pytest.approx(OFFICIAL["epoch_mean"], rel=1e-6)
    final = _ref.weight_stats(torch.load(run / "last.ckpt", weights_only=True)["model"])
    assert final["sum"] == pytest.approx(OFFICIAL["final"]["sum"], rel=1e-5)
    assert final["sumsq"] == pytest.approx(OFFICIAL["final"]["sumsq"], rel=1e-6)
    assert final["values"] == pytest.approx(OFFICIAL["final"]["values"], abs=1e-7)


def test_zero_dce_init_matches_official():
    from mon.models.lle.zero_dce import ZeroDCE

    torch.manual_seed(_ref.SEED)
    init = _ref.weight_stats(ZeroDCE().state_dict())
    assert init["sum"] == OFFICIAL["init"]["sum"] and init["values"] == OFFICIAL["init"]["values"]
