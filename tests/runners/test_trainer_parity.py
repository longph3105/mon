"""Training-loop parity with official scripts: Zero-DCE ``lowlight_train.py`` (see ``zero_dce_train_reference.py``)
and SCI++ ``TPAMI/train.py`` (reference loop below).

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


# --- SCI++ (two alternating optimizers) ---

def _official_scipp_loop(images: list[torch.Tensor], epochs: int, seed: int) -> dict[str, torch.Tensor]:
    """The official ``TPAMI/train.py`` loop (vis-opt-group/SCI ``f6f88fd``), statement for statement, on CPU.

    Kept: ``Variable(input, requires_grad=True)``, the ``requires_grad`` toggles *after* the forward pass,
    ``zero_grad`` / ``backward`` / ``step`` per branch, and ``clip_grad_norm_`` after the step. The network
    and loss are ``mon``'s ``SCIPP`` / ``SCILoss``, whose equivalence to the official ``Network`` /
    ``LossFunction`` is covered by the model tests; this checks the training loop.
    """
    from torch import nn
    from torch.nn import functional as F

    from mon.models.lle.sci import SCILoss, SCIPP

    torch.manual_seed(seed)
    model = SCIPP(stages=3)
    criterion = SCILoss()
    optimizer_a = torch.optim.Adam(model.ha.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=3e-4)
    optimizer_b = torch.optim.Adam(list(model.hb.parameters()) + list(model.calibrate.parameters()), lr=3e-4,
                                   betas=(0.9, 0.999), weight_decay=3e-4)
    total_step = 0
    for _ in range(epochs):
        model.train()
        for image in images:
            total_step += 1
            x = (image.float() / 255).unsqueeze(0).requires_grad_(True)
            stages = model.forward_stages(x)
            i_list, in_list = stages["illuminations"], stages["inputs"]
            loss1 = criterion(in_list[0], i_list[0])
            loss2 = F.l1_loss(i_list[0], i_list[1]) + 0.1 * criterion(in_list[0], i_list[1])
            loss3 = F.l1_loss(i_list[0], i_list[2]) + 0.1 * criterion(in_list[0], i_list[2])
            if total_step % 10 < 7:
                for p in model.ha.parameters():
                    p.requires_grad = True
                for p in [*model.hb.parameters(), *model.calibrate.parameters()]:
                    p.requires_grad = False
                optimizer_a.zero_grad()
                loss1.backward()
                optimizer_a.step()
            else:
                for p in model.ha.parameters():
                    p.requires_grad = False
                for p in [*model.hb.parameters(), *model.calibrate.parameters()]:
                    p.requires_grad = True
                optimizer_b.zero_grad()
                (loss2 + loss3).backward()
                optimizer_b.step()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
    return model.state_dict()


def test_scipp_loop_matches_official(tmp_path: StdPath):
    images = _ref.reference_images()[:5]
    for n, img in enumerate(images):
        write_image(img, tmp_path / "train" / f"{n:02d}.png")
    epochs = 5  # 25 steps: every phase boundary of the alternation, twice
    official = _official_scipp_loop(images, epochs, seed=2)
    run = train({"model": "sci++", "data": str(tmp_path / "train"), "epochs": epochs, "device": "cpu",
                 "num_workers": 0, "run_dir": str(tmp_path / "run")})
    ours = torch.load(run / "last.ckpt", weights_only=True)["model"]
    assert ours.keys() == official.keys()
    # float32 rounding (the official graph also differentiates the input) stays under 1e-6; running the
    # literal 7/3 schedule instead of its effective 6/2 one is off by ~2.6e-3.
    for k, v in official.items():
        assert torch.allclose(ours[k], v, rtol=0, atol=5e-6), (k, (ours[k] - v).abs().max().item())
    init  = _scipp_init()
    moved = [k for k in official if not torch.equal(official[k], init[k])]
    assert any(k.startswith("ha.") for k in moved) and any(k.startswith("hb.") for k in moved)


def _scipp_init() -> dict[str, torch.Tensor]:
    from mon.models.lle.sci import SCIPP

    torch.manual_seed(2)
    return SCIPP(stages=3).state_dict()
