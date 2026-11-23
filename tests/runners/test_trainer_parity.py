"""Training-loop parity with official scripts: Zero-DCE and Zero-DCE++ ``lowlight_train.py`` (see
``zero_dce_train_reference.py``, ``zero_dcepp_train_reference.py``), SCI ``CVPR/train.py``
(``sci_train_reference.py``), Zero-IG ``train.py`` (``zero_ig_train_reference.py``) and SCI++
``TPAMI/train.py`` (reference loop below).

Same seeded init, same 8-bit images in file order, the recipe's optimizer and clipping, on CPU: the
runner must reproduce the official per-epoch losses and final weights to float32 rounding (SCI: to the
tolerance its batch norms allow; see its test).
"""

import csv
import importlib.util
from pathlib import Path as StdPath

import pytest
import torch

from mon.data.io import write_image
from mon.runners.trainer import train

_spec = importlib.util.spec_from_file_location(
    "zero_dce_train_reference", StdPath(__file__).with_name("zero_dce_train_reference.py"),
)
assert _spec is not None and _spec.loader is not None
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

# Generated from the official Zero-DCE loop with torch 2.15.0+cu132.
OFFICIAL: dict = {
    "init": {
        "sum": -1.8475460590249462,
        "sumsq": 32.38603868190064,
        "values": [
            0.010812884196639061,
            -0.014962579123675823,
            0.0007421610062010586,
            0.018815496936440468,
            -0.001214824616909027,
            0.023221846669912338,
            -0.0062813907861709595,
        ],
    },
    "losses": [
        2.7027347087860107,
        2.686758518218994,
        2.65739369392395,
        2.642648220062256,
        2.6146130561828613,
        2.600914716720581,
    ],
    "epoch_mean": [2.6947466135025024, 2.650020956993103, 2.607763886451721],
    "final": {
        "sum": -2.3131460917100526,
        "sumsq": 31.3502140374019,
        "values": [
            0.01041758805513382,
            -0.015553772449493408,
            0.00023975536169018596,
            0.018217317759990692,
            -0.001798209617845714,
            0.022624755278229713,
            -0.006883133668452501,
        ],
    },
}


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


# --- Zero-DCE++ (official loop: zero_dcepp_train_reference.py) ---

# Generated from the official Zero-DCE++ loop with torch 2.15.0+cu132.
OFFICIAL_PP: dict = {
    "init": {
        "sum": -9.990068064325897,
        "sumsq": 176.2159106454262,
        "values": [
            -0.0024956068955361843,
            0.2642965018749237,
            0.389028936624527,
            -0.12805600464344025,
            -0.015029081143438816,
            -0.07392232865095139,
            0.09970849752426147,
        ],
    },
    "losses": [
        3.221865177154541,
        3.257899522781372,
        3.1870546340942383,
        3.2229838371276855,
        3.1528141498565674,
        3.1886701583862305,
    ],
    "epoch_mean": [3.2398823499679565, 3.205019235610962, 3.170742154121399],
    "final": {
        "sum": -9.971590512165676,
        "sumsq": 175.38909743437026,
        "values": [
            -0.003065943717956543,
            0.2636968493461609,
            0.38842910528182983,
            -0.1274656057357788,
            -0.014430544339120388,
            -0.07332700490951538,
            0.09910719841718674,
        ],
    },
}


def test_zero_dcepp_loop_matches_official(tmp_path: StdPath):
    for n, img in enumerate(_ref.reference_images()):
        write_image(img, tmp_path / "train" / f"{n:02d}.png")
    run = train({"model": "zero_dce++", "data": str(tmp_path / "train"), "resize": None, "shuffle": False,
                 "epochs": _ref.EPOCHS, "batch_size": _ref.BATCH, "seed": _ref.SEED, "device": "cpu",
                 "num_workers": 0, "run_dir": str(tmp_path / "run")})
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert [float(r["train/loss"]) for r in rows] == pytest.approx(OFFICIAL_PP["epoch_mean"], rel=1e-6)
    final = _ref.weight_stats(torch.load(run / "last.ckpt", weights_only=True)["model"])
    assert final["sum"] == pytest.approx(OFFICIAL_PP["final"]["sum"], rel=1e-5)
    assert final["sumsq"] == pytest.approx(OFFICIAL_PP["final"]["sumsq"], rel=1e-6)
    assert final["values"] == pytest.approx(OFFICIAL_PP["final"]["values"], abs=1e-7)


def test_zero_dcepp_default_init_matches_official():
    from mon.models.lle.zero_dce import ZeroDCEPP

    torch.manual_seed(_ref.SEED)
    init = _ref.weight_stats(ZeroDCEPP(scale_factor=1.0).state_dict())
    assert init["sum"] == OFFICIAL_PP["init"]["sum"] and init["values"] == OFFICIAL_PP["init"]["values"]


# --- SCI (official loop: sci_train_reference.py) ---

_sci_spec = importlib.util.spec_from_file_location(
    "sci_train_reference", StdPath(__file__).with_name("sci_train_reference.py"),
)
assert _sci_spec is not None and _sci_spec.loader is not None
_sci_ref = importlib.util.module_from_spec(_sci_spec)
_sci_spec.loader.exec_module(_sci_ref)

# Generated from the official SCI (CVPR) loop with torch 2.15.0+cu132.
OFFICIAL_SCI: dict = {
    "init": {
        "sum": 292.5374370350828,
        "sumsq": 309.3634275778634,
        "values": [
            0.006533919833600521,
            -0.000269236508756876,
            0.005544594023376703,
            -0.03960715979337692,
            0.048333972692489624,
            -0.032320648431777954,
            0.0,
        ],
    },
    "epoch_mean": [23.04537467956543, 16.39866943359375, 12.457106018066407],
    "final": {
        "sum": 1354.3013906779925,
        "sumsq": 152037.16930079783,
        "values": [
            0.005930294282734394,
            -0.00370827317237854,
            0.0010434042196720839,
            -0.042572785168886185,
            0.04871005937457085,
            -0.03295058384537697,
            -0.004436877090483904,
        ],
    },
}


def test_sci_loop_matches_official(tmp_path: StdPath):
    for n, img in enumerate(_sci_ref.reference_images()):
        write_image(img, tmp_path / "train" / f"{n:02d}.png")
    run = train({"model": "sci", "data": str(tmp_path / "train"), "shuffle": False, "epochs": _sci_ref.EPOCHS,
                 "device": "cpu", "num_workers": 0, "run_dir": str(tmp_path / "run")})  # recipe: seed 2, batch 1
    # Not float32-exact, unlike Zero-DCE: SCI's convolutions before a batch norm have biases with zero true
    # gradient, and Adam (eps 1e-8) turns their ~1e-7 rounding noise into ~lr steps, so ulp-level differences
    # between mon's network and the official one (same init, same first-step loss) grow. Measured 2026-11-23:
    # losses within 2.8e-4 (relative), weight sum 4.6e-5, values 9e-5. A recipe error is far outside these:
    # the init that re-drew the shared blocks gave losses 20% off.
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert [float(r["train/loss"]) for r in rows] == pytest.approx(OFFICIAL_SCI["epoch_mean"], rel=1e-3)
    final = _ref.weight_stats(torch.load(run / "last.ckpt", weights_only=True)["model"])
    assert final["sum"] == pytest.approx(OFFICIAL_SCI["final"]["sum"], rel=2e-4)
    assert final["sumsq"] == pytest.approx(OFFICIAL_SCI["final"]["sumsq"], rel=1e-6)
    assert final["values"] == pytest.approx(OFFICIAL_SCI["final"]["values"], abs=5e-4)


def test_sci_init_matches_official():
    from mon.models.lle.sci import SCI

    torch.manual_seed(_sci_ref.SEED)
    init = _ref.weight_stats(SCI().state_dict())
    assert init["sum"] == OFFICIAL_SCI["init"]["sum"] and init["values"] == OFFICIAL_SCI["init"]["values"]


# --- Zero-IG (official loop: zero_ig_train_reference.py; two epochs, the second in eval mode) ---

_zig_spec = importlib.util.spec_from_file_location("zero_ig_train_reference",
                                                   StdPath(__file__).with_name("zero_ig_train_reference.py"))
assert _zig_spec is not None and _zig_spec.loader is not None
_zig_ref = importlib.util.module_from_spec(_zig_spec)
_zig_spec.loader.exec_module(_zig_ref)

# Generated from the official Zero-IG loop with torch 2.15.0+cu132.
OFFICIAL_ZIG: dict = {
    "init": {
        "sum": 508.3129757771328,
        "sumsq": 641.4822872528225,
        "values": [
            -0.006652656476944685,
            0.021365534514188766,
            0.04261379316449165,
            -0.017105618491768837,
            -0.006231194827705622,
            -0.0014583473093807697,
            0.07257840037345886,
        ],
    },
    "losses": [
        1178.3555908203125,
        924.8494262695312,
        745.25830078125,
        643.4937133789062,
        412.1998596191406,
        366.7341613769531,
        335.34832763671875,
        324.8885498046875,
    ],
    "epoch_mean": [872.9892578125, 359.792724609375],
    "final": {
        "sum": 396.2874362687917,
        "sumsq": 987.5439622086118,
        "values": [
            -0.006967898458242416,
            0.021584337577223778,
            0.044639524072408676,
            -0.016623731702566147,
            -0.006514031905680895,
            -0.003357286099344492,
            0.07028605788946152,
        ],
    },
}


def test_zero_ig_loop_matches_official(tmp_path: StdPath):
    for n, img in enumerate(_zig_ref.reference_images()):
        write_image(img, tmp_path / "train" / f"{n:02d}.png")
    run = train({"model": "zero_ig", "data": str(tmp_path / "train"), "epochs": _zig_ref.EPOCHS, "device": "cpu",
                 "num_workers": 0, "run_dir": str(tmp_path / "run")})  # recipe: seed 2, batch 1, file order
    # The enhancer's batch norm makes the same noise as SCI's (see there), smaller over 4 steps; in eval mode
    # (epoch 2) its running statistics amplify it, though the eval-mode loss itself matches the official one
    # exactly (test_zero_ig.py). Measured 2026-11-24: epoch-1 loss within 2e-8 (relative), epoch 2 within
    # 3.9e-5; weight sum 3.0e-5, sumsq 1.8e-6, values 3.1e-6.
    rows = list(csv.DictReader((run / "log.csv").open()))
    first, second = (float(r["train/loss"]) for r in rows)
    assert first == pytest.approx(OFFICIAL_ZIG["epoch_mean"][0], rel=1e-6)
    assert second == pytest.approx(OFFICIAL_ZIG["epoch_mean"][1], rel=1e-4)
    final = _ref.weight_stats(torch.load(run / "last.ckpt", weights_only=True)["model"])
    assert final["sum"] == pytest.approx(OFFICIAL_ZIG["final"]["sum"], rel=1e-4)
    assert final["sumsq"] == pytest.approx(OFFICIAL_ZIG["final"]["sumsq"], rel=2e-5)
    assert final["values"] == pytest.approx(OFFICIAL_ZIG["final"]["values"], abs=1e-5)


def test_zero_ig_init_matches_official():
    from mon.models.lle.zero_ig import ZeroIG, _init_enhancer

    torch.manual_seed(_zig_ref.SEED)
    model = ZeroIG()
    _init_enhancer(model)
    init = _ref.weight_stats(model.state_dict())
    assert init["sum"] == OFFICIAL_ZIG["init"]["sum"] and init["values"] == OFFICIAL_ZIG["init"]["values"]


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
