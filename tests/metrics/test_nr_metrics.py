"""Tests for ``mon.metrics.brisque``, ``mon.metrics.nrqm``, ``mon.metrics.pi``: parity with pyiqa, helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from mon.core.path import datasets_dir
from mon.metrics import BRISQUE, METRICS, NIQE, NRQM, PI, rgb_to_y
from mon.metrics.nrqm import _dct_matrix, _forest_predict, _ggd_grid, _nearest, convert_nrqm_model

_spec = importlib.util.spec_from_file_location("metrics_reference", Path(__file__).with_name("metrics_reference.py"))
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

# pyiqa 0.1.16 scores: output of ``nr_metrics_reference.py``.
# Generated with pyiqa 0.1.16, torch 2.15.0+cu132; lol_v1 order: 111 image, 111 target, 146 image, 146 target, 179 image, 179 target.
NR_REFERENCE: dict = {'brisque': {'synthetic': [153.59100341796875, 153.59100341796875, 153.59100341796875, 153.59100341796875],
             'lol_v1': [19.48944091796875,
                        17.42010498046875,
                        -1.69073486328125,
                        -1.68878173828125,
                        -1.28546142578125,
                        4.92889404296875]},
 'nrqm': {'synthetic': [4.242570400238037, 4.311669826507568, 4.179855823516846, 4.193848609924316],
          'lol_v1': [3.4780619144439697,
                     8.00686264038086,
                     3.77205228805542,
                     8.913325309753418,
                     4.262078762054443,
                     8.336816787719727]},
 'pi': {'synthetic': [265.2607016428063, 272.7452334093147, 256.5809445634442, 259.00948510883586],
        'lol_v1': [7.638614857498662,
                   3.7283040238831133,
                   6.508076746014963,
                   3.3691607858779657,
                   5.726240464187046,
                   2.9716416338677494]}}

# Parity tolerances (absolute). BRISQUE: pyiqa runs in float32, and the GGD shapes are quantized to a 0.001 grid, so
# rounding flips move scores by up to ~0.07 (our own float32 run differs from our float64 one as much); NRQM: forest
# splits near a threshold; PI: NIQE's float32 steps in pyiqa (0.01% of ~260 on the synthetic noise).
TOL = {"brisque": 0.1, "nrqm": 2e-3, "pi": 2e-3}
TOL_SYNTHETIC_PI = 0.05


def _metric(name: str, device: torch.device):
    try:
        return METRICS.build(name).to(device)
    except FileNotFoundError as err:
        pytest.skip(f"metric model not available: {err}")


def _lol_batch() -> torch.Tensor:
    try:
        root = datasets_dir() / "lol_v1" / "test"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    from mon.data.io import read_image

    return torch.stack([read_image(root / sub / f"{stem}.png").float() / 255
                        for stem in _ref.REAL_STEMS for sub in ("images", "targets")])


# --- Parity with pyiqa ---

@pytest.mark.parametrize("name", ["brisque", "nrqm", "pi"])
def test_parity_synthetic(name: str, regression_device: torch.device):
    metric = _metric(name, regression_device)
    pred, target = _ref.reference_images()
    got = metric(torch.cat([pred, target]).to(regression_device)).cpu()
    tol = TOL_SYNTHETIC_PI if name == "pi" else TOL[name]
    assert got.dtype == torch.float64 and got.tolist() == pytest.approx(NR_REFERENCE[name]["synthetic"], abs=tol)


@pytest.mark.parametrize("name", ["brisque", "nrqm", "pi"])
def test_parity_on_lol_v1(name: str, regression_device: torch.device):
    batch = _lol_batch()
    metric = _metric(name, regression_device)
    got = metric(batch.to(regression_device)).cpu().tolist()
    assert got == pytest.approx(NR_REFERENCE[name]["lol_v1"], abs=TOL[name])


def test_directions_on_lol_v1(regression_device: torch.device):
    batch = _lol_batch()  # dark inputs vs their well-exposed targets
    nrqm = _metric("nrqm", regression_device)(batch.to(regression_device)).cpu()
    assert (nrqm[1::2] > nrqm[0::2] + 3).all()          # higher is better
    pi = _metric("pi", regression_device)(batch.to(regression_device)).cpu()
    assert (pi[1::2] < pi[0::2]).all()                  # lower is better


# --- Helpers ---

def test_nearest_matches_dense_argmin():
    grid, ratio = _ggd_grid(0.03, torch.device("cpu"))
    assert (ratio[1:] < ratio[:-1]).all()               # strictly decreasing, as _nearest needs
    values = torch.cat([torch.rand(2000, dtype=torch.float64) * 30 + 1, ratio[::97], torch.tensor([0.5, 1e6])])
    dense = (values[:, None] - ratio[None, :]).abs().argmin(dim=1)
    assert torch.equal(_nearest(ratio, values), dense)


def test_dct_matrix_is_orthonormal_dct2():
    c = _dct_matrix(7)
    assert torch.allclose(c @ c.T, torch.eye(7, dtype=torch.float64), atol=1e-12)
    x = torch.zeros(7, 7, dtype=torch.float64)
    x[...] = 1
    assert (c @ x @ c.T)[0, 0].item() == pytest.approx(7.0)  # DC of a constant block = n · value (orthonormal)


def _tree_loop(x: np.ndarray, ldau, rdau, upper, avnode, mbest) -> float:
    """Reference traversal of one tree (randomForest-MATLAB ``regRF_predict``)."""
    k = 0
    while ldau[k] != 0:
        k = (ldau[k] if x[mbest[k] - 1] <= upper[k] else rdau[k]) - 1
    return float(avnode[k])


def test_forest_predict_matches_tree_loop():
    rng = np.random.default_rng(0)
    trees, nodes, dim = 7, 15, 4
    ldau = np.zeros((nodes, trees), np.int64)
    rdau = np.zeros_like(ldau)
    for t in range(trees):  # complete binary trees of depth 3 (1-based children; leaves have 0)
        for k in range(7):
            ldau[k, t], rdau[k, t] = 2 * k + 2, 2 * k + 3
    upper  = rng.random((nodes, trees))
    avnode = rng.random((nodes, trees))
    mbest  = rng.integers(1, dim + 1, (nodes, trees)) * (ldau != 0)
    x = rng.random((5, dim))
    expected = [np.mean([_tree_loop(xi, ldau[:, t], rdau[:, t], upper[:, t], avnode[:, t], mbest[:, t])
                         for t in range(trees)]) for xi in x]
    forest = [torch.from_numpy(a) for a in (ldau, rdau)] + [torch.from_numpy(a) for a in (upper, avnode)] \
        + [torch.from_numpy(mbest)]
    assert _forest_predict(torch.from_numpy(x), forest).tolist() == pytest.approx(expected, abs=1e-15)


def test_rgb_to_y_and_color_space_option():
    x = torch.rand(1, 3, 4, 4, dtype=torch.float64)
    y = rgb_to_y(x, "ycbcr")
    assert torch.allclose(y * 255, 16 + (x * torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)).sum(1, True))
    assert torch.equal(rgb_to_y(x, "yiq"), rgb_to_y(x))
    with pytest.raises(ValueError, match="color_space"):
        rgb_to_y(x, "lab")  # type: ignore[arg-type]
    try:
        niqe = NIQE(crop_border=4, color_space="ycbcr")
    except FileNotFoundError as err:
        pytest.skip(f"NIQE model not available: {err}")
    assert "color_space='ycbcr'" in repr(niqe)


def test_nrqm_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="python -m mon.metrics.nrqm convert"):
        NRQM(weights=str(tmp_path / "missing.npz"))
    bad = tmp_path / "model.mat"
    bad.write_bytes(b"not the model")
    with pytest.raises(ValueError, match="SHA-256"):
        convert_nrqm_model(bad, tmp_path / "out.npz")
    try:
        nrqm = NRQM()
    except FileNotFoundError as err:
        pytest.skip(f"NRQM model not available: {err}")
    with pytest.raises(ValueError, match="H, W >= 70"):
        nrqm(torch.rand(1, 3, 64, 96))


def test_registry_and_build():
    assert METRICS.names(mode="nr") == ["brisque", "niqe", "nrqm", "pi"]
    assert METRICS.entry("nrqm").meta["higher_better"] and not METRICS.entry("pi").meta["higher_better"]
    assert all(cls.mode == "nr" for cls in (BRISQUE, NRQM, PI))
