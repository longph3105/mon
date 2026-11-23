"""Tests for ``mon.data.loader``."""

import multiprocessing as mp
import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from mon.core.runtime import seed_everything
from mon.data.loader import LIST_KEYS, collate, create_dataloader
from mon.data.structures import SampleMeta


def sample(i: int, n_boxes: int, h: int = 4, w: int = 6) -> dict:
    boxes = tv_tensors.BoundingBoxes(torch.rand(n_boxes, 4) * 2, format="XYXY", canvas_size=(h, w))
    return {
        "image" : tv_tensors.Image(torch.full((3, h, w), i, dtype=torch.uint8)),
        "mask"  : tv_tensors.Mask(torch.zeros(h, w, dtype=torch.uint8)),
        "boxes" : boxes,
        "labels": torch.arange(n_boxes),
        "meta"  : SampleMeta(index=i, stem=str(i)),
    }


class Synthetic(Dataset):
    """Returns its index and the worker's first draws from each RNG."""

    def __init__(self, n: int = 16) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        info = torch.utils.data.get_worker_info()
        import cv2
        return {
            "index"  : torch.tensor(i),
            "worker" : torch.tensor(-1 if info is None else info.id),
            "np"     : torch.tensor(float(np.random.rand())),
            "py"     : torch.tensor(random.random()),
            "torch"  : torch.rand(()),
            "threads": torch.tensor(cv2.getNumThreads()),
        }


# --- collate ---

def test_collate_stacks_images_and_lists_targets():
    batch = collate([sample(0, 2), sample(1, 0), sample(2, 5)])
    assert batch["image"].shape == (3, 3, 4, 6) and type(batch["image"]) is torch.Tensor
    assert batch["mask"].shape == (3, 4, 6)
    assert [len(b) for b in batch["boxes"]] == [2, 0, 5]
    assert all(isinstance(b, tv_tensors.BoundingBoxes) for b in batch["boxes"])
    assert [len(labels) for labels in batch["labels"]] == [2, 0, 5]
    assert [m.index for m in batch["meta"]] == [0, 1, 2]
    assert batch["image"][2].max() == 2


def test_collate_lists_boxes_under_any_key_and_non_tensors():
    s = [
        {"x": tv_tensors.BoundingBoxes(torch.zeros(k, 4), format="XYXY", canvas_size=(2, 2)), "name": f"n{k}",
         "y": torch.tensor(k)}
        for k in (1, 3)
    ]
    b = collate(s)
    assert [len(v) for v in b["x"]] == [1, 3] and b["name"] == ["n1", "n3"] and b["y"].tolist() == [1, 3]
    assert {"boxes", "labels", "meta"} <= LIST_KEYS


@pytest.mark.parametrize(
    ("samples", "match"),
    [
        ([], "empty batch"),
        ([sample(0, 1), {k: v for k, v in sample(1, 1).items() if k != "mask"}], "has keys"),
        ([sample(0, 1, h=4), sample(1, 1, h=5)], "cannot stack 'image': shapes differ"),
    ],
)
def test_collate_errors(samples, match: str):
    with pytest.raises(ValueError, match=match):
        collate(samples)


# --- create_dataloader: order and seeds ---

def _order(seed: int | None, **kw) -> list[int]:
    dl = create_dataloader(Synthetic(), batch_size=4, shuffle=True, seed=seed, **kw)
    return torch.cat([b["index"] for b in dl]).tolist()


def test_shuffle_is_reproducible_per_seed():
    assert _order(1) == _order(1) and _order(1) != _order(2)
    assert sorted(_order(1)) == list(range(16))


def test_seed_none_follows_seed_everything():
    seed_everything(123)
    a = _order(None)
    seed_everything(123)
    assert _order(None) == a


def test_batch_size_none_yields_single_samples():
    item = next(iter(create_dataloader(Synthetic(3), batch_size=None)))
    assert item["index"].shape == () and int(item["index"]) == 0


def test_defaults_and_kwargs():
    dl = create_dataloader(Synthetic(4), batch_size=2, drop_last=True, prefetch_factor=None)
    assert dl.pin_memory == torch.cuda.is_available() and not dl.persistent_workers and len(dl) == 2
    assert create_dataloader(Synthetic(4), collate_fn=lambda s: s).collate_fn(["x"]) == ["x"]


@pytest.fixture
def fork_ctx():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    return "fork"


def _worker_rows(seed: int, ctx: str) -> list[tuple[int, int, float, float, float, int]]:
    dl = create_dataloader(Synthetic(8), batch_size=None, seed=seed, num_workers=2, multiprocessing_context=ctx,
                           persistent_workers=False)
    return [
        (int(b["index"]), int(b["worker"]), float(b["np"]), float(b["py"]), float(b["torch"]), int(b["threads"]))
        for b in dl
    ]


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")
def test_workers_have_distinct_reproducible_rngs_and_no_cv2_threads(fork_ctx: str):
    rows = _worker_rows(7, fork_ctx)
    by_worker: dict[int, list] = {}
    for r in rows:
        by_worker.setdefault(r[1], []).append(r)
    assert set(by_worker) == {0, 1}
    first0, first1 = by_worker[0][0], by_worker[1][0]
    assert first0[2] != first1[2] and first0[3] != first1[3] and first0[4] != first1[4]  # numpy, random, torch differ
    assert _worker_rows(7, fork_ctx) == rows                                           # same seed → same draws
    assert _worker_rows(8, fork_ctx) != rows
    assert all(r[5] <= 1 for r in rows)  # cv2.setNumThreads(0) ran in each worker
