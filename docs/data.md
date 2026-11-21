# Data: datasets, transforms, loaders

How `mon.data` turns files on disk into training batches and how predictions get
back to image size. API details live in the docstrings (`help(mon.data)`); this
guide shows how the pieces fit together. Every Python block below runs in the
test suite (`tests/docs/test_data_md.py`), top to bottom, in one interpreter.

## 1. Where the data lives

`mon` never hardcodes data paths. It reads environment variables:

| Variable       | Meaning                               | Default              |
|----------------|---------------------------------------|----------------------|
| `DATA_DIR`     | Data root                             | none (required)      |
| `DATASETS_DIR` | Extracted datasets, one folder per id | `$DATA_DIR/datasets` |
| `ZOO_DIR`      | Pretrained weights                    | `$DATA_DIR/zoo`      |

Check them with `uv run python -m mon.core.path`.

Each dataset has one **id**, e.g. `lol_v1`: its folder (`$DATASETS_DIR/lol_v1/`)
and its registry name are the same string. Datasets are stored in a normalized
layout per task family. Paired restoration datasets (low-light, denoising, …)
use `<split>/images/` for inputs and `<split>/targets/` for ground truth, paired
by file name. Multi-exposure datasets (`sice`) keep one folder of exposures per
scene and one target per scene; a dataset maps each input to its target's name
with `FolderDataset.pair_key`. Unpaired test sets (`unpaired_layout`:
`<split>/images/` only) are bundled under one id, `unpaired_lol/<set>/`, and
registered per set (`dicm`, `fusion`, `lime`, `mef`, `npe`, `vv`), since papers
report them separately. The workspace `data/README.md` defines ids and layouts
and lists every dataset.

## 2. Loading a registered dataset

`DATASETS` maps ids to dataset classes. The first lookup imports `mon.data.zoo`,
which registers them.

```python
import torch
from torchvision.transforms import v2

from mon.core.enums import Split, Task
from mon.data import DATASETS

DATASETS.names(tasks=Task.LLE)           # ['lol_v1', 'lol_v2_real', …]
ds = DATASETS.build("lol_v1", "test")    # same as LOLv1("test")
sample = ds[0]
sorted(sample)                           # ['image', 'meta', 'target']
sample["image"].shape, sample["image"].dtype   # (torch.Size([3, H, W]), torch.uint8)
sample["meta"].stem, sample["meta"].orig_size  # file stem, Size(height=H, width=W)
```

A sample is a plain dict:

- Image-like values are `torchvision.tv_tensors` (`Image`, `Mask`), as stored on
  disk. 8-bit files load as `uint8`, 16-bit as `uint16`; nothing is scaled yet.
- `meta` is a `SampleMeta`: index, stem, split, source paths, and the size
  before transforms.

Pass `root=` to load a copy outside `DATASETS_DIR`; `on_missing="skip"` drops
samples with a missing file instead of raising.

## 3. Transforms

Transforms are `torchvision.transforms.v2`, applied to the whole sample dict.
Run them in two stages:

1. **Geometric** (crop, flip, resize) on the whole sample, so input and target
   stay aligned.
2. **Photometric** (color jitter, noise) only on the input, through `ApplyTo`. A
   v2 photometric transform changes *every* `Image` in the sample, including the
   ground-truth target.

```python
from mon.data import ApplyTo

train_tf = v2.Compose([
    v2.RandomCrop(256),
    v2.RandomHorizontalFlip(),
    ApplyTo(["image"], v2.ColorJitter(brightness=0.1, contrast=0.1)),  # input only
    v2.ToDtype(torch.float32, scale=True),                             # uint8 → float in [0, 1]
])
train = DATASETS.build("lol_v1", "train", transform=train_tf)
s = train[0]
s["image"].shape, s["target"].shape      # both (3, 256, 256), same crop and flip
```

Other helpers in `mon.data.transforms`, for what v2 lacks:

- `ResizeToMultiple(32)`: resize so height and width are multiples of a network
  stride (training).
- `PadToMultiple(32)`: pad bottom and right to a multiple instead (inference,
  see section 5).
- `MinMaxNormalize()`: scale each image to [0, 1] by its own range, for depth,
  RAW, or HDR data.

For detection data, add `v2.SanitizeBoundingBoxes()` after crops so labels are
dropped with their boxes.

## 4. Loaders and reproducibility

`create_dataloader` wraps `torch.utils.data.DataLoader` with `mon`'s `collate`
and a seeded generator.

```python
from mon.core.runtime import seed_everything
from mon.data import create_dataloader

seed = seed_everything(0)                # seeds Python, NumPy, torch; returns the seed to save
loader = create_dataloader(train, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))
batch["image"].shape                     # (2, 3, 256, 256)
len(batch["meta"])                       # 2: per-sample metadata stays a list
```

- `collate` stacks same-shape tensors into `[B, …]`; `meta`, boxes and labels
  stay per-sample lists.
- With `seed=None` (default) the loader draws its seed from torch's global RNG,
  so `seed_everything` alone makes the shuffle order and worker RNGs
  reproducible. Pass `seed=` to fix them independently.
- Full-size images only stack if they all have the same size; otherwise use
  crops or `batch_size=1`.

## 5. Inference: pad, predict, crop back

Networks with a stride need input sizes that are multiples of it (often 8–32).
At inference, **pad, don't resize**: padding is undone exactly by cropping,
while resizing in and back out resamples the output twice and shifts the
metrics.

```python
from mon.data import crop_to, pad_to_multiple

model = torch.nn.Conv2d(3, 3, 3, padding=1).eval()   # stand-in for an enhancement network

test = DATASETS.build("lol_v1", "test", transform=v2.ToDtype(torch.float32, scale=True))
batch = next(iter(create_dataloader(test, batch_size=1)))
x, size = pad_to_multiple(batch["image"], 32)        # reflect-padded, plus the original Size
with torch.no_grad():
    pred = crop_to(model(x), size)                   # back to the input size
assert pred.shape == batch["target"].shape
```

To pad inside the dataset instead (e.g. to batch images of different sizes after
padding), use `PadToMultiple(32)` in the transform and crop each prediction with
`crop_to(pred, meta.orig_size)`.

To predict on images that aren't a registered dataset (a folder, one file, or a
list of files), use `ImageFolder`. It searches subfolders, skips hidden files
(macOS `._*`), and keeps each image's path relative to the folder in
`meta.stem`, so outputs can mirror the input tree:

```python
from mon.data import ImageFolder, write_image

inputs = ImageFolder(test.root / "test" / "images", transform=v2.ToDtype(torch.float32, scale=True))
out_dir = test.root.parent.parent / "runs" / "demo"          # in real code: the run directory
for batch in create_dataloader(inputs, batch_size=1):
    x, size = pad_to_multiple(batch["image"], 32)
    with torch.no_grad():
        pred = crop_to(model(x), size).clamp(0, 1)
    meta = batch["meta"][0]
    write_image(pred[0], out_dir / f"{meta.stem}.png")      # night/0001.jpg → night/0001.png
```

`mon.runners.predict` does all of this for any registered model: it pads to the
model's `input_multiple`, fits zero-shot models per image, and saves outputs
clamped to `[0, 1]` with the resolved config. From the shell:

```text
uv run mon predict model=zero_dce weights=sice_me data=path/to/night   # or data=lol_v1 split=test
```

Outputs land in `$RUNS_DIR/predict/<model>[-<weights>]/<data>/`
(`$DATA_DIR/runs` if `RUNS_DIR` is unset); pass `run_dir=…` to choose.
`PredictConfig` lists every key (`device`, `tile`, `save_extra`, `fit_args`, …).

Score the saved outputs with `mon evaluate`; it pairs files with targets by stem
and writes `metrics.csv` and `metrics.yaml` into the run directory
(`gt_mean=true` adds the GT-mean protocol):

```text
uv run mon evaluate pred=$RUNS_DIR/predict/zero_dce-sice_me/lol_v1_test data=lol_v1 "metrics=[psnr,ssim,lpips]"
```

## 6. Classification datasets

`cifar10`, `cifar100`, `mnist` and `fashion_mnist` are stored as
`<split>/<class>/<index>.png` with a `classes.txt` listing class names in
label-id order. Samples are `{"image", "label", "meta"}`; `label` is an `int64`
scalar, so batches carry a `[B]` label tensor ready for `cross_entropy`.

```python
from collections import Counter

from torch.nn import functional as F

mnist = DATASETS.build("mnist", "train", transform=v2.Compose([
    v2.RandomCrop(28, padding=2),
    v2.ToDtype(torch.float32, scale=True),           # uint8 → [0, 1]
    v2.Normalize(mean=[0.1307], std=[0.3081]),       # MNIST statistics
]))
mnist.classes.names                                  # ['0', '1', …, '9']: label id = position
Counter(mnist.labels)                                # class counts without decoding any image

batch = next(iter(create_dataloader(mnist, batch_size=4, shuffle=True)))
batch["image"].shape, batch["label"].shape           # (4, 1, 28, 28), (4,)
classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 10))
loss = F.cross_entropy(classifier(batch["image"]), batch["label"])
```

- MNIST and Fashion-MNIST load as 1-channel gray images (`image_mode = "gray"`);
  CIFAR as RGB.
- `meta.stem` is `"<class>/<index>"`; `<index>` is the image's position in the
  upstream split.
- CIFAR-100 has 26 groups of byte-identical images, some with conflicting
  labels; see `mon.data.zoo.cifar`.

## 7. Adding a dataset

1. Put the data in `$DATASETS_DIR/<id>/` in the normalized layout and add its
   row to `data/README.md`.
2. Declare and register the class in a `mon.data.zoo` module:

   ```python
   from mon.data.dataset import FolderDataset, paired_layout, register_dataset

   @register_dataset("my_paired", tasks={Task.LLE})
   class MyPaired(FolderDataset):
       """My paired dataset: N train and M test pairs. Note any overlap with other datasets."""

       folder = "my_paired"
       layout = paired_layout("train", "test")
   ```

3. Import that module in `src/mon/data/zoo/__init__.py`; otherwise the decorator
   never runs and the id is missing from `DATASETS`.
4. Add tests in `tests/data/zoo/`: synthetic folders, plus `slow` checks on the
   real data (counts, format, no train/test overlap) that skip when the data
   isn't on the machine. See `tests/data/zoo/test_lol.py`.

For other layouts, write `layout` by hand with `Modality(dir, kind)`: kinds are
`image`, `mask`, `depth`, `yolo` and `voc`. Classification datasets
(`<split>/<class>/<image>` plus `classes.txt`) subclass `ClassFolderDataset`
with `layout = classification_layout("train", "test")` instead; their samples
are `{"image", "label", "meta"}`, and `collate` stacks `label` into a `[B]`
tensor.
