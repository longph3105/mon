# CONVENTIONS.md - mon

Code conventions for `src/mon/` and its tests in `tests/`. Follow them when
writing or changing code; `AGENTS.md` has the general rules.

## General

- Follow [PEP 8](https://peps.python.org/pep-0008/), except: lines up to 120
  characters, and the owner's alignment of `=` and `:` stays. Wrapped
  brackets use a hanging indent with a trailing comma (one item per line).
- `uv run ruff check` must pass (rules in `pyproject.toml`). No formatter or
  import sorting (`ruff format`, `I`): they undo the aligned `=` and `:`.
- Google-style docstrings; every module has a one-line docstring, plus
  `References:` (paper or source links) when ported or adapted. Strip
  `mon_old`'s shebang and `# -*- coding: utf-8 -*-` lines.
- `__all__` at the top of every public module. Package namespaces re-export
  submodules lazily: a new public name goes in the package's `_EXPORTS`,
  `__all__` and docstring table (`tests/test_namespaces.py` checks these and the
  layering in `ALLOWED_IMPORTS`). Inside `mon`, import from submodules
  (`from mon.core.path import Path`), never from a namespace. Top-level `mon`
  exposes only subpackages.
- Section comments in large modules: `# --- Section Name ---`. Keep the owner's
  alignment of `=` and `:`.
- Configs: OmegaConf via `mon.core.config.load_config`/`save_config`; schemas
  are dataclasses next to their code; resolve devices, weights and run dirs
  explicitly after loading (no property setters).
- Registries: one `Registry` (`mon.core.registry`) per kind, in the package that
  owns it; give an explicit name when `cls.__name__.lower()` isn't the paper's
  (`register("zero_dce++", ...)`).
- Enums subclass `AliasStrEnum` (`mon.core.enums`); no `"default"` member or
  `None` fallback: defaults go in signatures (`split: Split = Split.TRAIN`).
- Seeds: `mon.core.runtime.seed_everything()`; save the returned seed with the
  run.
- Logging: `logger = get_logger(__name__)` (`mon.core.console`); never configure
  logging or print at import. Only entry points call `setup_logging()`.
- Devices: `select_device(spec)` (`mon.core.runtime`); never fall back to CPU
  silently.
- Samples and datasets must pickle (DataLoader workers; Python 3.14
  `forkserver`): `mon.core.utils.ReadOnlyDict`, never `types.MappingProxyType`.

## Data

- A dataset subclasses `FolderDataset` (`ClassFolderDataset` for
  `<split>/<class>/`), sets `folder` (its root under `datasets_dir()`: the
  archive id, or a subfolder of it, e.g. `lsrw/huawei`; a split view of
  another dataset shares its folder, e.g. `sice_under`) and
  `layout` (its family's normalized layout: `paired_layout(...)`,
  `detection_layout(...)`, `classification_layout(...)`), and registers with
  `register_dataset` under the same id, in the `mon.data.zoo` module of its
  family (named after an id, e.g. `lol` for `lol_v1`/`lol_v2_real`; never per
  task: tasks are tags), imported in `zoo/__init__.py`. Record known train/test
  overlaps in its docstring.
- Loaders: `create_dataloader` (seeded). Image/video I/O: `mon.data.io`
  (OpenCV), never `torchvision.io` codecs. `uint16` tensors:
  `.float()`/`v2.ToDtype` before arithmetic.
- Samples are dicts of `tv_tensors`; wrap every image-like value
  (`Image`/`Mask`/`Video`): v2 treats only the first plain tensor as an image.
  Class-id maps are `Mask`; continuous maps (depth, targets) are `Image`;
  restore types after arithmetic with `tv_tensors.wrap`.
- Transforms (torchvision v2) run in two stages: geometric on the whole sample,
  photometric only on inputs (`ApplyTo`), since v2 photometric ops change every
  `Image`. After crops, `SanitizeBoundingBoxes`.
- Stride multiples: training resizes (`ResizeToMultiple`); inference pads
  (`PadToMultiple`, `pad_to_multiple`) and crops back with
  `crop_to(pred, meta.orig_size)`; never resize an output back for evaluation.

## Models, losses, weights

- Models are plain `nn.Module`s, tensors in and out (`forward(image) -> Tensor`
  or a small dict); no TensorDict, `in_keys`/`out_keys` or `__call__` override;
  constructors take architecture arguments only. One family per
  `mon/models/<task>/<family>.py` (weights enums, models, paper loss).
- Register a model with one `MODELS.add_lazy(name, "module:Class", tasks=...,
  arch=...)` line in `mon/models/registry.py`, not a decorator
  (`tests/models/test_registry.py` checks the index).
- Weights: a `WeightsEnum` of `Weights("<task>/<model>/<file>.pt")` (under
  `zoo_dir()`; naming: `data/README.md`, Zoo) with `sha256`, and `url` only when
  the zoo file is the released file byte for byte (a download is checked
  against `sha256`). A converted file (unwrapped, merged, pruned) has no `url`;
  its `meta` gives `source` and `source_sha256` (full, not shortened); an
  unreleased one (the owner's training) gives only `source`. The class
  sets `weights_enum: ClassVar[type[WeightsEnum]]`; load through
  `mon.models.load(name, weights=...)` and `resolve_weights(spec, Enum)`. A
  missing file raises, never skips.
- Large images: `mon.nn.tiling.tiled_forward`, never a per-model
  `forward_patch`. Stride requirement: `input_multiple: ClassVar[int]`.
- Classical methods are parameter-free modules. Multi-image inputs take an
  explicit axis (`[B, N, C, H, W]`). Zero-shot methods register with
  `zero_shot=True` and a `fit` function that fits a fresh network per image
  (`zs_n2n_denoise`); `forward` is the network only; never resize their inputs.
- Losses are separate modules named `…Loss`, returning per-sample values reduced
  by `reduction="mean"|"sum"|"none"`. Shared ones live in
  `mon.nn.losses.<family>`; one model's loss stays in its module until a second
  model needs it.
- Ported code follows the official code. Parity: losses against a reference copy
  of that code; models against reference values with the released weights, from
  a script next to the test (`tests/models/lle/zero_dce_reference.py`) run on
  the official code or `mon_old` (its `mon` conda env, when its code is a
  faithful copy), kept as literals. Record differences from the official code
  and `mon_old` in the module docstring.

## Metrics

- `Metric` modules score each image (`[B]` float64, no reduction), with `mode`
  (`"fr"`/`"nr"`) and `higher_better`, mirrored in `METRICS`. Defaults reproduce
  pyiqa (PSNR on RGB; SSIM and NIQE on luma rounded to 8-bit levels). Evaluate
  saved 8-bit outputs, not in-memory floats.

## Training and runners

- One loop (`mon.runners.trainer`); a model adds a `TrainRecipe`
  (`mon.nn.training`: step, loss, init, official defaults; alternating
  optimizers via `optimizers` + `schedule`) named by its `MODELS` entry
  (`train=`). Recipe defaults reproduce the official script (check with
  `tests/runners/test_trainer_parity.py`). Training resizes, never pads;
  checkpoints resume exactly.
- Runners take a dataclass config (`TrainConfig`, `PredictConfig`,
  `EvaluateConfig`) via `load_config` (YAML + dotlist), resolve it
  (`seed_everything`, `select_device`, run dir) and save it with the seed in the
  run dir: `train` and `predict` as `config.yaml`; `evaluate` and `benchmark`
  inside their result file (`metrics.yaml`, the `output` YAML), since evaluation
  writes into a predict run dir and a benchmark's `output` is optional.
- Run dirs: `runs_dir() / "train"` or `"predict"`, or a hand-set
  `"eval"`/`"benchmark"` dir (`$RUNS_DIR`, else `$DATA_DIR/runs`; layout:
  `data/README.md`, Runs). Evaluation writes next to the predictions. An
  existing, non-empty run dir raises unless `exist_ok`, and is never deleted.

## Tests

- Synthetic data only (NumPy/torch arrays, tiny files in `tmp_path`). Sample
  photos for manual work stay under `$DATA_DIR`, accessed by path.
- Regression tests on real data use the testkit (`mon.data.zoo.testkit`: 100
  samples at 256×256 per dataset, one per task family) on the
  `regression_device` fixture (CUDA without TF32, else CPU); extend it when a
  new task gets data and models. Whole-dataset checks are `full`.
