# 🐈 mon

A Python + PyTorch library for computer-vision research. It's named after my
cat, Mon.

## Status

**3.0 is a ground-up rewrite, in early development (pre-alpha): APIs may still
change.** It replaces the 2.x monorepo with a single package, and modules are
being ported one at a time.

| Version | Where | State |
| --- | --- | --- |
| 3.0.1 | [`3.0.1`](https://github.com/longph3105/mon/tree/3.0.1) branch (default) | In development |
| 3.0.0 | [`v3.0.0`](https://github.com/longph3105/mon/tree/v3.0.0) tag | First 3.0 release (2026-11-22) |
| 2.15.0 | [`2.15.0`](https://github.com/longph3105/mon/tree/2.15.0) branch | Previous version (2.x monorepo) |

Changes since 3.0.0 are listed in [CHANGELOG.md](CHANGELOG.md); earlier ones in
the [3.0.0 changelog](https://github.com/longph3105/mon/blob/v3.0.0/CHANGELOG.md).

## Requirements

- Python ≥ 3.13 (except 3.14.1)
- PyTorch ≥ 2.15 (installed automatically)
- Platforms: developed on macOS arm64 (MPS); Linux x86_64 (CUDA) runs training
  and the GPU tests

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended for development):

```shell
git clone https://github.com/longph3105/mon.git   # default branch: 3.0.1
cd mon
uv sync                # core
uv sync --extra gui    # core + display (matplotlib, PySide6)
```

As a dependency of another project:

```shell
uv add "mon @ git+https://github.com/longph3105/mon@v3.0.0"   # the 3.0.0 release
uv add "mon @ git+https://github.com/longph3105/mon@3.0.1"    # the development branch
# or
pip install "mon[gui] @ git+https://github.com/longph3105/mon@v3.0.0"
```

OpenCV is installed as `opencv-python-headless`, so `cv2.imshow` does not work
on Linux. Use the `gui` extra and matplotlib to display images.

## Configuration

Pretrained weights are looked up in `$ZOO_DIR`, datasets in `$DATASETS_DIR`, and
run outputs go to `$RUNS_DIR`; if unset, they default to `$DATA_DIR/zoo`,
`$DATA_DIR/datasets` and `$DATA_DIR/runs`:

```shell
export DATA_DIR=/path/to/data       # or set ZOO_DIR / DATASETS_DIR / RUNS_DIR individually
```

Set them where every process sees them, not only interactive terminals:

- macOS: `~/.zshenv` (zsh) or `~/.bash_profile` (bash). Linux servers:
  `~/.profile` or `~/.bashrc`.
- Containers: `docker run -e DATA_DIR=/data …`. Per-project overrides: a
  git-ignored `.env` file with `uv run --env-file .env …`.

Check what `mon` will use (exit code 1 if something is unset or missing):

```shell
uv run python -m mon.core.path
```

## Usage

```python
import mon.core as mc

cfg  = mc.load_config("exp.yaml", overrides=["optimizer.lr=1e-3"])
dev  = mc.select_device("auto")        # cuda:0 → mps → cpu
seed = mc.seed_everything(cfg.get("seed"))
```

Names load lazily, so `import mon.core` stays fast until a name that needs
PyTorch is used.

Models load by name with their released weights (downloaded if they aren't in
the zoo):

```python
import mon.models

mon.models.MODELS.names(tasks="lle")                # ['clode', 'colie', 'dccnet', ..., 'zero_dce++', 'zero_ig']
model = mon.models.load("zero_dce", device="auto")  # weights="default" → released SICE weights
enhanced = model(image)["enhanced"]                 # image: [B, 3, H, W] float in [0, 1]
```

Train, predict, evaluate and benchmark from the command line (OmegaConf dotlist
arguments; each run saves its resolved config in its run directory under
`$RUNS_DIR`):

```shell
uv run mon train model=zero_dce device=cuda:0
uv run mon predict model=zero_dce data=sice_me split=val device=cuda:0
uv run mon evaluate pred=<run dir> data=sice_me split=val "metrics=[psnr,ssim,lpips]"
uv run mon benchmark model=zero_dce size=[900,1200] device=cuda:0  # params, FLOPs/MACs, latency, memory
uv run mon benchmark model=zero_dce data=testkit_efficiency output=zero_dce.yaml  # per resolution, 256² to 8K
uv run mon benchmark model=colie data=testkit_efficiency  # zero-shot: times the per-image fit
```

Datasets, transforms and loaders: [docs/data.md](docs/data.md). Efficiency of
every LLE model, 256² to 8K on an RTX 3090:
[docs/efficiency.md](docs/efficiency.md).

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{Pham2022,
    author       = {Pham, Long Hoang and Tran, Duong Nguyen-Ngoc and Ho, Quoc Pham-Nam},
    title        = {mon},
    howpublished = {\url{https://github.com/longph3105/mon}},
    year         = {2022},
}
```

## License

[MIT](LICENSE)

## Contact

Long H. Pham: [longpham3105@gmail.com](mailto:longpham3105@gmail.com) or
[phlong@skku.edu](mailto:phlong@skku.edu)
