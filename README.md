# 🐈 mon

A Python + PyTorch library for computer-vision research. It's named after my cat, Mon.

## Status

**3.0.0 is a ground-up rewrite and is not usable yet.** It replaces the 2.x monorepo with a single package,
and modules are being ported one at a time. For the previous version, see the
[`2.15.0`](https://github.com/longph3105/mon/tree/2.15.0) branch.

Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Requirements

- Python ≥ 3.13 (except 3.14.1)
- PyTorch ≥ 2.15 (installed automatically)
- Tested platforms: macOS arm64 (MPS) and Linux x86_64 (CUDA 13)

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended for development):

```shell
git clone -b 3.0.0 https://github.com/longph3105/mon.git
cd mon
uv sync                # core
uv sync --extra gui    # core + display (matplotlib, PySide6)
```

As a dependency of another project:

```shell
uv add "mon @ git+https://github.com/longph3105/mon@3.0.0"
# or
pip install "mon[gui] @ git+https://github.com/longph3105/mon@3.0.0"
```

OpenCV is installed as `opencv-python-headless`, so `cv2.imshow` does not work on Linux.
Use the `gui` extra and matplotlib to display images.

## Configuration

Pretrained weights are looked up in `$ZOO_DIR` and datasets in `$DATASET_DIR`; if unset, they default to
`$DATA_DIR/zoo` and `$DATA_DIR/datasets`:

```shell
export DATA_DIR=/path/to/data       # or set ZOO_DIR / DATASET_DIR individually
```

Set them where every process sees them, not only interactive terminals:

- macOS: `~/.zshenv` (zsh) or `~/.bash_profile` (bash). Linux servers: `~/.profile` or `~/.bashrc`.
- Containers: `docker run -e DATA_DIR=/data …`. Per-project overrides: a git-ignored `.env` file with
  `uv run --env-file .env …`.

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

Names load lazily, so `import mon.core` stays fast until a name that needs PyTorch is used.

Datasets, transforms and loaders: [docs/data.md](docs/data.md).

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

Long H. Pham: [longpham3105@gmail.com](mailto:longpham3105@gmail.com) or [phlong@skku.edu](mailto:phlong@skku.edu)
