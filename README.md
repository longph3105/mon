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

## Usage

```python
import mon

print(mon.__version__)
```

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
