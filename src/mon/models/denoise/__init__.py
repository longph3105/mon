"""Denoising models.

One module per model family; each is listed in ``mon.models.MODELS`` and imported on use:

- ``tv_denoise``: total-variation denoising (classical, no parameters).
- ``zs_n2n``: Zero-Shot Noise2Noise (a tiny network fitted to each image; no weights).
"""

__all__: list[str] = []
