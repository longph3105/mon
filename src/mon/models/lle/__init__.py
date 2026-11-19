"""Low-light image enhancement models.

One module per model family; each is listed in ``mon.models.MODELS`` and imported on use:

- ``clode``: CLODE (curve adjustment as a neural ODE; exposure set by the end time ``T``).
- ``colie``: CoLIE (implicit illumination fitted per image; no weights).
- ``dccnet``: DCC-Net (gray, color, and restoration branches with pyramid color embedding).
- ``flol``: FLOL (FFT-amplitude enhancement, then SNR-guided spatial/frequency restoration).
- ``hvi_cidnet``: HVI-CIDNet (two cross-attending branches in the HVI color space).
- ``pairlie``: PairLIE (Retinex decomposition learned from low-light pairs).
- ``retinexnet``: RetinexNet (Retinex decomposition, then illumination adjustment).
- ``ruas``: RUAS (unrolled Retinex illumination stages and a denoiser, cells from architecture search).
- ``sci``: SCI and SCI++ (self-calibrated illumination).
- ``zero_dce``: Zero-DCE and Zero-DCE++ (zero-reference curve estimation).
- ``zero_ig``: Zero-IG (joint denoising and enhancement fitted per image; dataset-trained weights too).
"""

__all__: list[str] = []
