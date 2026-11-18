"""Low-light image enhancement models.

One module per model family; each is listed in ``mon.models.MODELS`` and imported on use:

- ``clode``: CLODE (curve adjustment as a neural ODE; exposure set by the end time ``T``).
- ``colie``: CoLIE (implicit illumination fitted per image; no weights).
- ``dccnet``: DCC-Net (gray, color, and restoration branches with pyramid color embedding).
- ``pairlie``: PairLIE (Retinex decomposition learned from low-light pairs).
- ``sci``: SCI and SCI++ (self-calibrated illumination).
- ``zero_dce``: Zero-DCE and Zero-DCE++ (zero-reference curve estimation).
"""

__all__: list[str] = []
