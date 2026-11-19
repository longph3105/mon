"""Models: plain ``nn.Module`` networks, a lazy registry, and pretrained loading.

Models live in task subpackages (``mon.models.lle``, …), one module per model family, and are
listed in ``MODELS`` without being imported. ``load(name, weights=...)`` builds one; import a
model module directly for its classes, weights enum, and paper loss. Names are re-exported
lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ========  ======================================  ============
    Module    Purpose                                 Key names
    ========  ======================================  ============
    registry  Model registry and pretrained loading.  MODELS, load
    ========  ======================================  ============

Task subpackages (imported explicitly, like ``mon.data.zoo``):

    - ``denoise``: denoising (``tv_denoise``: total variation; ``zs_n2n``: Zero-Shot Noise2Noise).
    - ``lle``: low-light image enhancement (``clode``: CLODE; ``colie``: CoLIE; ``dccnet``: DCC-Net;
      ``flol``: FLOL; ``hvi_cidnet``: HVI-CIDNet; ``pairlie``: PairLIE; ``retinexnet``: RetinexNet;
      ``ruas``: RUAS; ``sci``: SCI, SCI++; ``zero_dce``: Zero-DCE, Zero-DCE++; ``zero_ig``: Zero-IG).
    - ``mef``: multi-exposure fusion (``mertens``: Mertens et al. exposure fusion).
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("registry",)

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    "MODELS"         : "registry",
    "flush_denormals": "registry",
    "load"           : "registry",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "MODELS",
    "flush_denormals",
    "load",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.models.registry import MODELS, flush_denormals, load
