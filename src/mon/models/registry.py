"""Model registry and pretrained loading.

``MODELS`` lists every model lazily (the index below), so listing and filtering models
imports neither the model code nor PyTorch; ``load`` builds a model and loads its weights.

To add a model, add one ``MODELS.add_lazy`` line to the index. ``tests/models/test_registry.py``
checks that every entry resolves and every public model class in ``mon.models`` is listed.

Examples:
    >>> from mon.core.enums import Task
    >>> MODELS.names(tasks=Task.LLE)
    ['colie', 'pairlie', 'sci', 'sci++', 'zero_dce', 'zero_dce++']
    >>> load("mertens")  # no parameters: nothing to load
    Mertens(contrast=1.0, saturation=1.0, exposure=1.0, sigma=0.2, levels=None)
    >>> model = load("zero_dce", weights=None)  # random init; weights="default" loads SICE_ME
    >>> type(model).__name__, model.training
    ('ZeroDCE', False)
"""

__all__ = [
    "MODELS",
    "load",
]

from typing import TYPE_CHECKING, Any

from mon.core.enums import Task
from mon.core.registry import Registry
from mon.core.weights import WeightsLike, resolve_weights

if TYPE_CHECKING:
    import torch
    from torch import nn

MODELS: "Registry[type[nn.Module]]" = Registry("models")
"""All models by name. Metadata: ``tasks`` (set of ``Task``), ``arch`` (shared by variants), and
``zero_shot=True`` for networks fitted to each input at test time (no pretrained weights)."""

# --- Index: one line per model, grouped by task, sorted by name ---

# denoise
MODELS.add_lazy("tv_denoise", "mon.models.denoise.tv_denoise:TVDenoise", tasks={Task.DENOISE}, arch="tv_denoise")
MODELS.add_lazy("zs_n2n"    , "mon.models.denoise.zs_n2n:ZSN2N"        , tasks={Task.DENOISE}, arch="zs_n2n",
                zero_shot=True)  # fit per image: zs_n2n_denoise
# lle
MODELS.add_lazy("colie"     , "mon.models.lle.colie:CoLIE"       , tasks={Task.LLE}, arch="colie",
                zero_shot=True)  # fit per image: colie_enhance
MODELS.add_lazy("pairlie"   , "mon.models.lle.pairlie:PairLIE"   , tasks={Task.LLE}, arch="pairlie")
MODELS.add_lazy("sci"       , "mon.models.lle.sci:SCI"           , tasks={Task.LLE}, arch="sci")
MODELS.add_lazy("sci++"     , "mon.models.lle.sci:SCIPP"         , tasks={Task.LLE}, arch="sci")
MODELS.add_lazy("zero_dce"  , "mon.models.lle.zero_dce:ZeroDCE"  , tasks={Task.LLE}, arch="zero_dce")
MODELS.add_lazy("zero_dce++", "mon.models.lle.zero_dce:ZeroDCEPP", tasks={Task.LLE}, arch="zero_dce")
# mef
MODELS.add_lazy("mertens"   , "mon.models.mef.mertens:Mertens"   , tasks={Task.MEF}, arch="mertens")


def load(
    name   : str,
    weights: WeightsLike = "default",
    *,
    device : "str | int | torch.device | None" = None,
    strict : bool = True,
    **kwargs: Any,
) -> "nn.Module":
    """Build model ``name`` and load ``weights``; return it in eval mode.

    Args:
        name: Registry name (case-insensitive), e.g. ``"zero_dce"``.
        weights: ``"default"`` (the model's ``weights_enum.DEFAULT``; no weights for a model
            without parameters, e.g. ``"mertens"``), a member name (``"sice_me"``), a
            ``WeightsEnum`` member, ``Weights``, a checkpoint path (absolute or relative to
            ``zoo_dir()``), or ``None`` for the constructor's random init. Missing files are
            downloaded when the ``Weights`` has a ``url``.
        device: Move the model there (any ``select_device`` spec); ``None`` keeps it on CPU.
        strict: Passed to ``load_state_dict``.
        **kwargs: Constructor arguments; they must match the checkpoint's architecture.

    Raises:
        RegistryKeyError: If ``name`` isn't registered.
        ValueError: If ``weights="default"`` and a model with parameters has no ``weights_enum``.
        FileNotFoundError: If the checkpoint is missing and can't be downloaded.
        RuntimeError: If the checkpoint doesn't match the model (with ``strict``).
    """
    cls   = MODELS.get(name)
    enum  = getattr(cls, "weights_enum", None)
    model = cls(**kwargs)
    if enum is None and isinstance(weights, str) and weights.strip().lower() == "default":
        if any(True for _ in model.parameters()):
            raise ValueError(f"{name!r} has no pretrained weights; pass weights=None or a checkpoint path")
        weights = None  # parameter-free (classical) model: nothing to load
    spec  = resolve_weights(weights, enum)
    if spec is not None:
        model.load_state_dict(spec.load_state_dict(), strict=strict)
    if device is not None:
        from mon.core.runtime import select_device  # lazy: imports torch
        model.to(select_device(device))
    return model.eval()
