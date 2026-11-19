"""Model registry and pretrained loading.

``MODELS`` lists every model lazily (the index below), so listing and filtering models
imports neither the model code nor PyTorch; ``load`` builds a model and loads its weights.

To add a model, add one ``MODELS.add_lazy`` line to the index. ``tests/models/test_registry.py``
checks that every entry resolves and every public model class in ``mon.models`` is listed.

Examples:
    >>> from mon.core.enums import Task
    >>> MODELS.names(tasks=Task.LLE)
    ['clode', 'colie', 'dccnet', 'flol', 'hvi_cidnet', 'pairlie', 'retinexnet', 'sci', 'sci++', 'zero_dce', 'zero_dce++', 'zero_ig']
    >>> load("mertens")  # no parameters: nothing to load
    Mertens(contrast=1.0, saturation=1.0, exposure=1.0, sigma=0.2, levels=None)
    >>> model = load("zero_dce", weights=None)  # random init; weights="default" loads SICE_ME
    >>> type(model).__name__, model.training
    ('ZeroDCE', False)
"""

__all__ = [
    "MODELS",
    "flush_denormals",
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
``zero_shot=True`` for networks fitted to each input at test time (released weights, if any, are
dataset-trained alternatives, e.g. Zero-IG's), with ``fit`` = ``"module:function"`` of the function
that fits a fresh network per image (``fn(image, **kwargs) -> Tensor``, ``image`` ``[B, C, H, W]``).

``train`` = ``"module:RECIPE"`` names the model's ``mon.nn.training.TrainRecipe`` (``mon train``).

A model class may set ``input_multiple: ClassVar[int]`` when ``H`` and ``W`` must be multiples of
it; runners pad to it and crop back (default 1)."""

# --- Index: one line per model, grouped by task, sorted by name ---

# denoise
MODELS.add_lazy("tv_denoise", "mon.models.denoise.tv_denoise:TVDenoise", tasks={Task.DENOISE}, arch="tv_denoise")
MODELS.add_lazy("zs_n2n"    , "mon.models.denoise.zs_n2n:ZSN2N"        , tasks={Task.DENOISE}, arch="zs_n2n",
                zero_shot=True, fit="mon.models.denoise.zs_n2n:zs_n2n_denoise")
# lle
MODELS.add_lazy("clode"     , "mon.models.lle.clode:CLODE"       , tasks={Task.LLE}, arch="clode")
MODELS.add_lazy("colie"     , "mon.models.lle.colie:CoLIE"       , tasks={Task.LLE}, arch="colie",
                zero_shot=True, fit="mon.models.lle.colie:colie_enhance")
MODELS.add_lazy("dccnet"    , "mon.models.lle.dccnet:DCCNet"     , tasks={Task.LLE}, arch="dccnet")
MODELS.add_lazy("flol"      , "mon.models.lle.flol:FLOL"         , tasks={Task.LLE}, arch="flol")
MODELS.add_lazy("hvi_cidnet", "mon.models.lle.hvi_cidnet:HVICIDNet", tasks={Task.LLE}, arch="hvi_cidnet")
MODELS.add_lazy("pairlie"   , "mon.models.lle.pairlie:PairLIE"   , tasks={Task.LLE}, arch="pairlie")
MODELS.add_lazy("retinexnet", "mon.models.lle.retinexnet:RetinexNet", tasks={Task.LLE}, arch="retinexnet")
MODELS.add_lazy("sci"       , "mon.models.lle.sci:SCI"           , tasks={Task.LLE}, arch="sci",
                train="mon.models.lle.sci:SCI_RECIPE")
MODELS.add_lazy("sci++"     , "mon.models.lle.sci:SCIPP"         , tasks={Task.LLE}, arch="sci")
MODELS.add_lazy("zero_dce"  , "mon.models.lle.zero_dce:ZeroDCE"  , tasks={Task.LLE}, arch="zero_dce",
                train="mon.models.lle.zero_dce:ZERO_DCE_RECIPE")
MODELS.add_lazy("zero_dce++", "mon.models.lle.zero_dce:ZeroDCEPP", tasks={Task.LLE}, arch="zero_dce",
                train="mon.models.lle.zero_dce:ZERO_DCEPP_RECIPE")
MODELS.add_lazy("zero_ig"   , "mon.models.lle.zero_ig:ZeroIG"    , tasks={Task.LLE}, arch="zero_ig",
                zero_shot=True, fit="mon.models.lle.zero_ig:zero_ig_enhance", train="mon.models.lle.zero_ig:ZERO_IG_RECIPE")
# mef
MODELS.add_lazy("mertens"   , "mon.models.mef.mertens:Mertens"   , tasks={Task.MEF}, arch="mertens")


def flush_denormals(model: "nn.Module") -> int:
    """Set denormal float parameters and buffers (``0 < |x| < finfo.tiny``) to zero, in place; return how many.

    Weight decay leaves many trained weights in that range (about 25% of the released Zero-DCE weights), and CPU
    convolutions (oneDNN) are ~50× slower on them. Their products are far below float32 rounding, so outputs are
    unchanged.
    """
    import torch  # lazy: keep listing models free of torch

    count = 0
    with torch.no_grad():
        for t in [*model.parameters(), *model.buffers()]:
            if t.is_floating_point():
                tiny = (t != 0) & (t.abs() < torch.finfo(t.dtype).tiny)
                count += int(tiny.sum())
                t.masked_fill_(tiny, 0)
    return count


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
        strict: Passed to ``load_state_dict``. Loaded weights then go through ``flush_denormals``.
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
        flush_denormals(model)
    if device is not None:
        from mon.core.runtime import select_device  # lazy: imports torch
        model.to(select_device(device))
    return model.eval()
