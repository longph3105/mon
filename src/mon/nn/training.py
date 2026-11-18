"""Training recipes: a model's step, loss, init, and official defaults.

A model module defines one ``TrainRecipe`` per trainable model and names it in the model's
``MODELS`` entry (``train="module:RECIPE"``); ``mon.runners.trainer`` runs the generic loop
around it. The recipe's ``defaults`` reproduce the official training script and sit under
the user's config: every key stays overridable.
"""

__all__ = [
    "StepOutput",
    "TrainRecipe",
]

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

type StepOutput = dict[str, Tensor]
"""``{"loss": scalar to backpropagate, <term>: scalar, …}``; every value is logged (detached)."""


@dataclass(frozen=True)
class TrainRecipe:
    """One model's training definition.

    Attributes:
        loss: Builds the loss module (moved to the training device by the runner).
        step: ``(model, loss, batch) -> StepOutput`` for one training batch; ``batch`` is the
            collated dict (``"image"``, optional ``"target"``, ``"meta"``), already on the device.
            The model is in train mode; the runner does backward, clipping, and the optimizer step.
        init: Applied to a freshly built model (training from scratch) after seeding, when the
            official script re-initializes parts the constructor doesn't; ``None``: constructor init.
        defaults: ``TrainConfig`` values of the official recipe (``optimizer``, ``epochs``,
            ``batch_size``, ``resize``, ``grad_clip``, ``seed``, …), applied under the user's config.
        source: The official script these follow, for the record.
        notes: Known differences from the official script.
    """

    loss    : Callable[[], nn.Module]
    step    : Callable[[nn.Module, nn.Module, dict[str, Any]], StepOutput]
    init    : Callable[[nn.Module], None] | None = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    source  : str = ""
    notes   : str = ""
