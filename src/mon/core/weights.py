"""Pretrained-weights descriptors, per-model weights enums, and loading.

Pattern (as in torchvision): each model defines a ``WeightsEnum`` whose members are
``Weights``; user input (config value, CLI string, member) goes through
``resolve_weights``. A missing file raises; it is never silently skipped.

References:
    - torchvision multi-weight API: https://pytorch.org/vision/stable/models.html
"""

__all__ = [
    "Weights",
    "WeightsEnum",
    "WeightsLike",
    "resolve_weights",
]

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mon.core.fileio import download_file
from mon.core.path import Path, find_weights, zoo_dir
from mon.core.utils import has_value


@dataclass(frozen=True)
class Weights:
    """Where a checkpoint lives and what it is.

    Attributes:
        path: Absolute path, or a path relative to the model zoo (``zoo_dir()``). Kept
            as given; resolved only by ``resolve``.
        name: Human-readable name.
        url: Optional download URL used when ``path`` is missing.
        sha256: Optional expected digest of the file (checked on download).
        num_classes: Classes the checkpoint was trained with, if relevant.
        transforms: Factory for the matching preprocessing (not compared or hashed).
        meta: Free-form metadata, e.g. training recipe or metrics (not compared or hashed).
    """

    path       : str
    name       : str = ""
    url        : str | None = None
    sha256     : str | None = None
    num_classes: int | None = None
    transforms : Callable[..., Any] | None = field(default=None, compare=False)
    meta       : Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        path = os.fspath(self.path)
        if not path.strip():
            raise ValueError("Weights.path must be non-empty")
        object.__setattr__(self, "path", path)

    def resolve(self, *, download: bool = True) -> Path:
        """Return the local file: ``find_weights(path)``, else download ``url`` if allowed.

        Relative paths download to ``zoo_dir() / path``; absolute paths to themselves.

        Raises:
            FileNotFoundError: If the file is missing and can't be downloaded.
        """
        try:
            return find_weights(self.path)
        except FileNotFoundError as err:
            if not (download and self.url):
                raise FileNotFoundError(f"{err}{'' if self.url else '; no url to download from'}") from None
        given = Path(self.path).expanduser()
        target = given if given.is_absolute() else zoo_dir() / given
        return download_file(self.url, target, sha256=self.sha256)

    def load_state_dict(
        self,
        *,
        map_location: Any = "cpu",
        weights_only: bool = True,
        download    : bool = True,
    ) -> Any:
        """``torch.load`` the resolved file.

        ``weights_only=True`` (default) refuses arbitrary pickled objects; only pass
        ``False`` for checkpoints you trust.
        """
        import torch  # lazy: keep `import mon.core.weights` light

        return torch.load(self.resolve(download=download), map_location=map_location, weights_only=weights_only)


class WeightsEnum(Enum):
    """Base for per-model weights enums; members must be ``Weights``.

    Add ``DEFAULT = MEMBER`` (an alias) to support ``resolve_weights("default", ...)``.

    Examples:
        >>> class ZeroDCE_Weights(WeightsEnum):
        ...     LOL = Weights("enhance/zero_dce/lol.pt")
        ...     DEFAULT = LOL
        >>> resolve_weights("default", ZeroDCE_Weights) == ZeroDCE_Weights.LOL.value
        True
    """

    def __init__(self, value: object) -> None:
        if not isinstance(value, Weights):
            raise TypeError(f"{type(self).__name__}.{self.name} must be a Weights, got {type(value).__name__}")

    @property
    def weights(self) -> Weights:
        """The member's ``Weights``."""
        value: Weights = self.value
        return value


type WeightsLike = Weights | WeightsEnum | str | os.PathLike[str] | None


def resolve_weights(spec: WeightsLike, enum: type[WeightsEnum] | None = None) -> Weights | None:
    """Turn user input into ``Weights`` (or ``None`` for "no weights").

    - ``None``, ``""``, ``"none"``, ``"null"`` → ``None``.
    - ``Weights`` → itself; a ``WeightsEnum`` member → its ``Weights``.
    - With ``enum``: ``"default"`` → ``enum.DEFAULT`` (must exist); a member name
      (case-insensitive, e.g. ``"lol"``) → that member.
    - Any other string or path → ``Weights(path=spec)``; existence is checked on use.

    Raises:
        ValueError: For ``"default"`` without ``enum`` or without a ``DEFAULT`` member.
        TypeError: For unsupported input types.
    """
    if not has_value(spec):
        return None
    if isinstance(spec, Weights):
        return spec
    if isinstance(spec, WeightsEnum):
        return spec.weights
    if isinstance(spec, str):
        key = spec.strip()
        if key.lower() == "default":
            if enum is None:
                raise ValueError("'default' weights need the model's WeightsEnum")
            if "DEFAULT" not in enum.__members__:
                raise ValueError(f"{enum.__name__} has no DEFAULT member; choose one of: {', '.join(enum.__members__)}")
            return enum["DEFAULT"].weights
        if enum is not None:
            for name, member in enum.__members__.items():
                if name.lower() == key.lower():
                    return member.weights
        return Weights(path=key)
    if isinstance(spec, os.PathLike):
        return Weights(path=os.fspath(spec))
    raise TypeError(f"cannot resolve weights from {type(spec).__name__}")
