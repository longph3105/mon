"""Load and save OmegaConf configs.

Layers merge in order, later winning: ``schema`` (dataclass defaults) ← ``_base_`` files
← the config file ← dotlist ``overrides`` (``"optimizer.lr=1e-3"``). Interpolations such
as ``${oc.env:DATA_DIR}`` and ``${data.root}`` stay lazy until access or ``save_config``.

References:
    - OmegaConf: https://omegaconf.readthedocs.io/
    - ``_base_`` inheritance follows MMEngine configs: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html
"""

__all__ = [
    "BASE_KEY",
    "load_config",
    "save_config",
]

import os
from collections.abc import Iterable
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from mon.core.fileio import _atomic_write_text, _prepare_target
from mon.core.path import Path

BASE_KEY = "_base_"
"""Top-level key listing config files to inherit from (paths relative to the file)."""


# --- Helpers ---

def _load_file(path: Path, chain: tuple[Path, ...] = ()) -> DictConfig:
    """Load ``path`` and recursively merge its ``_base_`` files underneath it."""
    path = path.expanduser().resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"circular {BASE_KEY} inheritance: {cycle}")

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{path} must contain a mapping at the top level, got a list")

    bases = cfg.pop(BASE_KEY, None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    elif not OmegaConf.is_list(bases):
        raise TypeError(f"{BASE_KEY} in {path} must be a path or a list of paths, got {bases!r}")

    # Merging DictConfigs yields a DictConfig; ``cast`` narrows omegaconf's broad return type.
    merged = OmegaConf.create()
    for base in bases:
        base_path = Path(str(base)).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = cast(DictConfig, OmegaConf.merge(merged, _load_file(base_path, (*chain, path))))
    return cast(DictConfig, OmegaConf.merge(merged, cfg))


# --- Public API ---

def load_config(
    path     : str | os.PathLike[str] | None = None,
    overrides: Iterable[str] = (),
    *,
    schema   : Any = None,
) -> DictConfig:
    """Build a config from a schema, a YAML file (with ``_base_`` inheritance), and overrides.

    Args:
        path: YAML file to load. Its optional top-level ``_base_`` (a path or list of
            paths, relative to the file) is merged first, in order; bases may have bases.
        overrides: Dotlist overrides applied last, e.g. ``["optimizer.lr=1e-3", "seed=1"]``
            (typically ``sys.argv[1:]``). ``key=null`` sets ``None``.
        schema: Optional dataclass (class or instance) giving defaults and types. With a
            schema, wrong types and unknown keys raise ``omegaconf`` validation errors.

    Returns:
        The merged config. Lists are replaced, not concatenated, by later layers.

    Raises:
        FileNotFoundError: If ``path`` or a ``_base_`` file is missing.
        ValueError: On circular ``_base_`` inheritance.
        TypeError: If a file's top level isn't a mapping or ``_base_`` is malformed.
        omegaconf.errors.ValidationError: If a value doesn't match ``schema``.
    """
    layers: list[Any] = [OmegaConf.structured(schema) if schema is not None else OmegaConf.create()]
    if path is not None:
        layers.append(_load_file(Path(path)))
    overrides = list(overrides)
    if overrides:
        layers.append(OmegaConf.from_dotlist(overrides))
    return cast(DictConfig, OmegaConf.merge(*layers))


def save_config(
    cfg      : DictConfig | Any,
    path     : str | os.PathLike[str],
    *,
    resolve  : bool = True,
    overwrite: bool = True,
) -> Path:
    """Atomically write ``cfg`` (a ``DictConfig``, dataclass, or dict) as YAML; return the path.

    Args:
        cfg: Config to save.
        path: Destination file; parent directories are created.
        resolve: Resolve interpolations first, so the file records actual values
            (recommended for run directories). Missing values (``???``) are kept.
        overwrite: Replace an existing file.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is ``False``.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.structured(cfg) if not isinstance(cfg, dict) else OmegaConf.create(cfg)
    text   = OmegaConf.to_yaml(cfg, resolve=resolve)
    target = _prepare_target(path, overwrite)
    _atomic_write_text(target, text)
    return target
