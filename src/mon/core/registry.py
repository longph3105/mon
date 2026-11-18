"""Name-to-object registries with metadata filtering.

Create one ``Registry`` per kind of object in the package that owns it, e.g.
``MODELS = Registry[type[nn.Module]]("models")`` in ``mon.models``.

Entries are added eagerly (``add``/``register``, the object already exists) or lazily
(``add_lazy``, an import path resolved on first use), so a registry can list and filter
many entries without importing the modules that define them.
"""

__all__ = [
    "Entry",
    "Registry",
    "RegistryKeyError",
]

import difflib
import importlib
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_ABSENT = object()


class RegistryKeyError(KeyError):
    """``KeyError`` for unknown or duplicate registry names, with a readable message."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True, slots=True)
class Entry[T]:
    """A registered object, its (lowercase) name, and read-only metadata.

    ``target`` is the ``"module:attr"`` import path for entries added with ``add_lazy``.
    """

    name  : str
    obj   : T
    meta  : Mapping[str, Any]
    target: str | None = None


@dataclass(frozen=True, slots=True)
class _Pending:
    """A lazy entry whose ``target`` hasn't been imported yet."""

    name  : str
    target: str
    meta  : Mapping[str, Any]


def _check_target(target: str) -> str:
    module, sep, attr = target.partition(":") if isinstance(target, str) else ("", "", "")
    if not (sep and module.strip() and attr.strip()):
        raise ValueError(f"lazy target must look like 'package.module:Attr', got {target!r}")
    return target


def _import_target(target: str) -> Any:
    module_name, _, attr_path = target.partition(":")
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _target_of(obj: Any) -> str | None:
    """``"module:qualname"`` of ``obj``, or ``None`` if it has no module or qualified name."""
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    return f"{module}:{qualname}" if isinstance(module, str) and isinstance(qualname, str) else None


def _normalize(name: str) -> str:
    """Return the lookup key for ``name``: stripped and lowercased."""
    if not isinstance(name, str):
        raise TypeError(f"registry names must be str, got {type(name).__name__}")
    key = name.strip().lower()
    if not key:
        raise ValueError("registry names must be non-empty")
    return key


def _matches(actual: Any, wanted: Any) -> bool:
    """``actual == wanted``, or ``wanted in actual`` when ``actual`` is a non-string collection."""
    if actual is _ABSENT:
        return False
    if actual == wanted:
        return True
    return isinstance(actual, Collection) and not isinstance(actual, (str, bytes)) and wanted in actual


def _same_definition(a: Any, b: Any) -> bool:
    """Whether ``a`` and ``b`` come from the same source definition (same module and qualified name)."""
    names = [(getattr(x, "__module__", None), getattr(x, "__qualname__", None)) for x in (a, b)]
    return None not in names[0] and names[0] == names[1]


class Registry[T]:
    """A mapping from case-insensitive names to objects, with metadata for filtering.

    Names default to ``obj.__name__.lower()``; registering a different object under a
    taken name raises unless ``replace=True``. Registered objects are not modified.

    Examples:
        >>> LOSSES = Registry[type]("losses")
        >>> @LOSSES.register("l1", tasks={"denoise", "deblur"})
        ... class L1Loss: ...
        >>> LOSSES.get("L1") is L1Loss, LOSSES.names(tasks="deblur")
        (True, ['l1'])
    """

    def __init__(self, name: str, *, autoload: str | None = None) -> None:
        """Args:
            name: Label used in messages.
            autoload: Module (usually a package) imported on the first lookup so that its
                registrations are present, e.g. ``"mon.data.zoo"``. Registration itself never
                triggers it, so importing the module that defines the registry stays cheap.
        """
        self.name      = name
        self._autoload = autoload
        self._loaded   = autoload is None
        self._entries: dict[str, Entry[T] | _Pending] = {}

    def _ensure_loaded(self) -> None:
        """Import ``autoload`` once, before the first lookup."""
        if self._loaded:
            return
        self._loaded = True  # set first: the imported modules register into this registry
        try:
            importlib.import_module(self._autoload or "")
        except BaseException:
            self._loaded = False
            raise

    # --- Container ---

    def __repr__(self) -> str:
        pending = "" if self._loaded else f", autoload={self._autoload!r} pending"
        return f"{type(self).__name__}({self.name!r}, {len(self._entries)} entries{pending})"  # no autoload

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        self._ensure_loaded()
        return isinstance(name, str) and name.strip().lower() in self._entries

    def __iter__(self) -> Iterator[str]:
        """Iterate over names in sorted order."""
        self._ensure_loaded()
        return iter(sorted(self._entries))

    # --- Registration ---

    def add(self, obj: T, name: str | None = None, *, replace: bool = False, **meta: Any) -> Entry[T]:
        """Register ``obj`` and return its entry.

        Args:
            obj: Object to register (class, function, …).
            name: Registry name; defaults to ``obj.__name__``. Stored lowercase.
            replace: Overwrite an existing entry with the same name.
            **meta: Metadata stored on the entry, e.g. ``arch="yolo12"``, ``tasks={...}``.

        Raises:
            TypeError: If ``name`` is omitted and ``obj`` has no ``__name__``.
            RegistryKeyError: If the name is taken by a different object and not ``replace``.
                Re-registering the same object with the same metadata is a no-op; an object
                with the same ``__module__`` and ``__qualname__`` (its module imported again)
                replaces the old entry.
        """
        if name is None:
            name = getattr(obj, "__name__", None)
            if not isinstance(name, str):
                raise TypeError(f"{obj!r} has no __name__; pass name= explicitly")
        key   = _normalize(name)
        entry = Entry(name=key, obj=obj, meta=MappingProxyType(dict(meta)))

        existing = self._entries.get(key)
        if isinstance(existing, _Pending) and not replace:
            # The lazy target's own module registers the same object: same target and metadata
            # resolve the pending entry; anything else is a conflict.
            if _target_of(obj) == existing.target and dict(existing.meta) == dict(meta):
                entry = Entry(name=key, obj=obj, meta=existing.meta, target=existing.target)
                self._entries[key] = entry
                return entry
            raise RegistryKeyError(
                f"{key!r} is already registered in {self.name!r} (lazily, as {existing.target!r}); "
                "pass replace=True to overwrite"
            )
        if isinstance(existing, Entry) and not replace:
            if existing.obj is obj and dict(existing.meta) == dict(meta):
                return existing
            if existing.obj is not obj and _same_definition(existing.obj, obj):
                # The defining module was imported again (importlib.reload, notebooks, pytest's
                # importlib mode): the new object supersedes the old one.
                self._entries[key] = entry
                return entry
            raise RegistryKeyError(
                f"{key!r} is already registered in {self.name!r} (as {existing.obj!r}); pass replace=True to overwrite"
            )
        self._entries[key] = entry
        return entry

    def add_lazy(self, name: str, target: str, *, replace: bool = False, **meta: Any) -> None:
        """Register ``target`` (``"package.module:Attr"``) under ``name`` without importing it.

        ``names`` filters by metadata without importing; ``get``, ``build``, ``entry``, and
        ``entries`` import the target on first use and cache the object.

        Raises:
            ValueError: If ``target`` isn't of the form ``"module:attr"``.
            RegistryKeyError: If the name is taken (by another target, other metadata, or an
                eager entry) and not ``replace``. Repeating the same call is a no-op.

        Examples:
            >>> REG = Registry[type]("containers")
            >>> REG.add_lazy("ordered_dict", "collections:OrderedDict", family="dict")
            >>> REG.names(family="dict"), REG.entry("ordered_dict").target
            (['ordered_dict'], 'collections:OrderedDict')
        """
        key      = _normalize(name)
        pending  = _Pending(name=key, target=_check_target(target), meta=MappingProxyType(dict(meta)))
        existing = self._entries.get(key)
        if existing is not None and not replace:
            if existing.target == target and dict(existing.meta) == dict(meta):
                return
            shown = existing.target or (existing.obj if isinstance(existing, Entry) else None)
            raise RegistryKeyError(
                f"{key!r} is already registered in {self.name!r} (as {shown!r}); pass replace=True to overwrite"
            )
        self._entries[key] = pending

    def register(self, name: str | None = None, *, replace: bool = False, **meta: Any) -> Callable[[T], T]:
        """Decorator form of ``add``; returns the decorated object unchanged.

        Examples:
            >>> MODELS = Registry[type]("models")
            >>> @MODELS.register(arch="unet")
            ... class UNet: ...
            >>> MODELS.entry("unet").meta["arch"]
            'unet'
        """
        def decorator(obj: T) -> T:
            self.add(obj, name, replace=replace, **meta)
            return obj
        return decorator

    def remove(self, name: str) -> Entry[T]:
        """Unregister ``name`` and return its entry."""
        entry = self.entry(name)
        del self._entries[entry.name]
        return entry

    # --- Lookup ---

    def entry(self, name: str) -> Entry[T]:
        """Return the entry for ``name`` (case-insensitive).

        Raises:
            RegistryKeyError: If ``name`` isn't registered; suggests close matches.
        """
        key = _normalize(name)
        self._ensure_loaded()
        try:
            found = self._entries[key]
        except KeyError:
            close = difflib.get_close_matches(key, self._entries, n=3, cutoff=0.6)
            hint  = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
            raise RegistryKeyError(f"{name!r} is not registered in {self.name!r}{hint}") from None
        return self._resolve(found)

    def _resolve(self, found: Entry[T] | _Pending) -> Entry[T]:
        """Import a pending entry's target (once) and return the real entry.

        Raises:
            ImportError: If the target's module or attribute can't be imported.
        """
        if isinstance(found, Entry):
            return found
        try:
            obj = _import_target(found.target)
        except (ImportError, AttributeError) as err:
            raise ImportError(f"cannot load {found.name!r} in {self.name!r} from {found.target!r}: {err}") from err
        entry = Entry(name=found.name, obj=obj, meta=found.meta, target=found.target)
        self._entries[found.name] = entry
        return entry

    def get(self, name: str) -> T:
        """Return the object registered as ``name``. See ``entry`` for errors."""
        return self.entry(name).obj

    def build(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Call the object registered as ``name`` with ``args``/``kwargs`` and return the result.

        Raises:
            RegistryKeyError: If ``name`` isn't registered.
            TypeError: If the registered object isn't callable.
        """
        obj = self.get(name)
        if not callable(obj):
            raise TypeError(f"{name!r} in {self.name!r} is not callable: {obj!r}")
        return obj(*args, **kwargs)

    def entries(self, **criteria: Any) -> list[Entry[T]]:
        """Return entries whose metadata matches all ``criteria``, sorted by name.

        A criterion ``key=value`` matches when ``meta[key] == value``, or when
        ``meta[key]`` is a non-string collection containing ``value``. Entries lacking
        ``key`` don't match. Matching lazy entries are imported; ``names`` filters without
        importing.
        """
        return [self._resolve(self._entries[name]) for name in self.names(**criteria)]

    def names(self, **criteria: Any) -> list[str]:
        """Return sorted names of entries matching ``criteria`` (see ``entries``); never imports lazy targets."""
        self._ensure_loaded()
        return [
            key for key, e in sorted(self._entries.items())
            if all(_matches(e.meta.get(k, _ABSENT), v) for k, v in criteria.items())
        ]
