"""Name-to-object registries with metadata filtering.

Create one ``Registry`` per kind of object in the package that owns it, e.g.
``MODELS = Registry[type[nn.Module]]("models")`` in ``mon.nn``.
"""

__all__ = [
    "Entry",
    "Registry",
    "RegistryKeyError",
]

import difflib
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
    """A registered object, its (lowercase) name, and read-only metadata."""

    name: str
    obj : T
    meta: Mapping[str, Any]


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

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, Entry[T]] = {}

    # --- Container ---

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {len(self)} entries)"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._entries

    def __iter__(self) -> Iterator[str]:
        """Iterate over names in sorted order."""
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
                Re-registering the same object with the same metadata is a no-op.
        """
        if name is None:
            name = getattr(obj, "__name__", None)
            if not isinstance(name, str):
                raise TypeError(f"{obj!r} has no __name__; pass name= explicitly")
        key   = _normalize(name)
        entry = Entry(name=key, obj=obj, meta=MappingProxyType(dict(meta)))

        existing = self._entries.get(key)
        if existing is not None and not replace:
            if existing.obj is obj and dict(existing.meta) == dict(meta):
                return existing
            raise RegistryKeyError(
                f"{key!r} is already registered in {self.name!r} (as {existing.obj!r}); pass replace=True to overwrite"
            )
        self._entries[key] = entry
        return entry

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
        try:
            return self._entries[key]
        except KeyError:
            close = difflib.get_close_matches(key, self._entries, n=3, cutoff=0.6)
            hint  = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
            raise RegistryKeyError(f"{name!r} is not registered in {self.name!r}{hint}") from None

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
        ``key`` don't match.
        """
        return [
            e for _, e in sorted(self._entries.items())
            if all(_matches(e.meta.get(k, _ABSENT), v) for k, v in criteria.items())
        ]

    def names(self, **criteria: Any) -> list[str]:
        """Return sorted names of entries matching ``criteria`` (see ``entries``)."""
        return [e.name for e in self.entries(**criteria)]
