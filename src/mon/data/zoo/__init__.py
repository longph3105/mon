"""Concrete datasets, each a short ``FolderDataset`` declaration.

Imported automatically on the first ``DATASETS`` lookup; import each dataset module here so it
registers itself.
"""

__all__: list[str] = []

from mon.data.zoo import classify, lol, sice  # noqa: F401
