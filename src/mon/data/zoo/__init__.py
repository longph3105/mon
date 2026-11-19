"""Concrete datasets, each a short ``FolderDataset`` declaration.

Imported automatically on the first ``DATASETS`` lookup; import each dataset module here so it
registers itself.
"""

__all__: list[str] = []

from mon.data.zoo import classify, detect, lol, loli_street, ntire, segment, sice, testkit, unpaired_lol  # noqa: F401
