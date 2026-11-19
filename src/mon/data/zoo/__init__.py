"""Concrete datasets, each a short ``FolderDataset`` declaration.

Imported automatically on the first ``DATASETS`` lookup; import each dataset module here so it
registers itself.
"""

__all__: list[str] = []

from mon.data.zoo import (  # noqa: F401
    classify, detect, fivek, lol, loli_street, lsrw, ntire, segment, sice, testkit, track, uhd_ll, unpaired_lol,
)
