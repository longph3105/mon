"""Concrete datasets, each a short ``FolderDataset`` declaration.

One module per upstream dataset family, named after its id (``lol``: ``lol_v1``, ``lol_v2_real``;
``mot``: ``mot16``, ``mot20``; ``exdark``: ``exdark``), never by task: a dataset's tasks are its
``register_dataset(..., tasks=...)`` tags, and ``DATASETS.names(tasks=...)`` lists them. Shared
layouts live in ``mon.data.dataset``; shared label definitions in their family's module
(``cityscapes``).

Imported automatically on the first ``DATASETS`` lookup; import each dataset module here so it
registers itself.
"""

__all__: list[str] = []

from mon.data.zoo import (  # noqa: F401
    cifar, cityscapes, darkface, exdark, lol, loli_street, lsrw, mit_5k, mnist, mot, nightcity, ntire, sice, testkit,
    uhd_ll, unpaired_lol, widerface, ydld,
)
