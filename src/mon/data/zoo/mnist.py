"""MNIST and Fashion-MNIST: 28×28 grayscale image classification.

Stored as PNG class folders (``<split>/<class>/<index>.png`` + ``classes.txt``), decoded from
the official files after checking torchvision's MD5s; ``<index>`` is the position in the
upstream split.

Warning:
    No exact duplicates in either (near-duplicates not checked).

Examples:
    >>> from mon.core.enums import Task
    >>> from mon.data.dataset import DATASETS
    >>> {"mnist", "fashion_mnist"} <= set(DATASETS.names(tasks=Task.CLASSIFY))
    True
    >>> MNIST.image_mode, FashionMNIST.image_mode
    ('gray', 'gray')

References:
    - MNIST: LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.
    - Fashion-MNIST: Xiao et al., 2017. https://arxiv.org/abs/1708.07747,
      https://github.com/zalandoresearch/fashion-mnist
"""

__all__ = [
    "FashionMNIST",
    "MNIST",
]

from mon.core.enums import Task
from mon.data.dataset import ClassFolderDataset, classification_layout, register_dataset


@register_dataset("mnist", tasks={Task.CLASSIFY})
class MNIST(ClassFolderDataset):
    """MNIST: 60000 train / 10000 test grayscale 28×28 digits, classes ``0``–``9``."""

    folder     = "mnist"
    layout     = classification_layout("train", "test")
    image_mode = "gray"


@register_dataset("fashion_mnist", tasks={Task.CLASSIFY})
class FashionMNIST(ClassFolderDataset):
    """Fashion-MNIST: 60000 train / 10000 test grayscale 28×28 images, 10 clothing classes."""

    folder     = "fashion_mnist"
    layout     = classification_layout("train", "test")
    image_mode = "gray"
