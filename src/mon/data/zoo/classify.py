"""Image classification datasets: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST.

Stored as PNG class folders (``<split>/<class>/<index>.png`` + ``classes.txt``), decoded from
the official files after checking torchvision's MD5s; ``<index>`` is the position in the
upstream split. Ids ``cifar10``/``cifar100`` follow torchvision rather than the workspace id
rule (owner's choice).

Warning:
    CIFAR-100 contains 26 groups of byte-identical images (52 images): 10 test images also
    appear in train (6 with a different label), and 16 groups carry conflicting labels
    (girl/baby, otter/seal, mouse/shrew, snake/worm, spider/crab, whale/shark,
    oak_tree/willow_tree). It is kept as published so results stay comparable. CIFAR-10,
    MNIST and Fashion-MNIST have no exact duplicates (near-duplicates not checked).

Examples:
    >>> from mon.core.enums import Task
    >>> from mon.data.dataset import DATASETS
    >>> sorted(set(DATASETS.names(tasks=Task.CLASSIFY)) & {"cifar10", "cifar100", "mnist", "fashion_mnist"})
    ['cifar10', 'cifar100', 'fashion_mnist', 'mnist']
    >>> MNIST.image_mode, CIFAR10.image_mode
    ('gray', 'rgb')

    Loading needs the data on this machine (``DATA_DIR``):

    >>> ds = DATASETS.build("cifar10", "test")                    # doctest: +SKIP
    >>> len(ds), ds.classes.names[:3]                             # doctest: +SKIP
    (10000, ['airplane', 'automobile', 'bird'])
    >>> s = ds[0]                                                 # doctest: +SKIP
    >>> s["image"].shape, ds.classes.names[int(s["label"])]       # doctest: +SKIP
    (torch.Size([3, 32, 32]), 'airplane')

References:
    - CIFAR: Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.
      https://www.cs.toronto.edu/~kriz/cifar.html
    - MNIST: LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.
    - Fashion-MNIST: Xiao et al., 2017. https://arxiv.org/abs/1708.07747,
      https://github.com/zalandoresearch/fashion-mnist
"""

__all__ = [
    "CIFAR10",
    "CIFAR100",
    "FashionMNIST",
    "MNIST",
]

from mon.core.enums import Task
from mon.data.dataset import ClassFolderDataset, classification_layout, register_dataset


@register_dataset("cifar10", tasks={Task.CLASSIFY})
class CIFAR10(ClassFolderDataset):
    """CIFAR-10: 50000 train / 10000 test RGB 32×32 images, 10 classes."""

    folder = "cifar10"
    layout = classification_layout("train", "test")


@register_dataset("cifar100", tasks={Task.CLASSIFY})
class CIFAR100(ClassFolderDataset):
    """CIFAR-100: 50000 train / 10000 test RGB 32×32 images, 100 fine classes (no coarse labels)."""

    folder = "cifar100"
    layout = classification_layout("train", "test")


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
