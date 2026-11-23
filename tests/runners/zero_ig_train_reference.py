"""Reference losses and weights for the Zero-IG training-loop parity test (no ``mon`` imports).

``OFFICIAL_ZIG`` in ``test_trainer_parity.py`` comes from the official training loop
(https://github.com/Doyle59217/ZeroIG, ``train.py`` @ ``5af4b8a``; no license: never copied into this repo):
``Network()`` with ``enhance_weights_init`` on ``enhance.in_conv``/``conv``/``out_conv``, ``model._loss``, Adam
(lr 3e-4, betas 0.9/0.999, weight decay 3e-4; without the script's ``capturable=True``, which needs CUDA and
leaves the math unchanged), ``clip_grad_norm_`` 5 between ``backward`` and ``step``, batch 1, file order, from
``torch.manual_seed(2)``, on CPU. It runs ``EPOCHS`` = 2 epochs over the first ``COUNT`` reference images.

Two epochs on purpose: the script calls ``model.train()`` once, before its loop, and its epoch-0 test pass calls
``model.eval()`` without switching back, so from epoch 1 on the official training runs with batch norm in eval
mode (running statistics, not updated); ``mon``'s recipe reproduces this (``eval_mode_after: 1``). The test
pass itself (a no-grad forward, no RNG) changes nothing else and is left out. Inputs are ``uint8 · (1/255)``
as ``mon`` computes them (see ``sci_train_reference.py``: batch-norm biases turn one-ulp input differences into
``lr``-sized steps).

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run --with matplotlib python -I tests/runners/zero_ig_train_reference.py <checkout>  # utils.py imports it
"""

import importlib.util
import sys
from pathlib import Path

import torch

_fmt_spec = importlib.util.spec_from_file_location(
    "reference_format", Path(__file__).parents[1] / "reference_format.py",
)
assert _fmt_spec is not None and _fmt_spec.loader is not None
_fmt = importlib.util.module_from_spec(_fmt_spec)
_fmt_spec.loader.exec_module(_fmt)
format_literal = _fmt.format_literal

_spec = importlib.util.spec_from_file_location(
    "zero_dce_train_reference", Path(__file__).with_name("zero_dce_train_reference.py"),
)
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
weight_stats = _base.weight_stats

SEED, EPOCHS, COUNT = 2, 2, 4


def reference_images() -> list[torch.Tensor]:
    return _base.reference_images()[:COUNT]


def main(code_dir: str) -> None:
    sys.path.insert(0, str(Path(code_dir).resolve()))
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    from torch import nn

    import model  # official code

    torch.manual_seed(SEED)
    net = model.Network()
    for block in (net.enhance.in_conv, net.enhance.conv, net.enhance.out_conv):
        block.apply(net.enhance_weights_init)
    init = weight_stats(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=3e-4)
    net.train()
    losses, epoch_mean = [], []
    for epoch in range(EPOCHS):
        if epoch == 1:
            net.eval()  # train.py: the epoch-0 test pass, never switched back
        for image in reference_images():
            x = (image.float() * (1 / 255)).unsqueeze(0)  # mon's ToDtype(scale=True); see the module docstring
            optimizer.zero_grad()
            loss = net._loss(x)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        epoch_mean.append(sum(losses[-COUNT:]) / COUNT)
    result = {"init": init, "losses": losses, "epoch_mean": epoch_mean,
              "final": weight_stats(net.state_dict())}
    print(f"# Generated from the official Zero-IG loop with torch {torch.__version__}.")
    print(format_literal("OFFICIAL_ZIG", result))


if __name__ == "__main__":
    main(sys.argv[1])
