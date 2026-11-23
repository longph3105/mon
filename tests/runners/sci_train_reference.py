"""Reference losses and weights for the SCI (CVPR) training-loop parity test (no ``mon`` imports).

``OFFICIAL_SCI`` in ``test_trainer_parity.py`` comes from the official training loop
(https://github.com/vis-opt-group/SCI, ``CVPR/train.py`` @ ``f6f88fd``; MIT): ``Network(stage=3)`` with
``weights_init`` on the six sub-blocks in the script's order, ``model._loss`` (``LossFunction`` per stage),
Adam (lr 3e-4, betas 0.9/0.999, weight decay 3e-4), ``clip_grad_norm_`` 5 between ``backward`` and ``step``,
batch 1, from ``torch.manual_seed(2)``, on CPU (``Tensor.cuda`` is patched to a no-op: the loss hard-codes
CUDA). It runs on the first ``COUNT`` of ``zero_dce_train_reference.reference_images()`` in file order (the
script's loader shuffles: data handling, not the loop) for ``EPOCHS`` epochs.

Inputs are ``uint8 · (1/255)``, as ``mon``'s ``v2.ToDtype(scale=True)`` computes them, not the official
``ToTensor``'s ``/ 255``: the two differ by one float32 ulp on about 40% of pixels. SCI needs bit-identical
inputs: its convolutions before a batch norm have biases whose true gradient is zero (the norm subtracts the
batch mean), and Adam turns the rounding noise in those gradients into steps of about ``lr``. With ``/ 255``
the per-epoch losses drift by ~2e-4 (relative) after one epoch; the loop itself is unchanged. The same noise
keeps ``mon``'s network (identical init and first-step loss) from matching to float32 rounding; the test's
tolerances say how close it gets.

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/runners/sci_train_reference.py <checkout>/CVPR
"""

import importlib.util
import pprint
import sys
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("zero_dce_train_reference", Path(__file__).with_name("zero_dce_train_reference.py"))
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
weight_stats = _base.weight_stats

SEED, EPOCHS, COUNT = 2, 3, 5


def reference_images() -> list[torch.Tensor]:
    return _base.reference_images()[:COUNT]


def main(code_dir: str) -> None:
    sys.path.insert(0, str(Path(code_dir).resolve()))
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    from torch import nn

    import model  # official code

    torch.manual_seed(SEED)
    net = model.Network(stage=3)
    for block in (net.enhance.in_conv, net.enhance.conv, net.enhance.out_conv,
                  net.calibrate.in_conv, net.calibrate.convs, net.calibrate.out_conv):
        block.apply(net.weights_init)
    init = weight_stats(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=3e-4)
    epoch_mean = []
    for _ in range(EPOCHS):
        net.train()
        losses = []
        for image in reference_images():
            x = (image.float() * (1 / 255)).unsqueeze(0)  # mon's ToDtype(scale=True); see the module docstring
            optimizer.zero_grad()
            loss = net._loss(x)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        epoch_mean.append(sum(losses) / len(losses))
    result = {"init": init, "epoch_mean": epoch_mean, "final": weight_stats(net.state_dict())}
    print(f"# Generated from the official SCI (CVPR) loop with torch {torch.__version__}.")
    print("OFFICIAL_SCI: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
