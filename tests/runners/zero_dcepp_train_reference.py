"""Reference losses and weights for the Zero-DCE++ training-loop parity test (no ``mon`` imports).

``OFFICIAL_PP`` in ``test_trainer_parity.py`` comes from the official training loop
(https://github.com/Li-Chongyi/Zero-DCE_extension, ``Zero-DCE++/lowlight_train.py`` @ ``09f202b``; no license:
never copied into this repo): ``enhance_net_nopool(scale_factor=1)`` with PyTorch's default init (the script
leaves ``weights_init`` commented out), ``1600·L_TV + L_spa + 5·L_color + 10·L_exp(16)(E=0.6)``, Adam (lr 1e-4,
weight decay 1e-4), gradient clipping 0.1, on CPU (``Tensor.cuda`` is patched to a no-op: the losses hard-code
CUDA). Data, batch, epochs and seed are those of ``zero_dce_train_reference.py``.

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/runners/zero_dcepp_train_reference.py <checkout>/Zero-DCE++
"""

import importlib.util
import math
import pprint
import sys
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("zero_dce_train_reference", Path(__file__).with_name("zero_dce_train_reference.py"))
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
SEED, BATCH, EPOCHS, COUNT = _base.SEED, _base.BATCH, _base.EPOCHS, _base.COUNT
reference_images, weight_stats = _base.reference_images, _base.weight_stats


def main(code_dir: str) -> None:
    sys.path.insert(0, str(Path(code_dir).resolve()))
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    import Myloss  # official code
    import model  # official code

    torch.manual_seed(SEED)
    net = model.enhance_net_nopool(1)
    init = weight_stats(net.state_dict())
    images = torch.stack(reference_images()).float() / 255
    L_color, L_spa, L_exp, L_TV = Myloss.L_color(), Myloss.L_spa(), Myloss.L_exp(16), Myloss.L_TV()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    net.train()
    losses = []
    for _ in range(EPOCHS):
        for start in range(0, COUNT, BATCH):
            img = images[start:start + BATCH]
            enhanced, A = net(img)
            loss = (1600 * L_TV(A) + torch.mean(L_spa(enhanced, img)) + 5 * torch.mean(L_color(enhanced))
                    + 10 * torch.mean(L_exp(enhanced, 0.6)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)  # the script's deprecated clip_grad_norm
            optimizer.step()
            losses.append(loss.item())
    steps = math.ceil(COUNT / BATCH)
    result = {"init": init, "losses": losses,
              "epoch_mean": [sum(losses[e * steps:(e + 1) * steps]) / steps for e in range(EPOCHS)],
              "final": weight_stats(net.state_dict())}
    print(f"# Generated from the official Zero-DCE++ loop with torch {torch.__version__}.")
    print("OFFICIAL_PP: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
