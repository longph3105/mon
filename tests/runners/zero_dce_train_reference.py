"""Reference losses and weights for the Zero-DCE training-loop parity test (no ``mon`` imports).

``OFFICIAL`` in ``test_trainer_parity.py`` comes from the official training loop
(https://github.com/Li-Chongyi/Zero-DCE, ``Zero-DCE_code/lowlight_train.py`` @ ``e0f4adc``; no license:
never copied into this repo): ``enhance_net_nopool`` + ``weights_init``, the four ``Myloss`` terms with the
official weights, Adam (lr 1e-4, weight decay 1e-4), ``clip_grad_norm`` 0.1, on CPU (``Tensor.cuda`` is
patched to a no-op: the losses hard-code CUDA). It runs on ``reference_images()`` in file order (no
shuffle; the official loader's shuffle and PIL resize are data handling, not the loop) for
``EPOCHS`` epochs of batch ``BATCH``, from ``torch.manual_seed(SEED)``.

Running it executes third-party code: get the owner's go-ahead first. Regenerate with:

    uv run python -I tests/runners/zero_dce_train_reference.py <checkout>/Zero-DCE_code
"""

import math
import pprint
import sys
from pathlib import Path
from typing import Any

import torch

SEED, BATCH, EPOCHS, COUNT, SIZE = 0, 8, 3, 16, (48, 64)
STATS_INDICES = (0, 7, 101, 523, 1000, 2047, -1)


def reference_images() -> list[torch.Tensor]:
    """``COUNT`` deterministic low-light-like ``uint8`` ``[3, H, W]`` images (no RNG)."""
    h, w = SIZE
    i = torch.arange(h, dtype=torch.float64)[:, None]
    j = torch.arange(w, dtype=torch.float64)[None, :]
    out = []
    for n in range(COUNT):
        img = torch.empty(3, h, w, dtype=torch.float64)
        for c in range(3):
            wave = torch.sin(0.21 * (c + 1) * i + 0.13 * (n + 1) * j + c) * torch.cos(0.07 * j - 0.11 * i + n)
            noise = torch.frac(torch.sin((i * w + j) * 12.9898 + (n * 3 + c) * 78.233) * 43758.5453)
            img[c] = 0.04 + 0.1 * (1 + wave) * (0.6 + 0.1 * c) + 0.03 * noise
        out.append((img.clamp(0, 1) * 255).round().to(torch.uint8))
    return out


def weight_stats(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    flat = torch.cat([v.detach().double().flatten() for v in state.values()])
    return {"sum": flat.sum().item(), "sumsq": (flat**2).sum().item(),
            "values": [flat[k % flat.numel()].item() for k in STATS_INDICES]}


def main(code_dir: str) -> None:
    sys.path.insert(0, str(Path(code_dir).resolve()))
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    import Myloss  # official code
    import model  # official code
    from lowlight_train import weights_init  # official code

    torch.manual_seed(SEED)
    net = model.enhance_net_nopool()
    net.apply(weights_init)
    init = weight_stats(net.state_dict())
    images = torch.stack(reference_images()).float() / 255
    L_color, L_spa, L_exp, L_TV = Myloss.L_color(), Myloss.L_spa(), Myloss.L_exp(16, 0.6), Myloss.L_TV()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    net.train()
    losses = []
    for _ in range(EPOCHS):
        for start in range(0, COUNT, BATCH):
            img = images[start:start + BATCH]
            _, enhanced, A = net(img)
            loss = (200 * L_TV(A) + torch.mean(L_spa(enhanced, img)) + 5 * torch.mean(L_color(enhanced))
                     + 10 * torch.mean(L_exp(enhanced)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()
            losses.append(loss.item())
    steps = math.ceil(COUNT / BATCH)
    result = {"init": init, "losses": losses,
              "epoch_mean": [sum(losses[e * steps:(e + 1) * steps]) / steps for e in range(EPOCHS)],
              "final": weight_stats(net.state_dict())}
    print(f"# Generated from the official Zero-DCE loop with torch {torch.__version__}.")
    print("OFFICIAL: dict = " + pprint.pformat(result, width=118, sort_dicts=False))


if __name__ == "__main__":
    main(sys.argv[1])
