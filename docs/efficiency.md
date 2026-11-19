# Efficiency: LLE models across resolutions

Parameters, MACs, latency and peak memory of every low-light enhancement model in `mon`, on the
`testkit_efficiency` ladder (5 SICE scenes per size, 256² to 8K). Measured 2026-11-19 at commit `ca5765c`.

## Setup

- RTX 3090 (24 GiB), driver 595.99.02, CUDA 13.2, torch 2.15.0, Python 3.14, i7-7700; batch 1, float32,
  `torch.inference_mode`, no `torch.compile`, no AMP.
- Command per model (released default weights; `seed=0`):

  ```bash
  uv run mon benchmark model=<name> weights=default data=testkit_efficiency device=cuda:0 seed=0 \
      output=$DATA_DIR/runs/benchmark/efficiency_2026_11_19/<name>.yaml
  ```

  Timing: 5 untimed passes on the first image of a size, then 20 timed passes per image (CLODE: 1 and 3).
  Latency is the median over all timed passes. Memory is peak `torch.cuda.max_memory_allocated` over the
  images, so it excludes the CUDA context (~300 MiB) and the allocator's cache. MACs come from
  `FlopCounterMode` (FLOPs / 2). It counts matmuls and convolutions only, so element-wise work is not
  included (curve iterations, Retinex divisions, attention softmax).
- **Caveat:** these peaks were measured before a benchmarker fix (2026-11-19). Each one includes the other 4
  images of its size held on the GPU: +12 MiB at 600×400, +95 MiB at 1080p, +380 MiB at 4K, +1.5 GiB at 8K.
  To be re-measured.
- Raw YAML/CSV per model: `$DATA_DIR/runs/benchmark/efficiency_2026_11_19/`.
- `_s12` rows: Zero-DCE++ and SGZ with `model_args.scale_factor=12`, their official test setting. The curve is
  estimated on a 12× downsampled input and upsampled. Plain rows use the default `scale_factor=1`.
- Rows are sorted by 1080p latency. "OOM" means the forward pass ran out of the 24 GiB.

## Forward pass

### Median latency (ms)

| Model | Params | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sci++` | 22,898 | 0.262 | 0.279 | 0.264 | 0.916 | 1.94 | 3.45 | 7.59 | 29.8 |
| `sci` | 5,877 | 0.278 | 0.37 | 0.375 | 1.48 | 3.18 | 5.61 | 12.6 | 49.6 |
| `zero_dce++_s12` | 10,561 | 0.889 | 0.902 | 0.888 | 2.08 | 3.86 | 6.38 | 13.7 | 53.1 |
| `sgz_s12` | 10,561 | 0.938 | 0.961 | 0.937 | 2.09 | 3.88 | 6.31 | 13.4 | 53.1 |
| `flol` | 94,518 | 7.84 | 9.01 | 8.48 | 13 | 25.8 | 44.6 | 101 | 403 |
| `sgz` | 10,561 | 1.12 | 3.94 | 3.61 | 14.1 | 31.5 | 56 | 126 | OOM |
| `zero_dce++` | 10,561 | 1.11 | 3.99 | 3.64 | 14.2 | 31.6 | 56.2 | 126 | OOM |
| `ruas` | 3,438 | 3.49 | 4.55 | 4.24 | 15.2 | 33.3 | 59.3 | 133 | 553 |
| `zero_dce` | 79,416 | 1.04 | 3.7 | 3.38 | 16.6 | 37.1 | 63.5 | 142 | OOM |
| `zero_ig` | 86,572 | 2.6 | 8.85 | 7.98 | 29.5 | 65.6 | 116 | 262 | OOM |
| `retinexnet` | 555,205 | 3.58 | 13.9 | 12.5 | 47.5 | 106 | 192 | OOM | OOM |
| `pairlie` | 341,767 | 3.58 | 14.6 | 12.8 | 50 | 112 | 201 | 446 | OOM |
| `dccnet` | 13,188,724 | 12.9 | 45.4 | 41.9 | 162 | 365 | 643 | 1,446 | OOM |
| `hvi_cidnet` | 1,975,569 | 18 | 55.5 | 51.2 | 187 | 420 | 743 | 1,667 | OOM |
| `clode` | 207,105 | 379 | 1,551 | 1,215 | 4,580 | 10,183 | 18,069 | OOM | OOM |

### Peak CUDA memory (MiB)

| Model | Params | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sci++` | 22,898 | 6 | 24 | 23 | 84 | 192 | 338 | 759 | 3,040 |
| `sci` | 5,877 | 6 | 24 | 23 | 84 | 192 | 338 | 759 | 3,040 |
| `zero_dce++_s12` | 10,561 | 7 | 28 | 27 | 100 | 227 | 399 | 897 | 3,591 |
| `sgz_s12` | 10,561 | 7 | 28 | 27 | 100 | 227 | 399 | 897 | 3,591 |
| `flol` | 94,518 | 30 | 119 | 110 | 418 | 940 | 1,663 | 3,738 | 14,950 |
| `sgz` | 10,561 | 85 | 338 | 312 | 1,190 | 2,679 | 4,753 | 10,697 | OOM |
| `zero_dce++` | 10,561 | 85 | 338 | 312 | 1,190 | 2,679 | 4,753 | 10,697 | OOM |
| `ruas` | 3,438 | 25 | 99 | 91 | 353 | 788 | 1,393 | 3,134 | 12,536 |
| `zero_dce` | 79,416 | 74 | 295 | 273 | 1,348 | 3,036 | 5,386 | 12,120 | OOM |
| `zero_ig` | 86,572 | 54 | 214 | 196 | 752 | 1,687 | 2,996 | 6,743 | OOM |
| `retinexnet` | 555,205 | 162 | 642 | 588 | 2,244 | 5,041 | 8,954 | OOM | OOM |
| `pairlie` | 341,767 | 39 | 213 | 198 | 746 | 1,675 | 2,973 | 6,685 | OOM |
| `dccnet` | 13,188,724 | 125 | 350 | 327 | 1,219 | 2,681 | 4,719 | 10,557 | OOM |
| `hvi_cidnet` | 1,975,569 | 93 | 343 | 315 | 1,187 | 2,663 | 4,715 | 10,601 | OOM |
| `clode` | 207,105 | 170 | 669 | 610 | 2,342 | 5,266 | 9,355 | OOM | OOM |

### GMACs (mean over the 5 images)

| Model | Params | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sci++` | 22,898 | 0.0159 | 0.0637 | 0.0583 | 0.224 | 0.504 | 0.896 | 2.02 | 8.06 |
| `sci` | 5,877 | 0.0159 | 0.0637 | 0.0583 | 0.224 | 0.504 | 0.896 | 2.02 | 8.06 |
| `zero_dce++_s12` | 10,561 | 0.00444 | 0.0178 | 0.0166 | 0.0641 | 0.145 | 0.258 | 0.58 | 2.32 |
| `sgz_s12` | 10,561 | 0.00444 | 0.0178 | 0.0166 | 0.0641 | 0.145 | 0.258 | 0.58 | 2.32 |
| `flol` | 94,518 | 1.02 | 4.08 | 3.74 | 14.4 | 32.3 | 57.4 | 129 | 517 |
| `sgz` | 10,561 | 0.66 | 2.64 | 2.42 | 9.29 | 20.9 | 37.1 | 83.6 | OOM |
| `zero_dce++` | 10,561 | 0.66 | 2.64 | 2.42 | 9.29 | 20.9 | 37.1 | 83.6 | OOM |
| `ruas` | 3,438 | 0.214 | 0.856 | 0.784 | 3.01 | 6.77 | 12 | 27.1 | 108 |
| `zero_dce` | 79,416 | 5.19 | 20.8 | 19 | 73 | 164 | 292 | 657 | OOM |
| `zero_ig` | 86,572 | 10.5 | 41.9 | 38.4 | 147 | 331 | 589 | 1,326 | OOM |
| `retinexnet` | 555,205 | 21.7 | 86.9 | 79.6 | 305 | 687 | 1,222 | OOM | OOM |
| `pairlie` | 341,767 | 22.3 | 89.4 | 81.8 | 314 | 707 | 1,257 | 2,828 | OOM |
| `dccnet` | 13,188,724 | 63.9 | 256 | 234 | 898 | 2,021 | 3,593 | 8,085 | OOM |
| `hvi_cidnet` | 1,975,569 | 7.88 | 31.5 | 28.9 | 111 | 249 | 443 | 997 | OOM |
| `clode` | 207,105 | 655 | 2,923 | 2,262 | 8,950 | 19,541 | 33,678 | OOM | OOM |

## Zero-shot fitting

CoLIE and Zero-IG are zero-shot methods: by default they fit a fresh network to each image, so the
per-image cost is the fit, not one forward pass. The `zero_ig` rows above time the forward pass of its
released weights (`zero_shot=false`). CoLIE has no released weights, so it has no forward row (its
`forward` takes coordinates, not an image).

`mon benchmark` now times the fit for a zero-shot model by default (`zero_shot=false` for the forward pass):

```bash
uv run mon benchmark model=colie data=testkit_efficiency device=cuda:0  # also fit_args.<name>=...
```

With it, CoLIE on all 5 images per size takes 0.89–0.98 s per fit across two runs and 2,763 GMACs per fit at
every size. Its peak memory goes from 657 MiB at 256² to 1,933 MiB at 8K, which matches the table below.

Below: one image per size (`sice_1`), each model's official fit defaults (`colie_enhance`: 100 Adam steps on a
256² downsampled Value channel; `zero_ig_enhance`: 2001 steps at native resolution), wall time with CUDA
synchronized, `seed=0`. Raw values: `fit_timing.yml` in the run folder.

| Model | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `colie` time (s) | 1.28* | 0.90 | 0.91 | 0.89 | 0.88 | 0.88 | 0.88 | 0.90 |
| `colie` peak (MiB) | 657 | 662 | 662 | 680 | 711 | 754 | 877 | 1,932 |
| `zero_ig` time (s) | 43.5 | 91.3 | 80.7 | 274 | 610 | OOM | – | – |
| `zero_ig` peak (MiB) | 383 | 1,533 | 1,397 | 5,367 | 12,057 | OOM | – | – |

\* First fit, which includes CUDA and cuDNN warm-up.

## Observations

- **Curve and illumination nets are in a class of their own.** SCI, SCI++ and Zero-DCE++/SGZ at
  `scale_factor=12` do 4K in 8–14 ms with under 1 GiB, and 8K in 30–53 ms. Only these four, FLOL and
  RUAS fit 8K in 24 GiB.
- **Zero-DCE++ at `scale_factor=1` is barely faster than Zero-DCE.** It has 7.5× fewer
  parameters and 7.9× fewer MACs (depthwise separable), yet 1080p runs 31.6 ms against 37.1 ms. It still
  goes OOM at 8K: activations dominate (32 channels at full resolution, 8 curve iterations), not weights.
  The speed only comes from the downsampled curve (`scale_factor=12`: 3.9 ms at 1080p).
- **SCI++ has 4× SCI's parameters but runs faster.** At test time only its first illumination branch runs,
  which has the same three convolutions as SCI but no batch norm, so the MACs are identical. The extra
  parameters are training-only branches.
- **Parameters predict cost poorly.** RUAS has 3.4 k parameters but runs 33 ms at 1080p (many small
  sequential ops). HVI-CIDNet's 2.0 M parameters and 249 GMACs at 1080p are slower than DCC-Net's 13.2 M
  and 2,021 GMACs (420 vs 365 ms). That fits HVI-CIDNet's attention and color-space ops being outside the
  MAC count.
- **CLODE is the outlier.** Its adaptive ODE solver needs about 10 s per 1080p image and runs out of memory at
  4K. Its MACs vary with content (1080p mean 19.5 TMACs). It is far beyond every other model, so it
  is a quality reference, not a deployable option at these sizes.
- **Zero-IG's per-image fit is impractical above 720p.** It takes 10 min and 12 GiB at 1080p, and runs out
  of memory at 1440p. Its released weights run 66 ms at 1080p instead. CoLIE's fit is resolution-independent
  (under 1 s; only the final full-resolution mapping grows).
- **The LOL resolution (600×400) understates real cost.** At 1080p, latency is 8–9× the 600×400 value for
  most full-resolution nets, matching the 8.6× pixel ratio. Below 512², launch overhead dominates for the
  small nets (SCI: 0.26 ms flat).
