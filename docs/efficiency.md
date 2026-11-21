# Efficiency: LLE models across resolutions

Parameters, MACs, latency and peak memory of every low-light enhancement model
in `mon`, on the `testkit_efficiency` ladder (5 SICE scenes per size, 256² to
8K). Measured 2026-11-19 at commit `f610dfb`.

## Setup

- RTX 3090 (24 GiB), driver 595.99.02, CUDA 13.2, torch 2.15.0, Python 3.14,
  i7-7700; batch 1, float32, `torch.inference_mode`, no `torch.compile`, no AMP.
- Command per model (released default weights; `seed=0`):

  ```bash
  uv run mon benchmark model=<name> weights=default data=testkit_efficiency device=cuda:0 seed=0 \
      output=$DATA_DIR/runs/benchmark/efficiency_2026_11_19/<name>.yaml
  ```

  Timing: 5 untimed passes on the first image of a size, then 20 timed passes
  per image (CLODE: 1 and 3). Latency is the median over all timed passes.
  Memory is peak `torch.cuda.max_memory_allocated` over the images, so it
  excludes the CUDA context (~300 MiB) and the allocator's cache. MACs come from
  `FlopCounterMode` (FLOPs / 2). It counts matmuls and convolutions only, so
  element-wise work is not included (curve iterations, Retinex divisions,
  attention softmax).
- Latencies below 512² are launch-bound for the small nets and move between runs
  (FLOL at 256²: 7.8 and 12.8 ms in two runs); from 720p up, two runs agree
  within about 2%.
- Raw YAML/CSV per model: `$DATA_DIR/runs/benchmark/efficiency_2026_11_19/`.
- `_s12` rows: Zero-DCE++ and SGZ with `model_args.scale_factor=12`, their
  official test setting. The curve is estimated on a 12× downsampled input and
  upsampled. Plain rows use the default `scale_factor=1`.
- Rows are sorted by 1080p latency. "OOM" means the forward pass ran out of the
  24 GiB.

## Forward pass

### Median latency (ms)

| Model | Params | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sci++` | 22,898 | 0.281 | 0.282 | 0.29 | 0.934 | 1.93 | 3.44 | 7.52 | 29.8 |
| `sci` | 5,877 | 0.3 | 0.383 | 0.35 | 1.45 | 3.18 | 5.65 | 12.6 | 49.6 |
| `zero_dce++_s12` | 10,561 | 0.898 | 0.894 | 0.889 | 2.07 | 3.81 | 6.22 | 13.6 | 53 |
| `sgz_s12` | 10,561 | 1.17 | 0.934 | 0.94 | 2.09 | 3.92 | 6.34 | 13.4 | 53.1 |
| `flol` | 94,518 | 12.8 | 14 | 13.7 | 15.9 | 27 | 45.1 | 101 | 407 |
| `zero_dce++` | 10,561 | 1.13 | 4 | 3.69 | 14.1 | 31.4 | 55.9 | 126 | OOM |
| `sgz` | 10,561 | 1.12 | 3.95 | 3.62 | 14.1 | 31.5 | 55.9 | 126 | OOM |
| `ruas` | 3,438 | 3.4 | 4.44 | 4.23 | 15.1 | 33.5 | 59.3 | 133 | 554 |
| `zero_dce` | 79,416 | 1.02 | 3.65 | 3.32 | 16.5 | 36.4 | 63.3 | 141 | OOM |
| `zero_ig` | 86,572 | 2.6 | 8.57 | 7.72 | 29.4 | 65.5 | 116 | 262 | OOM |
| `retinexnet` | 555,205 | 3.61 | 14 | 12.5 | 47.6 | 106 | 192 | OOM | OOM |
| `pairlie` | 341,767 | 3.66 | 14.7 | 12.9 | 50.1 | 112 | 202 | 448 | OOM |
| `dccnet` | 13,188,724 | 12.7 | 45.5 | 42.1 | 162 | 367 | 641 | 1,440 | OOM |
| `hvi_cidnet` | 1,975,569 | 24.3 | 56 | 52 | 186 | 417 | 738 | 1,666 | OOM |
| `clode` | 207,105 | 378 | 1,554 | 1,223 | 4,567 | 10,170 | 18,086 | OOM | OOM |

### Peak CUDA memory (MiB)

| Model | Params | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sci++` | 22,898 | 3 | 12 | 11 | 42 | 96 | 169 | 380 | 1,520 |
| `sci` | 5,877 | 3 | 12 | 11 | 42 | 96 | 169 | 380 | 1,520 |
| `zero_dce++_s12` | 10,561 | 4 | 16 | 15 | 58 | 131 | 230 | 518 | 2,071 |
| `sgz_s12` | 10,561 | 4 | 16 | 15 | 58 | 131 | 230 | 518 | 2,071 |
| `flol` | 94,518 | 27 | 108 | 98 | 376 | 844 | 1,494 | 3,358 | 13,430 |
| `zero_dce++` | 10,561 | 82 | 326 | 301 | 1,148 | 2,583 | 4,584 | 10,317 | OOM |
| `sgz` | 10,561 | 82 | 326 | 301 | 1,148 | 2,583 | 4,584 | 10,317 | OOM |
| `ruas` | 3,438 | 22 | 87 | 80 | 310 | 692 | 1,224 | 2,755 | 11,016 |
| `zero_dce` | 79,416 | 71 | 283 | 262 | 1,306 | 2,940 | 5,218 | 11,740 | OOM |
| `zero_ig` | 86,572 | 51 | 202 | 185 | 710 | 1,591 | 2,827 | 6,363 | OOM |
| `retinexnet` | 555,205 | 159 | 629 | 577 | 2,202 | 4,946 | 8,785 | OOM | OOM |
| `pairlie` | 341,767 | 36 | 201 | 186 | 704 | 1,579 | 2,804 | 6,305 | OOM |
| `dccnet` | 13,188,724 | 122 | 338 | 317 | 1,177 | 2,585 | 4,550 | 10,177 | OOM |
| `hvi_cidnet` | 1,975,569 | 90 | 332 | 304 | 1,145 | 2,567 | 4,546 | 10,221 | OOM |
| `clode` | 207,105 | 167 | 658 | 602 | 2,300 | 5,170 | 9,186 | OOM | OOM |

### GMACs (mean over the 5 images)

| Model | Params | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sci++` | 22,898 | 0.0159 | 0.0637 | 0.0583 | 0.224 | 0.504 | 0.896 | 2.02 | 8.06 |
| `sci` | 5,877 | 0.0159 | 0.0637 | 0.0583 | 0.224 | 0.504 | 0.896 | 2.02 | 8.06 |
| `zero_dce++_s12` | 10,561 | 0.00444 | 0.0178 | 0.0166 | 0.0641 | 0.145 | 0.258 | 0.58 | 2.32 |
| `sgz_s12` | 10,561 | 0.00444 | 0.0178 | 0.0166 | 0.0641 | 0.145 | 0.258 | 0.58 | 2.32 |
| `flol` | 94,518 | 1.02 | 4.08 | 3.74 | 14.4 | 32.3 | 57.4 | 129 | 517 |
| `zero_dce++` | 10,561 | 0.66 | 2.64 | 2.42 | 9.29 | 20.9 | 37.1 | 83.6 | OOM |
| `sgz` | 10,561 | 0.66 | 2.64 | 2.42 | 9.29 | 20.9 | 37.1 | 83.6 | OOM |
| `ruas` | 3,438 | 0.214 | 0.856 | 0.784 | 3.01 | 6.77 | 12 | 27.1 | 108 |
| `zero_dce` | 79,416 | 5.19 | 20.8 | 19 | 73 | 164 | 292 | 657 | OOM |
| `zero_ig` | 86,572 | 10.5 | 41.9 | 38.4 | 147 | 331 | 589 | 1,326 | OOM |
| `retinexnet` | 555,205 | 21.7 | 86.9 | 79.6 | 305 | 687 | 1,222 | OOM | OOM |
| `pairlie` | 341,767 | 22.3 | 89.4 | 81.8 | 314 | 707 | 1,257 | 2,828 | OOM |
| `dccnet` | 13,188,724 | 63.9 | 256 | 234 | 898 | 2,021 | 3,593 | 8,085 | OOM |
| `hvi_cidnet` | 1,975,569 | 7.88 | 31.5 | 28.9 | 111 | 249 | 443 | 997 | OOM |
| `clode` | 207,105 | 655 | 2,923 | 2,262 | 8,950 | 19,541 | 33,678 | OOM | OOM |

## Zero-shot fitting

CoLIE and Zero-IG are zero-shot methods: by default they fit a fresh network to
each image, so the per-image cost is the fit, not one forward pass. The
`zero_ig` rows above time the forward pass of its released weights. That needs
`zero_shot=false`, because `mon benchmark` defaults to the fit for zero-shot
models. CoLIE has no released weights, so it has no forward row (its `forward`
takes coordinates, not an image).

`mon benchmark` now times the fit for a zero-shot model by default
(`zero_shot=false` for the forward pass):

```bash
uv run mon benchmark model=colie data=testkit_efficiency device=cuda:0  # also fit_args.<name>=...
```

With it (`colie_fit` in the run folder, 5 images per size), CoLIE's median fit
is 0.88–0.91 s at every size, with 2,763 GMACs per fit. Its peak memory goes
from 657 MiB at 256² to 1,933 MiB at 8K. Both match the table below.

Below: one image per size (`sice_1`), each model's official fit defaults
(`colie_enhance`: 100 Adam steps on a 256² downsampled Value channel;
`zero_ig_enhance`: 2001 steps at native resolution), wall time with CUDA
synchronized, `seed=0`. Raw values: `fit_timing.yml` in the run folder.

| Model | 256² | 512² | 600×400 | 720p | 1080p | 1440p | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `colie` time (s) | 1.28* | 0.90 | 0.91 | 0.89 | 0.88 | 0.88 | 0.88 | 0.90 |
| `colie` peak (MiB) | 657 | 662 | 662 | 680 | 711 | 754 | 877 | 1,932 |
| `zero_ig` time (s) | 43.5 | 91.3 | 80.7 | 274 | 610 | OOM | – | – |
| `zero_ig` peak (MiB) | 383 | 1,533 | 1,397 | 5,367 | 12,057 | OOM | – | – |

\* First fit, which includes CUDA and cuDNN warm-up.

## Observations

- **Curve and illumination nets are in a class of their own.** SCI, SCI++ and
  Zero-DCE++/SGZ at `scale_factor=12` do 4K in 8–14 ms with under 1 GiB, and 8K
  in 30–53 ms. Only these four, FLOL and RUAS fit 8K in 24 GiB.
- **Zero-DCE++ at `scale_factor=1` is barely faster than Zero-DCE.** It has 7.5×
  fewer parameters and 7.9× fewer MACs (depthwise separable), yet 1080p runs
  31.4 ms against 36.4 ms. It still goes OOM at 8K: activations dominate (32
  channels at full resolution, 8 curve iterations), not weights. The speed only
  comes from the downsampled curve (`scale_factor=12`: 3.9 ms at 1080p).
- **SCI++ has 4× SCI's parameters but runs faster.** At test time only its first
  illumination branch runs, which has the same three convolutions as SCI but no
  batch norm, so the MACs are identical. The extra parameters are training-only
  branches.
- **Parameters predict cost poorly.** RUAS has 3.4 k parameters but runs 33 ms
  at 1080p (many small sequential ops). HVI-CIDNet's 2.0 M parameters and 249
  GMACs at 1080p are slower than DCC-Net's 13.2 M and 2,021 GMACs (417 vs 367
  ms). That fits HVI-CIDNet's attention and color-space ops being outside the
  MAC count.
- **CLODE is the outlier.** Its adaptive ODE solver needs about 10 s per 1080p
  image and runs out of memory at 4K. Its MACs vary with content (1080p mean
  19.5 TMACs). It is far beyond every other model, so it is a quality reference,
  not a deployable option at these sizes.
- **Zero-IG's per-image fit is impractical above 720p.** It takes 10 min and 12
  GiB at 1080p, and runs out of memory at 1440p. Its released weights run 66 ms
  at 1080p instead. CoLIE's fit is resolution-independent (under 1 s; only the
  final full-resolution mapping grows).
- **The LOL resolution (600×400) understates real cost.** At 1080p, latency is
  8–9× the 600×400 value for most full-resolution nets, matching the 8.6× pixel
  ratio. Below 512², launch overhead dominates for the small nets (SCI: about
  0.3 ms flat).
