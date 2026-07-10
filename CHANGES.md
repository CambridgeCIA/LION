# Changes in LION Repository Since Forking

This document summarizes and explains all additions, modifications, and optimizations made to the `LION` library since it was forked from the upstream repository, structured according to the changed files.

---

## File Structure Tree of Changed Files

```text
LION/
├── CTtools/
│   └── ct_utils.py (modified)
├── classical_algorithms/
│   ├── __init__.py (modified)
│   ├── fbp.py (new)
│   ├── sirt.py (modified)
│   └── tv_min.py (modified)
├── data_loaders/
│   └── LIDC_IDRI.py (modified)
├── metrics/
│   ├── haarpsi.py (modified)
│   ├── psnr.py (modified)
│   └── ssim.py (modified)
└── models/
    └── score_inverse/ (new, MAIN)
        ├── .gitignore
        ├── __init__.py
        ├── ema.py
        ├── fst.py
        ├── layer.py
        ├── loss.py
        ├── ncsnpp.py
        ├── sampling.py
        ├── sde.py
        ├── sirt_adj.py
        └── utils.py
scripts/
└── score_inverse_scripts/ (new, MAIN)
    ├── configs.py
    ├── test.py
    ├── train.py
    └── validation.py
```

---

## Main Changes


### 1. `LION/models/score_inverse/` (new)

* **`.gitignore`, `__init__.py`**
  - Standard repository structure maintenance and exports.
* **`ema.py`**
  - Implemented the Exponential Moving Average (EMA) parameter tracker class for stabilizing model training and applying shadow weights during validation or inference.
* **`fst.py`**
  - Implemented direct and inverse parallel-beam Radon transforms using the Fourier Slice Theorem. Includes customizable k-space padding (expansion factor) to prevent interpolation errors.
* **`layer.py`**
  - Built convolutional layers, combine blocks for U-Net skip-connections, FIR up/down-samplers, and self-attention layers with choices between `einsum` and `bmm` computation.
* **`loss.py`**
  - Built the `SMLoss` class implementing Denoising Score Matching loss weighted across uniformly sampled time steps. Supports deterministic noise generation via custom PyTorch generators.
* **`ncsnpp.py`**
  - Built the Noise Conditional Score Network (NCSN++) architecture backbone, managing features across multi-resolution levels and conditioning them with Gaussian Fourier noise embeddings.
* **`sampling.py`**
  - Implemented Predictor-Corrector loops (`pc_sampler` and the JAX-equivalent `pc_sampler_new`), Tweedie denoising, conditional score modifications (`get_score_conditional`), and exact frequency-space or preconditioned right-inverse hijacking APIs (`get_hijack`, `get_hijack_new`).
* **`sde.py`**
  - Defined abstract SDEs, simple forward linear SDEs, simple reverse-time SDEs, and the Variance Exploding SDE (VESDE) transition distributions.
* **`sirt_adj.py`**
  - Implemented the `SIRTAdj` preconditioned adjoint backprojection operator ($x = C A^T R y$) using volume ($C$) and projection ($R$) scaling factors to serve as a stable pseudo-inverse for arbitrary projection operators.
* **`utils.py`**
  - Provided utilities for global random seed settings, checkpoint loading/applying for evaluation, and batch-wise application of projection operators.

### 2. `scripts/score_inverse_scripts/` (new)

* **`configs.py`**
  - Declares sparse geometries (parallel & fan), simulated projection noise presets, SDE bounds, and reference checkpoint paths.
* **`test.py`**
  - A command-line evaluation runner that executes reconstructions on sinograms, sweeps over time steps ($N$) and hijacking weights ($\lambda$), and logs runtime execution speed and peak GPU/RSS memory usage.
* **`train.py`**
  - Orchestrates multi-GPU DistributedDataParallel (DDP) model training using Denoising Score Matching, EMA shadow updates, and checkpoint saving.
* **`validation.py`**
  - Executes seed-controlled deterministic validation sweeps on saved checkpoints to calculate loss metrics.

## Side Changes

### 1. `LION/CTtools/`

* **`ct_utils.py` (modified)**
  - **Device Independence**: Replaced hardcoded CUDA/GPU current device queries in `_sinogram_add_noise` with `.device` attributes, supporting CPU-only runtimes gracefully.
  - **Dynamic Input Handling**: Added a check (`torch.is_tensor(proj)`) in `sinogram_add_noise` to avoid throwing attribute errors when invoking `.detach()` on NumPy arrays.
  - **Arbitrary Batch Reshaping**: Rewrote the 2D cross-talk convolution step using `.view(-1, 1, H, W)` to dynamically support batch-applied projection inputs with arbitrary leading dimensions, replacing the hardcoded single-batch `.unsqueeze(0)[0]` format.

### 2. `LION/classical_algorithms/`

* **`__init__.py` (modified)**
  - Imported and exposed the new `fbp` module under `__all__` alongside iterative solvers.
* **`sirt.py` (modified)**
  - Fixed a critical batch iteration bug where the entire multi-batch sinogram tensor `sino` was mistakenly passed to the underlying `ts_sirt(op, sino, ...)` call instead of the slice `sino[i]`.
* **`tv_min.py` (modified)**
  - Fixed a similar batch iteration bug where `sino` (full multi-batch tensor) was passed to `ts_tv_min(op, sino, ...)` instead of the individual batch slice `sino[i]`.
* **`fbp.py` (new)**
  - Added a Filtered Backprojection (FBP) baseline implementation for parallel and fan-beam geometries, implementing slice-by-slice batched reconstruction with optional non-negativity clipping.

### 3. `LION/data_loaders/`

* **`LIDC_IDRI.py` (modified)**
  - **New Task Mode**: Added `"image_only"` task mode support to retrieve only the raw lung CT slice tensor, skipping Radon transform simulations and speed-up training setups.
  - **CPU-Safety Fallback**: Added a fallback in default parameter configurations to set `param.device` to `"cpu"` if `torch.cuda.is_available()` is false, avoiding runtime exceptions on non-GPU instances.
  - **Dynamic Slice Count**: Overrode slice gathering behavior when `num_slices_per_patient == -1` to dynamically fetch, combine, and sort all nodule and non-nodule slices, replacing the legacy hardcoded cap of 1000.

### 4. `LION/metrics/`

* **`haarpsi.py` (modified)**
  - Added the missing `super().__init__()` call in the constructor of the `HAARPsi` class.
* **`psnr.py` (modified)**
  - **Dynamic Data Range**: Added support for an optional user-specified `data_range` parameter, removing the constraint where data range had to be calculated from `target.max() - target.min()`.
  - **Dimension Safety**: Delayed squeezing operations on batch elements (`xi = x_[i].squeeze()`) to inside the loops, resolving shape mismatch issues when dealing with single-batch inputs.
* **`ssim.py` (modified)**
  - **Squeeze Isolation**: Avoided early squeezing of batch dimensions which led to shape mismatches when batch size was 1 or when dimensions were squeezed unpredictably.
  - **Warning Triggers**: Added explicit warnings when squeezed shapes conflict with the requested `channel_axis`.
  - **Offset Correction**: Adjusted channel axis offset (`curr_channel_axis -= 1`) for squeezed batch elements to keep structural evaluations stable.
  - **Static Data Range**: Added support for an optional static `data_range` parameter.

