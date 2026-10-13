# Bringing sHSI into LION

This note tracks the incremental integration of the A3SmartML sHSI code.

Reference implementation: `UE4sHSI`, commit `9937dcd`.
Original repository: https://github.com/trung-vt/UE4sHSI

The goal is to run PCM and sHSI experiments in LION while retaining
the measurement model and reconstruction assumptions of each application.
The source sHSI repository should remain unchanged during the migration.

## Measurement model

PCM: subsampled Hadamard measurements of a 2D image.

sHSI: measurements of entries of an `(H, W, D)` hyperspectral cube.
The source code uses a full-size binary mask and dense, zero-filled
measurements:

```text
measurement = mask * cube + noise_on_observed_entries
```

It also accepts a flattened `(N, D)` matrix. Missing observations are
identified by the mask, not by checking whether a value is zero. A valid
observed value may itself be zero.

The reference examples `test_input_0.npz` and `test_input_1.npz`
contain `(25, 42, 800)` cubes at 5% and 10% sampling.
Each NPZ has `measurement`, `mask`, `sigma`, and optional `gt`.

## Stage 1: measurement support (implemented)

- `LION/operators/MaskedObservationOp.py` provides an operator with
  `forward(x) = mask * x`, `adjoint(y) = mask * y`, and `gram(x) = mask * x`.
  It supports matrices, cubes, and leading batch dimensions. Its range
  retains the full input shape to match the team's dense format.
- `LION/data_loaders/shsi/data.py` reads the reference NPZ format without
  changing numeric dtypes. It keeps the optional ground truth separate.
- Unit tests are in `pytests/operators/test_masked_observation_op.py`
  and `pytests/shsi/test_data.py`.

Example, when both Git checkouts are in the same parent directory:

```python
import torch

from LION.data_loaders.shsi.data import load_shsi_npz
from LION.operators.MaskedObservationOp import MaskedObservationOp

sample = load_shsi_npz("../UE4sHSI/example_data/test_input_0.npz")
op = MaskedObservationOp(sample.mask)
measured = torch.from_numpy(sample.measurement)

assert torch.equal(op(measured), measured)
```

This stage does not yet reconstruct the missing entries. It also does
not require the sibling repository at runtime, except for the example
data path shown above.

## Reconstruction code to migrate next

| Reference file | Functionality | Planned LION home |
| --- | --- | --- |
| `LowRank4MatrixCompletion.py` | Pipeline, modes, hyperparameter search | `LION/shsi/reconstruct.py` |
| `LowRank4MatrixCompletion_utils.py` | Low-rank alternating least squares (ALS), updates, selection | `LION/shsi/low_rank.py` |
| `AEProject.py` | 1D spectral autoencoder | `LION/models/shsi/` |
| `ApproxGibbsHighDim.py` | Approximate block Gibbs, posterior moments | `LION/shsi/uncertainty.py` |
| `compressedftir/utils.py` | Spatial neighbour regularisation and sampling utilities | Adapt only required parts; retain provenance |

Recommended migration order:

1. Port the classical low-rank ALS reconstruction. Keep the original
   lambda values configurable. Test against the original implementation
   on the same masked cube and compare both the output and objective.
2. Add optional automatic lambda search, then the trained spectral AE
   and `dual_ae` mode.
3. Add the source `pnp_dual` method. Its denoiser acts on spectral
   factors / their product, so this is **not** automatically the same
   method as LION's image-domain PnP-ADMM.
4. Port Gibbs-based uncertainty estimates after point reconstructions
   match. Test posterior mean, variance, sampling statistics and
   ordering conventions.
5. Only then adapt general LION reconstructor/experiment interfaces
   and compare methods across applications where their input dimensions
   and priors make sense.

## Compatibility risks

- The source pipeline uses row-major (C) flattening for ALS and
  column-major (Fortran) flattening for Gibbs. Preserve the explicit
  conversion of spatial factors and check round trips.
- The spectral axis is length 800 in supplied examples. A denoiser
  trained for 2D PCM images cannot simply be reused on a 3D cube.
- The source AE checkpoint has architecture / rank / spectral-length
  assumptions. Document them before including a checkpoint in LION.
- Gibbs uses dense spatial precision matrices in some paths. Avoid
  assuming large cube runs have the same runtime or memory needs
  as the small ALS smoke test.
- `LowRank4MatrixCompletion.py` supplies a zero-valued dummy ground
  truth to the underlying ALS routine for some evaluations. Do not
  mistake those ground-truth metrics for comparisons to real truth.
- LION includes GPL-3.0 license text. The source `LICENSE` specifies
  MIT No Attribution; its README says MIT; parts of
  `compressedftir/utils.py` carry a separate BSD-like license
  and an academic citation requirement. Preserve applicable notices,
  provenance, and attribution when copying source code.
- Do not copy hard-coded data paths or notebooks as library modules.

## Validation log

- Reference ALS smoke test: random `(4, 5, 8)` cube with rank 2,
  50% observations, `sigma=0.1`, `lambda_single=0.05`, two
  iterations, `gibbs_enabled=False`, `auto_lambda_search=False`.
  Result: finite `(4, 5, 8)` reconstruction; observed-entry MSE
  `0.002021880793780764`; exit code 0.
- LION sHSI unit tests: 14 passed.
- Existing LION PCM operator tests: 7 passed.
- LION loader and mask operator reproduce `test_input_0.npz`
  measurements exactly; no changes were made in UE4sHSI.

Run the new tests with:

```bash
python -m pytest -q -o addopts='' \
    pytests/operators/test_masked_observation_op.py \
    pytests/shsi/test_data.py
```
