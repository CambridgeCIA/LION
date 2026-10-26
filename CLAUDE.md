# LION

LION is a research library for learned tomographic (and other inverse-problem)
reconstruction in PyTorch. Keep it modular, light and close to plain torch.

Read `contributor-guide.md` before changing code: it has the conventions
(tests in flat `pytests/`, backends imported only where used, a missing
parameter means the old behaviour, models follow their papers, conda for
anything compiled, `(output, target)` order, the metric interface).

- Run the tests with `pytest pytests/ -o addopts=""` (or install `[dev]`).
- CI is CPU only; CT code also needs the GPU tests, run on a machine with
  CUDA and astra.
- The experiments' noise, doses and geometries are documented in
  `LION/experiments/ct_experiments.py`; the data loaders' docstrings document
  their data.
