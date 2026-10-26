# Contributor guide

LION is a research library: it should stay modular, light (few classes, few
dependencies) and as close to plain PyTorch as possible, so that methods are
easy to read and to change. These are the conventions the code follows. They
apply to people and to coding assistants alike (`CLAUDE.md` points here).

## Setting up

- Create the environment from `env_base.yml` (conda or mamba; see the README),
  then `pip install -e ".[dev]"`.
- **Everything compiled comes from conda, torch included** (`pytorch-gpu`). A pip
  torch loads the system's C++ runtime, and conda's scipy etc. then fail on
  machines with an older one (`CXXABI_... not found`). pip installs LION and
  pure-Python packages only.
- Pip-only installs (and CI) use the `standalone` extra.
- Datasets live under `LION_DATA_PATH` (see the README and
  `LION/data_loaders/README.md`).

## Tests

- Tests live in `pytests/`, **flat**, one file per topic. Shared helpers go in
  `pytests/helpers.py`, fixtures in `pytests/conftest.py`.
- Every fix or feature comes with a test, and a fix with a test that fails
  without it.
- Run them with `pytest pytests/`. `pytest-xdist` comes with `[dev]`; without
  it, add `-o addopts=""`.
- Tests that need tomosipo/astra or CUDA are marked `@pytest.mark.tomosipo` or
  `@pytest.mark.cuda` and are skipped where those are missing. CI runs on CPU
  only, so run the GPU tests yourself when you touch CT code.
- Test CT logic without tomosipo where possible (e.g. a matrix operator from
  `pytests/helpers.py`, or the `import_without_tomosipo` fixture).
- Format with Black (`pre-commit install`).

## Imports and dependencies

- **LION imports without its backends.** Each backend is imported only where it
  is used, through its own module:
  - CT: `LION/utils/tomosipo_import.py` (`import_tomosipo`, for tomosipo and
    ts_algorithms);
  - MRI: `LION/utils/mrpro_import.py` (`import_mrpro`);
  - Walsh-Hadamard transform: `LION/utils/spyrit_import.py` (`import_spyrit`).

  No module imports one of them at the top. `pytests/test_imports.py` checks
  this.
- **A dependency is something LION imports.** Packages used only by example
  scripts go in the `examples` extra, only by tests in `tests`.
- No torchvision in LION's code: image operations are plain torch
  (`LION/utils/image_transforms.py`).

## Parameters and backwards compatibility

- Models, solvers, experiments and data loaders take a `LIONParameter`, with a
  static `default_parameters()`.
- **A parameter that is missing means the old behaviour.** When a change alters
  results (a data split, a noise model, a model's architecture), add a
  parameter for it whose default in `default_parameters()` is the new
  behaviour, and read it with `getattr(params, "name", <old behaviour>)`. That
  way parameter files saved before keep reproducing their results. The
  exception is when the old behaviour was simply wrong; then say so in the
  docstring.
- Parameters saved as JSON lose their classes. Code that reads them converts
  them back, e.g. `as_model_parameter()` for models and
  `Geometry.init_from_parameter()` for geometries.

## Models

- Models derive from `LIONmodel` and take `(model_parameters, geometry=None)`,
  with the CT geometry as `geometry`.
- A model from a paper follows the paper and the authors' code. Where it
  deviates, the docstring says so.
- Models accept a plain `LIONParameter` (converted by `as_model_parameter()`), so
  a model can build another from nested parameters, and loading works at any
  depth.

## Solvers, losses and metrics

- Losses and metrics take `(output, target)`.
- Metrics (`PSNR`, `SSIM`, `HAARPsi`) share one interface:
  `metric(x, target, reduce=None, batched=True, data_range=None)` returns one
  score per sample. Scores set `higher_is_better = True`, and the solvers keep
  the model with the best validation value accordingly.
- Validation and testing go through `solver._forward()`, which for most
  solvers is `LIONsolver.model_forward()`: FDK for image-domain models, plus
  the model's normalisation.
- Self-supervised losses that are only derived for the MSE use the MSE whatever
  loss is given, and warn (`warn_if_not_mse`).
- Reconstruction is separate from training (`LION/reconstructors`) and uses
  `op.adjoint`, not autograd.

## Devices

- Use `LION.utils.device.default_device()` where no device is given; don't
  hard-code `"cuda"`.

## Data and experiments

- Changes to data loading are checked against the real data (or its record),
  not only against fake data in tests.
- Simulated CT data are densities (`ct_utils.from_HU_to_density`) in mm
  geometries. Noise is added with `ct_utils.sinogram_add_noise`, whose
  physical model is the default. The module docstring of
  `LION/experiments/ct_experiments.py` documents the experiments' noise, doses
  and geometries, and how they changed.
- Data loaders expect damaged files (interrupted downloads or copies) and leave
  them out with a warning rather than failing.
- `ct_benchmarking_experiments` reproduces a paper and is not changed.

## Pull requests

- One topic per PR, against `main`, with tests. Say in the description what
  changes for existing results.
- Analysis scripts written to make a decision are removed before merging
  (they stay in the branch history).
- New files start with the licence header (GPL-3, as ASTRA and tomosipo
  require).
