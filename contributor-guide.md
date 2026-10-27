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
- LION's version is the file `LION/VERSION` (`LION.__version__`; the package
  build reads it too).
- Results go to a solver's `save_folder`, by default `LION_RESULTS_PATH`
  (optional). A `save_folder` is where a machine writes, so it isn't recorded.

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
- **Components keep each constructor argument as an attribute of the same
  name** (`self.model_parameters`, `self.geometry`, ...). LION saves a component
  as the call that builds it (a spec, `LION/utils/spec.py`) from those
  attributes, so it can be built again from its file alone. A class that can't
  follow the rule defines `spec_kwargs()`.
- `X.instantiate(source)` makes a component as a saved file describes it
  (models with new weights; `X.load(source)` also restores the weights).
  Called on a base class (`LIONmodel`, `LIONDataset`, `Experiment`) it makes
  the class the file names. New base classes get the same method, through
  `LION.utils.spec.instantiate`.
- Specs keep classes, tuples, numpy arrays and enums. Parameter files saved as
  plain JSON lose them; code that reads such files converts them back, e.g.
  `as_model_parameter()` for models and `Geometry.init_from_parameter()` for
  geometries.
- Records (what LION writes about a component: `data_record`, the training
  record) are `LIONParameter`s too, read with attributes (`record.epochs.done`).
  The specs inside them stay dicts (a dict with `"_target_"` is never turned
  into a `LIONParameter`): they describe objects, made only on request
  (`instantiate`). Records are saved as plain JSON and read back as
  `LIONParameter`s.
- A parameter name that `default_parameters()` doesn't have gives a warning
  (it isn't used: likely a typo).
- A model's `.pt` holds only tensors and plain Python values (state dicts,
  epoch, loss); everything else goes in its `.json`. LION reads it with
  `torch.load(weights_only=True)`, which runs no code from the file (torch >= 2.6).

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
- Methods are evaluated with `LION.evaluation` (`Evaluation`, outside the
  solvers): every test sample, PSNR/SSIM/HaarPSI by default with one range for
  the test set, records saved and reused. The classical baseline of each
  modality is `LION.evaluation.baseline(operator)` (CT: FDK): add new
  modalities there, not on the operator; an experiment whose standard is
  another method overrides `Experiment.baseline()`. `Comparison` runs several
  methods against the baseline, after checking their records: a model trained
  or validated on test data (by patient) is an error, another problem or other
  LION code a warning.
- Solvers take a training `seed` (default 0; `None`: unseeded) and pass it to
  `LIONsolver`, which seeds torch, numpy and Python's `random` when training
  starts (`LION.utils.rng.seed_all`); checkpoints keep the random state, so a
  resumed run continues the same way. Validation doesn't change training's
  random numbers, and noise a solver adds in validation is drawn inside
  `rng.seeded(seed, ...)`, the same in every validation.
- Solvers write the record of their training (`LION/optimizers/training_record.py`)
  in every file they save. Values a solver computes during the run (e.g. an
  estimated noise level) are kept in `solver_params`, starting as `None`, so
  they are recorded and come back on resume.
- Solvers take `deterministic` (default False): torch's deterministic
  algorithms, so same-seed GPU runs give the same model (slower). Off, GPU runs
  with the same seed differ slightly.
- Every model file records the environment it was saved in
  (`LION/utils/environment.py`: LION's version, commit and uncommitted changes,
  torch/numpy/CT backends, GPU and driver, determinism settings); loading a file
  saved with other LION code warns.
- `LIONsolver.instantiate(file, model)` makes the solver from its own record;
  data, validation, testing and saving are set as usual.

## Devices

- Use `LION.utils.device.default_device()` where no device is given; don't
  hard-code `"cuda"`.

## Data and experiments

- Changes to data loading are checked against the real data (or its record),
  not only against fake data in tests.
- A CT `Geometry` is given by any two of shape, size and spacing, for the image
  (`image_shape`, `image_size`, `voxel_size`) and the detector
  (`detector_shape`, `detector_size`, `pixel_size`); the third is computed when
  read. `check_geometry()` checks a geometry when an operator is built from it
  and when it is saved.
- Simulated CT data are densities (`ct_utils.from_HU_to_density`) in mm
  geometries. Noise is added with `ct_utils.sinogram_add_noise`, whose
  physical model is the default.
- Data loaders derive from `LIONDataset` (a torch `Dataset` with
  `instantiate()`, nothing else), keep their constructor arguments under the
  same names (`mode`, `parameters`, the geometry) and list their samples
  (`sample_ids()`, `sample_files()`), so `data_record(dataset)`
  (`LION/data_loaders/data_record.py`) can describe them. Where samples come
  in groups that must stay in one split (LIDC-IDRI's patients, 2DeteCT's
  physical samples), `sample_groups()` gives each sample's, so evaluations
  check leakage by group; data are on
  `default_device()`. A noise function is a LION function or a
  `functools.partial` of one (not a lambda), so it can be recorded.
- Noise is part of the experiment, not the dataset: `LIONDataset.instantiate`
  makes a dataset without it; `Experiment.instantiate(...).get_dataset(split)`
  makes the data with it.
- Random numbers that make up the data (e.g. simulated noise) are drawn inside
  `LION.utils.rng.seeded(seed, <sample identity>)`, so each sample is the same
  every time it is read. Randomness of training (shuffling, noise a solver
  adds) is not seeded per sample. The module docstring of
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
