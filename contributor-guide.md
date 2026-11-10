# Contributor guide

The rules LION's code follows, for people and coding assistants alike
(`AGENTS.md` and `CLAUDE.md` point here). How LION is built, and what a new model, solver,
reconstructor, operator or dataset needs, is in the
[developer guide](https://cambridgecia.github.io/LION/developers.html); what
LION records, in [metadata.md](metadata.md).

LION is a research library: modular, light (few classes, few dependencies)
and as close to plain PyTorch as possible, so that methods are easy to read
and to change. Before adding a class, check that a function or a parameter
doesn't do.

- [Setting up](#setting-up)
- [Imports and dependencies](#imports-and-dependencies)
- [Names](#names)
- [Parameters, specs and backwards compatibility](#parameters-specs-and-backwards-compatibility)
- [Operators](#operators)
- [Models](#models)
- [Solvers](#solvers)
- [Losses and metrics](#losses-and-metrics)
- [Reconstructors and evaluation](#reconstructors-and-evaluation)
- [Data and experiments](#data-and-experiments)
- [Randomness](#randomness)
- [Messages, errors, devices](#messages-errors-devices)
- [Docstrings and documentation](#docstrings-and-documentation)
- [Tests](#tests)
- [Pull requests](#pull-requests)

## Setting up

- The environment comes from `env_base.yml` (conda or mamba; see the README),
  then `pip install -e ".[dev]"`.
- **Everything compiled comes from conda, torch included** (`pytorch-gpu`). A
  pip torch loads the system's C++ runtime, and conda's scipy etc. then fail
  on machines with an older one (`CXXABI_... not found`). pip installs LION and
  pure-Python packages only. Pip-only installs (and CI) use the `standalone`
  extra.
- Data live under `LION_DATA_PATH`, results under `LION_RESULTS_PATH`. Code
  reads them when a path is used (`LION.utils.paths`: `data_path(...)`,
  `results_path(...)`, the named folders), never at import.
- LION's version is the file `LION/VERSION` (`LION.__version__`).

## Imports and dependencies

- **LION imports without its backends.** Each is imported only where it is
  used, through its own module: `LION/utils/tomosipo_import.py` (tomosipo,
  ts_algorithms), `mrpro_import.py`, `spyrit_import.py`. No module imports one
  at the top (`pytests/test_imports.py`).
- A dependency is something LION imports. Packages used only by example
  scripts go in the `examples` extra, only by tests in `tests`.
- No torchvision: image operations are plain torch
  (`LION/utils/image_transforms.py`).
- Plots are plain matplotlib on `Figure`s (no pyplot state, no LION styling,
  `ax=` to draw into the user's figure), imported only in
  `LION/evaluation/figures.py`.

## Names

- **General words in general code, the operator's own in its code.** Code for
  any operator says `operator_params`, domain (the image), range (the
  measurements, `y`), and types them `OperatorParameters`/`Operator`. CT-only
  code says `Geometry`, `image_shape`, angles, sinogram.
- Where a general argument meets users, its docstring says what it is for CT
  ("for CT, the geometry"). The README's glossary has the general words.
- One name per thing: no aliases (a `measurement_shape` next to
  `range_shape`).
- Losses and metrics take `(output, target)`, in that order.

## Parameters, specs and backwards compatibility

- Models, solvers, reconstructors, experiments and data loaders take a
  `LIONParameter`, with a static `default_parameters()`.
- **`default_parameters()` lists every parameter that can be set**, each with
  a value: the defaults and the documentation of what can be set. A model from
  a paper defaults to the paper's.
- A parameter name `default_parameters()` doesn't have gives a warning (it
  isn't used: likely a typo).
- **Records say what happened, not "the default".** A value known only later
  is a placeholder replaced by the real value when known: `"default"` for a
  reconstruction becomes the operator's default function, `fdk` for CT
  (`LION.utils.parameter.resolve_reconstruction`); a value a solver estimates
  starts as `None` in `solver_params` and is set to what it estimated.
- **A missing parameter means the old behaviour** (from v1.0; until then no
  backwards compatibility, and parameters are read as `params.name`). A change
  that alters results (a data split, a noise model, an architecture) adds a
  parameter whose default in `default_parameters()` is the new behaviour, read
  with `getattr(params, "name", <old behaviour>)`. The exception is when the
  old behaviour was simply wrong: then the docstring says so.
- **Components keep each constructor argument as an attribute of the same
  name** (`self.model_parameters`, `self.operator_params`, ...): LION saves a
  component as the call that builds it (a spec, `LION/utils/spec.py`). A class
  that can't follow this defines `spec_kwargs()`.
- Each base class has `X.instantiate(source, **arguments)` (made as a file
  describes it; on a base class, the class the file names) and
  `X.parameters_from_file(source)`, through `LION.utils.spec`. A new base
  class gets them too. The argument that holds the parameters is
  `model_parameters` (models), `solver_params`, `experiment_params`,
  `parameters` (data loaders, reconstructors).
- Specs keep classes, tuples, numpy arrays and enums; plain-JSON parameter
  files lose them, and code that reads such files converts them back
  (`as_model_parameter()`, `Geometry.init_from_parameter()`).
- Records (`data_record`, the training record) are `LIONParameter`s, read with
  attributes; the specs inside them stay dicts (a dict with `"_target_"` is
  never turned into a `LIONParameter`). Records are saved as plain JSON.
- A model's `.pt` holds only tensors and plain Python values; everything else
  goes in its `.json`. LION reads it with `torch.load(weights_only=True)`.
- A component from a paper or a dataset defines `cite(cite_format="bib")`
  with `LION.utils.citation.citation` (BibTeX, or "MLA"): where it comes from,
  not the citation of a result made with it.

## Operators

- **Each kind of operator has its own folder**, `LION/operators/<kind>/`
  (`ct/`, `pcm/`): its parameters, operator, noise and its own
  reconstructions. Generic operators stay in `LION/operators/`, generic
  algorithms in `LION/classical_algorithms/`.
- A kind's parameters derive from `OperatorParameters` and make it
  (`make_operator()`); its operator derives from `Operator` and gives its
  default reconstruction as a function a record can name
  (`default_reconstruction_fn()`).
- Components that need an operator take its parameters (`operator_params`)
  and make it, so they work with any kind. Code for any operator asks the
  operator (`op.default_reconstruction(y)`) rather than importing a kind's
  folder.
- An operator takes one sample; batches go through `op.autograd()`.
- **Images and measurements have a singleton channel**, as CT's: an image is
  `(1, H, W)`, measurements `(1, ...)` (CT: `(1, angles, detectors)`), and a
  batch `(B, 1, ...)`. Models, solvers, reconstructors and data loaders rely
  on it.
- **`autograd()`** is the operator's author saying it can be differentiated:
  an operator of torch operations returns `(self, self.adjoint)`; one torch
  can't see into wraps itself in autograd functions (CT: tomosipo's
  `to_autograd`). The base class raises. Don't wrap an operator outside its
  own folder.
- Splitting the measurements (Noise2Inverse, Sparse2Inverse) goes through the
  parameters: `split_count`, `split(indices)`, `split_measurements(y,
  indices)`. A kind that can't be split doesn't define them.
- A CT `Geometry` is given by any two of shape, size and spacing, for the image
  and the detector; the third is computed. `check_geometry()` runs when an
  operator is made and when it is saved.

## Models

- Models derive from `LIONmodel` (any torch module: `TorchModel`).
- **A model that uses an operator takes `(operator_params,
  model_parameters=None)`**; one that doesn't, `(model_parameters=None)`.
  Gradients through the operator: `self._make_operator(autograd=True)`, then
  `self.A`, `self.AT`.
- `model_parameters.model_input_type`: what `forward` takes (measurements or
  an image).
- Models accept a plain `LIONParameter` (`as_model_parameter()`), so a model
  can build another from nested parameters.
- **A model from a paper follows the paper and the authors' code**; where it
  deviates, its docstring says so.
- **Models don't normalise inside `forward()`**: whoever calls them does. The
  normalisation (`set_normalisation()`) is a model parameter, saved with it;
  measurements are scaled by the images' factor.
- Initial weights are drawn from torch's CPU generator; the model keeps the
  random state it was made with (`initial_random_state`).

## Solvers

- Solvers derive from `LIONsolver` and take `(model, optimizer, loss_fn,
  operator_params, solver_params, model_regularization, scheduler,
  scheduler_step, verbose, device, seed, deterministic)`, in that order
  (`ARSolver` has no `loss_fn`). Where results go is `set_saving()`.
- A new solver defines `mini_batch_step(y, target)` (through `self._forward`
  and `self._regularized`) and `default_parameters()`.
- Validation and testing go through `solver._forward()`
  (`model_forward()`); a solver whose method reconstructs differently
  overrides it.
- Solvers don't hold reconstructors: one needed for validation is made outside
  and given, `set_validation(..., reconstructor=...)`.
- `model_regularization(model, step)`: a term added to the loss; every solver
  passes its own mini-batch in `step` (`input`, `output`, `target`,
  `forward`).
- Self-supervised losses derived only for the MSE use the MSE whatever loss is
  given, and warn (`warn_if_not_mse`).
- Every file a solver saves has the training record
  (`LION/optimizers/training_record.py`). Checkpoints keep the optimizer,
  scheduler and random state.

## Losses and metrics

- Metrics (`PSNR`, `SSIM`, `HAARPsi`) share one interface:
  `metric(x, target, reduce=None, batched=True, data_range=None)`, one score
  per sample; scores set `higher_is_better = True`.

## Reconstructors and evaluation

- Reconstruction is separate from training (`LION/reconstructors`) and uses
  `op.adjoint`, not autograd.
- **A reconstructor is made with everything that decides its result, then
  run:** `PnP(operator, model, parameters).reconstruct(y)`. `reconstruct`
  takes only the data, a starting image `x0` and `verbose`. A new one derives
  from `LIONReconstructor`, defines `default_parameters()` and
  `reconstruct_sample(y, x0, verbose)`, and reads `self.parameters`.
- Its spec holds the operator and the parameters, not the model;
  `instantiate(source, model)` makes it again.
- Methods are evaluated with `LION.evaluation` (`Evaluation`, `Comparison`),
  outside the solvers. The baseline is the operator's default reconstruction;
  an experiment whose standard is another method overrides
  `Experiment.baseline()`.
- `LION.package.publish` publishes only what LION alone can make again: LION's
  classes (and torch's layers, losses, optimizers), from committed and pushed
  code.

## Data and experiments

- Data loaders derive from `LIONDataset`, keep their arguments under the same
  names (`mode`, `parameters`, the geometry), and list their samples
  (`sample_ids()`, `sample_files()`, and `sample_groups()` where samples must
  stay in one split: patients, physical samples). Data are on
  `default_device()`.
- **Noise is part of the experiment, not the dataset.** A noise function is a
  LION function or a `functools.partial` of one (not a lambda).
- Simulated CT data are densities (`ct_utils.from_HU_to_density`) in mm
  geometries; noise is added with `ct_utils.sinogram_add_noise`.
- Changes to data loading are checked against the real data (or its record),
  not only against fake data.
- Data loaders leave out damaged files with a warning rather than failing.
- The module docstring of `LION/experiments/ct_experiments.py` documents the
  experiments' noise, doses and geometries, and how they changed.
- `LION/experiments/deteCT_benchmark.py` reproduces a paper: its parameters
  are not changed.

## Randomness

- Random numbers that make up the data (simulated noise) are drawn inside
  `LION.utils.rng.seeded(seed, <sample identity>)`: each sample is the same
  every time it is read.
- Randomness of training (shuffling, noise a solver adds) comes from the
  solver's `seed` (`rng.seed_all` when training starts), not per sample.
  Validation doesn't change training's random numbers; noise it adds is drawn
  inside `rng.seeded(...)`.

## Messages, errors, devices

- **`verbose`**, as in Keras: 0 nothing; 1 (default) one progress bar per long
  run; 2 also inner bars and messages. True is 1, False 0. Solvers take it
  when made; functions and `reconstruct` as an argument. A call inside
  another's progress bar gets one level less. A component without `verbose`
  prints nothing (download and preprocessing scripts excepted).
- Something the user should know is a `warnings.warn`, whatever `verbose` is.
- Wrong input raises `ValueError` or `TypeError`; a call before what it needs
  (saving before `set_saving()`) `RuntimeError`. A LION exception only for a
  case callers catch on its own, deriving from the matching built-in
  (`SpecError(ValueError)`).
- **Wrong input raises, not `assert`** (`python -O` removes asserts).
- `LION.utils.device.default_device()` where no device is given; never a
  hard-coded `"cuda"`.

## Docstrings and documentation

- Public classes and functions have a docstring: what it is or does, in prose,
  then its arguments as ``` ``name``: what it is ```. No `Args:` sections or
  type lists: the types are in the signature.
- A component from a paper says so, and where it deviates.
- **Docstrings are reStructuredText**: the site is built from them, and CI
  fails on anything Sphinx can't read. A list needs a blank line before it; a
  literal block a `::` before it.
- A new method gets a line in the site's Methods page (`docs/methods.rst`);
  a feature users will look for, a section in a tutorial (`demos/`) or an
  example script.
- Tutorials (`demos/dNN_*.py`) say "Previous"/"Next", what they need, and
  that their models are toys; data loaders on their own lines.

## Tests

- **Flat in `pytests/`**, one file per topic; helpers in
  `pytests/helpers.py`, fixtures in `pytests/conftest.py`.
- **A feature comes with one test**, from start to end, rather than one per
  detail; extend the feature's test before adding a new one. A test has to be
  worth the time it takes to run and to keep up to date.
- **A fix comes with a test only when it changes what a feature promises**
  (what records hold, what loads, the maths): add the check to that feature's
  test. Other fixes are checked by hand: say how in the PR.
- Tests needing tomosipo/astra or CUDA are marked `@pytest.mark.tomosipo` or
  `@pytest.mark.cuda`. CI runs on CPU only: run the GPU tests yourself when
  you touch CT code. Test CT logic without tomosipo where possible (a matrix
  operator from `helpers.py`, the `import_without_tomosipo` fixture).
- `pytest pytests/ -n auto`; format with Black (`pre-commit install`).

## Pull requests

- One topic per PR, against `main`. Say what changes for existing results,
  and add a line for it to `CHANGELOG.md` under Unreleased.
- Analysis scripts written to make a decision are removed before merging.
- Every Python file starts with the licence header (GPL-3, as ASTRA and
  tomosipo require), with its authors (`pytests/test_licence_headers.py`):

  ```
  # =============================================================================
  # This file is part of LION library
  # License : GPL-3
  #
  # Author  : ...
  # =============================================================================
  ```
