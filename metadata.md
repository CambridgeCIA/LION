# LION's metadata: results that can be made again

Every model LION saves carries the information needed to make it again: the
model, the data it was trained on and how they were simulated, how it was
trained, and the code and machine it ran on. From these records LION can
make any component again, change it and train again, evaluate and compare
models fairly, and publish a model as a self-contained package. This page
describes what is recorded, how to use it, and what a new component needs to
take part.

- [The model file](#the-model-file)
- [Specs: components written as calls](#specs-components-written-as-calls)
- [The records](#the-records)
- [Repeating a training exactly](#repeating-a-training-exactly)
- [Making things again from a file](#making-things-again-from-a-file)
- [Evaluating and comparing](#evaluating-and-comparing)
- [Publishing a model](#publishing-a-model)
- [Using a package](#using-a-package)
- [Folders: LION_DATA_PATH and LION_RESULTS_PATH](#folders-lion_data_path-and-lion_results_path)
- [For developers: making a component take part](#for-developers-making-a-component-take-part)
- [What isn't covered](#what-isnt-covered)

## The model file

A model is saved as two files with the same name, as a solver saves it
(`solver.save_final_results()`) or by hand (`model.save(file, dataset=...,
training=...)`):

- **`model.json`**: the metadata, plain JSON written by LION (read-only in
  intent):
  - `lion_format`: the metadata format's version;
  - `model`: the model's spec (its class and parameters, see below);
  - `environment`: the environment record (LION's code, packages, machine);
  - `dataset_params`: the data record of the training data;
  - `training`: the training record;
- **`model.pt`**: tensors and plain values only, read with torch's safe loader
  (`weights_only=True`), so loading a file runs no code from it:
  - `model_state_dict`: the weights;
  - `initial_random_state`: torch's random state when the model was made (see
    [Repeating a training exactly](#repeating-a-training-exactly));
  - for checkpoints, also the optimizer's state, the epoch, the losses and the
    random state of the run.

A shortened `model.json` of a DnCNN trained on a LIDC-IDRI experiment:

```json
{
 "lion_format": 1,
 "model": {
  "_target_": "LION.models.CNNs.dncnn.DnCNN",
  "model_parameters": {"_target_": "LION.models.LIONmodel.LIONModelParameter",
                       "in_channels": 1, "int_channels": 64, "blocks": 20, "...": "..."}
 },
 "environment": {"lion": {"version": "0.2.0", "commit": "a69eb0d6...", "uncommitted": [], "pushed": true},
                 "packages": {"torch": "2.14.1", "...": "..."}, "device": {"type": "cuda", "...": "..."}},
 "dataset_params": {
  "experiment": {"_target_": "LION.experiments.ct_experiments.LowDoseCTRecon", "...": "..."},
  "split": "train",
  "dataset": {"_target_": "LION.data_loaders.LIDC_IDRI.LIDC_IDRI", "mode": "train", "...": "..."},
  "sinogram_transform": {"_target_": "functools.partial", "I0": 11000, "...": "..."},
  "sinogram_transform_seed": 0,
  "samples": {"count": 4, "ids": ["LIDC-IDRI-0001/slice_0", "..."], "groups": ["LIDC-IDRI-0001", "..."]}
 },
 "training": {
  "solver": {"_target_": "LION.optimizers.SupervisedSolver.SupervisedSolver", "seed": 0, "...": "..."},
  "optimizer": {"_target_": "torch.optim.adam.Adam", "lr": 0.001, "...": "..."},
  "seed": 0,
  "initial_weights": {"from": "random_state"},
  "train": {"batch_size": 2, "shuffle": true, "drop_last": false},
  "epochs": {"asked": 2, "done": 2},
  "weights": {"kind": "final", "epoch": 2},
  "history": {"train_loss": [0.35, 0.33], "validation": [0.22, 0.21]}
 }
}
```

Loading it: `model, options, data = LIONmodel.load("model")` makes the class
the file names, with its weights; `DnCNN.load("model")` makes a `DnCNN`.
Loading a file saved with other LION code gives a warning saying how the code
differs.

## Specs: components written as calls

LION writes a component as the call that makes it (`LION/utils/spec.py`), in
JSON: a dict with `"_target_"` means "call this", its other keys are the
keyword arguments (`"_args_"`: positional ones), and `"_call_": false` refers
to a class or function without calling it. The keys follow Hydra's
`instantiate` convention.

```json
{"_target_": "LION.experiments.ct_experiments.LowDoseCTRecon",
 "experiment_params": {"_target_": "LION.utils.parameter.LIONParameter", "name": "...", "...": "..."},
 "dataset": "LIDC-IDRI"}
```

- **Writing.** A component's spec is made from its attributes with the names
  of its constructor's arguments (the *same-name rule*: LION's components keep
  each argument as an attribute of the same name; a class that can't defines
  `spec_kwargs()`). Tuples, numpy arrays, enums, paths, partials and classes
  are written so they come back as they were. A lambda, or a class that keeps
  no record of its arguments, can't be written: records then only *name* it
  (`{"unsaved": ..., "reason": ...}`), with a warning, and it can't be made
  again.
- **Building.** A spec names code to run, so only classes in allowed modules
  are imported and called: LION's, torch's layers, losses and optimizers, and
  a few builtins (`spec.ALLOWED_MODULES`, `ALLOWED_TARGETS`). Your own classes
  are allowed by passing their module: `allow=["my_package."]`. A file never
  widens what may be built; only the caller does.

## The records

Records are what LION writes about a component. They are `LIONParameter`s
(read with attributes, `record.epochs.done`), saved as plain JSON; the specs
inside them stay dicts, describing objects that are made only when asked.

| Record | Written by | Holds | Read with |
|---|---|---|---|
| **Data record** (`LION/data_loaders/data_record.py`) | `data_record(dataset)`, `experiment.data_record(split)`; saved as a model file's `dataset_params` | the dataset's spec; the experiment and split, when made by one; the noise (`sinogram_transform`) and its seed; every sample's ID, their count and hash, and their groups (e.g. LIDC-IDRI's patients); a `Subset`'s indices; on request, a hash of the files' contents | `read_record(source)` |
| **Training record** (`LION/optimizers/training_record.py`) | every solver, in every file it saves (final, best validation, checkpoint) | the solver's spec (its own arguments, `solver_params` with values it computed); the optimizer; the seed; how the initial weights are made again, and their hash; how data were fed (batch size, shuffle); validation (its data record, frequency, function, reconstructor); epochs asked and done; which weights the file holds; the loss history | `read_training_record(source)` |
| **Environment record** (`LION/utils/environment.py`) | every model save, and every test | LION's version, commit, uncommitted files and whether the commit is pushed; the packages that change the numbers (torch, numpy, CT backends); CUDA, cuDNN and determinism settings; device, GPUs, driver; platform. Nothing personal (no host, user or paths) | `options.environment` |
| **Test record** (`LION/evaluation/evaluation.py`) | `Evaluation.run(method)` | the method (e.g. a model's spec and weights hash); the evaluation's spec; the test data's record; the metrics; per-sample scores by sample ID; summary statistics; timing; the environment | `read_test_record(source)` |

## Repeating a training exactly

A training is repeated exactly when its data, initial weights and random
numbers are the same:

- **Data.** Simulated noise is drawn per sample from the experiment's noise
  seed (`noise_params.seed`, default 0), inside
  `LION.utils.rng.seeded(seed, <sample>)`: each sample is the same in every
  epoch, run and data loader worker.
- **Initial weights.** A model keeps torch's random state from when it was
  made (`initial_random_state`, saved in the `.pt`). When training starts, the
  solver checks that the model made again from that state has the same
  weights, and records `{"from": "random_state"}`. If not (a pretrained or
  custom start, layers made on a GPU), it saves the weights next to the
  results (`<name>_initial.pt`) and records `{"from": "file", ...}`.
  `LIONmodel.instantiate(file, initial_weights=True)` makes them again.
- **Training's random numbers.** Solvers take a `seed` (default 0; `None`:
  unseeded) and seed torch, numpy and Python's `random` when training starts.
  Checkpoints keep the random state, so a resumed run continues exactly as the
  uninterrupted one would. Validation doesn't change training's random numbers.
- **Determinism.** On the CPU, the same code and package versions give the same
  model. On a GPU, runs with the same seed differ slightly (about 1e-3 in the
  weights after a few epochs) unless the solver is made with
  `deterministic=True` (slower). torch's CPU and GPU random generators differ,
  so a seed gives different (equally valid) noise on each.

## Making things again from a file

Each component makes itself again from a saved file (a model file, a data,
training or test record, a spec, or a package's files):

| | made again | its parameters, to change | made with changes |
|---|---|---|---|
| Model | `LIONmodel.instantiate(file)` (new weights); `LIONmodel.load(file)` (trained weights); `instantiate(file, initial_weights=True)` | `LPD.parameters_from_file(file)` | `LPD.instantiate(file, model_parameters=parameters)` |
| Experiment | `Experiment.instantiate(file)` | `Experiment.parameters_from_file(file)` | `Experiment.instantiate(file, experiment_params=parameters)` |
| Dataset | `LIONDataset.instantiate(file)` (without noise; checked against the record's samples) | `LIDC_IDRI.parameters_from_file(file)` | `LIDC_IDRI.instantiate(file, parameters=parameters)` |
| Solver | `LIONsolver.instantiate(file, model)` | `LIONsolver.parameters_from_file(file)` | `LIONsolver.instantiate(file, model, solver_params=parameters)` |
| Reconstructor | `LIONReconstructor.instantiate(spec_or_test_record, model)` | `PnP.parameters_from_file(source)` | `PnP.instantiate(source, model, parameters=parameters)` |
| Evaluation | `Evaluation.instantiate(test_record)` | | `Evaluation.instantiate(test_record, samples=10)` |

- Called on a base class (`LIONmodel`, `Experiment`, ...) these make the class
  the file names; called on a class, that class, with a warning if the file
  names another.
- `parameters_from_file` gives what `default_parameters()` gives, but as the
  file recorded it: start from an old experiment and change one or two things.

  ```python
  parameters = LPD.parameters_from_file("results/lpd/final")
  parameters.n_iters = 5
  model = LPD.instantiate("results/lpd/final", model_parameters=parameters)  # new weights
  ```

- `instantiate(file, **arguments)` replaces any recorded argument.
- An experiment's data, noise included, are made with
  `Experiment.instantiate(file).get_dataset(record.split)`. The experiment
  classes differ only in their default parameters, so an experiment made with
  another experiment's parameters gives a warning: it would be that experiment,
  recorded under the wrong class.

## Evaluating and comparing

`LION.evaluation` scores reconstruction methods on a test set and keeps the
results as test records:

```python
from LION.evaluation import Comparison, Evaluation

evaluation = Evaluation(LowDoseCTRecon())            # the experiment's test split, PSNR/SSIM/HaarPSI
record = evaluation.run("results/lpd/final")          # a model file, a solver, a reconstructor or a function
comparison = Comparison(evaluation, ["results/lpd/final", "results/fbpconvnet/final"])
comparison.run()                                      # the baseline (FDK for CT) first
comparison.save("comparison")                         # records, tables, per-sample scores, figures
```

- Records are saved (in `$LION_RESULTS_PATH/evaluations` by default) and
  reused when the same method is run on the same evaluation again.
- From the models' records, a comparison checks that it is fair: a model
  trained or validated on test data is an error (checked by patient for
  LIDC-IDRI), a model trained on another problem or with other LION code gives
  a warning.
- `Comparison.instantiate(folder)` makes a saved comparison again.

## Publishing a model

`LION.package.publish` writes everything needed to make a model, train it
again and test it, in one folder:

```python
from LION.package import publish
publish("results/lpd/final", "lpd_package", evaluation=evaluation)  # evaluation: optional
```

```
lpd_package/
  model.json, model.pt   the model file, exactly as LION saved it
  <name>_initial.pt      the initial weights, when the solver saved them
  experiment.json        the experiment
  data/training.json     the training data's record
  data/validation.json   the validation data's record, if validated
  solver.json            the training record
  test.json              the test record, if an evaluation was given
  environment.txt        the packages installed where it was published (conda's and Python's)
  package.json           the files' sha256, the citations, LION's version and commit, notes
  README.md              a model card, generated (edit it freely: it isn't checksummed)
```

- **It refuses** (an error, before writing anything):
  - LION code that isn't committed and pushed, now and when the model was saved
    and tested;
  - a file without a training or data record;
  - records naming code outside LION, or what could only be named (a lambda);
  - what training again would miss: unsaved initial weights, `seed=None`, data
    not made by an `Experiment`, or data outside `LION_DATA_PATH`;
  - a test of other weights, or on data the model was trained or validated on;
  - a folder that isn't empty.
- **Notes** (in `package.json` and the model card): a package published in
  another environment than it was trained in, or tested on another problem.
- **Citations** are each component's `cite()`: where the model, solver and
  dataset come from (their papers), not the work being published.
- **The model card** describes the model, the problem, the training and the
  test, and shows how to use and train the model again, step by step.

## Using a package

```python
from LION.package import build, load, load_and_build, train_again

package = load("lpd_package")        # the recorded arguments, to change; nothing made yet
package.model.model_parameters.n_iters = 5
package.experiment.experiment_params.noise_params.I0 = 5000
package.optimizer.lr = 1e-4
package.training.epochs = 20

built = build(package)               # model, experiment, data, solver (data set), evaluation
built = load_and_build("lpd_package")        # as published
model = train_again(package, "retrained")    # trained again, changed or not
```

- `load` warns about files changed since publishing, and about LION code other
  than the model was saved with.
- `build` gives the model its trained weights, or with `initial_weights=True`
  the ones training started from; a model whose arguments were changed starts
  from new random weights.
- Everything runs on the device the model was trained on unless `device=` says
  otherwise (on the CPU if it was a GPU and there is none).
- Unchanged, `train_again` gives the published weights (on the CPU, or on a GPU
  with a deterministic solver).

## Folders: LION_DATA_PATH and LION_RESULTS_PATH

- `LION_DATA_PATH` is where the datasets are; `LION_RESULTS_PATH` (optional) is
  where solvers save by default. Both are read when a path is used.
- Paths inside them are written relative to them, as `data_path("...")` and
  `results_path("...")`, so a file means the same folder on any machine. A path
  outside them is written as it is, and `publish` refuses data outside
  `LION_DATA_PATH`.
- Where a machine saves (`save_folder`) and where it runs (`device`,
  `verbose`) are not recorded.

## For developers: making a component take part

A new model, dataset, experiment, solver or reconstructor takes part when:

- it derives from LION's base class (`LIONmodel`, `LIONDataset`, `Experiment`,
  `LIONsolver`, `LIONReconstructor`), which gives it `instantiate`,
  `parameters_from_file` and its records;
- it keeps each constructor argument as an attribute of the same name
  (`self.model_parameters`, `self.geometry`), or defines `spec_kwargs()`;
- its settings are a `LIONParameter` with a static `default_parameters()`; a
  setting added later reads `getattr(parameters, "name", <old behaviour>)`, so
  files saved before keep giving their results;
- it is made only from LION's classes and functions (a `functools.partial` of
  a LION function, not a lambda), so its spec can be built again;
- **datasets** list their samples (`sample_ids()`, and `sample_files()`), and
  `sample_groups()` where samples come in groups that must stay in one split;
  random numbers that make up the data are drawn inside `rng.seeded(...)`;
- **solvers** keep values they compute during a run in `solver_params`
  (starting as `None`), so they are recorded and come back on resume;
- it defines `cite()` if it comes from a paper.

The conventions are in `contributor-guide.md`. Where the code is:

| | |
|---|---|
| specs, building, `allow` | `LION/utils/spec.py` |
| the data record | `LION/data_loaders/data_record.py` |
| the training record | `LION/optimizers/training_record.py`, `LION/optimizers/LIONsolver.py` |
| the environment record | `LION/utils/environment.py` |
| random numbers | `LION/utils/rng.py` |
| model files | `LION/models/LIONmodel.py` |
| evaluation, comparison, figures | `LION/evaluation/` |
| publishing and packages | `LION/package.py` |
| folders | `LION/utils/paths.py` |

## What isn't covered

- **The data themselves**: records identify the data (sample IDs, how they
  were simulated) but don't contain them, and a package doesn't either (they
  are large and licensed by others). A file-content hash can be asked for
  (`data_record(dataset, hash_files=True)`).
- **Code outside LION**: specs can name it (with `allow=`), but LION only
  records its own code's commit, and only LION's classes can be published.
- **Bitwise GPU results** without `deterministic=True`, or across different
  versions of torch, CUDA or the hardware.
- **Files saved before this system** (October 2026) load as before, with
  whatever they recorded.
