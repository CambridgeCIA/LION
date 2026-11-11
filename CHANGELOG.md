# Changelog

What changes in each release of LION. Each release has the sections below
that apply; the newest is first. Until a release, changes go under
**Unreleased** (the template, at the end of this file, is copied there after
each release).

## Unreleased: 1.0.0

LION's first stable release. Since 0.2.0, most of LION has been reviewed,
fixed and made consistent, and it gained what research needs to be repeated:
records, evaluation and publishing. From 1.0, a model or experiment saved with
1.0 loads in any 1.x, and a parameter missing from an older file means the
old behaviour. Until 1.0, nothing was kept backwards compatible: files saved
with 0.2.0 don't load.

### Added

- **Records** (`metadata.md`; #271–#298, #311, #312):
  - every saved model is a spec (the call that makes it again), read with
    torch's safe loader;
  - the data record (experiment, dataset, samples), the training record
    (solver, optimizer, scheduler, seed, losses, initial weights) and the
    environment record (LION's code, packages, machine);
  - `instantiate()` and `parameters_from_file()` on every base class;
  - reconstructors save their own record; a fine-tuned model records its
    original.
- **Evaluation** (`LION.evaluation`; #282–#284, #370): `Evaluation` on a test
  set, `Comparison` against a baseline (with checks for data leakage by
  patient, another problem or other code), tables and figures.
- **Publishing** (`LION.package`; #292, #295): `publish` a trained model with
  its experiment, data, solver and test; `load`, `build`, `train_again`.
- **Reproducible randomness** (#277, #280, #291): simulated noise seeded per
  sample; a training seed; initial weights that can be made again;
  `deterministic` training.
- **Operators** (#327–#333): operator parameters make their operator
  (`operator_params` everywhere; for CT, the geometry); each operator gives
  its default reconstruction; `autograd()`; splitting of the measurements for
  Noise2Inverse and Sparse2Inverse with any operator; `LION.operators.ct` and
  `LION.operators.pcm`.
- **Reconstructors** (#286–#288): made with operator, model and parameters,
  then run; `ForwardReconstructor`; PnP with HQS, a starting image and a
  weight relative to the operator (#229, #252, #375).
- **Training** (#341, #342, #347): learning-rate schedulers;
  `model_regularization` for every solver, with Jacobian penalties for PnP
  denoisers; `GaussianDenoiserSolver` trained as in the literature.
- **Models**: `TorchModel`, any torch module as a LION model (#316); GSD with
  any denoiser (#315, #317, #323); a model's own initial reconstruction
  (#321); `DRUNet.pretrained()`, the authors' weights (#365); a model's
  normalisation is a parameter (#287); `cite()` on every component from a
  paper (#289).
- **CT**: a geometry from any two of shape, size and spacing, checked (#271);
  the physical noise model by default, with photon counting and cross-talk
  (#248, #256); LIDC-IDRI resampled to one pixel spacing (#264); the
  experiments run on the measured 2DeteCT data (#255); the 2DeteCT benchmark
  paper's experiments (`deteCT_benchmark`, #335).
- **Documentation** (#348–#384): a docstring for every public definition; the
  site (https://cambridgecia.github.io/LION/) with 14 tutorials (the last on
  operators other than CT and sparse recovery), the API reference, "Methods
  in LION", "Datasets" (getting and preparing the data), "Records"
  (`metadata.md`) and the developer guide; eight example scripts; the
  contributor guide; `AGENTS.md`; this changelog and `CITATION.cff`.
- **Paper scripts**: the noise paper's (Kiss et al., 2025; from #103): its
  eight denoisers' training and evaluation, with its U-Net.

### Changed

- One interface for each kind of piece: models take the operator's
  parameters first (#340); solvers take one argument order (#341); metrics
  take `(x, target, reduce, batched, data_range)` (#220, #261); `verbose` 0,
  1 or 2 (#339); Python's exceptions (#344).
- `default_parameters()` lists every parameter; records say the value used,
  not "the default" (#330).
- One default device, following torch's (#223, #224); loaders return samples
  on it (#324).
- LION imports without tomosipo, mrpro or spyrit (#212, #262); no
  torchvision or kornia; dependencies trimmed; compiled packages from conda
  (#238, #257, #267).
- Tests in `pytests/`, one test per feature, run by CI on every pull request
  (#218, #239, #354).
- Models fixed against their papers: LPD, cLPD, LG, ItNet, UNet,
  cFBPConvNet, the learned FBPs, E2I (#226–#228, #251, #253, #254, #263).
- The cLPD paper's scripts rewritten for LION's current API: a training per
  model and problem, and the evaluation of each problem.

### Fixed

Many bugs, found by four reviews (#212–#270) and since; the PRs list them.
Among them: saving parameters (#219), PSNR/SSIM with batch size 1 (#220),
conjugate gradient stopping (#222), the 2DeteCT splits (#245), LIDC-IDRI
patients left out (#249), validation keeping the best model for scores
(#247), `DebiasOp` (#275).

### Removed

- FBPMSDNet (#325), GSDRUNet (#315), `CTNoiseExperiment` (#337; the noise
  paper's data are a paper script), the LUNA16 code (#301), the hackathon
  scripts (#310), `setup.py` (#306), `Geometry.default_geo` (#351), the
  empty `LION.models.fully_learned`.

## 0.2.0 (30 March 2026)

The last release before the 1.0 work; see the GitHub release.

<!--
Template for the next release: copy it under the heading of the release,
keep the sections that apply, one line per change with its PR.

## Unreleased: X.Y.Z

What the release is about, in a sentence or two, and what it changes for
existing results or saved files.

### Added
### Changed
### Fixed
### Removed
-->
