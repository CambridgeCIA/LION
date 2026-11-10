# LION: instructions for coding assistants

LION is a PyTorch research library for learned tomographic (and other
inverse-problem) reconstruction. Keep it **modular, light (few classes, few
dependencies) and close to plain torch**: before adding a class, check that a
function or a parameter doesn't do.

## Read first

- [`contributor-guide.md`](contributor-guide.md): the rules the code follows
  (names, parameters and backwards compatibility, operators, models, solvers,
  data, randomness, errors, docstrings, tests, PRs). Follow it.
- The developer guide, [`docs/developers.rst`](docs/developers.rst) (on the
  site: https://cambridgecia.github.io/LION/developers.html): how the pieces
  connect, one training followed through the code, and what a new model,
  solver, reconstructor, operator or dataset needs.
- [`metadata.md`](metadata.md), when a change touches what LION saves or
  records (specs, records, `instantiate`, publishing).

## Where things are

- `LION/operators/`: `Operator` and generic operators; one folder per kind,
  with its parameters, operator and reconstructions: `ct/` (`Geometry`,
  `CTProjectionOp`, FDK, SIRT, TV, noise and HU in `ct_utils`), `pcm/`.
- `LION/classical_algorithms/`: CG, FISTA, SPGL1, for any operator.
- `LION/data_loaders/`: `LIONDataset`, the datasets (`LIDC_IDRI`, `deteCT`),
  `data_record`; each dataset's download and preprocessing scripts.
- `LION/experiments/`: `Experiment`; `ct_experiments.py` (LION's simulated
  LIDC-IDRI problems), `deteCT_benchmark.py` (a paper's: don't change it).
- `LION/models/`: `LIONmodel`, `TorchModel`; models by kind: `CNNs/`,
  `post_processing/`, `iterative_unrolled/`, `learned_fbp/`,
  `learned_regularizer/`, `PnP/`, `fully_learned/`.
- `LION/optimizers/`: `LIONsolver`, the solvers, `training_record`.
- `LION/losses/`, `LION/metrics/`: losses (SURE, Jacobian penalties); PSNR,
  SSIM, HaarPSI.
- `LION/reconstructors/`: `LIONReconstructor`, `ForwardReconstructor`, `PnP`,
  `LearnedRegularizerReconstructor`.
- `LION/evaluation/`: `Evaluation`, `Comparison`, tables and figures.
- `LION/utils/`: parameters, specs, paths, device, random seeds,
  normalisation, citations, environment record, backend imports.
- `LION/package.py`: `publish`, `load`, `build`, `train_again`.
- `pytests/`: the tests, **flat** (no `tests/` folder); `helpers.py`
  (small operators and models), `conftest.py` (markers, fixtures).
- `demos/`: the tutorials (`dNN_*.py`, `# %%` cells), the site's gallery.
- `scripts/`: example scripts (`example_scripts/`) and paper scripts.
- `docs/`: the documentation site (Sphinx, furo, sphinx-gallery):
  `methods.rst` (every method, with its paper), `developers.rst`, `api.rst`.
- `utils/`: shell scripts (conda environments, `publish_docs.sh`).
- `.github/workflows/`: tests (CPU), pre-commit, docs build.

`README.md` is for users; `CHANGELOG.md` lists each release's changes (add
yours under Unreleased); `papers_in_LION.md`, `developers.md` and
`LION/models/README.md` point to the site.

## Commands

- Tests: `pytest pytests/ -n auto`. CI runs them on CPU; tests marked
  `cuda` or `tomosipo` are skipped where those are missing.
- Format: `pre-commit run --files <files>` (Black 22.10.0).
- Docs: `sphinx-build -W -b html docs docs/_build/html` (CI fails on any
  warning; docstrings are reStructuredText). Tutorials run only with
  `LION_DOCS_RUN_DEMOS=1`, on a GPU machine with the data.

## Working on LION

- Inspect only the files relevant to the change and their direct
  dependencies; follow the existing patterns; no unrelated refactoring.
- One topic per PR, against `main`. Say what changes for existing results.
- A feature comes with one test; a fix only when it changes what a feature
  promises. Run the targeted tests, not the whole suite.
- A change that alters results (data, noise, architecture) or a convention
  is a design decision: raise it before making it.
- No backwards compatibility until v1.0; from v1.0, a missing parameter
  means the old behaviour.
- Models from papers follow the paper and the authors' code.
- Comments say what the code does, not its history (history goes in commit
  messages and PRs).
- Every Python file starts with the GPL-3 licence header (see the
  contributor guide).
- Data are under `LION_DATA_PATH`, results under `LION_RESULTS_PATH`; GPU
  tests and anything with real data run on a GPU machine with the data.
