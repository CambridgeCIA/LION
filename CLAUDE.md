# LION

LION is a research library for learned tomographic (and other inverse-problem)
reconstruction in PyTorch. Keep it modular, light and close to plain torch.

Read `contributor-guide.md` before changing code: it has the conventions
(tests in flat `pytests/`, backends imported only where used, a missing
parameter means the old behaviour, models follow their papers, conda for
anything compiled, `(output, target)` order, the metric interface).

# Project Instructions

## Project Structure

LION is a modular Python/PyTorch research library for learned tomographic reconstruction and inverse problems.

### Main Package: `LION/`

- `LION/classical_algorithms/`: Classical reconstruction for any operator: FISTA, conjugate gradient, SPGL1.
- `LION/data_loaders/`: Datasets (`LIONDataset` and the loaders derived from it), their data records, and per-dataset download and preprocessing scripts (`LIDC_IDRI/`, `2deteCT/`, `walnuts/`).
- `LION/evaluation/`: Testing trained methods: `Evaluation`, `Comparison`, tables and figures.
- `LION/exceptions/`: LION's exceptions.
- `LION/experiments/`: Experiments: geometry, noise, doses and data for each benchmark problem.
- `LION/losses/`: Loss functions (SURE).
- `LION/metrics/`: Image quality metrics (PSNR, SSIM, HaarPSI).
- `LION/models/`: Neural networks and learned reconstruction methods (`LIONmodel` and the models derived from it), by type: `CNNs/`, `post_processing/`, `iterative_unrolled/`, `learned_regularizer/`, `PnP/`, `fully_learned/`, `learned_fbp/`.
- `LION/operators/`: Forward operators. Generic ones here (`Operator`, wavelets, composite and debiasing operators); each kind of operator has its own folder with its parameters, operator and reconstructions:
  - `ct/`: computed tomography: the geometry (`Geometry`, the CT operator's parameters), `CTProjectionOp`, simulated sinogram noise and HU conversions (`ct_utils`), FDK, SIRT and TV minimisation.
  - `pcm/`: photocurrent mapping: `PhotocurrentMapOp`, its `Subsampler`, the Walsh-Hadamard transform.
- `LION/optimizers/`: Training: `LIONsolver` and the solvers derived from it (supervised, self-supervised, adversarial, ...), and the training record.
- `LION/reconstructors/`: Reconstructing with trained models (`LIONReconstructor`): PnP, learned regulariser, forward.
- `LION/utils/`: Shared utilities: parameters, specs, paths, device, random seeds, normalisation, citations, environment record.
- `LION/package.py`: Publishing and loading trained models (`publish`, `load`, `build`, `train_again`).

### Supporting Directories

- `pytests/`: Automated tests, flat. Keep tests in this directory rather than creating a new `tests/` directory.
- `demos/`: Demonstrations and usage notebooks.
- `scripts/`: Standalone scripts: example scripts and paper scripts.
- `utils/`: Repository-level shell scripts for saving and updating conda environments.
- `misc/`: Miscellaneous files (e.g. environment exports from holly, the lab's GPU machine).
- `.github/workflows/`: CI workflows (CPU tests, pre-commit).

### Documentation and Configuration

- `README.md`: Project overview and installation.
- `CLAUDE.md`: AI development instructions.
- `contributor-guide.md`: Coding conventions and contribution guidelines.
- `metadata.md`: The metadata and reproducibility system (records, specs, publishing).
- `papers_in_LION.md`: Papers whose methods LION implements.
- `developers.md`: Developer documentation.
- `pyproject.toml`: Python project and package installation configuration.
- `env_base.yml`: Conda environment configuration.

## Development efficiency

- Inspect only files relevant to the requested change and their direct dependencies.
- Avoid exploring unrelated modules or the entire repository.
- Follow existing implementations and patterns.
- Avoid unrelated refactoring.
- Run targeted tests rather than the full test suite.
- Keep responses concise and avoid repeating established context.
